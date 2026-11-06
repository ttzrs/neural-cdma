"""Prototipo CDMA SELECTIVO con evaluación held-out independiente (wikitext validation).

Copia de prototype_selective.py: LoRA entrena SOLO con train. Sorpresa en K=1,
terciles, recovery por K y retención se calculan separadamente en train y held-out.
Los umbrales de tercil se estiman sobre el propio conjunto evaluado, no sobre train
para aplicarlos a validation. Por tanto, son estratos relativos dentro de cada split.
Las ventanas se comparten entre K y las semillas coinciden entre splits.
"""

from __future__ import annotations

import itertools
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

dev = torch.device("cuda")
torch.manual_seed(0)
MODEL = "Qwen/Qwen2.5-0.5B"
T, BS, STEPS, KMAX = 32, 8, 4000, 8
LORA_R, LORA_ALPHA = 32, 32
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
EVAL_KS = [1, 2, 4, 8]
N_EVAL_BATCHES, EVAL_SEED = 32, 10000
EVAL_WINDOWS_PER_BATCH = 64


class LoRALinear(nn.Module):
    def __init__(self, lin, r, alpha):
        super().__init__()
        self.base = lin
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.randn(r, lin.in_features) * 0.02)
        self.B = nn.Parameter(torch.zeros(lin.out_features, r))
        self.s = alpha / r

    def forward(self, x):
        return self.base(x) + ((x @ self.A.t()) @ self.B.t()) * self.s


tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
base.config.use_cache = False
base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
for p in base.parameters():
    p.requires_grad = False
for mod in base.modules():
    for cn, child in list(mod.named_children()):
        if isinstance(child, nn.Linear) and cn in TARGETS:
            setattr(mod, cn, LoRALinear(child, LORA_R, LORA_ALPHA))
base.to(dev)
D, VOCAB = base.config.hidden_size, base.config.vocab_size
qwen, lm_head, embed = base.model, base.lm_head, base.get_input_embeddings()
codes = torch.sign(torch.randn(KMAX, D, device=dev)); codes[codes == 0] = 1

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
buf = [r["text"] for r in itertools.islice(iter(ds), 4000) if len(r["text"]) > 1]
ids_all = torch.tensor(tok(" ".join(buf)).input_ids, device=dev)
ds_val = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation", streaming=True)
buf_val = [r["text"] for r in ds_val if len(r["text"]) > 1]
ids_val = torch.tensor(tok(" ".join(buf_val)).input_ids, device=dev)
N = ids_all.numel()
print(f"tokens={N} heldout_tokens={ids_val.numel()} d={D}")


def windows(source, n, generator=None):
    starts = torch.randint(0, source.numel() - T - 1, (n,), device=dev, generator=generator)
    off = torch.arange(T + 1, device=dev)
    seqs = source[(starts.unsqueeze(-1) + off)]
    return seqs[..., :T], seqs[..., 1:]


def forward(x):                                       # x:(bs,K,T)
    K = x.shape[1]; c = codes[:K].view(1, K, 1, D)
    z = (embed(x) * c).sum(1)
    h = qwen(inputs_embeds=z).last_hidden_state
    return lm_head(h.unsqueeze(1) * c)                # (bs,K,T,vocab)


def train():
    opt = torch.optim.AdamW([p for p in base.parameters() if p.requires_grad], lr=2e-4)
    base.train()
    for s in range(STEPS):
        kcur = 1 + int((KMAX - 1) * min(1.0, (s / STEPS) / 0.75))
        K = int(torch.randint(1, kcur + 1, (1,)))
        x, y = windows(ids_all, BS * K)
        x = x.view(BS, K, T)
        loss = F.cross_entropy(forward(x).reshape(-1, VOCAB), y.view(BS, K, T)[:, :, :].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 1000 == 0:
            print(f"  step {s} K={K} loss={loss.item():.3f}", flush=True)


@torch.no_grad()
def stratified_eval(source):
    base.eval()
    # 32 recovery batches por K, 64 ventanas por lote (agrupadas según K).
    M = N_EVAL_BATCHES * EVAL_WINDOWS_PER_BATCH
    generator = torch.Generator(device=dev).manual_seed(EVAL_SEED)
    x_all, y_all = windows(source, M, generator=generator)
    surp = torch.zeros(M, T, device=dev)
    for i in range(0, M, 32):
        xb = x_all[i:i+32].unsqueeze(1)
        lp = F.log_softmax(forward(xb)[:, 0], dim=-1)
        surp[i:i+32] = -lp.gather(-1, y_all[i:i+32].unsqueeze(-1)).squeeze(-1)
    # Terciles sobre TODOS los tokens del split evaluado, también en held-out.
    flat = surp.flatten()
    q1, q2 = torch.quantile(flat, torch.tensor([1/3, 2/3], device=dev))
    res = {"surprise_terciles": [round(float(q1), 2), round(float(q2), 2)], "by_K": {}}
    for K in EVAL_KS:
        ng = M // K
        xg = x_all[:ng*K].view(ng, K, T)
        yg = y_all[:ng*K].view(ng, K, T)
        sg = surp[:ng*K].view(ng, K, T)
        rec = torch.zeros(ng, K, T, device=dev)
        gb = max(1, EVAL_WINDOWS_PER_BATCH // K)
        for i in range(0, ng, gb):
            pred = forward(xg[i:i+gb]).argmax(-1)
            rec[i:i+gb] = (pred == yg[i:i+gb]).float()
        rflat, sflat = rec.flatten(), sg.flatten()
        easy = rflat[sflat <= q1].mean()
        mid = rflat[(sflat > q1) & (sflat <= q2)].mean()
        hard = rflat[sflat > q2].mean()
        res["by_K"][K] = {"acc_easy": round(float(easy), 3), "acc_mid": round(float(mid), 3),
                          "acc_hard": round(float(hard), 3), "acc_all": round(float(rflat.mean()), 3)}
        print(f"  K={K}: FÁCIL={float(easy):.3f}  MEDIO={float(mid):.3f}  DIFÍCIL={float(hard):.3f}  (todos={float(rflat.mean()):.3f})")
    # Prior-only ajustado SOLO con train; se puntúa sobre el split evaluado.
    const = int(torch.bincount(ids_all, minlength=VOCAB).argmax())
    rprior = (y_all == const).float().flatten()
    sflat = surp.flatten()
    res["prior_only"] = {
        "token": const,
        "acc_easy": round(float(rprior[sflat <= q1].mean()), 3),
        "acc_mid": round(float(rprior[(sflat > q1) & (sflat <= q2)].mean()), 3),
        "acc_hard": round(float(rprior[sflat > q2].mean()), 3),
        "acc_all": round(float(rprior.mean()), 3)}
    print(f"  prior-only(tok={const}): FÁCIL={res['prior_only']['acc_easy']:.3f}  "
          f"DIFÍCIL={res['prior_only']['acc_hard']:.3f}")
    base_e, base_m, base_h = (res["by_K"][1]["acc_easy"], res["by_K"][1]["acc_mid"], res["by_K"][1]["acc_hard"])
    res["retention_vs_K1"] = {K: {
        "easy": round(res["by_K"][K]["acc_easy"] / base_e, 3) if base_e > 0 else None,
        "mid": round(res["by_K"][K]["acc_mid"] / base_m, 3) if base_m > 0 else None,
        "hard": round(res["by_K"][K]["acc_hard"] / base_h, 3) if base_h > 0 else None,
    } for K in EVAL_KS}
    res["k_cap_rel80"] = max([0] + [K for K in EVAL_KS
                                     if res["by_K"][K]["acc_all"] >= 0.8 * res["by_K"][1]["acc_all"]])
    print("  retención fáciles vs K=1:", {K: res["retention_vs_K1"][K]["easy"] for K in EVAL_KS})
    return res


def evaluate():
    return {split: stratified_eval(source)
            for split, source in (("train", ids_all), ("heldout", ids_val))}


def main():
    train()
    print("\n=== Recovery estratificado por dificultad: train y held-out ===")
    res = evaluate()
    res.update({"n_eval_batches": N_EVAL_BATCHES, "eval_window_seed": EVAL_SEED,
                "eval_windows_per_batch": EVAL_WINDOWS_PER_BATCH,
                "n_eval_windows": N_EVAL_BATCHES * EVAL_WINDOWS_PER_BATCH,
                "n_tokens": {"train": N, "heldout": ids_val.numel()},
                "dataset": "Salesforce/wikitext/wikitext-2-raw-v1",
                "heldout_split": "validation",
                "tercile_thresholds": "estimated independently on each evaluated split using K=1 surprise",
                "prior_only_fit_split": "train",
                "k_cap_rel80_heldout": res["heldout"]["k_cap_rel80"]})
    with open("prototype_selective_heldout_result.json", "w") as f:
        json.dump(res, f, indent=2)
    print("→ prototype_selective_heldout_result.json")


if __name__ == "__main__":
    main()
