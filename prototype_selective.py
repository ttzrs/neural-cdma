"""Prototipo CDMA SELECTIVO: ¿la dificultad por token (sorpresa) predice su multiplexabilidad?

Valida la pieza central de la teoría: D_task es HETEROGÉNEA. Si los tokens fáciles (baja sorpresa) se
recuperan a K alto mientras los difíciles colapsan, multiplexar SOLO los fáciles rompe el K_cap=2 uniforme.

Qwen2.5-0.5B + wikitext, fine-tune LoRA para multiplexar. Luego, sobre ventanas test:
  - K=1: sorpresa de cada token target = -log p(target).
  - K∈{2,4,8}: multiplexar y medir recovery (argmax==target) por token.
  - Estratificar accuracy(K) por TERCIL de sorpresa (fácil/medio/difícil).
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
EVAL_KS = [1, 2, 4, 8]   # K=1 = baseline estratificado (referencia para saber si multiplexar fáciles es GRATIS)


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
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(dev)
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
N = ids_all.numel()
print(f"tokens={N} d={D}")


def windows(n):
    starts = torch.randint(0, N - T - 1, (n,), device=dev)
    off = torch.arange(T + 1, device=dev)
    seqs = ids_all[(starts.unsqueeze(-1) + off)]
    return seqs[..., :T], seqs[..., 1:]              # x(n,T), y(n,T)


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
        x, y = windows(BS * K)
        x = x.view(BS, K, T)
        loss = F.cross_entropy(forward(x).reshape(-1, VOCAB), y.view(BS, K, T)[:, :, :].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 1000 == 0:
            print(f"  step {s} K={K} loss={loss.item():.3f}", flush=True)


@torch.no_grad()
def stratified_eval(n_groups=128):
    base.eval()
    # 1) sorpresa por token en K=1
    M = n_groups * max(EVAL_KS)
    x_all, y_all = windows(M)                          # (M,T)
    surp = torch.zeros(M, T, device=dev)
    for i in range(0, M, 32):
        xb = x_all[i:i+32].unsqueeze(1)               # (b,1,T)
        lp = F.log_softmax(forward(xb)[:, 0], dim=-1) # (b,T,vocab)
        surp[i:i+32] = -lp.gather(-1, y_all[i:i+32].unsqueeze(-1)).squeeze(-1)
    # terciles de sorpresa (global, sobre tokens válidos)
    flat = surp.flatten()
    q1, q2 = torch.quantile(flat, torch.tensor([1/3, 2/3], device=dev))
    res = {"surprise_terciles": [round(float(q1), 2), round(float(q2), 2)], "by_K": {}}
    # 2) recovery por K, estratificado por sorpresa del token
    for K in EVAL_KS:
        ng = M // K
        xg = x_all[:ng*K].view(ng, K, T)
        yg = y_all[:ng*K].view(ng, K, T)
        sg = surp[:ng*K].view(ng, K, T)
        rec = torch.zeros(ng, K, T, device=dev)
        gb = max(1, 64 // K)                           # batch de grupos adaptado a K (controla memoria)
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
    # --- control prior-only: predecir SIEMPRE el token marginal más frecuente, estratificado ---
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
    # --- retención: acc_easy(K) / acc_easy(K=1). ~1 = multiplexar fáciles es GRATIS ---
    base_e, base_m, base_h = (res["by_K"][1]["acc_easy"], res["by_K"][1]["acc_mid"], res["by_K"][1]["acc_hard"])
    res["retention_vs_K1"] = {K: {
        "easy": round(res["by_K"][K]["acc_easy"] / base_e, 3) if base_e > 0 else None,
        "mid": round(res["by_K"][K]["acc_mid"] / base_m, 3) if base_m > 0 else None,
        "hard": round(res["by_K"][K]["acc_hard"] / base_h, 3) if base_h > 0 else None,
    } for K in EVAL_KS}
    print("  retención fáciles vs K=1:", {K: res["retention_vs_K1"][K]["easy"] for K in EVAL_KS})
    return res


def main():
    train()
    print("\n=== Recovery estratificado por dificultad de token (sorpresa) ===")
    res = stratified_eval()
    # throughput efectivo selectivo: a cada K, fracción de tokens 'fáciles' que se recuperan bien
    json.dump(res, open("prototype_selective_result.json", "w"), indent=2)
    print("\nValida la teoría si acc_easy >> acc_hard a K alto (heterogeneidad de demanda → selectivo viable).")
    print("→ prototype_selective_result.json")


if __name__ == "__main__":
    main()
