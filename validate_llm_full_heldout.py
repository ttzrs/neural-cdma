"""Cota JUSTA de K_cap en LLM real: FULL fine-tune (no LoRA) de Qwen2.5-0.5B para multiplexar K ventanas
de wikitext. Copia held-out: entrena SOLO en train; evalúa train y validation independientemente.
Las semillas de ventanas son iguales entre splits para cada K y no alteran el RNG de entrenamiento.
"""

from __future__ import annotations

import itertools
import json

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

dev = torch.device("cuda")
torch.manual_seed(0)
MODEL = "Qwen/Qwen2.5-0.5B"
T, BS, STEPS, KMAX, LR = 32, 4, 4000, 8, 2e-5
EVAL_KS = [1, 2, 4, 6, 8]
N_EVAL_BATCHES, EVAL_SEED = 32, 10000

tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
base.config.use_cache = False
base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
D = base.config.hidden_size
VOCAB = base.config.vocab_size
qwen, lm_head, embed = base.model, base.lm_head, base.get_input_embeddings()
codes = torch.sign(torch.randn(KMAX, D, device=dev)); codes[codes == 0] = 1

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
buf = [r["text"] for r in itertools.islice(iter(ds), 4000) if len(r["text"]) > 1]
ids_all = torch.tensor(tok(" ".join(buf)).input_ids, device=dev)
ds_val = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation", streaming=True)
buf_val = [r["text"] for r in ds_val if len(r["text"]) > 1]
ids_val = torch.tensor(tok(" ".join(buf_val)).input_ids, device=dev)
N = ids_all.numel()
print(f"FULL FT. tokens={N} heldout_tokens={ids_val.numel()} d={D} vocab={VOCAB} log2V={torch.log2(torch.tensor(float(VOCAB))):.1f}")


def batch(source, bs, K, generator=None):
    starts = torch.randint(0, source.numel() - T - 1, (bs, K), device=dev, generator=generator)
    off = torch.arange(T + 1, device=dev)
    seqs = source[(starts.unsqueeze(-1) + off)]
    return seqs[..., :T], seqs[..., 1:]


def forward(x):
    K = x.shape[1]; c = codes[:K].view(1, K, 1, D)
    z = (embed(x) * c).sum(1)
    h = qwen(inputs_embeds=z).last_hidden_state
    return lm_head(h.unsqueeze(1) * c)


@torch.no_grad()
def evaluate():
    base.eval(); out = {}
    for split, source in (("train", ids_all), ("heldout", ids_val)):
        out[split] = {}
        for K in EVAL_KS:
            generator = torch.Generator(device=dev).manual_seed(EVAL_SEED + K)
            acc = 0.0
            for _ in range(N_EVAL_BATCHES):
                x, y = batch(source, 8, K, generator=generator)
                acc += (forward(x).argmax(-1) == y).float().mean().item()
            out[split][K] = round(acc / N_EVAL_BATCHES, 3)
    base.train()
    return out


def main():
    untrained = evaluate()
    print("sin fine-tune:", untrained)
    opt = torch.optim.AdamW(base.parameters(), lr=LR)
    base.train()
    for s in range(STEPS):
        kcur = 1 + int((KMAX - 1) * min(1.0, (s / STEPS) / 0.75))
        K = int(torch.randint(1, kcur + 1, (1,)))
        x, y = batch(ids_all, BS, K)
        loss = F.cross_entropy(forward(x).reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 500 == 0:
            print(f"  step {s} K={K} loss={loss.item():.3f}", flush=True)
    trained = evaluate()
    res = {"n_eval_batches": N_EVAL_BATCHES, "eval_window_seed": EVAL_SEED,
           "n_tokens": {"train": N, "heldout": ids_val.numel()},
           "dataset": "Salesforce/wikitext/wikitext-2-raw-v1",
           "heldout_split": "validation"}
    for split in ("train", "heldout"):
        a1 = trained[split][1]
        kcap = max([0] + [K for K in EVAL_KS if trained[split][K] >= 0.8 * a1])
        res[split] = {"model": MODEL, "mode": "full_ft", "untrained": untrained[split],
                      "trained": trained[split], "acc_K1": a1, "k_cap_rel80": kcap}
        print(f"\n{split}:  K | sin-FT | FULL fine-tuned")
        for K in EVAL_KS:
            print(f"  {K:2d} | {untrained[split][K]:.3f} | {trained[split][K]:.3f}")
        print(f"  acc(K=1)={a1:.3f}. K_cap (≥80% K=1) = {kcap}")
    res["k_cap_rel80_heldout"] = res["heldout"]["k_cap_rel80"]
    with open("validate_llm_full_heldout_result.json", "w") as f:
        json.dump(res, f, indent=2)
    print("→ validate_llm_full_heldout_result.json")


if __name__ == "__main__":
    main()
