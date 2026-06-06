"""Cota JUSTA de K_cap en LLM real: FULL fine-tune (no LoRA) de Qwen2.5-0.5B para multiplexar K ventanas
de wikitext. Decide si K_cap=2 era el LoRA o el techo real del lenguaje. Gradient checkpointing + bs bajo."""

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

tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(dev)
base.config.use_cache = False
base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
D = base.config.hidden_size
VOCAB = base.config.vocab_size
qwen, lm_head, embed = base.model, base.lm_head, base.get_input_embeddings()
codes = torch.sign(torch.randn(KMAX, D, device=dev)); codes[codes == 0] = 1

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
buf = [r["text"] for r in itertools.islice(iter(ds), 4000) if len(r["text"]) > 1]
ids_all = torch.tensor(tok(" ".join(buf)).input_ids, device=dev)
N = ids_all.numel()
print(f"FULL FT. tokens={N} d={D} vocab={VOCAB} log2V={torch.log2(torch.tensor(float(VOCAB))):.1f}")


def batch(bs, K):
    starts = torch.randint(0, N - T - 1, (bs, K), device=dev)
    off = torch.arange(T + 1, device=dev)
    seqs = ids_all[(starts.unsqueeze(-1) + off)]
    return seqs[..., :T], seqs[..., 1:]


def forward(x):
    K = x.shape[1]; c = codes[:K].view(1, K, 1, D)
    z = (embed(x) * c).sum(1)
    h = qwen(inputs_embeds=z).last_hidden_state
    return lm_head(h.unsqueeze(1) * c)


@torch.no_grad()
def evaluate():
    base.eval(); out = {}
    for K in EVAL_KS:
        acc = 0.0
        for _ in range(8):
            x, y = batch(8, K)
            acc += (forward(x).argmax(-1) == y).float().mean().item()
        out[K] = round(acc / 8, 3)
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
        x, y = batch(BS, K)
        loss = F.cross_entropy(forward(x).reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 500 == 0:
            print(f"  step {s} K={K} loss={loss.item():.3f}", flush=True)
    trained = evaluate()
    a1 = trained[1]
    kcap = max([0] + [K for K in EVAL_KS if trained[K] >= 0.8 * a1])
    res = {"model": MODEL, "mode": "full_ft", "untrained": untrained, "trained": trained,
           "acc_K1": a1, "k_cap_rel80": kcap}
    print(f"\n  K | sin-FT | FULL fine-tuned")
    for K in EVAL_KS:
        print(f"  {K:2d} | {untrained[K]:.3f} | {trained[K]:.3f}")
    print(f"\n  acc(K=1)={a1:.3f} (nativo Qwen ~0.45). K_cap (≥80% K=1) = {kcap}")
    json.dump(res, open("validate_llm_full_result.json", "w"), indent=2)
    print("→ validate_llm_full_result.json")


if __name__ == "__main__":
    main()
