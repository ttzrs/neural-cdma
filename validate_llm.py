"""¿Sobrevive la paralelización en superposición a un LLM REAL con lenguaje real?

Qwen2.5-0.5B. K ventanas de wikitext-2 superpuestas en inputs_embeds (moduladas por códigos ±1),
fine-tune LoRA (attn+MLP, todas las capas) para procesarlas en paralelo; la salida de cada hilo se lee
des-modulando antes del lm_head. Métrica: next-token accuracy por hilo vs K. Compara con K=1.
Decide si las optimizaciones (throughput/KV/calidad) tienen base real o el K_cap colapsa en lenguaje.
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
EVAL_KS = [1, 2, 4, 6, 8]
LORA_R, LORA_ALPHA = 32, 32
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


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
D = base.config.hidden_size
for p in base.parameters():
    p.requires_grad = False
# LoRA manual: reemplazar in-place las proyecciones objetivo
for mod in base.modules():
    for cn, child in list(mod.named_children()):
        if isinstance(child, nn.Linear) and cn in TARGETS:
            setattr(mod, cn, LoRALinear(child, LORA_R, LORA_ALPHA))
base.to(dev)
peft_model = base                                   # alias (entrenables = A,B)
qwen = base.model
lm_head = base.lm_head
embed = base.get_input_embeddings()
codes = torch.sign(torch.randn(KMAX, D, device=dev)); codes[codes == 0] = 1

# datos: wikitext-2 tokenizado en un tensor 1D
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
buf = []
for r in itertools.islice(iter(ds), 4000):
    if len(r["text"]) > 1:
        buf.append(r["text"])
ids_all = torch.tensor(tok(" ".join(buf)).input_ids, device=dev)
N = ids_all.numel()
print(f"tokens wikitext: {N}  d={D} vocab={base.config.vocab_size} log2V={torch.log2(torch.tensor(float(base.config.vocab_size))):.1f}")


def batch(bs, K):
    starts = torch.randint(0, N - T - 1, (bs, K), device=dev)
    off = torch.arange(T + 1, device=dev)
    seqs = ids_all[(starts.unsqueeze(-1) + off)]      # (bs,K,T+1)
    return seqs[..., :T], seqs[..., 1:]               # x, y (next-token)


def forward(x):                                        # x:(bs,K,T)
    K = x.shape[1]; c = codes[:K].view(1, K, 1, D)
    z = (embed(x) * c).sum(1)                          # (bs,T,d) superposición
    h = qwen(inputs_embeds=z).last_hidden_state        # (bs,T,d)
    return lm_head(h.unsqueeze(1) * c)                 # (bs,K,T,vocab)


@torch.no_grad()
def evaluate():
    peft_model.eval(); out = {}
    for K in EVAL_KS:
        acc = 0.0
        for _ in range(8):
            x, y = batch(8, K)
            acc += (forward(x).argmax(-1) == y).float().mean().item()
        out[K] = round(acc / 8, 3)
    return out


def main():
    untrained = evaluate()
    print("sin fine-tune (multiplexado):", untrained)
    opt = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=2e-4)
    ntr = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    peft_model.train()
    for s in range(STEPS):
        kcur = 1 + int((KMAX - 1) * min(1.0, (s / STEPS) / 0.75))
        K = int(torch.randint(1, kcur + 1, (1,)))
        x, y = batch(BS, K)
        loss = F.cross_entropy(forward(x).reshape(-1, base.config.vocab_size), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 500 == 0:
            print(f"  step {s} K={K} loss={loss.item():.3f}", flush=True)
    trained = evaluate()
    res = {"model": MODEL, "d": D, "lora_params": ntr, "untrained": untrained, "trained": trained}
    print(f"\n  K | sin-FT | fine-tuned (next-token acc por hilo)")
    for K in EVAL_KS:
        print(f"  {K:2d} | {untrained[K]:.3f}  | {trained[K]:.3f}")
    a1 = trained[1]
    kcap = max([0] + [K for K in EVAL_KS if trained[K] >= 0.8 * a1])
    res["k_cap_rel80"] = kcap
    print(f"\n  acc(K=1)={a1:.3f}. K_cap (≥80% de K=1) = {kcap}")
    json.dump(res, open("validate_llm_result.json", "w"), indent=2)
    print("→ validate_llm_result.json")


if __name__ == "__main__":
    main()
