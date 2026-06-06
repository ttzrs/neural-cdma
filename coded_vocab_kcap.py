"""¿Diseñar los códigos de binding (tokens-como-código) mejora la multiplexación vs random?

Test de la idea de Jose: ¿códigos DISEÑADOS ortogonales baten a random para soportar multiplexación?
Misma arquitectura entrenada (transformer masked-multiplex, text_bpe V=512/T=64/d=128/KMAX=32), única
variable = la estructura de los códigos c_k que ligan cada hilo:
  - random_pm1 : sign(randn) — cuasi-ortogonal por Johnson-Lindenstrauss (lo que usábamos).
  - hadamard   : filas de una matriz Hadamard 128×128 → PERFECTAMENTE ortogonales (diseñado).
  - correlated : códigos de bajo rango (alta coherencia mutua) → el extremo MALO (sanity).
Mide K_cap(señal≥80%) + throughput + loss@step (convergencia). Hipótesis: hadamard ≈ random (el binding
ya está resuelto por random); correlated << (el código solo importa cuando es malo). El cuello es D_task, no el código.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import itertools
import json

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import hadamard

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

V, T, D = 512, 64, 128
NLAYERS, NHEAD = 3, 4
KMAX_TRAIN = 32
EVAL_KS = [1, 2, 4, 8, 16, 32]
STEPS, BS, LR = 6000, 256, 3e-4
MASK_FRAC = 0.5
MASK_ID = V
EVAL_N = 256


def make_codes(kind: str) -> torch.Tensor:
    if kind == "random_pm1":
        c = torch.sign(torch.randn(KMAX_TRAIN, D)); c[c == 0] = 1
    elif kind == "hadamard":
        H = torch.tensor(hadamard(D), dtype=torch.float32)   # (D,D) filas ±1 ortogonales
        idx = torch.randperm(D)[:KMAX_TRAIN]
        c = H[idx]
    elif kind == "correlated":
        # bajo rango: pocos ejes base compartidos → alta coherencia mutua (códigos malos)
        rank = 4
        base = torch.randn(rank, D)
        mix = torch.randn(KMAX_TRAIN, rank)
        c = torch.sign(mix @ base); c[c == 0] = 1
    else:
        raise ValueError(kind)
    return c.to(dev)


def code_coherence(c: torch.Tensor) -> float:
    cn = c / c.norm(dim=1, keepdim=True)
    g = (cn @ cn.t()).abs()
    g.fill_diagonal_(0)
    return float(g.max())


class Model(nn.Module):
    def __init__(self, codes: torch.Tensor):
        super().__init__()
        self.E = nn.Embedding(V + 1, D)
        self.pos = nn.Parameter(torch.randn(1, T, D) * 0.02)
        self.register_buffer("codes", codes)
        enc = nn.TransformerEncoderLayer(D, NHEAD, 4 * D, batch_first=True, activation="gelu", dropout=0.0)
        self.enc = nn.TransformerEncoder(enc, NLAYERS)
        self.head = nn.Linear(D, V)

    def forward(self, xin):
        K = xin.shape[1]
        c = self.codes[:K].view(1, K, 1, D)
        z = (self.E(xin) * c).sum(1) + self.pos
        h = self.enc(z)
        return self.head(h.unsqueeze(1) * c)


def mask_batch(x):
    m = torch.rand(x.shape, device=dev) < MASK_FRAC
    m[(~m).all(-1, keepdim=True).expand_as(m) & (torch.arange(T, device=dev) == 0)] = True
    xin = torch.where(m, torch.full_like(x, MASK_ID), x)
    return xin, x, m


def text_source():
    from collections import Counter
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
    txt = " ".join(r["text"] for r in itertools.islice(iter(ds), 20000) if r["text"].strip())
    ids = tok(txt).input_ids
    common = [i for i, _ in Counter(ids).most_common(V - 1)]
    remap = {i: r for r, i in enumerate(common)}
    seq = torch.tensor([remap.get(i, V - 1) for i in ids], dtype=torch.long, device=dev)
    N = seq.numel()

    def sample(bs, K):
        starts = torch.randint(0, N - T, (bs, K), device=dev)
        off = torch.arange(T, device=dev)
        return seq[(starts.unsqueeze(-1) + off)]
    return sample


def evaluate(model, sample):
    model.eval(); out = {}
    with torch.no_grad():
        for K in EVAL_KS:
            x = sample(EVAL_N, K)
            xin, y, m = mask_batch(x)
            pred = model(xin).argmax(-1)
            out[K] = round(float(((pred == y) & m).sum() / m.sum()), 4)
    model.train()
    return out


def run_codes(kind: str, sample):
    print(f"\n===== códigos {kind} =====", flush=True)
    codes = make_codes(kind)
    coh = code_coherence(codes)
    model = Model(codes).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    loss_at = {}
    for step in range(STEPS):
        kcur = 1 + int((KMAX_TRAIN - 1) * min(1.0, (step / STEPS) / 0.75))
        K = int(torch.randint(1, kcur + 1, (1,)))
        x = sample(BS, K)
        xin, y, m = mask_batch(x)
        loss = (nn.functional.cross_entropy(model(xin).reshape(-1, V), y.reshape(-1), reduction="none")
                .reshape(x.shape)[m]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step in (1000, 3000):
            loss_at[step] = round(float(loss), 3)
        if step % 1500 == 0:
            print(f"  step {step:5d} K={K} loss={loss.item():.3f}", flush=True)
    trained = evaluate(model, sample)
    ch = 1 / V
    s1 = trained[1] - ch
    kcap = 1 if s1 <= 0.02 else max([1] + [K for K in EVAL_KS if (trained[K] - ch) >= 0.8 * s1])
    thru = {K: round(K * (trained[K] - ch), 3) for K in EVAL_KS}
    print(f"  coherencia={coh:.3f} acc(K=1)={trained[1]:.4f} K_cap={kcap} thru_max={max(thru.values()):.2f} loss@1k={loss_at.get(1000)} loss@3k={loss_at.get(3000)}")
    return {"coherence": round(coh, 4), "trained": trained, "k_cap_signal80": kcap,
            "thru_max": max(thru.values()), "loss_at": loss_at, "chance": round(ch, 5)}


def main():
    sample = text_source()
    res = {"meta": {"V": V, "T": T, "D": D, "kmax": KMAX_TRAIN, "steps": STEPS, "eval_ks": EVAL_KS}}
    for kind in ["random_pm1", "hadamard", "correlated"]:
        res[kind] = run_codes(kind, sample)
    json.dump(res, open("coded_vocab_kcap_result.json", "w"), indent=2)
    r, h, c = res["random_pm1"], res["hadamard"], res["correlated"]
    print(f"\nVEREDICTO (coherencia | K_cap | thru | loss@3k):")
    for k, v in (("random", r), ("hadamard", h), ("correlated", c)):
        print(f"  {k:10s}: coh={v['coherence']:.3f}  K_cap={v['k_cap_signal80']}  thru={v['thru_max']:.2f}  loss@3k={v['loss_at'].get(3000)}")
    print("→ hadamard≈random confirma: el binding ya está resuelto por random (JL), diseñar no ayuda." if abs(h['thru_max']-r['thru_max']) < 0.15*max(r['thru_max'],1e-6)
          else "→ hadamard difiere de random: diseñar SÍ cambia (revisar).")
    print("→ coded_vocab_kcap_result.json")


if __name__ == "__main__":
    main()
