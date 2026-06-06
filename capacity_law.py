"""LEY DE CAPACIDAD: ¿cuántos hilos K_cap se paralelizan vs ancho d y complejidad de tarea (vocab V)?

Fix sobre v1: CURRÍCULUM de K (rampa de fácil→difícil) + más steps, para que converja antes de medir.
Grid d ∈ {64,128,256} × V ∈ {16,64}. Tarea reverse, K secuencias multiplexadas por código CDMA ±1.
K_cap = mayor K con accuracy ≥ 0.9. Comprueba la ley K_cap ∝ d / log2(V).
"""

from __future__ import annotations

import json
import math

import torch
import torch.nn as nn

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

T, NLAYERS, NHEAD = 8, 3, 4
STEPS, BS, LR = 10000, 256, 3e-4
ACC_THRESH = 0.9
GRID_D = [64, 128, 256]
GRID_V = [16, 64]


class Model(nn.Module):
    def __init__(self, d, V, kmax):
        super().__init__()
        self.d, self.V = d, V
        self.E = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.randn(1, T, d) * 0.02)
        codes = torch.sign(torch.randn(kmax, d)); codes[codes == 0] = 1
        self.register_buffer("codes", codes)
        enc = nn.TransformerEncoderLayer(d, NHEAD, 4 * d, batch_first=True, activation="gelu", dropout=0.0)
        self.enc = nn.TransformerEncoder(enc, NLAYERS)
        self.head = nn.Linear(d, V)

    def forward(self, x):
        K = x.shape[1]
        c = self.codes[:K].view(1, K, 1, self.d)
        z = (self.E(x) * c).sum(1) + self.pos
        h = self.enc(z)
        return self.head(h.unsqueeze(1) * c)


def run_cell(d, V, kmax_train, eval_ks):
    model = Model(d, V, kmax_train).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for step in range(STEPS):
        # currículum: Kmax efectivo sube de 1 a kmax_train en el primer 75% del entrenamiento
        kcur = 1 + int((kmax_train - 1) * min(1.0, (step / STEPS) / 0.75))
        K = int(torch.randint(1, kcur + 1, (1,)))
        x = torch.randint(0, V, (BS, K, T), device=dev)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, V), x.flip(-1).reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    accs = {}
    with torch.no_grad():
        for K in eval_ks:
            x = torch.randint(0, V, (1024, K, T), device=dev)
            accs[K] = float((model(x).argmax(-1) == x.flip(-1)).float().mean())
    kcap = max([0] + [K for K in eval_ks if accs[K] >= ACC_THRESH])
    return kcap, {k: round(v, 3) for k, v in accs.items()}


def main():
    results = {"meta": {"T": T, "layers": NLAYERS, "steps": STEPS, "acc_thresh": ACC_THRESH}, "cells": []}
    print(f"{'d':>4} {'V':>3} {'bits':>5} {'K_cap':>6}  | accuracy(K)")
    for V in GRID_V:
        for d in GRID_D:
            kmax_train = min(d // 4, 32)
            eval_ks = sorted(set([1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32]))
            eval_ks = [k for k in eval_ks if k <= kmax_train]
            kcap, accs = run_cell(d, V, kmax_train, eval_ks)
            results["cells"].append({"d": d, "V": V, "bits": round(math.log2(V), 2),
                                     "kmax_train": kmax_train, "k_cap": kcap, "accs": accs})
            print(f"{d:>4} {V:>3} {math.log2(V):>5.1f} {kcap:>6}  | {accs}", flush=True)
    print("\n=== Ley: K_cap/d y K_cap·log2(V)/d (¿constante = ley K_cap∝d/log2V?) ===")
    for c in results["cells"]:
        print(f"  d={c['d']:>4} V={c['V']:>3} bits={c['bits']:.0f} | K_cap={c['k_cap']:>3} | "
              f"K_cap/d={c['k_cap']/c['d']:.3f} | K_cap·bits/d={c['k_cap']*c['bits']/c['d']:.2f}")
    json.dump(results, open("capacity_law_result.json", "w"), indent=2)
    print("\n→ capacity_law_result.json")


if __name__ == "__main__":
    main()
