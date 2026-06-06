"""Función K_cap(rank): barre el rango del LoRA (multiplexador en attn+MLP) sobre un base congelado.
Base entrenado UNA vez (K=1) y reusado; para cada rank se entrena solo el adaptador. Objetivo: la forma
matemática de K_cap(rank) (¿lineal? ¿saturante hacia el techo full?). Setup d=128, V=16."""

from __future__ import annotations

import copy
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
V, T, D, H, NLAYERS, KMAX = 16, 8, 128, 4, 3, 16
RANKS = [2, 4, 8, 16, 32, 64, 128]
EVAL_KS = [1, 2, 4, 6, 8, 10, 12, 14, 16]


class LoRALinear(nn.Module):
    def __init__(self, lin, r):
        super().__init__()
        self.base = lin
        self.A = nn.Parameter(torch.randn(r, lin.in_features) * 0.02)
        self.B = nn.Parameter(torch.zeros(lin.out_features, r))
        self.on = False

    def forward(self, x):
        y = self.base(x)
        return y + (x @ self.A.t()) @ self.B.t() if self.on else y


class Block(nn.Module):
    def __init__(self, r):
        super().__init__()
        self.Wq, self.Wk, self.Wv, self.Wo = (LoRALinear(nn.Linear(D, D), r) for _ in range(4))
        self.W1, self.W2 = LoRALinear(nn.Linear(D, 4 * D), r), LoRALinear(nn.Linear(4 * D, D), r)
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)

    def attn(self, x):
        B, Tt, _ = x.shape
        q, k, v = (f(x).view(B, Tt, H, D // H).transpose(1, 2) for f in (self.Wq, self.Wk, self.Wv))
        a = F.softmax(q @ k.transpose(-2, -1) / (D // H) ** 0.5, dim=-1)
        return self.Wo((a @ v).transpose(1, 2).reshape(B, Tt, D))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.W2(F.gelu(self.W1(self.ln2(x))))


class Net(nn.Module):
    def __init__(self, r):
        super().__init__()
        self.E = nn.Embedding(V, D)
        self.pos = nn.Parameter(torch.randn(1, T, D) * 0.02)
        codes = torch.sign(torch.randn(KMAX, D)); codes[codes == 0] = 1
        self.register_buffer("codes", codes)
        self.blocks = nn.ModuleList(Block(r) for _ in range(NLAYERS))
        self.head = nn.Linear(D, V)

    def forward(self, x):
        K = x.shape[1]; c = self.codes[:K].view(1, K, 1, D)
        h = (self.E(x) * c).sum(1) + self.pos
        for b in self.blocks:
            h = b(h)
        return self.head(h.unsqueeze(1) * c)

    def loras(self):
        return [m for m in self.modules() if isinstance(m, LoRALinear)]


def batch(bs, K):
    x = torch.randint(0, V, (bs, K, T), device=dev); return x, x.flip(-1)


def train(model, steps, kmax, full=False):
    if full:
        for p in model.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = False
        model.head.weight.requires_grad = model.head.bias.requires_grad = True
        for lo in model.loras():
            lo.on = True; lo.A.requires_grad = lo.B.requires_grad = True
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    model.train()
    for s in range(steps):
        kcur = 1 + int((kmax - 1) * min(1.0, (s / steps) / 0.75))
        K = int(torch.randint(1, kcur + 1, (1,)))
        x, y = batch(256, K)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def kcap(model):
    model.eval(); accs = {}
    with torch.no_grad():
        for K in EVAL_KS:
            x, y = batch(1024, K); accs[K] = round(float((model(x).argmax(-1) == y).float().mean()), 3)
    return max([0] + [K for K in EVAL_KS if accs[K] >= 0.9]), accs


def main():
    base = Net(2).to(dev)
    train(base, 3000, 1, full=True)         # base single-thread
    base_sd = {k: v for k, v in base.state_dict().items() if ".A" not in k and ".B" not in k}
    print("base entrenado (K=1). Barriendo rank:")
    res = {"meta": {"D": D, "V": V}, "points": []}
    for r in RANKS:
        m = Net(r).to(dev)
        m.load_state_dict(base_sd, strict=False)   # carga pesos base; A,B frescos
        ntr = train(m, 8000, KMAX, full=False)
        kc, accs = kcap(m)
        res["points"].append({"rank": r, "n_trainable": ntr, "k_cap": kc, "accs": accs})
        print(f"  r={r:3d} | params={ntr:>7d} | K_cap={kc:>2d} | K_cap/r={kc/r:.3f}", flush=True)
    # techo full
    mf = Net(2).to(dev); mf.load_state_dict(base_sd, strict=False)
    ntr = train(mf, 8000, KMAX, full=True); kc, _ = kcap(mf)
    res["full"] = {"n_trainable": ntr, "k_cap": kc}
    print(f"  full | params={ntr:>7d} | K_cap={kc} (techo)")
    json.dump(res, open("rank_sweep_result.json", "w"), indent=2)
    print("\nfunción K_cap(rank): inspeccionar si lineal (K_cap/r const) o saturante hacia el techo.")
    print("→ rank_sweep_result.json")


if __name__ == "__main__":
    main()
