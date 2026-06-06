"""¿El multiplexado del canal de PROCESADO es añadible barato (LoRA) sobre un base single-thread?

Fase 1: entrenar un transformer base en la tarea con K=1 (reverse single-thread). Congelar.
Fase 2: añadir multiplexado para K hilos, variando QUÉ se entrena (códigos FIJOS en todas; el residual
        stream/transporte queda libre). Condiciones:
  - full      : todo el base re-entrenable (TECHO).
  - lora_both : LoRA en attention + MLP (head entrenable).
  - lora_mlp  : LoRA solo en MLP.
  - lora_attn : LoRA solo en attention.
  - frozen    : base 100% congelado, solo head (lectura) entrenable (PISO).
Mide K_cap y nº de parámetros entrenados por condición. Valida si el multiplexado vive en el procesado
y es barato de añadir (LoRA ≈ techo con muchos menos params).
"""

from __future__ import annotations

import copy
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

V, T, D, H, NLAYERS = 16, 8, 128, 4, 3
KMAX = 16
LORA_R = 16
EVAL_KS = [1, 2, 4, 6, 8, 10, 12, 14, 16]


class LoRALinear(nn.Module):
    def __init__(self, lin: nn.Linear, r=LORA_R):
        super().__init__()
        self.base = lin
        self.A = nn.Parameter(torch.randn(r, lin.in_features) * 0.02)
        self.B = nn.Parameter(torch.zeros(lin.out_features, r))
        self.on = False

    def forward(self, x):
        y = self.base(x)
        if self.on:
            y = y + (x @ self.A.t()) @ self.B.t()
        return y


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wq, self.Wk, self.Wv, self.Wo = (LoRALinear(nn.Linear(D, D)) for _ in range(4))
        self.W1, self.W2 = LoRALinear(nn.Linear(D, 4 * D)), LoRALinear(nn.Linear(4 * D, D))
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)

    def attn(self, x):
        B, Tt, _ = x.shape
        q, k, v = (f(x).view(B, Tt, H, D // H).transpose(1, 2) for f in (self.Wq, self.Wk, self.Wv))
        a = F.softmax(q @ k.transpose(-2, -1) / (D // H) ** 0.5, dim=-1)
        o = (a @ v).transpose(1, 2).reshape(B, Tt, D)
        return self.Wo(o)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.W2(F.gelu(self.W1(self.ln2(x))))
        return x


class Net(nn.Module):
    def __init__(self, kmax):
        super().__init__()
        self.E = nn.Embedding(V, D)
        self.pos = nn.Parameter(torch.randn(1, T, D) * 0.02)
        codes = torch.sign(torch.randn(kmax, D)); codes[codes == 0] = 1
        self.register_buffer("codes", codes)
        self.blocks = nn.ModuleList(Block() for _ in range(NLAYERS))
        self.head = nn.Linear(D, V)

    def forward(self, x):
        K = x.shape[1]
        c = self.codes[:K].view(1, K, 1, D)
        h = (self.E(x) * c).sum(1) + self.pos
        for b in self.blocks:
            h = b(h)
        return self.head(h.unsqueeze(1) * c)

    def loras(self):
        return [m for m in self.modules() if isinstance(m, LoRALinear)]

    def set_mode(self, mode):
        for p in self.parameters():
            p.requires_grad = False
        for lo in self.loras():
            lo.on = False
        self.head.weight.requires_grad = self.head.bias.requires_grad = True  # lectura siempre
        if mode == "full":
            for p in self.parameters():
                p.requires_grad = True
        elif mode == "frozen":
            pass
        else:
            for b in self.blocks:
                grp = ([b.Wq, b.Wk, b.Wv, b.Wo] if "attn" in mode or "both" in mode else []) + \
                      ([b.W1, b.W2] if "mlp" in mode or "both" in mode else [])
                for lo in grp:
                    lo.on = True
                    lo.A.requires_grad = lo.B.requires_grad = True

    def n_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def batch(bs, K):
    x = torch.randint(0, V, (bs, K, T), device=dev)
    return x, x.flip(-1)


def train(model, steps, kmax, curriculum=True):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    model.train()
    for s in range(steps):
        kcur = 1 + int((kmax - 1) * min(1.0, (s / steps) / 0.75)) if curriculum else kmax
        K = int(torch.randint(1, kcur + 1, (1,)))
        x, y = batch(256, K)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()


def kcap(model):
    model.eval(); accs = {}
    with torch.no_grad():
        for K in EVAL_KS:
            x, y = batch(1024, K)
            accs[K] = round(float((model(x).argmax(-1) == y).float().mean()), 3)
    return max([0] + [K for K in EVAL_KS if accs[K] >= 0.9]), accs


def main():
    # Fase 1: base single-thread (K=1)
    base = Net(KMAX).to(dev)
    base.set_mode("full")
    train(base, 3000, 1, curriculum=False)
    bc, ba = kcap(base)
    print(f"BASE (K=1 only): acc@K=1={ba[1]}  (K_cap del base={bc})")

    results = {"meta": {"D": D, "V": V, "layers": NLAYERS, "lora_r": LORA_R}, "base_acc1": ba[1], "conds": []}
    for mode in ["full", "lora_both", "lora_mlp", "lora_attn", "frozen"]:
        m = copy.deepcopy(base)
        m.set_mode(mode)
        ntr = m.n_trainable()
        train(m, 8000, KMAX, curriculum=True)
        kc, accs = kcap(m)
        results["conds"].append({"mode": mode, "n_trainable": ntr, "k_cap": kc, "accs": accs})
        print(f"  {mode:10s} | params_entren={ntr:>8d} | K_cap={kc:>2d} | {accs}", flush=True)
    json.dump(results, open("hybrid_result.json", "w"), indent=2)
    tot = sum(p.numel() for p in base.parameters())
    print(f"\n(total params modelo ≈ {tot}. K_cap full-scratch de referencia (capacity_law) = 12.)")
    print("Valida si: lora_* alcanza K_cap≈full con params_entren << full → multiplexado barato en procesado.")
    print("→ hybrid_result.json")


if __name__ == "__main__":
    main()
