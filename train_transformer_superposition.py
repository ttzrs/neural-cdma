"""¿Puede un TRANSFORMER entrenarse a procesar K secuencias SUPERPUESTAS en paralelo?

- K secuencias x_k ∈ V^T (símbolos aleatorios). Tarea por hilo: REVERSE (target_k[t] = x_k[T-1-t]) — exige
  atención (cada posición atiende a su espejo).
- Multiplexado CDMA en la dimensión de feature: z[t] = Σ_k E[x_k[t]] ⊙ c_k, con códigos c_k ∈ {±1}^d.
- El transformer (bidireccional, atención real) procesa z; la salida del hilo k se lee des-modulando por c_k.
- Mide accuracy de token por hilo vs K. Compara: ENTRENADO vs sin entrenar (azar=1/V); accuracy vs K.
Responde: ¿sobrevive la paralelización en superposición a un transformer con atención + tarea de secuencia?
"""

from __future__ import annotations

import json

import torch
import torch.nn as nn

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

V, T, D = 16, 8, 128
KMAX_TRAIN = 8
EVAL_KS = [1, 2, 3, 4, 6, 8]
STEPS, BS, LR = 6000, 256, 3e-4
NLAYERS, NHEAD = 3, 4


class Model(nn.Module):
    def __init__(self, kmax):
        super().__init__()
        self.E = nn.Embedding(V, D)
        self.pos = nn.Parameter(torch.randn(1, T, D) * 0.02)
        codes = torch.sign(torch.randn(kmax, D)); codes[codes == 0] = 1
        self.register_buffer("codes", codes)               # (kmax,d) ±1, fijos
        enc = nn.TransformerEncoderLayer(D, NHEAD, 4 * D, batch_first=True, activation="gelu", dropout=0.0)
        self.enc = nn.TransformerEncoder(enc, NLAYERS)
        self.head = nn.Linear(D, V)

    def forward(self, x):                                   # x: (bs,K,T)
        K = x.shape[1]
        c = self.codes[:K].view(1, K, 1, D)
        emb = self.E(x)                                     # (bs,K,T,d)
        z = (emb * c).sum(1) + self.pos                     # (bs,T,d) superposición
        h = self.enc(z)                                     # (bs,T,d)
        hk = h.unsqueeze(1) * c                             # (bs,K,T,d) des-modular por hilo
        return self.head(hk)                                # (bs,K,T,V)


def make_batch(bs, K):
    x = torch.randint(0, V, (bs, K, T), device=dev)
    y = x.flip(-1)                                          # reverse
    return x, y


def evaluate(model):
    model.eval(); out = {}
    with torch.no_grad():
        for K in EVAL_KS:
            x, y = make_batch(1024, K)
            pred = model(x).argmax(-1)
            out[K] = round(float((pred == y).float().mean()), 3)
    return out


def main():
    model = Model(KMAX_TRAIN).to(dev)
    untrained = evaluate(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for step in range(STEPS):
        K = int(torch.randint(1, KMAX_TRAIN + 1, (1,)))
        x, y = make_batch(BS, K)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0:
            print(f"  step {step:5d} K={K} loss={loss.item():.3f}", flush=True)
    trained = evaluate(model)
    res = {"meta": {"V": V, "T": T, "D": D, "layers": NLAYERS, "kmax_train": KMAX_TRAIN, "chance": round(1/V, 3)},
           "untrained": untrained, "trained": trained}
    print(f"\n  chance (azar) = {1/V:.3f}")
    print("  K  | sin-entrenar | ENTRENADO (accuracy por hilo)")
    for K in EVAL_KS:
        print(f"  {K:2d} |    {untrained[K]:.3f}     |   {trained[K]:.3f}")
    json.dump(res, open("train_transformer_superposition_result.json", "w"), indent=2)
    print("\n→ train_transformer_superposition_result.json")
    print("Paraleliza si accuracy entrenada >> azar para K>1 (procesa K secuencias en un forward).")


if __name__ == "__main__":
    main()
