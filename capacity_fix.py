"""Cierra los puntos flojos de la ley de capacidad: d=64 (kmax_train apropiado) y d=128/V=64 (más steps)."""

from __future__ import annotations
import json, math
import torch, torch.nn as nn

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
T, NLAYERS, NHEAD, BS, LR = 8, 3, 4, 256, 3e-4
ACC = 0.9


class Model(nn.Module):
    def __init__(self, d, V, kmax):
        super().__init__()
        self.d = d
        self.E = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.randn(1, T, d) * 0.02)
        codes = torch.sign(torch.randn(kmax, d)); codes[codes == 0] = 1
        self.register_buffer("codes", codes)
        enc = nn.TransformerEncoderLayer(d, NHEAD, 4 * d, batch_first=True, activation="gelu", dropout=0.0)
        self.enc = nn.TransformerEncoder(enc, NLAYERS)
        self.head = nn.Linear(d, V)

    def forward(self, x):
        K = x.shape[1]; c = self.codes[:K].view(1, K, 1, self.d)
        h = self.enc((self.E(x) * c).sum(1) + self.pos)
        return self.head(h.unsqueeze(1) * c)


def run(d, V, kmax_train, steps, eval_ks):
    m = Model(d, V, kmax_train).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=LR); m.train()
    for s in range(steps):
        kcur = 1 + int((kmax_train - 1) * min(1.0, (s / steps) / 0.75))
        K = int(torch.randint(1, kcur + 1, (1,)))
        x = torch.randint(0, V, (BS, K, T), device=dev)
        loss = nn.functional.cross_entropy(m(x).reshape(-1, V), x.flip(-1).reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); accs = {}
    with torch.no_grad():
        for K in eval_ks:
            x = torch.randint(0, V, (1024, K, T), device=dev)
            accs[K] = round(float((m(x).argmax(-1) == x.flip(-1)).float().mean()), 3)
    kcap = max([0] + [K for K in eval_ks if accs[K] >= ACC])
    return kcap, accs


CFG = [
    (64, 16, 10, 10000, [1, 2, 3, 4, 5, 6, 8, 10]),
    (64, 64, 6, 10000, [1, 2, 3, 4, 5, 6]),
    (128, 64, 12, 16000, [1, 2, 4, 6, 8, 10, 12]),
]
out = []
for d, V, kmax, steps, eks in CFG:
    kcap, accs = run(d, V, kmax, steps, eks)
    bits = math.log2(V)
    print(f"d={d} V={V} bits={bits:.0f} | K_cap={kcap} | K_cap·bits/d={kcap*bits/d:.2f} | {accs}", flush=True)
    out.append({"d": d, "V": V, "bits": bits, "k_cap": kcap, "ratio": round(kcap * bits / d, 3), "accs": accs})
json.dump(out, open("capacity_fix_result.json", "w"), indent=2)
print("→ capacity_fix_result.json")
