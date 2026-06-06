"""¿Se puede ENTRENAR a una red a computar K hilos en superposición (paralelización entrenada)?

Tarea canónica de computación-en-superposición:
- N tareas posibles; en cada ejemplo K están activas (sparse). a ∈ R^N (sparse), valores ~U[-1,1].
- Entrada superpuesta: x = C·a ∈ R^d con d < N (superposición forzada; C = códigos).
- Target: y = g(a) elementwise (g no-lineal); el modelo debe recuperar g(aₖ) por tarea activa y 0 en inactivas.
- Mide R² y sign-accuracy por hilo vs K. Compara: ENTRENADO vs SIN ENTRENAR; códigos FIJOS vs APRENDIDOS;
  g lineal vs no-lineal. Responde: ¿el entrenamiento habilita procesar K hilos en paralelo, y hasta qué K?
"""

from __future__ import annotations

import json

import torch
import torch.nn as nn

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

D, N, H = 64, 128, 512          # dim canal, nº tareas, ancho oculto
KMAX_TRAIN = 16
STEPS, BS, LR = 4000, 256, 2e-3
EVAL_KS = [1, 2, 4, 8, 16, 24, 32, 48]
GS = {"identity": lambda a: a, "square": lambda a: a * a, "relu": lambda a: torch.relu(a)}


def make_codes(learn):
    C = torch.randn(D, N, device=dev)
    C = C / C.norm(dim=0, keepdim=True)
    return nn.Parameter(C) if learn else C


def batch(C, bs, K, g):
    # vectorizado: K activos por ejemplo via top-K de ruido
    idx = torch.rand(bs, N, device=dev).topk(K, dim=1).indices       # (bs,K)
    a = torch.zeros(bs, N, device=dev)
    vals = torch.empty(bs, K, device=dev).uniform_(-1, 1)
    a.scatter_(1, idx, vals)
    x = a @ C.T
    y = g(a)
    mask = (a != 0).float()
    return x, y, mask, a


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, H), nn.GELU(), nn.Linear(H, H), nn.GELU(), nn.Linear(H, N))

    def forward(self, x):
        return self.net(x)


def evaluate(model, C, g):
    model.eval()
    out = {}
    with torch.no_grad():
        for K in EVAL_KS:
            x, y, mask, a = batch(C, 1024, K, g)
            pred = model(x)
            # métricas SOLO en posiciones activas
            err = (((pred - y) * mask).pow(2).sum() / (y * mask).pow(2).sum().clamp_min(1e-9)).sqrt()
            r2 = 1 - ((pred - y) * mask).pow(2).sum() / (((y - (y*mask).sum()/mask.sum()) * mask).pow(2).sum().clamp_min(1e-9))
            sign_acc = ((torch.sign(pred) == torch.sign(y)) * mask).sum() / mask.sum()
            out[K] = {"rel_err": round(float(err), 3), "r2": round(float(r2), 3), "sign_acc": round(float(sign_acc), 3)}
    return out


def train(g, learn_codes):
    C = make_codes(learn_codes)
    model = MLP().to(dev)
    params = list(model.parameters()) + ([C] if learn_codes else [])
    opt = torch.optim.Adam(params, lr=LR)
    untrained = evaluate(model, C, g)
    model.train()
    for step in range(STEPS):
        K = int(torch.randint(1, KMAX_TRAIN + 1, (1,)))
        x, y, mask, a = batch(C, BS, K, g)
        pred = model(x)
        loss = (pred - y).pow(2).mean()          # MSE sobre todas las N (inactivas->0 enseña a suprimir)
        opt.zero_grad(); loss.backward(); opt.step()
    trained = evaluate(model, C, g)
    return {"untrained": untrained, "trained": trained}


def main():
    results = {"meta": {"D": D, "N": N, "H": H, "kmax_train": KMAX_TRAIN, "steps": STEPS}, "runs": {}}
    for g_name in ["identity", "square"]:
        for learn in [False, True]:
            tag = f"{g_name}_{'learnedC' if learn else 'fixedC'}"
            print(f"\n=== {tag} (g={g_name}, códigos {'aprendidos' if learn else 'fijos'}) ===")
            res = train(GS[g_name], learn)
            results["runs"][tag] = res
            print("  K  | sin-entrenar r2 | ENTRENADO r2 | ENTRENADO sign_acc | rel_err")
            for K in EVAL_KS:
                u, t = res["untrained"][K], res["trained"][K]
                print(f"  {K:3d} |     {u['r2']:+.2f}      |   {t['r2']:+.2f}     |     {t['sign_acc']:.2f}        | {t['rel_err']:.2f}")
    json.dump(results, open("train_superposition_result.json", "w"), indent=2)
    print("\n→ train_superposition_result.json")
    print(f"(D={D} canal, N={N} tareas → sobre-suscripción {N/D:.1f}×. 'Paraleliza' si r2 entrenado alto a K>1.)")


if __name__ == "__main__":
    main()
