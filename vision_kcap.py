"""¿Tiene la VISIÓN un D_task por token MENOR que el lenguaje → más hilos CDMA-multiplexables (K_cap mayor)?

Mirror EXACTO de train_transformer_superposition.py (mismo mecanismo de superposición: códigos ±1 por hilo,
z = Σ_k E[x_k] ⊙ c_k + pos, transformer, des-modular salida ⊙ c_k, cabeza por hilo). Única variable = D_task
de los datos. Tres modos con TODO idéntico (V, T, D, capas, pasos, tarea):

- image : parches CIFAR-10 (8×8×3) → V=64 tokens visuales (k-means). Redundancia espacial ⇒ D_task bajo.
- text  : char-level wikitext-2, alfabeto de 64 chars. D_task lenguaje real.
- random: i.i.d. uniforme V=64. Predecir enmascarado es imposible ⇒ piso=azar (valida que el harness no fuga).

Tarea común = MASKED-TOKEN PREDICTION: se enmascara MASK_FRAC posiciones por hilo (embedding [MASK]); el
modelo predice el token original en las posiciones enmascaradas. Loss/accuracy SOLO en enmascarados, por hilo.
K_cap = max K con acc(K) ≥ 0.8·acc(K=1). Hipótesis: K_cap_image > K_cap_text > K_cap_random≈1.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import itertools
import json

import numpy as np
import torch
import torch.nn as nn

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

V, T, D = 64, 64, 128  # T=64: parches 4×4 (grid 8×8) → más contexto espacial / redundancia
NLAYERS, NHEAD = 3, 4
KMAX_TRAIN = 8
EVAL_KS = [1, 2, 3, 4, 6, 8]
STEPS, BS, LR = 6000, 256, 3e-4
MASK_FRAC = 0.5
MASK_ID = V  # fila extra en el embedding


class Model(nn.Module):
    def __init__(self, kmax):
        super().__init__()
        self.E = nn.Embedding(V + 1, D)  # +1 = [MASK]
        self.pos = nn.Parameter(torch.randn(1, T, D) * 0.02)
        codes = torch.sign(torch.randn(kmax, D)); codes[codes == 0] = 1
        self.register_buffer("codes", codes)
        enc = nn.TransformerEncoderLayer(D, NHEAD, 4 * D, batch_first=True, activation="gelu", dropout=0.0)
        self.enc = nn.TransformerEncoder(enc, NLAYERS)
        self.head = nn.Linear(D, V)

    def forward(self, xin):                                   # xin: (bs,K,T) con MASK_ID en huecos
        K = xin.shape[1]
        c = self.codes[:K].view(1, K, 1, D)
        emb = self.E(xin)                                    # (bs,K,T,d)
        z = (emb * c).sum(1) + self.pos                      # (bs,T,d) superposición
        h = self.enc(z)                                      # (bs,T,d)
        hk = h.unsqueeze(1) * c                              # (bs,K,T,d) des-modular
        return self.head(hk)                                 # (bs,K,T,V)


def mask_batch(x):
    """x:(bs,K,T) tokens limpios → (xin con MASK, target, mask_bool)."""
    m = torch.rand(x.shape, device=dev) < MASK_FRAC
    m[(~m).all(-1, keepdim=True).expand_as(m) & (torch.arange(T, device=dev) == 0)] = True  # ≥1 mask/fila
    xin = torch.where(m, torch.full_like(x, MASK_ID), x)
    return xin, x, m


# ---------- fuentes de datos: cada una devuelve sampler(bs,K)->(bs,K,T) de tokens en [0,V) ----------

def make_image_source():
    from torchvision import datasets, transforms
    ds = datasets.CIFAR10("./data", train=True, download=True, transform=transforms.ToTensor())
    imgs = torch.stack([ds[i][0] for i in range(len(ds))])           # (N,3,32,32)
    P = 4                                                            # grid 8×8 = 64 parches
    G = 32 // P
    patches = imgs.unfold(2, P, P).unfold(3, P, P)                   # (N,3,G,G,P,P)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(imgs.shape[0], G * G, 3 * P * P)  # (N,64,48)
    flat = patches.reshape(-1, patches.shape[-1]).numpy()
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    idx = np.random.choice(flat.shape[0], 20000, replace=False)
    scaler = StandardScaler().fit(flat[idx])
    km = KMeans(n_clusters=V, n_init=4, random_state=0).fit(scaler.transform(flat[idx]))
    toks = km.predict(scaler.transform(flat)).reshape(patches.shape[0], G * G)  # (N,64) ids en [0,V)
    toks = torch.tensor(toks, dtype=torch.long, device=dev)
    N = toks.shape[0]

    def sample(bs, K):
        sel = torch.randint(0, N, (bs, K), device=dev)
        return toks[sel]                                            # (bs,K,64)
    return sample


def make_text_source():
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
    txt = " ".join(r["text"] for r in itertools.islice(iter(ds), 20000) if r["text"].strip()).lower()
    from collections import Counter
    common = [ch for ch, _ in Counter(txt).most_common(V - 1)]      # V-1 + 1 OOV = V
    cmap = {ch: i for i, ch in enumerate(common)}
    oov = V - 1
    ids = torch.tensor([cmap.get(ch, oov) for ch in txt], dtype=torch.long, device=dev)
    N = ids.numel()

    def sample(bs, K):
        starts = torch.randint(0, N - T, (bs, K), device=dev)
        off = torch.arange(T, device=dev)
        return ids[(starts.unsqueeze(-1) + off)]                    # (bs,K,T)
    return sample


def make_random_source():
    def sample(bs, K):
        return torch.randint(0, V, (bs, K, T), device=dev)
    return sample


SOURCES = {"image": make_image_source, "text": make_text_source, "random": make_random_source}


def evaluate(model, sample):
    model.eval(); out = {}
    with torch.no_grad():
        for K in EVAL_KS:
            x = sample(1024, K)
            xin, y, m = mask_batch(x)
            pred = model(xin).argmax(-1)
            out[K] = round(float(((pred == y) & m).sum() / m.sum()), 3)
    model.train()
    return out


def run_mode(name):
    print(f"\n===== modo {name} =====", flush=True)
    sample = SOURCES[name]()
    model = Model(KMAX_TRAIN).to(dev)
    untrained = evaluate(model, sample)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for step in range(STEPS):
        K = int(torch.randint(1, KMAX_TRAIN + 1, (1,)))
        x = sample(BS, K)
        xin, y, m = mask_batch(x)
        logits = model(xin)
        loss = (nn.functional.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="none")
                .reshape(x.shape)[m]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0:
            print(f"  step {step:5d} K={K} loss={loss.item():.3f}", flush=True)
    trained = evaluate(model, sample)
    ch = 1 / V
    a1 = trained[1]
    s1 = a1 - ch                                                    # señal sobre azar en K=1
    # K_cap honesto: señal(K) ≥ 0.8·señal(K=1). Requiere señal real (s1>0.05) si no → K_cap=1 (sin multiplexar).
    if s1 <= 0.05:
        kcap = 1
    else:
        kcap = max([1] + [K for K in EVAL_KS if (trained[K] - ch) >= 0.8 * s1])
    print(f"  chance={ch:.3f}  acc(K=1)={a1:.3f}  señal(K=1)={s1:.3f}  K_cap(señal≥80%)={kcap}")
    print("  K  | sin-entrenar | ENTRENADO | señal=acc-chance")
    for K in EVAL_KS:
        print(f"  {K:2d} |    {untrained[K]:.3f}     |   {trained[K]:.3f}   |  {trained[K]-ch:.3f}")
    return {"untrained": untrained, "trained": trained, "acc_K1": a1, "signal_K1": round(s1, 4),
            "k_cap_signal80": kcap, "chance": round(ch, 4)}


def main():
    meta = {"V": V, "T": T, "D": D, "layers": NLAYERS, "head": NHEAD, "steps": STEPS,
            "mask_frac": MASK_FRAC, "eval_ks": EVAL_KS}
    res = {"meta": meta}
    for name in ["random", "text", "image"]:
        res[name] = run_mode(name)
    json.dump(res, open("vision_kcap_result.json", "w"), indent=2)
    ki, kt, kr = res["image"]["k_cap_signal80"], res["text"]["k_cap_signal80"], res["random"]["k_cap_signal80"]
    print(f"\nVEREDICTO  K_cap: image={ki}  text={kt}  random={kr}")
    print("Teoría confirmada si image > text (visión multiplexa más por D_task menor)." if ki > kt
          else "Teoría NO confirmada: image no supera a text.")
    print("→ vision_kcap_result.json")


if __name__ == "__main__":
    main()
