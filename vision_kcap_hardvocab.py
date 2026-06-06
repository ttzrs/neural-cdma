"""Test LIMPIO visión vs lenguaje a VOCAB DURO emparejado (V=512, D_task alto en ambos).

La 1ª corrida (vision_kcap.py, V=64) comparó image vs CHAR-text: char es D_task bajo (K_cap=8) → no era
test limpio. Aquí ambos modos tienen vocab grande y por tanto D_task alto por token:
- image    : parches CIFAR 4×4 → kmeans V=512 (vocab visual fino, parche menos predecible).
- text_bpe : tokens BPE Qwen sobre wikitext, top-(V-1) ids + OOV → lenguaje real D_task alto (cf. K_cap=2 en validate_llm_full).
- random   : i.i.d. uniforme V=512 (piso).

Mismo transformer/superposición/máscara que vision_kcap.py. Reporta K_cap(señal≥80%) Y throughput=K·señal(K)
(bits agregados recuperados por forward). Hipótesis afinada: a vocab duro, image throughput > text_bpe.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import itertools
import json

import numpy as np
import torch
import torch.nn as nn

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

V, T, D = 512, 64, 128
NLAYERS, NHEAD = 3, 4
KMAX_TRAIN = 8
EVAL_KS = [1, 2, 3, 4, 6, 8]
STEPS, BS, LR = 8000, 256, 3e-4
MASK_FRAC = 0.5
MASK_ID = V


class Model(nn.Module):
    def __init__(self, kmax):
        super().__init__()
        self.E = nn.Embedding(V + 1, D)
        self.pos = nn.Parameter(torch.randn(1, T, D) * 0.02)
        codes = torch.sign(torch.randn(kmax, D)); codes[codes == 0] = 1
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


def make_image_source():
    from torchvision import datasets, transforms
    ds = datasets.CIFAR10("./data", train=True, download=True, transform=transforms.ToTensor())
    imgs = torch.stack([ds[i][0] for i in range(len(ds))])
    P = 4; G = 32 // P
    patches = imgs.unfold(2, P, P).unfold(3, P, P)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(imgs.shape[0], G * G, 3 * P * P)
    flat = patches.reshape(-1, patches.shape[-1]).numpy()
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler
    idx = np.random.choice(flat.shape[0], 50000, replace=False)
    scaler = StandardScaler().fit(flat[idx])
    km = MiniBatchKMeans(n_clusters=V, n_init=3, batch_size=4096, random_state=0).fit(scaler.transform(flat[idx]))
    toks = torch.tensor(km.predict(scaler.transform(flat)).reshape(imgs.shape[0], G * G),
                        dtype=torch.long, device=dev)
    N = toks.shape[0]

    def sample(bs, K):
        return toks[torch.randint(0, N, (bs, K), device=dev)]
    return sample


def make_text_bpe_source():
    from collections import Counter
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
    txt = " ".join(r["text"] for r in itertools.islice(iter(ds), 20000) if r["text"].strip())
    ids = tok(txt).input_ids
    common = [i for i, _ in Counter(ids).most_common(V - 1)]   # V-1 ids frecuentes + 1 OOV
    remap = {i: r for r, i in enumerate(common)}
    oov = V - 1
    seq = torch.tensor([remap.get(i, oov) for i in ids], dtype=torch.long, device=dev)
    N = seq.numel()

    def sample(bs, K):
        starts = torch.randint(0, N - T, (bs, K), device=dev)
        off = torch.arange(T, device=dev)
        return seq[(starts.unsqueeze(-1) + off)]
    return sample


def make_random_source():
    def sample(bs, K):
        return torch.randint(0, V, (bs, K, T), device=dev)
    return sample


SOURCES = {"image": make_image_source, "text_bpe": make_text_bpe_source, "random": make_random_source}


def evaluate(model, sample):
    model.eval(); out = {}
    with torch.no_grad():
        for K in EVAL_KS:
            x = sample(1024, K)
            xin, y, m = mask_batch(x)
            pred = model(xin).argmax(-1)
            out[K] = round(float(((pred == y) & m).sum() / m.sum()), 4)
    model.train()
    return out


def run_mode(name):
    print(f"\n===== modo {name} =====", flush=True)
    sample = SOURCES[name]()
    model = Model(KMAX_TRAIN).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for step in range(STEPS):
        K = int(torch.randint(1, KMAX_TRAIN + 1, (1,)))
        x = sample(BS, K)
        xin, y, m = mask_batch(x)
        loss = (nn.functional.cross_entropy(model(xin).reshape(-1, V), y.reshape(-1), reduction="none")
                .reshape(x.shape)[m]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0:
            print(f"  step {step:5d} K={K} loss={loss.item():.3f}", flush=True)
    trained = evaluate(model, sample)
    ch = 1 / V
    s1 = trained[1] - ch
    kcap = 1 if s1 <= 0.02 else max([1] + [K for K in EVAL_KS if (trained[K] - ch) >= 0.8 * s1])
    thru = {K: round(K * (trained[K] - ch), 4) for K in EVAL_KS}
    print(f"  chance={ch:.4f}  acc(K=1)={trained[1]:.4f}  señal(K=1)={s1:.4f}  K_cap={kcap}")
    print("  K  | acc | señal=acc-ch | throughput=K·señal")
    for K in EVAL_KS:
        print(f"  {K:2d} | {trained[K]:.4f} | {trained[K]-ch:.4f} | {thru[K]:.4f}")
    return {"trained": trained, "signal_K1": round(s1, 4), "k_cap_signal80": kcap,
            "throughput": thru, "thru_max": max(thru.values()), "chance": round(ch, 5)}


def main():
    res = {"meta": {"V": V, "T": T, "D": D, "steps": STEPS, "mask_frac": MASK_FRAC, "eval_ks": EVAL_KS}}
    for name in ["random", "text_bpe", "image"]:
        res[name] = run_mode(name)
    json.dump(res, open("vision_kcap_hardvocab_result.json", "w"), indent=2)
    ki, kt = res["image"]["k_cap_signal80"], res["text_bpe"]["k_cap_signal80"]
    ti, tt = res["image"]["thru_max"], res["text_bpe"]["thru_max"]
    print(f"\nVEREDICTO vocab duro V={V}:  K_cap image={ki} text_bpe={kt} random={res['random']['k_cap_signal80']}")
    print(f"  throughput_max image={ti} text_bpe={tt}")
    print(f"  K_cap: {'IMAGE gana' if ki > kt else 'text gana' if kt > ki else 'empate'} | "
          f"throughput: {'IMAGE gana' if ti > tt else 'text gana' if tt > ti else 'empate'}")
    print("→ vision_kcap_hardvocab_result.json")


if __name__ == "__main__":
    main()
