"""¿CÓDIGO multiplexa mejor que LENGUAJE bajo CDMA? (predicción del negativo de visión).

Lógica: visión colisiona porque tokens correlacionados/bajo-rango; lenguaje multiplexa porque iid-like.
→ predicción: fuente MÁS decorrelacionada multiplexa MEJOR. Código (independencia local alta) debería dar
K_cap > prosa. Clave: extender EVAL_KS más allá de 8 (text_bpe ya satura en 8) para VER la diferencia.

Mismo harness que vision_kcap_hardvocab.py (superposición CDMA, masked-pred, V=512/T=64/d=128) pero KMAX=32.
Fuentes (Qwen BPE top-511, mismo régimen): code (.py locales) vs text_bpe (wikitext) vs random. K_cap + throughput.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import glob
import itertools
import json

import torch
import torch.nn as nn

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

V, T, D = 512, 64, 128
NLAYERS, NHEAD = 3, 4
KMAX_TRAIN = 32
EVAL_KS = [1, 2, 4, 8, 12, 16, 24, 32]
STEPS, BS, LR = 8000, 256, 3e-4
MASK_FRAC = 0.5
MASK_ID = V
EVAL_N = 256


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


def _bpe_seq_source(text):
    from collections import Counter
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    ids = tok(text).input_ids
    common = [i for i, _ in Counter(ids).most_common(V - 1)]
    remap = {i: r for r, i in enumerate(common)}
    oov = V - 1
    seq = torch.tensor([remap.get(i, oov) for i in ids], dtype=torch.long, device=dev)
    N = seq.numel()
    # autocorrelación lag-1 (proxy de decorrelación): corr entre token y siguiente
    a = seq[:-1].float(); b = seq[1:].float()
    ac = float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std() + 1e-8))

    def sample(bs, K):
        starts = torch.randint(0, N - T, (bs, K), device=dev)
        off = torch.arange(T, device=dev)
        return seq[(starts.unsqueeze(-1) + off)]
    return sample, {"tokens": N, "autocorr_lag1": round(ac, 4)}


def make_code_source():
    # Fuente de "código" portable: los .py instalados de torch/transformers (sin rutas hardcoded).
    import os
    import torch as _torch
    import transformers as _tf
    paths = []
    for mod in (_torch, _tf):
        root = os.path.dirname(mod.__file__)
        paths += glob.glob(root + "/**/*.py", recursive=True)
    buf, tot = [], 0
    for p in paths:
        try:
            s = open(p, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        buf.append(s); tot += len(s)
        if tot > 6_000_000:
            break
    return _bpe_seq_source("\n".join(buf))


def make_text_source():
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
    txt = " ".join(r["text"] for r in itertools.islice(iter(ds), 20000) if r["text"].strip())
    return _bpe_seq_source(txt)


def make_random_source():
    def sample(bs, K):
        return torch.randint(0, V, (bs, K, T), device=dev)
    return sample, {"autocorr_lag1": 0.0}


SOURCES = {"code": make_code_source, "text_bpe": make_text_source, "random": make_random_source}


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


def run_mode(name):
    print(f"\n===== modo {name} =====", flush=True)
    sample, meta = SOURCES[name]()
    print(f"  meta={meta}", flush=True)
    model = Model(KMAX_TRAIN).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for step in range(STEPS):
        kcur = 1 + int((KMAX_TRAIN - 1) * min(1.0, (step / STEPS) / 0.75))  # currículum K
        K = int(torch.randint(1, kcur + 1, (1,)))
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
    print(f"  acc(K=1)={trained[1]:.4f} señal={s1:.4f} K_cap={kcap} thru_max={max(thru.values()):.3f}")
    for K in EVAL_KS:
        print(f"  {K:2d} | acc {trained[K]:.4f} | señal {trained[K]-ch:.4f} | thru {thru[K]:.3f}")
    return {"trained": trained, "signal_K1": round(s1, 4), "k_cap_signal80": kcap,
            "throughput": thru, "thru_max": max(thru.values()), "chance": round(ch, 5), "meta": meta}


def main():
    res = {"meta": {"V": V, "T": T, "D": D, "steps": STEPS, "kmax": KMAX_TRAIN, "eval_ks": EVAL_KS}}
    for name in ["random", "text_bpe", "code"]:
        res[name] = run_mode(name)
    json.dump(res, open("code_kcap_result.json", "w"), indent=2)
    kc, kt = res["code"]["k_cap_signal80"], res["text_bpe"]["k_cap_signal80"]
    tc, tt = res["code"]["thru_max"], res["text_bpe"]["thru_max"]
    print(f"\nVEREDICTO: K_cap code={kc} text={kt} random={res['random']['k_cap_signal80']} | thru code={tc:.3f} text={tt:.3f}")
    print(f"  autocorr: code={res['code']['meta'].get('autocorr_lag1')} text={res['text_bpe']['meta'].get('autocorr_lag1')}")
    print("  → predicción CONFIRMADA: código multiplexa más (decorrelado)." if kc > kt or tc > tt
          else "  → predicción NO confirmada: código no supera a lenguaje.")
    print("→ code_kcap_result.json")


if __name__ == "__main__":
    main()
