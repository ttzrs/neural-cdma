"""¿Se pueden procesar K HILOS de cómputo a la vez en UN forward, multiplexados con los códigos extraídos?

Ground-truth secuencial vs superpuesto:
- Hilo k = perturbación δₖ = amp·cₖ inyectada en L_INJECT (cₖ = código de hilo).
- SECUENCIAL (verdad): un forward por hilo → Δh_k (en L_READ) y target-token_k (el que más sube en logits).
- SUPERPUESTO: UN forward con Σₖ δₖ → de-mezclar.
- Métricas:
   (1) h_{L'} (canal ~lineal): corr(demix_k, secuencial_k) + control nulo.
   (2) LOGITS (cabeza NO-lineal, test fuerte): ¿el target_k de cada hilo sigue elevado en el superpuesto?
       fracción de hilos cuyo target sobrevive (rank<=R en el superpuesto). = cómputo paralelo real.
Comparado EXTRAÍDA vs random ; barriendo amplitud y K.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B"
L_INJECT, L_READ, POS = 6, 16, -1
M_SENSE, B = 2700, 64
EPS_FRAC = 0.05
KS = [2, 4, 8, 16, 32]
AMPS = [0.05, 0.15, 0.30]
RANK_OK = 10  # un target "sobrevive" si queda en el top-RANK_OK del forward superpuesto
torch.manual_seed(0); np.random.seed(0)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(dev).eval()
d = model.config.hidden_size
layers = model.model.layers
_inject, _cap = None, {}


def inj_hook(mod, inp, out):
    if _inject is None:
        return None
    h = out[0] if isinstance(out, tuple) else out
    h = h.clone(); h[:, POS, :] = h[:, POS, :] + _inject.to(h.dtype)
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


def cap_hook(mod, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    _cap["h"] = h[:, POS, :].detach().float()


layers[L_INJECT].register_forward_hook(inj_hook)
layers[L_READ].register_forward_hook(cap_hook)
ids = tok("The contract stipulates that the delivery must be completed within thirty days of signing.",
          return_tensors="pt").input_ids.to(dev)


def run(inject=None, bs=1):
    """inject: (bs,d) o None. Devuelve h_{L'} (bs,d) y logits (bs,vocab)."""
    global _inject
    _inject = inject
    with torch.no_grad():
        out = model(input_ids=ids.repeat(bs, 1))
    _inject = None
    return _cap["h"].clone(), out.logits[:, POS, :].float()


h_ref_b, logits_ref_b = run(None, 1)
h_ref = h_ref_b[0]; logits_ref = logits_ref_b[0]
ref_norm = float(h_ref.norm()); eps = EPS_FRAC * ref_norm

# --- sensing -> J -> SVD ---
V = torch.randn(M_SENSE, d, device=dev); V = V / V.norm(dim=1, keepdim=True) * eps
dH = torch.zeros(M_SENSE, d, device=dev)
for i in range(0, M_SENSE, B):
    c = V[i:i+B]; h, _ = run(c, c.shape[0]); dH[i:i+c.shape[0]] = h - h_ref
JT = torch.linalg.lstsq(V, dH).solution; J = JT.T
U, S, Vh = torch.linalg.svd(J)
print(f"d={d} L{L_INJECT}->L{L_READ} ‖h_ref‖={ref_norm:.2f} σ_max={float(S.max()):.3f}")


def codes(kind, K):
    if kind == "extracted":
        return Vh[:K].T.contiguous()
    g = torch.randn(d, K, device=dev); return g / g.norm(dim=0, keepdim=True)


def corr(a, b):
    a = a - a.mean(); b = b - b.mean(); den = a.norm() * b.norm()
    return float((a @ b) / den) if den > 0 else 0.0


results = {"meta": {"model": MODEL, "L": [L_INJECT, L_READ], "rank_ok": RANK_OK}, "sweep": []}
for amp_f in AMPS:
    amp = amp_f * ref_norm
    for K in KS:
        for kind in ["extracted", "random"]:
            C = codes(kind, K)                       # (d,K) códigos de hilo (norma 1 c/u)
            scale = amp                              # energía por hilo
            # --- SECUENCIAL (verdad) ---
            inj_seq = (C * scale).T                  # (K,d): hilo k aislado
            h_seq, logits_seq = run(inj_seq, K)      # (K,d),(K,vocab)
            dh_seq = h_seq - h_ref                   # (K,d)
            targets = (logits_seq - logits_ref).argmax(dim=1)  # token-objetivo por hilo
            # --- SUPERPUESTO (un forward) ---
            inj_mix = (C * scale).sum(dim=1, keepdim=True).T   # (1,d) = Σ δₖ
            h_mix, logits_mix = run(inj_mix, 1)
            dh_mix = (h_mix[0] - h_ref)
            # (1) recuperación en h_{L'}: demix con readout R=J·C
            R = J @ C                                # (d,K)
            coef = torch.linalg.lstsq(R, dh_mix.unsqueeze(1)).solution.squeeze(1)  # (K,)
            # reconstrucción por hilo del demix vs secuencial (dirección)
            recon = R * coef                         # (d,K) contribución estimada de cada hilo
            h_corr = float(np.mean([corr(recon[:, k], (J @ inj_seq[k])) for k in range(K)]))
            # control nulo: corr del demix del hilo k con un hilo NO inyectado
            Cnull = codes(kind, K)
            Rnull = J @ Cnull
            coef_null = torch.linalg.lstsq(Rnull, dh_mix.unsqueeze(1)).solution.squeeze(1)
            null_energy = float(coef_null.abs().mean() / (coef.abs().mean() + 1e-9))
            # (2) supervivencia en LOGITS: ranking de cada target en el forward superpuesto
            order = torch.argsort(logits_mix[0], descending=True)
            rank_of = {int(t): int((order == t).nonzero()[0, 0]) for t in targets.tolist()}
            survive = float(np.mean([rank_of[int(t)] < RANK_OK for t in targets.tolist()]))
            # ¿los targets son distintos entre hilos? (si no, el test es trivial)
            distinct = len(set(targets.tolist())) / K
            row = {"amp": amp_f, "K": K, "kind": kind,
                   "h_corr": round(h_corr, 3), "null_ratio": round(null_energy, 3),
                   "logit_survival": round(survive, 3), "targets_distinct": round(distinct, 2)}
            results["sweep"].append(row)
    # imprimir bloque por amplitud
    print(f"\n=== amp={amp_f} (‖δ‖/hilo={amp:.2f}) ===")
    for K in KS:
        e = next(r for r in results["sweep"] if r["amp"] == amp_f and r["K"] == K and r["kind"] == "extracted")
        rnd = next(r for r in results["sweep"] if r["amp"] == amp_f and r["K"] == K and r["kind"] == "random")
        print(f"  K={K:2d} | EXTR h_corr={e['h_corr']:.2f} logit_surv={e['logit_survival']:.2f} "
              f"distinct={e['targets_distinct']:.2f} null={e['null_ratio']:.2f}"
              f" || RAND logit_surv={rnd['logit_survival']:.2f} h_corr={rnd['h_corr']:.2f}")

json.dump(results, open("multithread_result.json", "w"), indent=2)
print("\n→ multithread_result.json")
print("Lectura: 'procesar K hilos' funciona si logit_survival alto (targets distintos sobreviven el "
      "forward superpuesto) con null bajo. h_corr alto = el canal intermedio sí separa; la caída de "
      "logit_survival vs K/amp mide cuánto la no-linealidad de la cabeza ahoga el multiplexado.")
