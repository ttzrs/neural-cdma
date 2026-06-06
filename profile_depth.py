"""¿Qué FRACCIÓN del transformer preserva la superposición de K hilos (= es paralelizable)?

Inyecta K perturbaciones δₖ (códigos de hilo) en una capa temprana sobre un contexto compartido, y mide
por CADA capa el error de superposición:
    err(L) = ‖Δh_L(Σδₖ) − Σ_k Δh_L(δₖ)‖ / ‖Δh_L(Σδₖ)‖
err≈0 hasta la capa L* ⇒ el tramo [inject, L*] es lineal en los hilos ⇒ esos K hilos comparten ese cómputo.
% paralelizable = L* / n_layers. Comparado EXTRAÍDA vs random, barriendo K y amplitud.

(Honesto: mide hilos como PERTURBACIONES de un contexto compartido — paraleliza K variantes cercanas, no K
inputs arbitrarios. La cabeza/logits ya se sabe que colapsa: aquí medimos cuánto del stack aguanta antes.)
"""

from __future__ import annotations

import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B"
L_INJECT, POS = 2, -1          # inyectar temprano para abarcar casi todo el stack
M_SENSE, B, EPS_FRAC = 1500, 64, 0.05
KS = [2, 4, 8, 16]
AMPS = [0.05, 0.15]
THRESH = 0.10                  # err por debajo = superposición preservada
torch.manual_seed(0); np.random.seed(0)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(dev).eval()
d = model.config.hidden_size
nL = model.config.num_hidden_layers
layers = model.model.layers
_inject = None


def inj_hook(mod, inp, out):
    if _inject is None:
        return None
    h = out[0] if isinstance(out, tuple) else out
    h = h.clone(); h[:, POS, :] = h[:, POS, :] + _inject.to(h.dtype)
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


layers[L_INJECT].register_forward_hook(inj_hook)
ids = tok("The contract stipulates that the delivery must be completed within thirty days of signing.",
          return_tensors="pt").input_ids.to(dev)


def run(inject=None, bs=1):
    """Devuelve hidden_states: tensor (n_layers+1, bs, d) en la posición POS."""
    global _inject
    _inject = inject
    with torch.no_grad():
        out = model(input_ids=ids.repeat(bs, 1), output_hidden_states=True)
    _inject = None
    hs = torch.stack([h[:, POS, :] for h in out.hidden_states], dim=0)  # (nL+1, bs, d)
    return hs.float()


hs_ref = run(None, 1)[:, 0, :]                       # (nL+1, d)
ref_norm = float(hs_ref[L_INJECT].norm()); eps = EPS_FRAC * ref_norm

# sensing en L_INJECT -> J -> SVD (para la base extraída)
V = torch.randn(M_SENSE, d, device=dev); V = V / V.norm(dim=1, keepdim=True) * eps
dH = torch.zeros(M_SENSE, d, device=dev)
read_at = min(L_INJECT + 8, nL)
for i in range(0, M_SENSE, B):
    c = V[i:i+B]; hs = run(c, c.shape[0]); dH[i:i+c.shape[0]] = hs[read_at] - hs_ref[read_at]
JT = torch.linalg.lstsq(V, dH).solution
_, _, Vh = torch.linalg.svd(JT.T)
print(f"d={d} n_layers={nL} inject@L{L_INJECT} ‖h‖={ref_norm:.2f}")


def codes(kind, K):
    if kind == "extracted":
        return Vh[:K].T.contiguous()
    g = torch.randn(d, K, device=dev); return g / g.norm(dim=0, keepdim=True)


results = {"meta": {"model": MODEL, "n_layers": nL, "inject": L_INJECT, "thresh": THRESH}, "sweep": []}
for amp_f in AMPS:
    amp = amp_f * ref_norm
    for K in KS:
        for kind in ["extracted", "random"]:
            C = codes(kind, K) * amp                 # (d,K)
            hs_seq = run(C.T, K)                      # (nL+1, K, d): hilo k aislado
            dseq = hs_seq - hs_ref.unsqueeze(1)       # (nL+1, K, d)
            sum_seq = dseq.sum(dim=1)                  # (nL+1, d) = Σ Δh_k
            hs_mix = run(C.sum(1, keepdim=True).T, 1)[:, 0, :]   # (nL+1, d)
            dmix = hs_mix - hs_ref                     # (nL+1, d)
            err = (dmix - sum_seq).norm(dim=1) / (dmix.norm(dim=1) + 1e-9)   # (nL+1,)
            err_np = err.cpu().numpy()
            # primera capa (>L_INJECT) donde err supera el umbral
            brk = nL
            for L in range(L_INJECT + 1, nL + 1):
                if err_np[L] > THRESH:
                    brk = L; break
            frac = (brk - L_INJECT) / (nL - L_INJECT)
            results["sweep"].append({"amp": amp_f, "K": K, "kind": kind,
                                     "break_layer": brk, "frac_parallel": round(frac, 3),
                                     "err_curve": [round(float(x), 3) for x in err_np]})
    print(f"\n=== amp={amp_f} (‖δ‖/hilo={amp:.2f}) ; umbral err<{THRESH} ===")
    for K in KS:
        e = next(r for r in results["sweep"] if r["amp"] == amp_f and r["K"] == K and r["kind"] == "extracted")
        rnd = next(r for r in results["sweep"] if r["amp"] == amp_f and r["K"] == K and r["kind"] == "random")
        print(f"  K={K:2d} | EXTR rompe@L{e['break_layer']:2d} ({e['frac_parallel']*100:4.0f}% del stack) "
              f"|| RAND rompe@L{rnd['break_layer']:2d} ({rnd['frac_parallel']*100:4.0f}%)")

json.dump(results, open("profile_depth_result.json", "w"), indent=2)
print(f"\n(n_layers={nL}. frac = capas hasta ruptura / capas tras la inyección.)")
print("→ profile_depth_result.json")
