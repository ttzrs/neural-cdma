"""Test de viabilidad: ¿se pueden EXTRAER del LLM códigos de canal (eigen-coding del residual
stream) y multiplexar K señales mejor que Gold codes externos o aleatorios — sobreviviendo el forward?

Procedimiento (honesto, mide en el modelo REAL, no solo en la linealización):
1. Canal = mapa residual stream capa L_INJECT -> capa L_READ en la última posición de token.
2. Estimar la Jacobiana J (d×d) del canal por *sensing*: B perturbaciones aleatorias pequeñas inyectadas
   en h_L, regresión lineal sobre Δh_{L'}. SVD(J) -> base propia del canal.
3. Tres alfabetos de K códigos en R^d:
   - extracted : top-K vectores singulares derechos de J (los modos que más sobreviven).
   - gold_like : vectores ±1/√d de baja coherencia (código externo, agnóstico al canal).
   - random    : gaussianos normalizados.
4. Multiplexar: inyectar Σ aₖcₖ (a = signos ±1) a energía total fija; correr el modelo REAL; leer Δh_{L'};
   de-mezclar con el readout R=J·C (lstsq) y medir sign-recovery accuracy y correlación, en función de K.
5. Honestidad: medir el gap no-lineal ‖Δh_real − J·inj‖/‖Δh_real‖ y el coste en logits (KL).
6. Capacidad: rango efectivo del canal (σ sobre el ruido del sensing).
"""

from __future__ import annotations

import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B"
L_INJECT, L_READ, POS = 6, 16, -1
M_SENSE = 2700           # nº de perturbaciones de sensing (~3·d)
B = 64                   # batch
EPS_FRAC = 0.05          # norma de perturbación = EPS_FRAC · ‖h_ref‖
AMP_FRAC = 0.25          # energía total de la inyección de prueba = AMP_FRAC · ‖h_ref‖
KS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
torch.manual_seed(0); np.random.seed(0)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(dev).eval()
d = model.config.hidden_size
layers = model.model.layers

_inject = None       # (batch, d) o None
_captured = {}


def inj_hook(mod, inp, out):
    if _inject is None:
        return None
    h = out[0] if isinstance(out, tuple) else out
    h = h.clone()
    h[:, POS, :] = h[:, POS, :] + _inject.to(h.dtype)
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


def cap_hook(mod, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    _captured["h"] = h[:, POS, :].detach().float()


layers[L_INJECT].register_forward_hook(inj_hook)
layers[L_READ].register_forward_hook(cap_hook)

text = "The contract stipulates that the delivery must be completed within thirty days of signing."
ids = tok(text, return_tensors="pt").input_ids.to(dev)


def run(inject=None, bs=1):
    global _inject
    batch = ids.repeat(bs, 1)
    _inject = inject
    with torch.no_grad():
        out = model(input_ids=batch)
    _inject = None
    return _captured["h"].clone(), out.logits[:, POS, :].float()


# --- referencia ---
h_ref, logits_ref = run(None, 1)
h_ref = h_ref[0]
ref_norm = float(h_ref.norm())
eps = EPS_FRAC * ref_norm
print(f"device={dev} d={d} L{L_INJECT}->L{L_READ}  ‖h_ref‖={ref_norm:.2f}  eps={eps:.3f}")

# --- 1) sensing del canal ---
V = torch.randn(M_SENSE, d, device=dev)
V = V / V.norm(dim=1, keepdim=True) * eps
dH = torch.zeros(M_SENSE, d, device=dev)
for i in range(0, M_SENSE, B):
    chunk = V[i:i + B]
    h, _ = run(chunk, chunk.shape[0])
    dH[i:i + chunk.shape[0]] = h - h_ref
# J: dH ≈ V @ J^T  ->  J^T = pinv(V) @ dH
JT = torch.linalg.lstsq(V, dH).solution            # (d,d)
J = JT.T
# residual del ajuste lineal (ruido/no-linealidad del sensing)
pred = V @ JT
sense_res = float((dH - pred).norm() / dH.norm())
U, S, Vh = torch.linalg.svd(J)                      # J = U diag(S) Vh ; modos entrada = filas de Vh
S_np = S.cpu().numpy()
noise = S_np.max() * sense_res
eff_rank = int((S_np > noise).sum())
print(f"sensing residual (no-lin/ruido)={sense_res:.3f}  σ_max={S_np.max():.3f}  "
      f"σ_min={S_np.min():.3e}  rango_efectivo≈{eff_rank}/{d}")


def alphabet(kind, K):
    if kind == "extracted":
        return Vh[:K].T.contiguous()                       # (d,K) top-K input singular vectors
    if kind == "gold_like":
        C = torch.sign(torch.randn(d, K, device=dev)) / (d ** 0.5)
        return C
    g = torch.randn(d, K, device=dev)
    return g / g.norm(dim=0, keepdim=True)


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    den = (a.norm() * b.norm())
    return float((a @ b) / den) if den > 0 else 0.0


results = {"meta": {"model": MODEL, "L_inject": L_INJECT, "L_read": L_READ,
                    "sense_residual": sense_res, "eff_rank": eff_rank, "d": d},
           "sweep": []}
amp = AMP_FRAC * ref_norm
for K in KS:
    row = {"K": K}
    for kind in ["extracted", "gold_like", "random"]:
        C = alphabet(kind, K)                              # (d,K)
        a = torch.sign(torch.randn(K, device=dev))         # señales ±1
        inj = C @ a                                        # (d,)
        inj = inj / inj.norm() * amp                       # energía total fija
        h_inj, logits_inj = run(inj.unsqueeze(0), 1)
        dH_real = (h_inj[0] - h_ref)
        dH_lin = J @ inj
        gap = float((dH_real - dH_lin).norm() / dH_real.norm())
        # de-mezcla con readout predicho R=J·C
        R = J @ C                                          # (d,K)
        a_hat = torch.linalg.lstsq(R, dH_real.unsqueeze(1)).solution.squeeze(1)
        sign_acc = float((torch.sign(a_hat) == a).float().mean())
        c = corr(a.float(), a_hat.float())
        kl = float(torch.nn.functional.kl_div(
            torch.log_softmax(logits_inj[0], -1), torch.softmax(logits_ref[0], -1),
            reduction="sum"))
        row[kind] = {"sign_acc": round(sign_acc, 3), "corr": round(c, 3),
                     "nonlin_gap": round(gap, 3), "logit_kl": round(kl, 4)}
    results["sweep"].append(row)
    e, g, r = row["extracted"], row["gold_like"], row["random"]
    print(f"K={K:4d} | extracted acc={e['sign_acc']:.2f} corr={e['corr']:+.2f} gap={e['nonlin_gap']:.2f}"
          f" || gold acc={g['sign_acc']:.2f} corr={g['corr']:+.2f} || rand acc={r['sign_acc']:.2f}"
          f" | KL(ext)={e['logit_kl']:.3f}")

json.dump(results, open("viability_result.json", "w"), indent=2)
print("\n→ viability_result.json")

# veredicto rápido: K máximo con sign_acc>=0.95 por alfabeto
def kmax(kind):
    ks = [r["K"] for r in results["sweep"] if r[kind]["sign_acc"] >= 0.95]
    return max(ks) if ks else 0
print(f"\nK_max(acc>=0.95):  extracted={kmax('extracted')}  gold_like={kmax('gold_like')}  random={kmax('random')}")
