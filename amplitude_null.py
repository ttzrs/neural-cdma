"""Red-team del test de viabilidad: (1) barrido de amplitud para localizar el régimen lineal-útil,
(2) CONTROL NULO para descartar que la recuperación sea un artefacto del readout bien-condicionado.

Control nulo: se inyecta la señal `a`, se obtiene Δh_real, se de-mezcla -> â. Se mide:
  - sign_acc(â, a)       : recuperación de la señal inyectada (debe ser alta si funciona)
  - sign_acc(â, a_null)  : contra una señal aleatoria NO inyectada (debe ser ~0.5 = azar)
Si ambos son altos, la "recuperación" es un artefacto. Si el primero >> el segundo, es señal real.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B"
L_INJECT, L_READ, POS = 6, 16, -1
M_SENSE, B = 2700, 64
EPS_FRAC = 0.05
K = 128
AMPS = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5]
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
    global _inject
    _inject = inject
    with torch.no_grad():
        out = model(input_ids=ids.repeat(bs, 1))
    _inject = None
    return _cap["h"].clone(), out.logits[:, POS, :].float()


h_ref, logits_ref = run(None, 1); h_ref = h_ref[0]
ref_norm = float(h_ref.norm()); eps = EPS_FRAC * ref_norm

# sensing -> J -> SVD
V = torch.randn(M_SENSE, d, device=dev); V = V / V.norm(dim=1, keepdim=True) * eps
dH = torch.zeros(M_SENSE, d, device=dev)
for i in range(0, M_SENSE, B):
    c = V[i:i+B]; h, _ = run(c, c.shape[0]); dH[i:i+c.shape[0]] = h - h_ref
JT = torch.linalg.lstsq(V, dH).solution; J = JT.T
U, S, Vh = torch.linalg.svd(J)
print(f"d={d} L{L_INJECT}->L{L_READ} ‖h_ref‖={ref_norm:.2f}  σ_max={float(S.max()):.3f}")


def alpha(kind):
    if kind == "extracted":
        return Vh[:K].T.contiguous()
    if kind == "gold_like":
        return torch.sign(torch.randn(d, K, device=dev)) / d**0.5
    g = torch.randn(d, K, device=dev); return g / g.norm(dim=0, keepdim=True)


print(f"\nK={K}.  Por amplitud: gap=‖Δreal−J·inj‖/‖Δreal‖ ; acc=recuperación ; null=control(azar~0.5) ; KL=coste salida")
for af in AMPS:
    amp = af * ref_norm
    line = f"amp={af:.2f}(‖inj‖={amp:.1f}) | "
    for kind in ["extracted", "gold_like", "random"]:
        C = alpha(kind)
        a = torch.sign(torch.randn(K, device=dev))
        a_null = torch.sign(torch.randn(K, device=dev))
        inj = C @ a; inj = inj / inj.norm() * amp
        h_inj, logits_inj = run(inj.unsqueeze(0), 1)
        dHr = h_inj[0] - h_ref
        gap = float((dHr - J @ inj).norm() / dHr.norm())
        R = J @ C
        a_hat = torch.linalg.lstsq(R, dHr.unsqueeze(1)).solution.squeeze(1)
        acc = float((torch.sign(a_hat) == a).float().mean())
        acc_null = float((torch.sign(a_hat) == a_null).float().mean())
        kl = float(torch.nn.functional.kl_div(torch.log_softmax(logits_inj[0], -1),
                                               torch.softmax(logits_ref[0], -1), reduction="sum"))
        line += f"{kind[:4]}: acc={acc:.2f} null={acc_null:.2f} gap={gap:.2f} KL={kl:.2f} | "
    print(line)
print("\nLectura: viable si existe una amplitud con (acc_extracted alto) Y (null~0.5) Y (KL bajo) Y (gap bajo).")
