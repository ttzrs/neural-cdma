"""Experimento de proveniencia por marca de agua CDMA en el residual stream.

Pregunta honesta: ¿puede un watermark CDMA (sign-codes inyectados en el residual)
(1) recuperarse del logit output (lectura de caja negra) con bit-accuracy >> 0.5, y
(2) dejar el texto generado esencialmente inalterado (KL≈0, token_agreement≈1)?

Si la recuperación desde logits sólo funciona cuando el texto ya está corrompido (KL alto),
el watermark es sólo white-box (residual). Eso es el negativo honesto.

Diseño:
  - Modelo: Qwen/Qwen2.5-0.5B fp32 en GPU
  - Inyección: capa L_INJECT, posición last-token del prompt
  - Lectura residual: capa L_READ (hook)
  - Lectura logits: lm_head(norm(h_read))  →  full vocab
  - B ∈ {16, 64, 256} códigos sign ±1/√d
  - α/‖h‖ ∈ {0.01, 0.02, 0.05, 0.1, 0.2, 0.4}
  - 16 prompts × 8 posiciones de continuación = 128 muestras por (B, α)
  - Métricas:
      kl              : KL(watermarked || clean) en distribución next-token
      token_agreement : fracción posiciones donde argmax es idéntico
      bitacc_residual : bit-accuracy decodificando desde Δh en L_READ
      bitacc_logits   : bit-accuracy decodificando desde Δlogits
      bitacc_control  : control (códigos aleatorios no inyectados → debe ser ≈0.5)
"""
from __future__ import annotations

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------- #
# Configuración                                                                  #
# ---------------------------------------------------------------------------- #
MODEL      = "Qwen/Qwen2.5-0.5B"
L_INJECT   = 8           # capa de inyección
L_READ     = 20          # capa de lectura residual (antes de lm_head)
N_LAST     = -1          # posición: último token del prompt
B_LIST     = [16, 64, 256]
ALPHA_LIST = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
N_PROMPTS  = 16
N_POS      = 8           # posiciones de continuación por prompt
GEN_LEN    = N_POS + 2   # tokens a generar (descartamos los primeros 2 para warm-up)

PROMPTS = [
    "The contract stipulates that the delivery must be completed within thirty days",
    "Researchers at the university announced a breakthrough in quantum computing",
    "The central bank raised interest rates by fifty basis points to combat inflation",
    "She carefully opened the envelope and read the letter inside with great attention",
    "The software update introduced several new features including improved performance",
    "Scientists discovered a new species of deep-sea fish near the Mariana Trench",
    "The city council voted unanimously to approve the new public transit expansion",
    "His theory of relativity fundamentally changed our understanding of space and time",
    "The ancient manuscript revealed details about trading routes in the medieval period",
    "Climate models predict more frequent and severe weather events in coming decades",
    "The startup secured twenty million in series A funding from venture capital firms",
    "Engineers designed a bridge capable of withstanding category five hurricane winds",
    "The chef prepared a traditional dish using locally sourced seasonal ingredients",
    "A new algorithm can detect fraudulent transactions with ninety eight percent accuracy",
    "The museum unveiled its largest collection of impressionist paintings since opening",
    "Astronomers observed a rare alignment of planets visible from the northern hemisphere",
]
assert len(PROMPTS) == N_PROMPTS

# ---------------------------------------------------------------------------- #
# Cargar modelo                                                                  #
# ---------------------------------------------------------------------------- #
dev = torch.device("cuda")
print(f"Cargando {MODEL} fp32 en {dev}...")
tokenizer = AutoModelForCausalLM.from_pretrained  # placeholder
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(dev).eval()
d = model.config.hidden_size
n_layers = model.config.num_hidden_layers
print(f"d={d}  n_layers={n_layers}  L_inject={L_INJECT}  L_read={L_READ}")

# ---------------------------------------------------------------------------- #
# Hooks                                                                          #
# ---------------------------------------------------------------------------- #
_inject_delta: torch.Tensor | None = None   # (d,) perturbación a añadir
_h_read: dict[str, torch.Tensor] = {}


def _make_inject_hook(pos: int):
    def hook(mod, inp, out):
        if _inject_delta is None:
            return None
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        h[:, pos, :] = h[:, pos, :] + _inject_delta.to(h.dtype).unsqueeze(0)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    return hook


def _make_read_hook(key: str, pos: int):
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        _h_read[key] = h[:, pos, :].detach().float().clone()
    return hook


_hinj_handle = model.model.layers[L_INJECT].register_forward_hook(
    _make_inject_hook(N_LAST)
)
_hread_handle = model.model.layers[L_READ].register_forward_hook(
    _make_read_hook("L_read", N_LAST)
)

# ---------------------------------------------------------------------------- #
# Helpers de forward                                                             #
# ---------------------------------------------------------------------------- #

def run_single(input_ids: torch.Tensor, inject: torch.Tensor | None = None):
    """Corre el modelo en un prompt (1, T), devuelve (h_read (d,), logits (V,))."""
    global _inject_delta
    _inject_delta = inject
    with torch.no_grad():
        out = model(input_ids=input_ids)
    _inject_delta = None
    h = _h_read["L_read"][0]          # (d,)
    logits = out.logits[0, -1, :].float()   # último token → next-token dist
    return h, logits


def generate_continuation(input_ids: torch.Tensor, n_new: int,
                           inject: torch.Tensor | None = None) -> torch.Tensor:
    """Genera n_new tokens (greedy). inject se aplica en CADA step de generación.
    Devuelve tensor (n_new, d) de h_read y (n_new, V) de logits en cada paso."""
    global _inject_delta
    ids = input_ids.clone()
    hs_list, logits_list = [], []
    for _ in range(n_new):
        _inject_delta = inject
        with torch.no_grad():
            out = model(input_ids=ids)
        _inject_delta = None
        h = _h_read["L_read"][0].clone()   # (d,)
        lg = out.logits[0, -1, :].float()  # (V,)
        hs_list.append(h)
        logits_list.append(lg)
        next_tok = lg.argmax(dim=-1, keepdim=True).unsqueeze(0)  # (1,1)
        ids = torch.cat([ids, next_tok], dim=1)
    return torch.stack(hs_list), torch.stack(logits_list)  # (n, d), (n, V)


# ---------------------------------------------------------------------------- #
# Construir códigos sign CDMA                                                    #
# ---------------------------------------------------------------------------- #

def make_codes(B: int, d: int, device: torch.device) -> torch.Tensor:
    """Genera B sign-codes ±1/√d, shape (d, B)."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(123)
    C = torch.sign(torch.randn(d, B, generator=gen)).to(device) / (d ** 0.5)
    return C   # (d, B)


# ---------------------------------------------------------------------------- #
# Codificación / decodificación                                                  #
# ---------------------------------------------------------------------------- #

def encode_bits(bits: torch.Tensor, C: torch.Tensor, alpha_abs: float) -> torch.Tensor:
    """bits (B,) ∈ {±1}, C (d,B) → vector delta (d,) con energía=alpha_abs."""
    delta = C @ bits.float()   # (d,)
    delta = delta / (delta.norm() + 1e-12) * alpha_abs
    return delta


def decode_from_vector(delta_obs: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Decodifica bits proyectando delta_obs sobre C. Devuelve bits_hat (B,) ∈ {±1}."""
    scores = C.T @ delta_obs   # (B,)
    return torch.sign(scores)


def bit_accuracy(bits_true: torch.Tensor, bits_hat: torch.Tensor) -> float:
    return float((bits_true == bits_hat).float().mean().item())


# ---------------------------------------------------------------------------- #
# Experimento principal                                                          #
# ---------------------------------------------------------------------------- #
results: dict = {}

for B in B_LIST:
    print(f"\n{'='*60}")
    print(f"B={B} códigos")
    C = make_codes(B, d, dev)    # (d, B) fijos para este B
    results[str(B)] = {}

    for alpha_frac in ALPHA_LIST:
        kl_list, agree_list = [], []
        bitacc_res_list, bitacc_log_list, bitacc_ctrl_list = [], [], []

        for p_idx, prompt in enumerate(PROMPTS):
            ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
            T0 = ids.shape[1]

            # referencia (sin inyección): obtener h_ref, logits_ref para los N_POS pasos
            h_ref_base, logits_ref_base = run_single(ids)
            h_ref_norm = float(h_ref_base.norm())
            alpha_abs = alpha_frac * h_ref_norm

            # bits a inyectar (fijos para este (B, alpha, prompt))
            rng = torch.Generator(device="cpu")
            rng.manual_seed(p_idx * 1000 + B)
            bits_true = torch.sign(torch.randn(B, generator=rng)).to(dev)
            delta = encode_bits(bits_true, C, alpha_abs)  # (d,)

            # Forward clean y watermarked para N_POS posiciones
            # Usamos generación paso a paso para leer el residual en CADA paso
            hs_clean, logits_clean = generate_continuation(ids, N_POS, inject=None)
            hs_wm,    logits_wm    = generate_continuation(ids, N_POS, inject=delta)
            # hs: (N_POS, d)   logits: (N_POS, V)

            for pos_i in range(N_POS):
                # --- fidelidad output ---
                log_p_wm  = F.log_softmax(logits_wm[pos_i], dim=-1)
                p_clean   = F.softmax(logits_clean[pos_i], dim=-1)
                kl = float(F.kl_div(log_p_wm, p_clean, reduction="sum").item())
                agree = float((logits_wm[pos_i].argmax() == logits_clean[pos_i].argmax()).float().item())
                kl_list.append(max(kl, 0.0))
                agree_list.append(agree)

                # --- recuperación desde residual ---
                delta_res = hs_wm[pos_i] - hs_clean[pos_i]   # (d,)
                bits_hat_res = decode_from_vector(delta_res, C)
                bitacc_res_list.append(bit_accuracy(bits_true, bits_hat_res))

                # --- recuperación desde logits ---
                delta_log = logits_wm[pos_i] - logits_clean[pos_i]   # (V,)
                # proyectar delta_log sobre espacio de C: necesitamos C en vocab space
                # lm_head transporte: si h → logits = lm_head(norm(h)), la perturbación
                # en logits es ≈ J_lmhead @ delta_h (linealización).
                # En Qwen2.5-0.5B lm_head es Linear(d, V) sin bias.
                # Proyectamos delta_log sobre W @ C (cada columna de C mapeada por W):
                W = model.lm_head.weight.float()       # (V, d)
                # Necesitamos el efecto de la norma de la capa final.
                # Simplificación honesta: usamos delta_logits directamente,
                # correlando con W @ C[:,k] para cada código k.
                WC = W @ C   # (V, B)   = logit-space representation de cada código
                scores_log = WC.T @ delta_log   # (B,)
                bits_hat_log = torch.sign(scores_log)
                bitacc_log_list.append(bit_accuracy(bits_true, bits_hat_log))

                # --- control: códigos aleatorios no inyectados ---
                rng_ctrl = torch.Generator(device="cpu")
                rng_ctrl.manual_seed(p_idx * 9999 + pos_i * 37 + B * 7)
                bits_ctrl = torch.sign(torch.randn(B, generator=rng_ctrl)).to(dev)
                bits_hat_ctrl = decode_from_vector(delta_res, C)
                # control = comparar bits_ctrl (no inyectados) contra bits_hat_res
                bitacc_ctrl_list.append(bit_accuracy(bits_ctrl, bits_hat_res))

        kl_mean    = float(np.mean(kl_list))
        agree_mean = float(np.mean(agree_list))
        res_mean   = float(np.mean(bitacc_res_list))
        log_mean   = float(np.mean(bitacc_log_list))
        ctrl_mean  = float(np.mean(bitacc_ctrl_list))

        results[str(B)][str(alpha_frac)] = {
            "kl":              round(kl_mean,  5),
            "token_agreement": round(agree_mean, 4),
            "bitacc_residual": round(res_mean,  4),
            "bitacc_logits":   round(log_mean,  4),
            "bitacc_control":  round(ctrl_mean, 4),
        }
        print(
            f"  α/‖h‖={alpha_frac:.2f}  "
            f"KL={kl_mean:.4f}  agree={agree_mean:.3f}  "
            f"res={res_mean:.3f}  log={log_mean:.3f}  ctrl={ctrl_mean:.3f}"
        )

# ---------------------------------------------------------------------------- #
# Guardar y reportar                                                             #
# ---------------------------------------------------------------------------- #
out_path = "watermark_provenance_result.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResultados guardados en {out_path}")

# Tabla compacta
print("\n" + "="*90)
print(f"{'B':>6} {'α/‖h‖':>7} {'KL':>8} {'agree':>7} {'res_acc':>8} {'log_acc':>8} {'ctrl_acc':>8}")
print("-"*90)
for B in B_LIST:
    for alpha_frac in ALPHA_LIST:
        r = results[str(B)][str(alpha_frac)]
        print(
            f"{B:>6} {alpha_frac:>7.2f}  "
            f"{r['kl']:>8.4f}  "
            f"{r['token_agreement']:>7.3f}  "
            f"{r['bitacc_residual']:>8.3f}  "
            f"{r['bitacc_logits']:>8.3f}  "
            f"{r['bitacc_control']:>8.3f}"
        )
print("="*90)

# Veredicto
print("\nVEREDICTO:")
usable = []
for B in B_LIST:
    for alpha_frac in ALPHA_LIST:
        r = results[str(B)][str(alpha_frac)]
        if (r["bitacc_logits"] > 0.65
                and r["token_agreement"] >= 0.95
                and r["kl"] < 0.05):
            usable.append((B, alpha_frac, r))

if usable:
    best = max(usable, key=lambda x: x[2]["bitacc_logits"])
    B_b, a_b, r_b = best
    print(
        f"WATERMARK USABLE DETECTADO: B={B_b}, α/‖h‖={a_b:.2f} → "
        f"log_acc={r_b['bitacc_logits']:.3f}, KL={r_b['kl']:.4f}, "
        f"agree={r_b['token_agreement']:.3f}  (señal detectable SIN corromper el texto)"
    )
else:
    # Buscar si hay recuperación logit pero sólo a costa del texto
    logit_works = [(B, a, results[str(B)][str(a)])
                   for B in B_LIST for a in ALPHA_LIST
                   if results[str(B)][str(a)]["bitacc_logits"] > 0.65]
    if logit_works:
        print(
            "NEGATIVO PARCIAL: recuperación logit>0.65 existe pero sólo cuando "
            "KL>0.05 o token_agreement<0.95 — watermark sólo funciona corrompiendo el texto."
        )
    else:
        print(
            "NEGATIVO: la marca de agua CDMA NO sobrevive al lm_head como señal "
            "detectable (log_acc≈0.5 en todo el rango). "
            "Watermark sólo existe en el residual (white-box). "
            "La no-linealidad del lm_head destruye la proyección lineal del código."
        )
