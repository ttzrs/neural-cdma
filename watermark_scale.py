"""Watermark a escala: N sesiones de generación larga, p-valor POR SESIÓN, paired vs blind.

Corrige el protocolo de watermark_provenance.py (v1) en cuatro puntos:
  1. Generación autoregresiva larga con KV-cache (gen_len tokens por sesión), inyectando
     el delta en cada token emitido (la marca persiste en el contexto, como en producción).
  2. Métricas por sesión (una generación = un litigio): bits acertados y p binomial de
     ESA sesión; nunca se promedian deltas entre prompts.
  3. Dos protocolos de verificación:
       paired : el verificador re-ejecuta el TEXTO disputado sin clave (teacher-forced) y
                decodifica  Cᵀ mean_t (h'_t − h_t).
       blind  : el verificador solo tiene el log  h'_t  y una calibración estática
                calib = mean Cᵀ h_t  sobre un corpus limpio disjunto:  Cᵀ mean_t h'_t − calib.
  4. Controles: clave errónea (2), sin marca verificada con la clave correcta (FPR),
     y "sustitución": delta sumado directamente en la capa de lectura sin ejecutar las
     capas intermedias (¿la marca prueba ejecución o solo etiqueta?).

Uso:
  python watermark_scale.py --model Qwen/Qwen2.5-0.5B --dtype bf16 \
      --n-sessions 128 --n-calib 128 --gen-len 128 --out results/watermark_scale_qwen0.5b_bf16.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as scipy_stats
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent / "watermark_pkg"))
from cdma_watermark.codec import make_codes  # noqa: E402
from cdma_watermark.ecc import ecc_decode, ecc_encode  # noqa: E402

# ---------------------------------------------------------------------------- #
# Configuración                                                                  #
# ---------------------------------------------------------------------------- #
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
ap.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
ap.add_argument("--n-sessions", type=int, default=128)
ap.add_argument("--n-calib", type=int, default=128)
ap.add_argument("--prompt-len", type=int, default=32)
ap.add_argument("--gen-len", type=int, default=128)
ap.add_argument("--alpha", type=float, default=0.01)
ap.add_argument("--inject-layer", type=int, default=None, help="por defecto round(L/3)")
ap.add_argument("--read-layer", type=int, default=None, help="por defecto round(5L/6)")
ap.add_argument("--owner-key", default="acme-corp-2026")
ap.add_argument("--wrong-keys", default="thief-key,competitor-xyz")
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--only-nomark", action="store_true",
                help="solo el control nulo paired sin marca: generación limpia verificada con la clave "
                     "correcta por re-ejecución teacher-forced (mide el ruido numérico caché vs pase completo)")
args = ap.parse_args()

DTYPE = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
B_LOGICAL = 16            # payload lógico (session ID)
ECC_R = 4                 # B=64 físico = 16 lógicos × 4 repeticiones
B_PHYS = B_LOGICAL * ECC_R
T_CURVE = [8, 16, 32, 64, 128]   # tokens promediados para la curva p(T)
P_THRESH = [1e-2, 1e-3, 1e-4, 1e-5]
WRONG_KEYS = [k for k in args.wrong_keys.split(",") if k]

dev = torch.device("cuda")
torch.manual_seed(args.seed)

# ---------------------------------------------------------------------------- #
# Modelo                                                                         #
# ---------------------------------------------------------------------------- #
print(f"Cargando {args.model} {args.dtype}...", flush=True)
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=DTYPE).to(dev).eval()
d = model.config.hidden_size
n_layers = model.config.num_hidden_layers
L_INJECT = args.inject_layer if args.inject_layer is not None else round(n_layers / 3)
L_READ = args.read_layer if args.read_layer is not None else round(5 * n_layers / 6)
print(f"d={d} L={n_layers} inject={L_INJECT} read={L_READ}", flush=True)

# ---------------------------------------------------------------------------- #
# Datos: WikiText-2 validation → prompts de prompt_len tokens                     #
# ---------------------------------------------------------------------------- #
from datasets import load_dataset  # noqa: E402

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
need = args.n_sessions + args.n_calib
prompts: list[torch.Tensor] = []
for row in ds:
    txt = row["text"].strip()
    if len(txt) < 200 or txt.startswith("="):
        continue
    ids = tok(txt, return_tensors="pt").input_ids[0]
    if ids.numel() < args.prompt_len + 16:
        continue
    prompts.append(ids[: args.prompt_len])
    if len(prompts) >= need:
        break
assert len(prompts) == need, f"solo {len(prompts)} prompts válidos"
session_prompts = prompts[: args.n_sessions]
calib_prompts = prompts[args.n_sessions:]
print(f"prompts: {len(session_prompts)} sesiones + {len(calib_prompts)} calibración", flush=True)

# ---------------------------------------------------------------------------- #
# Códigos                                                                        #
# ---------------------------------------------------------------------------- #
codes = {  # (d, B_PHYS) — los primeros 16 códigos sirven para B=16; los 64 para B=64
    "owner": make_codes(args.owner_key, B_PHYS, d, dev),
}
for k in WRONG_KEYS:
    codes[k] = make_codes(k, B_PHYS, d, dev)


def payload_for_session(i: int) -> torch.Tensor:
    """16 bits lógicos ±1 deterministas por sesión."""
    g = torch.Generator(device="cpu")
    g.manual_seed(10_000 + i)
    return torch.sign(torch.randn(B_LOGICAL, generator=g)).to(dev)


# ---------------------------------------------------------------------------- #
# Hooks: inyección en la última posición (cada token emitido) y lectura           #
# ---------------------------------------------------------------------------- #
_state = {"dir": None, "read": None, "sub_delta": None}


def _inject_hook(mod, inp, out):
    if _state["dir"] is None:
        return None
    h = out[0] if isinstance(out, tuple) else out
    h = h.clone()
    last = h[:, -1, :]
    delta = args.alpha * last.float().norm(dim=-1, keepdim=True) * _state["dir"].unsqueeze(0)
    h[:, -1, :] = last + delta.to(h.dtype)
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


def _read_hook(mod, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    _state["read"] = h.detach().float().clone()   # (1, T, d)


model.model.layers[L_INJECT].register_forward_hook(_inject_hook)
model.model.layers[L_READ].register_forward_hook(_read_hook)


def set_payload(bits_phys: torch.Tensor | None):
    """bits_phys (B,) ±1 con B∈{16,64} → dirección unitaria C b / ‖C b‖ (o None = sin marca)."""
    if bits_phys is None:
        _state["dir"] = None
        return
    C = codes["owner"][:, : bits_phys.numel()]
    v = C @ bits_phys.float()
    _state["dir"] = v / (v.norm() + 1e-12)


@torch.no_grad()
def generate(prompt_ids: torch.Tensor, n_new: int):
    """Greedy con KV-cache. Devuelve (tokens (n_new,), h_read (n_new, d), logits (n_new, V) fp32 en CPU)."""
    ids = prompt_ids.unsqueeze(0).to(dev)
    out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    toks, hs, lgs = [], [], []
    lg = out.logits[0, -1, :].float()
    for _ in range(n_new):
        hs.append(_state["read"][0, -1, :].clone())
        lgs.append(lg.cpu())
        nxt = lg.argmax().view(1, 1)
        toks.append(int(nxt))
        out = model(input_ids=nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        lg = out.logits[0, -1, :].float()
    return torch.tensor(toks), torch.stack(hs), torch.stack(lgs)


@torch.no_grad()
def teacher_forced_read(prompt_ids: torch.Tensor, gen_tokens: torch.Tensor) -> torch.Tensor:
    """Pase limpio (sin marca) sobre prompt+texto disputado. Devuelve h_read en las posiciones
    que predicen cada token generado: (n_new, d). Posición i-ésima = la que emitió gen_tokens[i]."""
    set_payload(None)
    full = torch.cat([prompt_ids, gen_tokens]).unsqueeze(0).to(dev)
    model(input_ids=full, use_cache=False)
    P = prompt_ids.numel()
    return _state["read"][0, P - 1: P - 1 + gen_tokens.numel(), :].clone()


# ---------------------------------------------------------------------------- #
# Decodificación / estadística                                                    #
# ---------------------------------------------------------------------------- #
def pval(matches: int, n: int) -> float:
    return float(scipy_stats.binom.sf(matches - 1, n, 0.5))


def decode_scores(score: torch.Tensor, bits_true: torch.Tensor, ecc: bool) -> dict:
    """score (B_phys,) → bits; si ecc, mayoría ×R a 16 lógicos. Devuelve matches/p para físico y lógico."""
    bhat = torch.sign(score)
    bhat[bhat == 0] = 1.0
    n = bhat.numel()
    bits_phys_true = ecc_encode(bits_true, ECC_R) if ecc else bits_true
    m_phys = int((bhat == bits_phys_true).sum())
    res = {"n_phys": n, "m_phys": m_phys, "p_phys": pval(m_phys, n)}
    if ecc:
        blog = ecc_decode(bhat, ECC_R)
        m_log = int((blog == bits_true).sum())
        res.update({"m_log": m_log, "p_log": pval(m_log, B_LOGICAL)})
    return res


def proj(h: torch.Tensor, key: str, B: int) -> torch.Tensor:
    """h (T, d) → (T, B) proyecciones sobre los primeros B códigos de key."""
    return h @ codes[key][:, :B]


# ---------------------------------------------------------------------------- #
# 1) Calibración estática: mean_t Cᵀ h_t sobre generaciones limpias disjuntas      #
# ---------------------------------------------------------------------------- #
t0 = time.time()
set_payload(None)
calib_acc = {k: torch.zeros(B_PHYS, device=dev) for k in codes}
n_calib_tok = 0
for i, p in enumerate([] if args.only_nomark else calib_prompts):
    _, h, _ = generate(p, args.gen_len)
    for k in codes:
        calib_acc[k] += proj(h, k, B_PHYS).sum(dim=0)
    n_calib_tok += h.shape[0]
    if (i + 1) % 16 == 0:
        print(f"  calib {i+1}/{len(calib_prompts)}  {time.time()-t0:.0f}s", flush=True)
calib = {k: v / max(n_calib_tok, 1) for k, v in calib_acc.items()}
# también la desviación típica del score por token (para el paper: SNR del canal)
print(f"calibración lista ({n_calib_tok} tokens, {time.time()-t0:.0f}s)", flush=True)

# ---------------------------------------------------------------------------- #
# 2) Sesiones                                                                     #
# ---------------------------------------------------------------------------- #
sessions: list[dict] = []
t0 = time.time()
for i, p in enumerate(session_prompts):
    bits = payload_for_session(i)                       # (16,) lógicos
    rec: dict = {"session": i}

    # --- clean generation (referencia de fidelidad + control sin marca) ---
    set_payload(None)
    tok_c, h_c, lg_c = generate(p, args.gen_len)

    # --- control nulo paired: texto limpio re-ejecutado sin clave y decodificado con la clave correcta ---
    h_tf_c = teacher_forced_read(p, tok_c)
    for B, ecc in ((B_LOGICAL, False), (B_PHYS, True)):
        S_c = proj(h_c, "owner", B)
        S_tfc = proj(h_tf_c, "owner", B)
        rec[f"B{B}"] = {
            "nomark_paired": decode_scores((S_c - S_tfc).mean(0), bits, ecc),
            "nomark_paired_T": {str(T): decode_scores((S_c[:T] - S_tfc[:T]).mean(0), bits, ecc)
                                for T in T_CURVE if T <= args.gen_len},
            "nomark_delta_norm_mean": float((h_c - h_tf_c).norm(dim=-1).mean()),
        }
    if args.only_nomark:
        sessions.append(rec)
        if (i + 1) % 8 == 0 or i == 0:
            print(f"  sesión {i+1}/{len(session_prompts)}  nomark_paired={rec['B16']['nomark_paired']['m_phys']}/16 "
                  f"{time.time()-t0:.0f}s", flush=True)
        continue

    for B, ecc in ((B_LOGICAL, False), (B_PHYS, True)):
        bits_phys = ecc_encode(bits, ECC_R) if ecc else bits
        set_payload(bits_phys)
        tok_w, h_w, lg_w = generate(p, args.gen_len)
        set_payload(None)

        # --- fidelidad: divergencia autoregresiva ---
        eq = (tok_w == tok_c)
        first_div = int((~eq).nonzero()[0]) if (~eq).any() else args.gen_len
        # KL(wm‖clean) por paso, solo mientras el contexto es idéntico (hasta first_div incluido)
        n_kl = min(first_div + 1, args.gen_len)
        logp_w = F.log_softmax(lg_w[:n_kl], dim=-1)
        p_c = F.softmax(lg_c[:n_kl], dim=-1)
        kl = (logp_w.exp() * (logp_w - torch.log(p_c + 1e-30))).sum(dim=-1)
        # el pase limpio teacher-forced sobre el texto disputado (verificador paired)
        h_tf = teacher_forced_read(p, tok_w)

        r: dict = {
            "first_div": first_div,
            "identical": bool(first_div == args.gen_len),
            "kl_mean_pre_div": float(kl.mean()),
            "kl_max_pre_div": float(kl.max()),
            "agree_frac": float(eq.float().mean()),
        }
        # --- owner, paired / blind ---
        S_w = proj(h_w, "owner", B)            # (T, B)
        S_tf = proj(h_tf, "owner", B)
        r["paired"] = decode_scores((S_w - S_tf).mean(0), bits, ecc)
        r["blind"] = decode_scores(S_w.mean(0) - calib["owner"][:B], bits, ecc)
        r["blind_T"] = {str(T): decode_scores(S_w[:T].mean(0) - calib["owner"][:B], bits, ecc)
                        for T in T_CURVE if T <= args.gen_len}
        r["paired_T"] = {str(T): decode_scores((S_w[:T] - S_tf[:T]).mean(0), bits, ecc)
                         for T in T_CURVE if T <= args.gen_len}
        # --- claves erróneas (blind: lo que un tercero sin K haría) ---
        r["wrong"] = {}
        for k in WRONG_KEYS:
            S_wk = proj(h_w, k, B)
            S_tfk = proj(h_tf, k, B)
            r["wrong"][k] = {
                "paired": decode_scores((S_wk - S_tfk).mean(0), bits, ecc),
                "blind": decode_scores(S_wk.mean(0) - calib[k][:B], bits, ecc),
            }
        # --- control sin marca: generación limpia verificada con la clave correcta ---
        S_c = proj(h_c, "owner", B)
        r["nomark_blind"] = decode_scores(S_c.mean(0) - calib["owner"][:B], bits, ecc)
        # --- sustitución: delta sumado en la capa de lectura sin ejecutar inject..read ---
        # un nodo que no ejecuta las capas intermedias devuelve h_tf + α‖h‖·dir
        dirv = codes["owner"][:, :B] @ bits_phys.float()
        dirv = dirv / dirv.norm()
        h_sub = h_tf + args.alpha * h_tf.norm(dim=-1, keepdim=True) * dirv.unsqueeze(0)
        S_sub = proj(h_sub, "owner", B)
        r["substitution_paired"] = decode_scores((S_sub - S_tf).mean(0), bits, ecc)
        r["substitution_blind"] = decode_scores(S_sub.mean(0) - calib["owner"][:B], bits, ecc)
        # --- SNR del canal: energía del delta observado vs ruido de calibración ---
        r["delta_norm_mean"] = float((h_w - h_tf).norm(dim=-1).mean())
        r["h_norm_mean"] = float(h_tf.norm(dim=-1).mean())
        rec[f"B{B}"].update(r)

    sessions.append(rec)
    if (i + 1) % 8 == 0 or i == 0:
        r16 = rec["B16"]
        print(f"  sesión {i+1}/{len(session_prompts)}  first_div={r16['first_div']} "
              f"paired={r16['paired']['m_phys']}/16 blind={r16['blind']['m_phys']}/16 "
              f"sub={r16['substitution_paired']['m_phys']}/16  {time.time()-t0:.0f}s", flush=True)

# ---------------------------------------------------------------------------- #
# 3) Agregados                                                                    #
# ---------------------------------------------------------------------------- #
def agg_mode(get, key_p="p_phys", key_m="m_phys", n_key="n_phys") -> dict:
    ps = np.array([get(s)[key_p] for s in sessions])
    ms = np.array([get(s)[key_m] for s in sessions])
    n = get(sessions[0])[n_key]
    return {
        "n": int(n),
        "bitacc_mean": float(ms.mean() / n),
        "matches_median": float(np.median(ms)),
        "p_median": float(np.median(ps)),
        "p_q25": float(np.quantile(ps, 0.25)),
        "p_q75": float(np.quantile(ps, 0.75)),
        "p_geomean": float(np.exp(np.log(ps).mean())),
        "frac_p_below": {f"{t:g}": float((ps < t).mean()) for t in P_THRESH},
    }


summary: dict = {}
for B in (B_LOGICAL, B_PHYS):
    k = f"B{B}"
    if args.only_nomark:
        S = {"nomark_paired": agg_mode(lambda s: s[k]["nomark_paired"]),
             "nomark_paired_T": {T: agg_mode(lambda s, T=T: s[k]["nomark_paired_T"][T])
                                 for T in sessions[0][k]["nomark_paired_T"]},
             "nomark_delta_norm_mean": float(np.mean([s[k]["nomark_delta_norm_mean"] for s in sessions]))}
        if B == B_PHYS:
            S["nomark_paired_logical"] = agg_mode(lambda s: s[k]["nomark_paired"], "p_log", "m_log", "n_phys")
            S["nomark_paired_logical"]["n"] = B_LOGICAL
            S["nomark_paired_logical"]["bitacc_mean"] *= B_PHYS / B_LOGICAL
        summary[k] = S
        continue
    g = lambda s, m: s[k][m]  # noqa: E731
    fd = np.array([s[k]["first_div"] for s in sessions])
    S: dict = {
        "fidelity": {
            "frac_identical": float(np.mean([s[k]["identical"] for s in sessions])),
            "first_div_mean": float(fd.mean()),
            "first_div_median": float(np.median(fd)),
            "agree_frac_mean": float(np.mean([s[k]["agree_frac"] for s in sessions])),
            "kl_mean_pre_div": float(np.mean([s[k]["kl_mean_pre_div"] for s in sessions])),
            "kl_max_pre_div": float(np.max([s[k]["kl_max_pre_div"] for s in sessions])),
        },
        "paired": agg_mode(lambda s: s[k]["paired"]),
        "blind": agg_mode(lambda s: s[k]["blind"]),
        "paired_T": {T: agg_mode(lambda s, T=T: s[k]["paired_T"][T]) for T in sessions[0][k]["paired_T"]},
        "blind_T": {T: agg_mode(lambda s, T=T: s[k]["blind_T"][T]) for T in sessions[0][k]["blind_T"]},
        "wrong": {w: {"paired": agg_mode(lambda s, w=w: s[k]["wrong"][w]["paired"]),
                      "blind": agg_mode(lambda s, w=w: s[k]["wrong"][w]["blind"])} for w in WRONG_KEYS},
        "nomark_blind": agg_mode(lambda s: s[k]["nomark_blind"]),
        "nomark_paired": agg_mode(lambda s: s[k]["nomark_paired"]),
        "nomark_paired_T": {T: agg_mode(lambda s, T=T: s[k]["nomark_paired_T"][T]) for T in sessions[0][k]["nomark_paired_T"]},
        "substitution_paired": agg_mode(lambda s: s[k]["substitution_paired"]),
        "substitution_blind": agg_mode(lambda s: s[k]["substitution_blind"]),
        "delta_norm_mean": float(np.mean([s[k]["delta_norm_mean"] for s in sessions])),
        "h_norm_mean": float(np.mean([s[k]["h_norm_mean"] for s in sessions])),
    }
    if B == B_PHYS:  # también los 16 bits lógicos tras ECC
        for m in ("paired", "blind", "nomark_blind", "substitution_paired"):
            S[m + "_logical"] = agg_mode(lambda s, m=m: s[k][m], "p_log", "m_log", "n_phys")
            S[m + "_logical"]["n"] = B_LOGICAL
            S[m + "_logical"]["bitacc_mean"] = S[m + "_logical"]["bitacc_mean"] * B_PHYS / B_LOGICAL
        S["blind_T_logical"] = {}
        for T in sessions[0][k]["blind_T"]:
            a = agg_mode(lambda s, T=T: s[k]["blind_T"][T], "p_log", "m_log", "n_phys")
            a["n"] = B_LOGICAL
            a["bitacc_mean"] = a["bitacc_mean"] * B_PHYS / B_LOGICAL
            S["blind_T_logical"][T] = a
    summary[k] = S

out = {
    "config": {
        "model": args.model, "dtype": args.dtype, "d": d, "n_layers": n_layers,
        "inject_layer": L_INJECT, "read_layer": L_READ, "alpha": args.alpha,
        "n_sessions": len(sessions), "n_calib_prompts": len(calib_prompts), "n_calib_tokens": n_calib_tok,
        "prompt_len": args.prompt_len, "gen_len": args.gen_len,
        "B_logical": B_LOGICAL, "ecc_R": ECC_R, "B_phys": B_PHYS,
        "dataset": "Salesforce/wikitext wikitext-2-raw-v1 validation",
        "owner_key": args.owner_key, "wrong_keys": WRONG_KEYS,
        "model_revision": getattr(getattr(model, "config", None), "_commit_hash", None),
        "only_nomark": bool(args.only_nomark),
        "p_value": "one-sided binomial P(X>=m | n, 1/2), per session",
        "pval_examples": {f"{m}/16": pval(m, 16) for m in (12, 13, 14, 15, 16)},
    },
    "summary": summary,
    "sessions": sessions,
}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(out, indent=1))
print("\n=== RESUMEN ===")
for B in (B_LOGICAL, B_PHYS):
    S = summary[f"B{B}"]
    if args.only_nomark:
        print(f"B={B}: nomark_paired acc={S['nomark_paired']['bitacc_mean']:.3f} p_med={S['nomark_paired']['p_median']:.2e} "
              f"frac<1e-2={S['nomark_paired']['frac_p_below']['0.01']:.3f}")
        continue
    print(f"B={B}: identical={S['fidelity']['frac_identical']:.3f} first_div_med={S['fidelity']['first_div_median']:.0f} "
          f"KL={S['fidelity']['kl_mean_pre_div']:.2e} | paired acc={S['paired']['bitacc_mean']:.3f} "
          f"p_med={S['paired']['p_median']:.1e} | blind acc={S['blind']['bitacc_mean']:.3f} p_med={S['blind']['p_median']:.1e} "
          f"| nomark p_med={S['nomark_blind']['p_median']:.2f} | subst acc={S['substitution_paired']['bitacc_mean']:.3f}")
print(f"→ {args.out}")
