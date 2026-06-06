"""CLI cdma-watermark: embed | verify | extract.

WHITE-BOX ONLY — requiere acceso al residual stream (modelo local).
"""
from __future__ import annotations
import argparse
import json
import sys


# Prompts por defecto: varias frases largas → suficientes tokens para promediar la señal del watermark.
_DEFAULT_PROMPTS = [
    "The contract stipulates that the delivery must be completed within thirty days of signing.",
    "Researchers at the university announced a major breakthrough in quantum computing today.",
    "The central bank raised interest rates again to combat persistent inflation this quarter.",
    "She carefully opened the envelope and read the handwritten letter inside the package.",
]


def _parse_payload(s: str, n_bits: int) -> list[int]:
    """Payload: o bits separados por espacio/coma ('1 0 1 1...'), o un entero (owner-ID) → binario de n_bits."""
    toks = s.replace(",", " ").split()
    if len(toks) == 1 and set(toks[0]) - {"0", "1"}:
        val = int(toks[0], 0)  # acepta decimal o 0x...
        if val < 0 or val >= (1 << n_bits):
            sys.exit(f"Error: payload entero {val} no cabe en {n_bits} bits")
        return [(val >> (n_bits - 1 - i)) & 1 for i in range(n_bits)]
    bits = [int(b) for b in toks]
    if len(bits) != n_bits:
        sys.exit(f"Error: payload tiene {len(bits)} bits, se esperaban {n_bits}")
    if set(bits) - {0, 1}:
        sys.exit("Error: los bits deben ser 0 o 1")
    return bits


def _load_model_tok(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cargando {model_name} en {dev}...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    ).to(dev).eval()
    return model, tok, dev


def cmd_embed(args: argparse.Namespace) -> None:
    """Inyecta watermark y genera texto con el modelo."""
    import torch
    from .watermark import Watermark

    payload_bits = _parse_payload(args.payload, args.n_bits)

    model, tok, dev = _load_model_tok(args.model)
    inject_layer = args.layer if args.layer is not None else args.inject_layer
    read_layer = args.layer if args.layer is not None else args.read_layer
    wm = Watermark(
        args.key,
        n_bits=args.n_bits,
        alpha=args.alpha,
        inject_layer=inject_layer,
        read_layer=read_layer,
    )

    print(f"Payload: {payload_bits}")
    prompts = args.prompts if args.prompts else _DEFAULT_PROMPTS
    for prompt in prompts:
        toks = tok(prompt, return_tensors="pt").input_ids.to(dev)
        with wm.embed(model, payload_bits):
            with torch.no_grad():
                gen = model.generate(toks, max_new_tokens=args.gen_len, do_sample=False)
        text = tok.decode(gen[0], skip_special_tokens=True)
        print(f"[prompt] {prompt}")
        print(f"[output] {text}\n")


def cmd_verify(args: argparse.Namespace) -> None:
    """Verifica proveniencia: compara payload esperado con el recuperado."""
    from .watermark import Watermark

    payload_bits = _parse_payload(args.payload, args.n_bits)
    model, tok, dev = _load_model_tok(args.model)
    inject_layer = args.layer if args.layer is not None else args.inject_layer
    read_layer = args.layer if args.layer is not None else args.read_layer
    wm = Watermark(
        args.key,
        n_bits=args.n_bits,
        alpha=args.alpha,
        inject_layer=inject_layer,
        read_layer=read_layer,
    )
    prompts = args.prompts if args.prompts else _DEFAULT_PROMPTS
    result = wm.verify(model, prompts, tok, payload_bits, device=dev)
    print(json.dumps(result, indent=2))
    verdict = "PROVENANCE CONFIRMED" if result["provenance_confirmed"] else "NOT CONFIRMED"
    print(f"\n{verdict}  (p={result['p_value']:.2e}, bit_acc={result['bit_accuracy']:.3f})")


def cmd_extract(args: argparse.Namespace) -> None:
    """Extrae bits watermark sin conocer el payload esperado."""
    from .watermark import Watermark

    model, tok, dev = _load_model_tok(args.model)
    inject_layer = args.layer if args.layer is not None else args.inject_layer
    read_layer = args.layer if args.layer is not None else args.read_layer
    wm = Watermark(
        args.key,
        n_bits=args.n_bits,
        alpha=args.alpha,
        inject_layer=inject_layer,
        read_layer=read_layer,
    )
    prompts = args.prompts if args.prompts else _DEFAULT_PROMPTS
    result = wm.extract(model, prompts, tok, device=dev)
    print(f"Recovered bits: {result['bits']}")
    conf = result["confidence"].tolist()
    print(f"Confidence:     {[round(c, 3) for c in conf]}")


def cmd_audit(args: argparse.Namespace) -> None:
    """Demo HONESTO de proveniencia desacoplada: inyecta con la key del propietario,
    decodifica el delta inyectado con la key del propietario VS keys de atacantes.
    Owner confirma (p diminuto); atacante sin el secreto → chance (no puede leer)."""
    import torch
    from .watermark import Watermark
    from .codec import make_codes, decode
    from scipy import stats

    payload_bits = _parse_payload(args.payload, args.n_bits)
    model, tok, dev = _load_model_tok(args.model)
    inject_layer = args.layer if args.layer is not None else args.inject_layer
    read_layer = args.layer if args.layer is not None else args.read_layer
    wm = Watermark(args.key, n_bits=args.n_bits, alpha=args.alpha,
                   inject_layer=inject_layer, read_layer=read_layer)
    prompts = args.prompts if args.prompts else _DEFAULT_PROMPTS
    d = model.config.hidden_size
    bits_pm1 = torch.tensor([1.0 if b else -1.0 for b in payload_bits], device=dev)

    # Captura el delta REALMENTE inyectado por la key del propietario (protocolo 2-partes).
    h_cap: dict = {}
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        h_cap["h"] = h[:, -1, :].detach().float().clone()
    rh = model.model.layers[read_layer].register_forward_hook(hook)
    deltas = []
    for p in prompts:
        toks = tok(p, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            model(input_ids=toks)
        h_clean = h_cap["h"][0].clone()
        with wm.embed(model, payload_bits):
            with torch.no_grad():
                model(input_ids=toks)
        deltas.append(h_cap["h"][0].clone() - h_clean)
    rh.remove()
    mean_delta = torch.stack(deltas).mean(dim=0).unsqueeze(0)

    def decode_p(key: str) -> tuple[int, float]:
        bits, _ = decode(mean_delta, make_codes(key, args.n_bits, d, dev))
        m = int((bits_pm1 == bits).sum().item())
        return m, float(stats.binom.sf(m - 1, args.n_bits, 0.5))

    thr = args.p_threshold
    m_o, p_o = decode_p(args.key)
    print(f"\nProveniencia desacoplada (inyectado con key del propietario):")
    print(f"  OWNER  '{args.key}': {m_o}/{args.n_bits} bits  p={p_o:.2e}  → "
          f"{'PROVENANCE CONFIRMED' if p_o < thr else 'no confirmado'}")
    attackers = args.attacker_key or ["attacker-A", "attacker-B", "attacker-C"]
    for ak in attackers:
        m_a, p_a = decode_p(ak)
        print(f"  ATTACK '{ak}': {m_a}/{args.n_bits} bits  p={p_a:.2e}  → "
              f"{'¡FUGA!' if p_a < thr else 'chance (no puede leer sin el secreto)'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cdma-watermark",
        description="CDMA activation watermark — WHITE-BOX provenance (residual stream only).",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B", help="HF model name or path")
    parser.add_argument("--key", default="owner-secret-key", help="Watermark secret key")
    parser.add_argument("--n-bits", type=int, default=16, dest="n_bits")
    parser.add_argument("--alpha", type=float, default=0.01)
    # Defaults = punto de operación VALIDADO en el experimento (inject_layer=8, read_layer=20, B=16, alpha=0.01).
    parser.add_argument("--layer", type=int, default=None, help="Si se da, fija inject=read=layer (override)")
    parser.add_argument("--inject-layer", type=int, default=8, dest="inject_layer")
    parser.add_argument("--read-layer", type=int, default=20, dest="read_layer")
    sub = parser.add_subparsers(dest="command", required=True)

    p_embed = sub.add_parser("embed", help="Generate watermarked text")
    p_embed.add_argument("--payload", required=True, help="Bits to embed, e.g. '1 0 1 1 ...'")
    p_embed.add_argument("--gen-len", type=int, default=30, dest="gen_len")
    p_embed.add_argument("--prompts", nargs="+", default=None)
    p_embed.set_defaults(func=cmd_embed)

    p_verify = sub.add_parser("verify", help="Verify provenance (known payload)")
    p_verify.add_argument("--payload", required=True, help="Expected bits, e.g. '1 0 1 1 ...'")
    p_verify.add_argument("--prompts", nargs="+", default=None)
    p_verify.set_defaults(func=cmd_verify)

    p_extract = sub.add_parser("extract", help="Extract payload (unknown)")
    p_extract.add_argument("--prompts", nargs="+", default=None)
    p_extract.set_defaults(func=cmd_extract)

    p_audit = sub.add_parser("audit", help="Honest decoupled provenance demo (owner vs attacker keys)")
    p_audit.add_argument("--payload", required=True, help="Owner payload: integer ID or '1 0 1 ...' bits")
    p_audit.add_argument("--attacker-key", nargs="+", default=None, dest="attacker_key",
                         help="Attacker keys to test (default: 3 synthetic)")
    p_audit.add_argument("--p-threshold", type=float, default=1e-3, dest="p_threshold")
    p_audit.add_argument("--prompts", nargs="+", default=None)
    p_audit.set_defaults(func=cmd_audit)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
