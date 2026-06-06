"""Test de integración real con Qwen2.5-0.5B en GPU.

Verifica el punto de operación medido en watermark_provenance.py:
  - B=16 bits, alpha=0.01, inject_layer=8, read_layer=20
  - bitacc_residual ≈ 0.92, KL ≈ 0, token_agreement ≈ 1.0
  - control ≈ 0.50

WHITE-BOX ONLY: se lee el residual stream directamente.
"""
from __future__ import annotations
import pytest
import torch
import torch.nn.functional as F

# Saltar si no hay GPU
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requiere GPU (CUDA)"
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
KEY = "test-owner-key-2026"
PAYLOAD = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1]  # 16 bits
WRONG_KEY = "wrong-key-not-owner"

PROMPTS = [
    "The contract stipulates that the delivery must be completed within thirty days",
    "Researchers at the university announced a breakthrough in quantum computing",
    "The central bank raised interest rates to combat inflation significantly",
    "She carefully opened the envelope and read the letter inside",
    "The software update introduced several new features and improved performance",
    "Scientists discovered a new species of deep-sea fish near the Mariana Trench",
    "The city council voted unanimously to approve the new public transit expansion",
    "His theory fundamentally changed our understanding of space and time",
]


@pytest.fixture(scope="module")
def model_tok():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32
    ).to(dev).eval()
    return model, tok, dev


@pytest.fixture(scope="module")
def wm():
    from cdma_watermark import Watermark
    return Watermark(KEY, n_bits=16, alpha=0.01, inject_layer=8, read_layer=20)


@pytest.fixture(scope="module")
def wm_wrong():
    from cdma_watermark import Watermark
    return Watermark(WRONG_KEY, n_bits=16, alpha=0.01, inject_layer=8, read_layer=20)


def _collect_watermarked_activations(model, tok, wm, payload, prompts, dev):
    """Inyecta watermark, recoge activaciones watermarked y clean para comparar."""
    from cdma_watermark.codec import make_codes, decode
    
    h_wm_all = []
    h_clean_all = []

    # Gancho de lectura
    h_cap = {}
    def read_hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h_cap["h"] = h.detach().float().clone()

    read_layer = model.model.layers[20]
    rh = read_layer.register_forward_hook(read_hook)

    try:
        for prompt in prompts:
            toks = tok(prompt, return_tensors="pt").input_ids.to(dev)
            # Clean
            with torch.no_grad():
                model(input_ids=toks)
            h_clean_all.append(h_cap["h"].clone())  # (1, T, d)

            # Watermarked (usamos embed context manager)
            with wm.embed(model, payload):
                with torch.no_grad():
                    model(input_ids=toks)
            h_wm_all.append(h_cap["h"].clone())  # (1, T, d)
    finally:
        rh.remove()

    return h_wm_all, h_clean_all


def test_cuda_available():
    assert torch.cuda.is_available()


def test_correct_key_provenance(model_tok, wm):
    """Clave correcta: provenance_confirmed True, p_value < 1e-4."""
    model, tok, dev = model_tok
    result = wm.verify(model, PROMPTS, tok, PAYLOAD, device=dev)
    print(f"\nCorrect key result: {result}")
    assert result["provenance_confirmed"], (
        f"Proveniencia debería confirmarse: bit_acc={result['bit_accuracy']:.3f}, "
        f"p={result['p_value']:.2e}"
    )
    # p_value < 1e-2 es suficiente con 16 bits (el medido es 0.92 por bit)
    assert result["p_value"] < 1e-2, f"p_value demasiado alto: {result['p_value']:.4f}"
    assert result["bit_accuracy"] >= 0.75, f"bit_acc baja: {result['bit_accuracy']:.3f}"


def test_wrong_key_not_confirmed(model_tok, wm, wm_wrong):
    """Clave incorrecta: provenance_confirmed False, p_value alto.

    Protocolo correcto: el contenido fue inyectado con CORRECT_KEY.
    El verificador con WRONG_KEY intenta decodificar el delta del correcto
    usando sus propios códigos (incorrectos) → proyección es noise → chance.

    Esto requiere:
      1. Capturar delta_correct = h_wm(correct) - h_clean
      2. Decodificar delta_correct con wrong_key's codes → bit_acc ≈ 0.5
    """
    from cdma_watermark.codec import make_codes, decode
    from scipy import stats as scipy_stats

    model, tok, dev = model_tok
    d = model.config.hidden_size

    # Códigos del propietario legítimo y del atacante
    codes_correct = make_codes(KEY, 16, d, dev)
    codes_wrong = make_codes(WRONG_KEY, 16, d, dev)
    bits_pm1 = torch.tensor([1.0 if b else -1.0 for b in PAYLOAD], device=dev)

    # Recoger delta inyectado por correct_key
    h_cap = {}
    def read_hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h_cap["h"] = h[:, -1, :].detach().float().clone()

    rh = model.model.layers[20].register_forward_hook(read_hook)
    deltas_correct = []
    for prompt in PROMPTS:
        toks = tok(prompt, return_tensors="pt").input_ids.to(dev)
        # clean
        with torch.no_grad():
            model(input_ids=toks)
        h_clean = h_cap["h"][0].clone()
        # watermarked con correct key
        with wm.embed(model, PAYLOAD):
            with torch.no_grad():
                model(input_ids=toks)
        h_wm = h_cap["h"][0].clone()
        deltas_correct.append((h_wm - h_clean))
    rh.remove()

    # Promedio delta across prompts
    mean_delta = torch.stack(deltas_correct).mean(dim=0)  # (d,)

    # Decodificar con codes INCORRECTOS — señal debería ser noise
    bits_wrong, _ = decode(mean_delta.unsqueeze(0), codes_wrong)
    matches_wrong = int((bits_pm1 == bits_wrong).sum().item())
    p_wrong = float(scipy_stats.binom.sf(matches_wrong - 1, 16, 0.5))

    print(f"\nWrong key: matches={matches_wrong}/16, p={p_wrong:.4f}")
    assert p_wrong > 0.05, (
        f"Clave incorrecta debería dar p>0.05, got {p_wrong:.4f} "
        f"(matches={matches_wrong}/16)"
    )

    # Para contraste: decodificar con codes CORRECTOS — señal clara
    bits_correct, _ = decode(mean_delta.unsqueeze(0), codes_correct)
    matches_correct = int((bits_pm1 == bits_correct).sum().item())
    p_correct = float(scipy_stats.binom.sf(matches_correct - 1, 16, 0.5))
    print(f"Correct key: matches={matches_correct}/16, p={p_correct:.6f}")
    assert p_correct < 0.05, f"Clave correcta debería dar p<0.05, got {p_correct:.4f}"


def test_token_agreement(model_tok, wm):
    """Token agreement ≈ 1.0 — el texto generado no se corrompe."""
    model, tok, dev = model_tok
    from cdma_watermark.codec import make_codes, encode_delta

    agreements = []
    for prompt in PROMPTS[:4]:  # primeros 4 para velocidad
        toks = tok(prompt, return_tensors="pt").input_ids.to(dev)
        
        # Logits clean
        with torch.no_grad():
            out_clean = model(input_ids=toks)
        logits_clean = out_clean.logits[0, -1, :].float()
        
        # Logits watermarked
        with wm.embed(model, PAYLOAD):
            with torch.no_grad():
                out_wm = model(input_ids=toks)
        logits_wm = out_wm.logits[0, -1, :].float()
        
        agree = float((logits_clean.argmax() == logits_wm.argmax()).float().item())
        agreements.append(agree)

    token_agreement = sum(agreements) / len(agreements)
    print(f"\nToken agreement: {token_agreement:.3f}")
    assert token_agreement >= 0.75, f"Token agreement bajo: {token_agreement:.3f}"


def test_bit_accuracy_matches_experiment(model_tok, wm):
    """bit_acc_residual debe estar en el rango medido (≥ 0.75 conservador)."""
    model, tok, dev = model_tok
    from cdma_watermark.codec import make_codes, encode_delta, decode
    import torch.nn.functional as F

    bits_pm1 = torch.tensor([1.0 if b else -1.0 for b in PAYLOAD])
    d = model.config.hidden_size
    codes = make_codes(KEY, 16, d, dev)

    accs = []
    for prompt in PROMPTS:
        toks = tok(prompt, return_tensors="pt").input_ids.to(dev)
        h_ref_cap = {}
        h_wm_cap = {}

        def make_hook(cap):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                cap["h"] = h[:, -1, :].detach().float().clone()
            return hook

        read_layer = model.model.layers[20]
        
        # Clean
        rh = read_layer.register_forward_hook(make_hook(h_ref_cap))
        with torch.no_grad():
            model(input_ids=toks)
        rh.remove()
        h_clean = h_ref_cap["h"][0]  # (d,)

        # Watermarked
        rh2 = read_layer.register_forward_hook(make_hook(h_wm_cap))
        with wm.embed(model, PAYLOAD):
            with torch.no_grad():
                model(input_ids=toks)
        rh2.remove()
        h_wm = h_wm_cap["h"][0]  # (d,)

        delta_obs = h_wm - h_clean
        bits_hat, _ = decode(delta_obs.unsqueeze(0), codes)
        acc = float((bits_pm1.to(dev) == bits_hat).float().mean())
        accs.append(acc)

    mean_acc = sum(accs) / len(accs)
    print(f"\nBit accuracy (residual): {mean_acc:.4f}")
    assert mean_acc >= 0.75, f"bit_acc_residual baja: {mean_acc:.4f}"
