"""Generación determinista de códigos CDMA y encode/decode del payload.

WHITE-BOX ONLY: la detección requiere acceso a las activaciones internas del modelo
(hook en el residual stream). La señal NO sobrevive al lm_head (black-box).
"""
from __future__ import annotations
import hashlib
import numpy as np
import torch


def _key_to_seed(key: str) -> int:
    """Hash estable de una clave string → entero de 64 bits para Generator."""
    h = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(h[:8], "big")


def make_codes(key: str, n_bits: int, d: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Genera n_bits sign-codes ±1/√d, shape (d, n_bits), determinista desde key.

    Misma clave → mismos códigos. Clave distinta → códigos distintos (ortogonales con alta prob).
    """
    seed = _key_to_seed(key)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed % (2**63))  # Generator acepta int64
    C = torch.sign(torch.randn(d, n_bits, generator=gen)) / (d ** 0.5)
    return C.to(device)


def encode_delta(
    bits: torch.Tensor,   # (n_bits,) valores ±1
    codes: torch.Tensor,  # (d, n_bits)
    alpha_abs: float,
) -> torch.Tensor:
    """Codifica bits como perturbación en el residual.

    Retorna delta (d,) con energía normalizada a alpha_abs.
    """
    delta = codes @ bits.float()       # (d,)
    norm = delta.norm() + 1e-12
    return delta / norm * alpha_abs


def decode(
    activations: torch.Tensor,           # (T, d) o (d,) — activaciones observadas
    codes: torch.Tensor,                 # (d, n_bits)
    calibration: torch.Tensor | None = None,  # (n_bits,) proyección media sin watermark
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decodifica bits y confianzas desde activaciones.

    Si activations es (T, d), promedia proyecciones sobre tokens (mejora SNR).
    calibration: proyección media de activaciones SIN watermark sobre los códigos.
    Restamos el sesgo para que el bias de activación del modelo no enmascare la señal.

    Retorna:
        bits_hat   (n_bits,) ∈ {±1}
        confidence (n_bits,) ∈ [0, 1] — |score| normalizado a [0,1] via sigmoid
    """
    if activations.dim() == 1:
        activations = activations.unsqueeze(0)   # (1, d)
    # Proyectar cada token sobre los códigos y promediar
    scores = activations.float() @ codes.float()  # (T, n_bits)
    score = scores.mean(dim=0)                    # (n_bits,)
    if calibration is not None:
        score = score - calibration
    bits_hat = torch.sign(score)
    bits_hat[bits_hat == 0] = 1.0   # desempate
    # Confianza: sigmoid del score absoluto (heurística interpretable)
    confidence = torch.sigmoid(score.abs() * 10.0)
    return bits_hat, confidence
