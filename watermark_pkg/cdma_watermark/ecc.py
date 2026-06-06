"""Corrección de errores simple para el path de extracción (payload desconocido).

Usa repetition code: cada bit lógico se replica R veces en bits físicos (CDMA codes).
El decodificador vota por mayoría. Sirve cuando bit_acc ≈ 0.92 y queremos un ID corto
con recuperación casi perfecta.

Tradeoff documentado:
  R=1: 16 bits lógicos, bit_acc≈0.92 (el basal medido)
  R=2:  8 bits lógicos, bit_acc≈0.994  [1-(1-0.92)^? ver nota]
  R=3:  5 bits lógicos, bit_acc≈0.999  (2^5=32 IDs propietario)
  R=5:  3 bits lógicos, bit_acc≈0.9998 (2^3=8 IDs)

Nota: con repetition code de tamaño impar R, error = P(mayoría equivocada) = sum_{k>R/2} C(R,k) p_err^k (1-p_err)^(R-k)
donde p_err = 1 - base_acc.
"""
from __future__ import annotations
import math
import torch


def ecc_encode(logical_bits: torch.Tensor, R: int = 3) -> torch.Tensor:
    """Expande B bits lógicos a B*R bits físicos por repetición.

    logical_bits (B,) ∈ {±1} → physical_bits (B*R,) ∈ {±1}
    """
    if R == 1:
        return logical_bits.clone()
    return logical_bits.repeat_interleave(R)


def ecc_decode(physical_bits: torch.Tensor, R: int = 3) -> torch.Tensor:
    """Vota por mayoría para recuperar bits lógicos desde bits físicos.

    physical_bits (B*R,) ∈ {±1} → logical_bits (B,) ∈ {±1}
    """
    if R == 1:
        return physical_bits.clone()
    B = len(physical_bits) // R
    reshaped = physical_bits[:B * R].view(B, R)   # (B, R)
    majority = reshaped.sum(dim=1)
    result = torch.sign(majority)
    result[result == 0] = 1.0
    return result


def ecc_bit_accuracy(R: int, base_acc: float = 0.92) -> float:
    """Calcula la bit-accuracy post-ECC bajo repetition code de tamaño R.

    Asume que los errores son i.i.d. con probabilidad p_err = 1 - base_acc.
    """
    p_err = 1.0 - base_acc
    # P(mayoría falla) = P(más de R/2 errores en R trials)
    threshold = R // 2  # necesitamos MÁS de threshold errores para fallar
    p_fail = sum(
        math.comb(R, k) * (p_err ** k) * ((1 - p_err) ** (R - k))
        for k in range(threshold + 1, R + 1)
    )
    return 1.0 - p_fail
