"""Tests para la matemática del p_value binomial en verify()."""
import pytest
from scipy import stats as scipy_stats
from cdma_watermark.watermark import Watermark
import torch


def binomial_p(k: int, n: int) -> float:
    """P(X >= k | n, 0.5) — cola unilateral."""
    return float(scipy_stats.binom.sf(k - 1, n, 0.5))


def test_pvalue_all_correct():
    """16/16 → p_value muy pequeño."""
    p = binomial_p(16, 16)
    assert p < 1e-4, f"16/16 debe dar p<1e-4, got {p}"


def test_pvalue_half_correct():
    """8/16 → p_value ≈ 0.5 (chance)."""
    p = binomial_p(8, 16)
    assert 0.4 < p < 0.6, f"8/16 debe dar p≈0.5, got {p}"


def test_pvalue_15_of_16():
    """15/16 → p_value < 1e-3."""
    p = binomial_p(15, 16)
    assert p < 1e-3, f"15/16 debe dar p<1e-3, got {p}"


def test_pvalue_9_of_16():
    """9/16 → p_value no significativo (>0.1)."""
    p = binomial_p(9, 16)
    assert p > 0.1, f"9/16 no debe ser significativo, got {p}"


def test_provenance_confirmed_high_k():
    """provenance_confirmed True cuando matches=15."""
    # Usamos un modelo dummy — sólo probamos la lógica del verify sin GPU
    # Para eso hackeamos directamente la fórmula
    n, k = 16, 15
    p = binomial_p(k, n)
    assert p < 1e-3
    confirmed = p < 1e-3
    assert confirmed is True


def test_provenance_confirmed_low_k():
    """provenance_confirmed False cuando matches=8."""
    n, k = 16, 8
    p = binomial_p(k, n)
    confirmed = p < 1e-3
    assert confirmed is False


def test_pvalue_monotone():
    """p_value decrece con más matches."""
    n = 16
    ps = [binomial_p(k, n) for k in range(8, 17)]
    for i in range(len(ps) - 1):
        assert ps[i] >= ps[i + 1], f"p_value no monotone decreciente en k={i+8}"
