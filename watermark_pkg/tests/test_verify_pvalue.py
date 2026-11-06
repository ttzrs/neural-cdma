"""Tests para la matemática del p_value binomial en verify()."""
import pytest
from cdma_watermark.watermark import Watermark
from cdma_watermark.codec import session_pvalue
import torch


def binomial_p(k: int, n: int) -> float:
    """P(X >= k | n, 0.5) — cola unilateral."""
    return session_pvalue(k, n)


@pytest.mark.parametrize("matches, expected", [
    (14, 2.0905e-3),
    (16, 1.5259e-5),
    (10, 0.2272),
])
def test_session_pvalue_exact(matches, expected):
    assert session_pvalue(matches, 16) == pytest.approx(expected, rel=3e-4)


def test_session_pvalue_extremes():
    assert session_pvalue(0, 16) == 1.0
    assert session_pvalue(16, 16) == 2.0 ** -16


@pytest.mark.parametrize("matches, n_bits", [(-1, 16), (17, 16), (0, 0)])
def test_session_pvalue_invalid_counts(matches, n_bits):
    with pytest.raises(ValueError):
        session_pvalue(matches, n_bits)


def test_pvalue_all_correct():
    """16/16 → p_value muy pequeño."""
    p = binomial_p(16, 16)
    assert p < 1e-4, f"16/16 debe dar p<1e-4, got {p}"


def test_pvalue_half_correct():
    """8/16 → cola inclusiva exacta ≈ 0.59819, no 0.5."""
    p = binomial_p(8, 16)
    assert p == pytest.approx(0.5981903076171875)


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
