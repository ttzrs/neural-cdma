"""cdma_watermark — White-box CDMA activation watermark for LLM provenance.

WHITE-BOX ONLY: verification requires access to the model's internal residual
stream activations. The signal does NOT survive the lm_head non-linearity.
"""
from .codec import make_codes, encode_delta, decode
from .ecc import ecc_encode, ecc_decode
from .watermark import Watermark

__all__ = [
    "Watermark",
    "make_codes",
    "encode_delta",
    "decode",
    "ecc_encode",
    "ecc_decode",
]
__version__ = "0.1.0"
