# cdma-watermark

White-box CDMA activation watermark for **forensic provenance and legal non-repudiation** of LLM generations.

An LLM provider tags each generation, in the residual stream, with a per-session payload (session/user ID,
timestamp) using a secret key — without changing the generated text. From its own activation logs it can later
prove, to a binomial p-value, that a disputed generation came from its model under that session key
(*attribution*), or that a text carries no valid signature and was therefore not produced by its deployment
(*exculpation*). This is a forensic / audit-logging primitive for the party that runs the model — **not** an
anti-distillation / copyright tool, and not a text-detectable watermark (see the white-box note below).

> **WHITE-BOX ONLY — IMPORTANT LIMITATION**
> 
> This package embeds and verifies watermarks by reading the model's **internal residual
> stream activations** via forward hooks. It is **NOT a black-box / text-only detector**.
> The signal does not survive the `lm_head` non-linearity: measuring only generated text
> gives ~50% bit accuracy (chance level). Verification requires running the original model
> and reading intermediate layer activations.

## What it does

Embeds a B-bit owner payload into the LLM's residual stream during inference at low
amplitude. The generated text is essentially unchanged (KL≈0, token-agreement≈1.0 at
the measured operating point). The payload can be recovered from activations at the same
layer by correlating with key-derived sign codes.

**Measured operating point** (from `watermark_provenance.py` in this repo, Qwen2.5-0.5B fp32):

| Metric | Value |
|---|---|
| B (payload bits) | 16 |
| α / ‖h‖ | 0.01 |
| KL divergence | ≈ 0.00035 |
| Token agreement | 1.0 |
| Bit accuracy (residual) | 0.919 |
| Bit accuracy (logits/black-box) | 0.568 (not useful) |
| Control (chance) | 0.502 |

## Installation

```bash
pip install -e /path/to/watermark_pkg
```

## Quickstart

```python
from cdma_watermark import Watermark
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B").cuda().eval()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

wm = Watermark("my-secret-key", n_bits=16, alpha=0.01, inject_layer=8, read_layer=20)
payload = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1]

# Embed: generate text with watermark injected
with wm.embed(model, payload):
    inputs = tok("Hello, world", return_tensors="pt").input_ids.cuda()
    out = model.generate(inputs, max_new_tokens=50)

# Verify provenance
prompts = ["Hello, world", "The contract stipulates..."]
result = wm.verify(model, prompts, tok, payload)
print(result)
# {'match_bits': 15, 'n_bits': 16, 'bit_accuracy': 0.9375,
#  'p_value': 0.000244, 'provenance_confirmed': True, 'recovered_bits': [...]}
```

## CLI

Defaults reproduce the validated operating point (`--inject-layer 8 --read-layer 20`, `--n-bits 16`,
`--alpha 0.01`). `--payload` accepts a bit string *or* an integer owner-ID (e.g. `43981` or `0xABCD`).

```bash
# Verify a known watermark (re-injects with the key and confirms recovery)
cdma-watermark --model Qwen/Qwen2.5-0.5B --key acme-corp-2026 verify --payload 43981
# → PROVENANCE CONFIRMED  (p=1.53e-05, bit_acc=1.000)
```

### `audit` — the honest decoupled provenance demo (recommended)

`verify` re-injects with the supplied key, so it confirms *any* key's own payload — it does not, by itself,
prove discrimination. The **`audit`** command runs the true two-party protocol in one shot: it injects with
the owner key, then decodes the injected delta with the owner key **and** with attacker keys. Only the owner
recovers the payload; attackers without the secret get chance.

```bash
cdma-watermark --model Qwen/Qwen2.5-0.5B --key acme-corp-2026 \
    audit --payload 43981 --attacker-key thief-key competitor-xyz
# OWNER  'acme-corp-2026': 16/16 bits  p=1.53e-05  → PROVENANCE CONFIRMED
# ATTACK 'thief-key':      10/16 bits  p=2.27e-01  → chance (cannot read without the secret)
# ATTACK 'competitor-xyz':  9/16 bits  p=4.02e-01  → chance
```

## Threat model

- **Who can embed**: anyone running the model locally with the secret key
- **Who can verify**: anyone with (a) the model, (b) the secret key, (c) the injection/read layer indices
- **What it proves**: that a specific payload was injected during inference with a specific key — even imperfect recovery (e.g. 14/16 bits) gives p < 0.004, confirming a KNOWN watermark with high confidence
- **What it does NOT prove**: black-box text authenticity; text could be copied/regenerated

### Wrong-key discrimination

The codes are key-derived sign vectors. The discriminative test is: inject content with
CORRECT_KEY, then decode the resulting delta with WRONG_KEY's codes → bit_acc ≈ 0.5
(measured: 6/16 matches, p=0.89). An attacker who doesn't know the owner's key cannot
claim ownership of content watermarked with the owner's key.

## Limitations (honest)

1. **White-box only**: requires model access and forward hooks
2. **Same model required**: verification fails if the model is fine-tuned or quantized (activations change)
3. **Layer sensitivity**: inject/read layer indices must match between embed and verify
4. **No public verifiability**: the secret key must be shared for third-party verification
5. **bit_acc ≈ 0.92** at the operating point: ~1 bit in 13 is recovered incorrectly on average. Use the binomial p_value — not raw bit accuracy — as the verification criterion
6. **Text preservation measured** on Qwen2.5-0.5B fp32 only; other models/dtypes untested

## Error correction (ECC)

For applications where the payload is a short owner ID and near-perfect recovery is needed,
use `ecc_encode`/`ecc_decode` with repetition factor R:

```python
from cdma_watermark.ecc import ecc_encode, ecc_decode, ecc_bit_accuracy
# R=3: 5 logical bits (32 owner IDs), post-ECC bit_acc ≈ 0.9997
print(ecc_bit_accuracy(R=3, base_acc=0.92))  # 0.9997
```
