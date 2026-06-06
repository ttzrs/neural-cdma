# neural-cdma — Engineered Superposition in Pretrained LLMs: a forensic tool and an honest map of the wall

This repository studies **CDMA-style superposition** (binding multiple signals into one activation with
key-derived sign codes) inside *off-the-shelf, pretrained* transformer language models. It contains:

1. **A working system** — `cdma-watermark`, a white-box **forensic provenance** tool that lets an LLM provider
   prove (or disprove) that a specific generation came from its own model under a specific session key, from
   logged activations, without changing the generated text.
2. **An honest empirical study** — where engineered superposition works, and the many places it does **not**,
   on pretrained models, with a single law that explains both.

The paper is [`PAPER.md`](PAPER.md). Every number below comes from a real run in this repo (`*_result.json`).

> **Honesty first.** Most of the application results here are *negative*. We think the map of where a tempting
> idea fails — measured, not hand-waved — is the useful contribution, alongside the one application the physics
> actually supports.

## The one law

> **CDMA superposition survives only in the linear channel** (a linear / white-box read, or a *trained*
> de-multiplexer). **Every untrained non-linear readout collapses it.**

Confirmed independently at five non-linear points:

| Non-linear point | Probe (script) | Result |
|---|---|---|
| Unembedding / logits | text watermark (`watermark_provenance.py`) | logit recovery → chance; residual recovery 0.92 |
| Attention softmax | KV-cache merge (`kv_selective.py`) | perplexity 8.6 → **172,904**; plain eviction gives 1.5× context at +0.29 |
| MLP non-linearity | knowledge ablation (`cdma_neurosurgery.py`) | concept localizes (116 neurons, 0.40→0.00) but ppl +50–266%; multiplexed scan non-separable |
| Downstream stack | multi-concept steering (`eigen_steering.py`) | orthogonal codes cut cross-talk 2.2%, halve on-target retention |
| Output | parallel compute (`multithread.py`) | logits collapse |

In the *linear / trained* regime it works: the residual watermark below, and trained masked-multiplex rides to
K=32.

## The system: forensic provenance (`watermark_pkg/`)

A provider injects a per-session payload (session/user ID, timestamp) into the residual stream with a secret
key, at an amplitude that leaves the **text identical** (KL ≈ 3×10⁻⁴, token-agreement 1.0). From its own
activation logs it later proves provenance with a binomial p-value:

```
OWNER  'acme-corp-2026': 16/16 bits  p=1.5e-05  → PROVENANCE CONFIRMED
ATTACK 'thief-key':      10/16 bits  p=2.3e-01  → chance (cannot read without the secret)
```

White-box by necessity: the same non-linear unembedding that destroys a *text* watermark is what makes the
*activation* one sound. Use case: **legal non-repudiation** — attribute a disputed generation to a session, or
exculpate ("no signature → not our deployment"). See [`watermark_pkg/README.md`](watermark_pkg/README.md).

```bash
cd watermark_pkg && pip install -e .
cdma-watermark --model Qwen/Qwen2.5-0.5B --key acme-corp-2026 audit --payload 43981 --attacker-key thief-key
```

## Findings, in one place

- **Capacity law.** Trained masked-multiplex: `K_cap ≈ 0.37·d/log₂V` (toy). Real causal Qwen-0.5B: **K_cap = 2**
  (LoRA and full FT) — language's per-token demand ≫ log₂V (`capacity_law.py`, `validate_llm*.py`).
- **Surprise predicts multiplexability** (near-binary): easy/low-surprise tokens recover under superposition,
  hard ones collapse even at K=1 (`prototype_selective.py`).
- **Modality is a real effect, token design is not.** Text and code ride to K_cap=32; images collapse at K=4
  for both k-means and a trained VQ-VAE (`vision_kcap*.py`, `code_kcap.py`, `vqvae_kcap.py`). Under training,
  orthogonal vs random vs near-collinear binding codes are identical (`coded_vocab_kcap.py`).

## Honest positioning

The *mechanism* (trained MIMO via orthogonal keys) is [MIMONets](https://arxiv.org/abs/2312.02829); the
linear-readout floor is theory in [arXiv:2605.01192](https://arxiv.org/abs/2605.01192); natural task
superposition is [arXiv:2410.05603](https://arxiv.org/abs/2410.05603); parallel streams are
[PARSCALE](https://arxiv.org/abs/2505.10475). Our contributions are the **forensic system**, the **applied
negative-results map on off-the-shelf models**, and the **surprise→multiplexability** link. This is an
empirical / systems note, not a methods paper.

## Reproducibility

Python 3.x, PyTorch, transformers; experiments run on a single 16 GB GPU (Qwen2.5-0.5B). Each `*.py` writes its
`*_result.json`. License: [MIT](LICENSE).
