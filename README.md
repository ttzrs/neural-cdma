# neural-cdma — Engineered CDMA Superposition in Pretrained LLMs

[![CI](https://github.com/ttzrs/neural-cdma/actions/workflows/ci.yml/badge.svg)](https://github.com/ttzrs/neural-cdma/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20570543.svg)](https://doi.org/10.5281/zenodo.20570543)

This repository studies **CDMA-style superposition** (binding several signals into one activation with
key-derived sign codes) inside *off-the-shelf, pretrained* transformer language models. Version 2 splits the
work into two papers, both built from the result JSONs in this repo (no hand-typed numbers):

| Paper | PDF | What it is |
|---|---|---|
| **B — Channel physics** | [`papers/channel/main.pdf`](papers/channel/main.pdf) | *Engineered CDMA Superposition in Pretrained Transformers: Nonlinear Readout Failures, Capacity Limits, and Per-Token Surprise.* The empirical map: superposition survives only in the linear channel or with a trained de-multiplexer; five untrained non-linear readouts collapse it; language caps at K_cap = 2; per-token surprise predicts multiplexability. |
| **A — Audit-log tag** | [`papers/forensic/main.pdf`](papers/forensic/main.pdf) | *Activation-Side Session Tagging for LLM Audit Logs: What a Residual-Stream Watermark Can and Cannot Prove.* The one application that survives, with a corrected threat model (commit-then-verify), an exact verifier protocol, per-session statistics and a scale study on Qwen2.5-0.5B/1.5B/7B. |

The previous single paper (v1, DOI 10.5281/zenodo.20570544) is kept verbatim in [`archive/v1/`](archive/v1/).
Its claims of legal non-repudiation, "text unchanged" and pooled p-values are withdrawn in v2; see Paper A §1.

> **Honesty first.** Most results here are *negative*, measured rather than hand-waved. The v2 scale study
> adds three more: the mark does **not** prove model execution, blind (no-replay) verification is at chance,
> and a KL ≈ 10⁻⁴ perturbation still changes most 128-token greedy continuations on larger models.

## Paper B in one table — the law

> **CDMA superposition survives only in the linear channel** (a linear / white-box read, or a *trained*
> de-multiplexer). **Every untrained non-linear readout tested collapses it.**

| Non-linear point | Probe (script) | Result |
|---|---|---|
| Unembedding / logits | residual vs logit decoding (`watermark_provenance.py`) | logit recovery ≈ chance; residual recovery 0.92 |
| Attention softmax | KV-cache merge (`kv_selective.py`) | perplexity 8.6 → **172,904**; plain eviction gives 1.5× context at +0.29 |
| MLP non-linearity | knowledge ablation (`cdma_neurosurgery.py`) | concept localizes (116 neurons, 0.40→0.00) but ppl +50–266%; multiplexed scan non-separable |
| Downstream stack | multi-concept steering (`eigen_steering.py`) | orthogonal codes cut cross-talk 2.2%, halve on-target retention |
| Output | parallel compute (`multithread.py`) | logits collapse |

Plus: an empirical capacity fit `K_cap ≈ 0.38·d/log₂V` on toy transformers, but on real causal Qwen-0.5B the
largest K keeping ≥80 % of single-stream accuracy is **1–2** (held-out re-evaluation in `validate_llm_full_heldout.py`,
`prototype_selective_heldout.py`; the 80 % threshold statistic is fragile, the monotone collapse is not); **surprise predicts multiplexability** (`prototype_selective.py`);
oracle early-exit by surprise saves only 6% of layers (`early_exit_surprise.py`); modality matters, binding-code
design does not (`vision_kcap*.py`, `vqvae_kcap.py`, `code_kcap.py`, `coded_vocab_kcap.py`).

## Paper A in one table — the scale study (`watermark_scale.py`)

128 WikiText-2 sessions × 128 greedy tokens, α = 0.01, injection at round(L/3), read at round(5L/6), B = 16
and 64 physical bits (= 16 logical × 4 repetition). Per-session one-sided binomial p; **no deltas pooled across
sessions**. Results in `results/watermark_scale_*.json`.

| Model (dtype) | d | Identical 128-tok continuations | Paired bit-acc (B=16) | Sessions p<10⁻⁵ (64 physical bits, paired) | Blind bit-acc | Substitution bit-acc |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B (fp32) | 896 | 0.69 | 0.978 | 0.86 | 0.49 | 1.00 |
| Qwen2.5-0.5B (bf16) | 896 | 0.26 | 0.904 | 0.27 | 0.50 | 1.00 |
| Qwen2.5-1.5B (bf16) | 1536 | 0.20 | 0.957 | 0.70 | 0.51 | 1.00 |
| Qwen2.5-7B (bf16) | 3584 | 0.21 | 0.914 | 0.45 | 0.53 | 1.00 |

Reading: the tag is recoverable from logged activations **only by paired replay** (the verifier re-runs the
disputed text without the key and subtracts); static-calibration blind decoding is at chance; a tag added
directly at the read layer decodes perfectly, so the mark **does not prove the model ran**; and the
"text identical" claim of v1 holds only for short windows — and **precision dominates**: the same 0.5B model in
bf16 loses most of its fidelity and recovery (0.69 → 0.26 identical, 0.86 → 0.27 sessions at p<10⁻⁵), so
α = 0.01 in bf16 is a different operating point. Paired no-mark controls (clean text verified with the owner key)
give 0/128 sessions below p<10⁻³ (`results/nomark_paired_*.json`). The sound use is a key-conditioned session tag on
an activation log under *commit-then-verify* (Merkle-committed records before any dispute) — a supplement to,
never a replacement for, a signed transcript hash. See Paper A §2.

```bash
cd watermark_pkg && pip install -e .
cdma-watermark --model Qwen/Qwen2.5-0.5B --key acme-corp-2026 audit --payload 43981 --attacker-key thief-key
python watermark_scale.py --model Qwen/Qwen2.5-0.5B --dtype fp32 --out results/watermark_scale_qwen0.5b_fp32.json
```

## Reproducibility

```bash
pip install -r requirements.txt          # experiments
cd watermark_pkg && pip install -e .      # the tagging package (paired/blind modes, per-session p-values)
pytest watermark_pkg/tests -q             # GPU tests included; CPU subset runs in CI
cd papers/channel  && bash build.sh       # tables + figures from JSON, then tectonic
cd papers/forensic && bash build.sh
```

Experiments ran on a 16 GB RTX A5000 (0.5B, 1.5B) and a 20 GB RTX 3080 (7B, bf16). Each `*.py` writes its
`*_result.json` (committed). Papers need `tectonic` and `matplotlib`. License: [MIT](LICENSE) (code),
CC-BY-4.0 (papers).

## Citation

See [`CITATION.cff`](CITATION.cff). The concept DOI resolves to the latest Zenodo version; v1 is 10.5281/zenodo.20570544.

```bibtex
@software{vargas_neural_cdma_2026,
  author  = {Vargas Altalaguerri, Jose Joaqu\'in},
  title   = {neural-cdma: Engineered CDMA Superposition in Pretrained LLMs ---
             channel physics (Paper B) and activation-side session tagging (Paper A)},
  year    = {2026},
  doi     = {10.5281/zenodo.20570543},
  url     = {https://doi.org/10.5281/zenodo.20570543},
  version = {0.2.0}
}
```
