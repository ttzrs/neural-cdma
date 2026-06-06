"""Prototipo KV selectivo: compresión real de caché en Qwen mediante mezcla CDMA.

Mide si tokens fáciles (baja sorpresa) pueden fusionarse en la KV-cache de un LLM real sin
degradar demasiado la perplexity de una continuación. Resultado negativo válido: si la lectura
softmax no tolera la superposición, el experimento debe mostrarlo.
"""

from __future__ import annotations

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import itertools
import json
import math
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


MODEL = "Qwen/Qwen2.5-0.5B"
DATASET = ("Salesforce/wikitext", "wikitext-2-raw-v1")
N_CTX = 1024
M_EVAL = 256
N_PASSAGES = 8
TARGET_REMOVED = 1 / 3
SEED = 1234
OUT = "kv_selective_result.json"


@dataclass(frozen=True)
class Pairs:
    easy: list[tuple[int, int]]
    random: list[tuple[int, int]]


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible: torch.cuda.is_available() es False; no se generan números CPU.")
    return torch.device("cuda")


def load_tokens(tok, dev: torch.device) -> torch.Tensor:
    need = (N_CTX + M_EVAL) * (N_PASSAGES + 2)
    ds = load_dataset(*DATASET, split="validation", streaming=True)
    buf: list[str] = []
    for row in itertools.islice(ds, 5000):
        text = row["text"].strip()
        if text:
            buf.append(text)
        if len(" ".join(buf)) > need * 5:
            break
    ids = tok(" \n ".join(buf), return_tensors="pt").input_ids[0]
    if ids.numel() < need:
        raise RuntimeError(f"dataset/tokenizer produjo solo {ids.numel()} tokens; se necesitan al menos {need}")
    return ids.to(dev)


def passage_starts(total: int) -> list[int]:
    span = N_CTX + M_EVAL
    max_start = total - span - 1
    if max_start <= 0:
        raise RuntimeError("no hay suficientes tokens para las ventanas")
    return [int(x) for x in torch.linspace(0, max_start, N_PASSAGES + 2)[1:-1]]


def legacy(cache) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    return tuple(cache)


def dynamic_from_legacy(items: tuple[tuple[torch.Tensor, torch.Tensor], ...], config) -> DynamicCache:
    return DynamicCache(ddp_cache_data=items, config=config)


def nll_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")


@torch.no_grad()
def plain_eval_nll(model, seq: torch.Tensor) -> torch.Tensor:
    out = model(seq, use_cache=False)
    logits = out.logits[:, N_CTX - 1 : N_CTX + M_EVAL - 1]
    targets = seq[:, N_CTX : N_CTX + M_EVAL]
    return nll_from_logits(logits, targets)


@torch.no_grad()
def context_cache_and_surprise(model, context: torch.Tensor):
    prefix = context[:, :-1]
    out = model(prefix, use_cache=True)

    # Sorpresa por posición de contexto: -log p(token_i | prefix_<i). La posición 0 no existe;
    # la última se reserva como token puente para poder puntuar el primer token de eval.
    logits = out.logits[:, :-1]
    targets = prefix[:, 1:]
    surprise = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
    pos = torch.arange(1, context.shape[1] - 1, device=context.device)
    return out.past_key_values, surprise.detach(), pos


def choose_pairs(surprise: torch.Tensor, candidate_pos: torch.Tensor, rng: random.Random) -> Pairs:
    n_pairs = int(round(N_CTX * TARGET_REMOVED))
    if candidate_pos.numel() < 2 * n_pairs:
        raise RuntimeError("no hay suficientes posiciones candidatas para la tasa objetivo")

    easy_order = torch.argsort(surprise)[: 2 * n_pairs]
    easy_positions = candidate_pos[easy_order].tolist()
    easy_pairs = [(int(easy_positions[i]), int(easy_positions[i + 1])) for i in range(0, len(easy_positions), 2)]

    random_positions = candidate_pos.tolist()
    rng.shuffle(random_positions)
    random_positions = random_positions[: 2 * n_pairs]
    random_pairs = [(int(random_positions[i]), int(random_positions[i + 1])) for i in range(0, len(random_positions), 2)]
    return Pairs(easy=easy_pairs, random=random_pairs)


def coded_merge(a: torch.Tensor, b: torch.Tensor, codes: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    c1, c2 = codes
    return c1 * a + c2 * b


def compress_legacy(
    base_items: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    method: str,
    pairs: list[tuple[int, int]],
    codes: tuple[torch.Tensor, torch.Tensor],
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    if method == "FULL":
        return tuple((k.clone(), v.clone()) for k, v in base_items)

    pair_for: dict[int, tuple[int, int]] = {}
    paired = set()
    for a, b in pairs:
        pair_for[min(a, b)] = (a, b)
        paired.add(a)
        paired.add(b)

    out = []
    for key, val in base_items:
        k_slots: list[torch.Tensor] = []
        v_slots: list[torch.Tensor] = []
        length = key.shape[2]
        for pos in range(length):
            if pos in pair_for:
                a, b = pair_for[pos]
                if method.startswith("CDMA"):
                    # Qwen ya guardó keys con RoPE aplicado. Fusionar esas keys RoPE'd es la aproximación honesta.
                    k_slots.append(coded_merge(key[:, :, a], key[:, :, b], codes))
                    v_slots.append(coded_merge(val[:, :, a], val[:, :, b], codes))
                elif method == "MEAN-MERGE-EASY":
                    k_slots.append((key[:, :, a] + key[:, :, b]) * 0.5)
                    v_slots.append((val[:, :, a] + val[:, :, b]) * 0.5)
                elif method == "EVICT-EASY":
                    k_slots.append(key[:, :, a])
                    v_slots.append(val[:, :, a])
                else:
                    raise ValueError(method)
            elif pos not in paired:
                k_slots.append(key[:, :, pos])
                v_slots.append(val[:, :, pos])
        out.append((torch.stack(k_slots, dim=2).contiguous(), torch.stack(v_slots, dim=2).contiguous()))
    return tuple(out)


@torch.no_grad()
def cached_eval_nll(model, cache_items, context: torch.Tensor, eval_ids: torch.Tensor) -> torch.Tensor:
    cache = dynamic_from_legacy(cache_items, model.config)
    compact_len = cache.get_seq_length()

    bridge_pos = torch.tensor([[N_CTX - 1]], device=context.device)
    bridge_cache_pos = torch.tensor([compact_len], device=context.device)
    bridge = model(
        context[:, -1:],
        past_key_values=cache,
        use_cache=True,
        position_ids=bridge_pos,
        cache_position=bridge_cache_pos,
    )
    first = nll_from_logits(bridge.logits[:, -1:], eval_ids[:, :1])

    rest_inputs = eval_ids[:, :-1]
    rest_targets = eval_ids[:, 1:]
    rest_pos = torch.arange(N_CTX, N_CTX + M_EVAL - 1, device=context.device).unsqueeze(0)
    rest_cache_pos = torch.arange(compact_len + 1, compact_len + M_EVAL, device=context.device)
    rest = model(
        rest_inputs,
        past_key_values=bridge.past_key_values,
        use_cache=True,
        position_ids=rest_pos,
        cache_position=rest_cache_pos,
    )
    return torch.cat([first, nll_from_logits(rest.logits, rest_targets)])


def summarize(vals: list[float], ratio: float) -> dict[str, float]:
    nll = sum(vals) / len(vals)
    return {
        "nll": nll,
        "perplexity": math.exp(nll),
        "compression_ratio": ratio,
        "effective_context_x": 1.0 / (1.0 - ratio),
    }


def print_table(results: dict[str, dict[str, float]]) -> None:
    full = results["FULL"]["perplexity"]
    print("\nmethod            | ratio | eff_ctx | perplexity | Δppl vs FULL")
    print("------------------|-------|---------|------------|-------------")
    for name, row in results.items():
        print(
            f"{name:<17} | {row['compression_ratio']:.3f} | {row['effective_context_x']:.3f} "
            f"| {row['perplexity']:10.3f} | {row['perplexity'] - full:+11.3f}"
        )


def verdict(results: dict[str, dict[str, float]]) -> str:
    cdma = results["CDMA-MERGE-EASY"]["perplexity"]
    rivals = {
        "MEAN-MERGE-EASY": results["MEAN-MERGE-EASY"]["perplexity"],
        "EVICT-EASY": results["EVICT-EASY"]["perplexity"],
        "CDMA-MERGE-RANDOM": results["CDMA-MERGE-RANDOM"]["perplexity"],
    }
    beats = [k for k, v in rivals.items() if cdma < v]
    cost = cdma - results["FULL"]["perplexity"]
    if len(beats) == len(rivals):
        return f"VEREDICTO: CDMA-MERGE-EASY gana a todos los controles; {results['CDMA-MERGE-EASY']['effective_context_x']:.2f}x contexto cuesta Δppl={cost:.3f}."
    return f"VEREDICTO: CDMA-MERGE-EASY no gana a todos los controles ({', '.join(beats) or 'ninguno'}); {results['CDMA-MERGE-EASY']['effective_context_x']:.2f}x contexto cuesta Δppl={cost:.3f}."


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    dev = require_cuda()
    dtype = torch.float32  # fp32: el path de caché reconstruida debe igualar al forward plano a <1e-3 (bf16 no)

    tok = AutoTokenizer.from_pretrained(MODEL, fix_mistral_regex=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(dev).eval()
    ids = load_tokens(tok, dev)
    starts = passage_starts(ids.numel())

    sample_cache, _, _ = context_cache_and_surprise(model, ids[starts[0] : starts[0] + N_CTX].unsqueeze(0))
    sample_key = legacy(sample_cache)[0][0]
    code_shape = (1, sample_key.shape[1], sample_key.shape[-1])
    c1 = torch.sign(torch.randn(code_shape, device=dev, dtype=sample_key.dtype))
    c2 = torch.sign(torch.randn(code_shape, device=dev, dtype=sample_key.dtype))
    c1[c1 == 0] = 1
    c2[c2 == 0] = 1
    codes = (c1, c2)

    methods = ["FULL", "CDMA-MERGE-EASY", "MEAN-MERGE-EASY", "EVICT-EASY", "CDMA-MERGE-RANDOM"]
    token_nlls: dict[str, list[float]] = {m: [] for m in methods}
    ratios: dict[str, float] = {m: 0.0 for m in methods}
    rng = random.Random(SEED)
    sanity_diffs: list[float] = []

    for i, start in enumerate(starts, 1):
        seq = ids[start : start + N_CTX + M_EVAL].unsqueeze(0)
        context, eval_ids = seq[:, :N_CTX], seq[:, N_CTX:]
        plain = plain_eval_nll(model, seq)
        base_cache, surprise, candidate_pos = context_cache_and_surprise(model, context)
        base_items = legacy(base_cache)
        pair_sets = choose_pairs(surprise, candidate_pos, rng)

        plan = {
            "FULL": [],
            "CDMA-MERGE-EASY": pair_sets.easy,
            "MEAN-MERGE-EASY": pair_sets.easy,
            "EVICT-EASY": pair_sets.easy,
            "CDMA-MERGE-RANDOM": pair_sets.random,
        }
        for method in methods:
            comp_items = compress_legacy(base_items, method, plan[method], codes)
            ratio = 1.0 - (comp_items[0][0].shape[2] + 1) / N_CTX
            ratios[method] += ratio
            got = cached_eval_nll(model, comp_items, context, eval_ids)
            token_nlls[method].extend(float(x) for x in got.detach().cpu())
            if method == "FULL":
                diff = abs(float(got.mean().detach().cpu()) - float(plain.mean().detach().cpu()))
                sanity_diffs.append(diff)
                if diff > 1e-3:
                    raise RuntimeError(f"sanity FULL falló en pasaje {i}: diff NLL={diff:.6g}")
        torch.cuda.empty_cache()
        print(f"pasaje {i}/{N_PASSAGES}: sanity FULL diff={sanity_diffs[-1]:.6g}", flush=True)

    results = {m: summarize(token_nlls[m], ratios[m] / N_PASSAGES) for m in methods}
    payload = {
        "model": MODEL,
        "dataset": "/".join(DATASET),
        "n_context": N_CTX,
        "m_eval": M_EVAL,
        "n_passages": N_PASSAGES,
        "dtype": str(dtype).replace("torch.", ""),
        "codes": "dos códigos aleatorios ±1 por kv-head/head_dim, compartidos entre capas y pares",
        "positioning": "position_ids absolutos continúan desde N; cache_position usa slots compactos",
        "full_cache_sanity_max_abs_nll_diff": max(sanity_diffs),
        "methods": results,
    }
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSanity FULL cache vs full_sequence: max abs mean-NLL diff={max(sanity_diffs):.6g}")
    print_table(results)
    print("\n" + verdict(results))
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
