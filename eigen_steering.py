"""
Experimento: eigen_steering.py
===============================
Compara el INTERFERENCIA entre K=4 vectores de dirección simultáneos cuando se aplican
de forma NAIVE vs ORTOGONALIZADA (eigen-coding). Mide la matriz de interferencia K×K
para cada condición: SINGLE, NAIVE-MULTI, ORTHO-MULTI, RANDOM-CONTROL.

Behaviors (K=4):
  0: Sentiment     — positivo vs negativo (logprob proxy sobre palabras clave)
  1: Language      — inglés vs español (token-set overlap)
  2: Verbosity     — largo vs corto (longitud de generación)
  3: Topic-food    — presencia de léxico "food" (token-mass en vocabulario food)

Protocolo:
- Extraer steering vectors via diff-of-means en activaciones residuales (capa media)
  sobre sets positivo/negativo por comportamiento.
- Evaluar en prompts neutros held-out.
- Condiciones: SINGLE (1 vector a la vez), NAIVE (suma cruda), ORTHO (Gram-Schmidt),
  RANDOM (direcciones ortogonales aleatorias — control).
- Métrica: score_k(output) para cada comportamiento k, normalizado a [-1,1] donde
  +1 = dirección positiva del steering, -1 = negativa.
- Interference matrix [i,j]: steer hacia i, medir efecto en j (j≠i debe ser ~0).

Salida: eigen_steering_result.json
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Dict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-0.5B"
DEVICE = "cuda"
LAYER = 12          # mid-network layer to inject/extract (0-indexed, total=24)
ALPHA = 15.0        # steering amplitude (matched norm across conditions)
N_STEER_PROMPTS = 10   # prompts per direction for diff-of-means
N_EVAL_PROMPTS = 15    # neutral prompts for evaluation
MAX_NEW_TOKENS = 30    # tokens generated for scoring
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Prompt sets per behavior (positive / negative)
# ---------------------------------------------------------------------------
STEER_PROMPTS = {
    "sentiment": {
        "pos": [
            "This is absolutely wonderful and I love it!",
            "What a fantastic day, everything is great.",
            "I feel so happy and joyful today.",
            "This is the best experience I have ever had.",
            "Amazing results, truly excellent work here.",
            "Wonderful news, I am thrilled and excited.",
            "This makes me so glad and pleased.",
            "Brilliant outcome, very positive feelings.",
            "Great progress, really impressive and good.",
            "Superb quality, this is outstanding!",
        ],
        "neg": [
            "This is terrible and I hate everything about it.",
            "What an awful day, nothing is working.",
            "I feel so sad and miserable today.",
            "This is the worst experience I have ever had.",
            "Horrible results, truly dreadful work here.",
            "Terrible news, I am devastated and upset.",
            "This makes me angry and disappointed.",
            "Dreadful outcome, very negative feelings.",
            "Bad progress, really disappointing and poor.",
            "Awful quality, this is an absolute failure.",
        ],
    },
    "language": {
        "pos": [  # Spanish target
            "Hola, ¿cómo estás hoy? Espero que bien.",
            "Me llamo Juan y vivo en Madrid, España.",
            "El tiempo en Barcelona es muy agradable.",
            "Quiero aprender más sobre la cultura española.",
            "Los libros en español son muy interesantes.",
            "Me gusta mucho la comida española tradicional.",
            "Voy a la tienda para comprar algunas cosas.",
            "El museo de arte tiene obras muy bonitas.",
            "Mañana iré al mercado con mi familia.",
            "La música latina tiene ritmos muy alegres.",
        ],
        "neg": [  # English target (negative = not Spanish)
            "Hello, how are you today? I hope well.",
            "My name is John and I live in London.",
            "The weather in New York is quite pleasant.",
            "I want to learn more about English culture.",
            "Books in English are very interesting to read.",
            "I really enjoy traditional American food cuisine.",
            "I am going to the store to buy some things.",
            "The art museum has very beautiful works.",
            "Tomorrow I will go to the market with family.",
            "Latin music has very lively rhythms.",
        ],
    },
    "verbosity": {
        "pos": [  # verbose (long, detailed)
            "Please provide an extensive, comprehensive, and thoroughly detailed explanation of everything.",
            "Describe in great length and considerable detail all the various aspects involved.",
            "Give me a very long, elaborate, and exhaustive response with many specifics.",
            "Write a complete and thorough description with maximum detail and explanation.",
            "Elaborate extensively on all points, providing rich context and full information.",
            "Discuss at great length every nuance and detail of the topic comprehensively.",
            "Please expand on this with full detail, covering every aspect thoroughly.",
            "Write a detailed and comprehensive analysis covering all points extensively.",
            "Provide a thorough, in-depth, and exhaustive examination of the subject matter.",
            "Give a complete, detailed, and lengthy response covering everything possible.",
        ],
        "neg": [  # terse/short
            "Brief.",
            "Short answer only.",
            "Minimal.",
            "Quick.",
            "Yes.",
            "No.",
            "OK.",
            "Done.",
            "Fine.",
            "Got it.",
        ],
    },
    "topic_food": {
        "pos": [  # food-related
            "The restaurant serves delicious pasta with tomato sauce.",
            "I love eating fresh bread, cheese, and vegetables.",
            "The recipe calls for eggs, flour, and butter.",
            "We had pizza for dinner with lots of cheese.",
            "The market sells fresh fruit, meat, and seafood.",
            "Cooking a good meal requires fresh ingredients.",
            "The soup was made with carrots, onions, and broth.",
            "I enjoy breakfast with coffee, eggs, and toast.",
            "The bakery makes wonderful cakes and cookies.",
            "Grilled chicken with salad is a healthy meal.",
        ],
        "neg": [  # non-food / tech
            "The computer program processes data very efficiently.",
            "The software algorithm sorts numbers in memory.",
            "This mathematical equation describes the function.",
            "The circuit board connects electronic components.",
            "The telescope observes distant stars and galaxies.",
            "The database stores records for quick retrieval.",
            "The airplane flies at high altitude and speed.",
            "The satellite orbits the planet every hour.",
            "The library contains thousands of technical books.",
            "The laboratory conducts scientific experiments daily.",
        ],
    },
}

# Neutral evaluation prompts
EVAL_PROMPTS = [
    "The situation is",
    "Consider the following",
    "Looking at this topic",
    "In general terms",
    "One thing to note is",
    "The main point here",
    "When thinking about",
    "This particular case",
    "A common observation",
    "From one perspective",
    "The relevant aspect",
    "It is worth noting",
    "Taking this into account",
    "The standard approach",
    "In most situations",
]

# ---------------------------------------------------------------------------
# Vocabulary sets for scoring
# ---------------------------------------------------------------------------
SENTIMENT_POS_WORDS = {
    "great", "good", "wonderful", "excellent", "amazing", "fantastic", "happy",
    "love", "best", "joy", "positive", "beautiful", "brilliant", "perfect",
    "outstanding", "nice", "superb", "glad", "pleased", "thrilled"
}
SENTIMENT_NEG_WORDS = {
    "bad", "terrible", "awful", "horrible", "hate", "sad", "worst", "poor",
    "dreadful", "negative", "ugly", "fail", "failure", "wrong", "broken",
    "disappointed", "angry", "miserable", "upset", "devastated"
}

SPANISH_WORDS = {
    "el", "la", "los", "las", "un", "una", "de", "en", "que", "es", "se",
    "no", "al", "del", "con", "por", "para", "como", "pero", "más", "muy",
    "su", "le", "lo", "me", "mi", "te", "tu", "yo", "él", "ella", "nos",
    "están", "este", "esta", "ese", "esa", "bien", "también", "hay"
}

FOOD_WORDS = {
    "food", "eat", "meal", "dinner", "lunch", "breakfast", "cook", "cooking",
    "recipe", "restaurant", "kitchen", "hungry", "taste", "flavor", "delicious",
    "bread", "rice", "meat", "fish", "vegetable", "fruit", "cheese", "soup",
    "pasta", "pizza", "chicken", "beef", "sauce", "butter", "cream", "sweet",
    "drink", "coffee", "tea", "wine", "beer", "juice", "water", "milk"
}


# ---------------------------------------------------------------------------
# Model + tokenizer loading
# ---------------------------------------------------------------------------
def load_model():
    print(f"[load] Loading {MODEL_ID} fp32 on {DEVICE}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        device_map=DEVICE,
    )
    model.eval()
    print(f"[load] Done. Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    return tok, model


# ---------------------------------------------------------------------------
# Activation extraction helpers
# ---------------------------------------------------------------------------
def get_residual_activations(
    tok, model, prompts: List[str], layer: int
) -> torch.Tensor:
    """
    Returns mean residual stream at `layer` over the last token of each prompt.
    Shape: (len(prompts), hidden_size).
    """
    acts = []
    for prompt in prompts:
        inputs = tok(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        # hidden_states: tuple of (n_layers+1) × (1, seq, d)
        h = out.hidden_states[layer + 1]  # after layer `layer`
        acts.append(h[0, -1, :].cpu())
    return torch.stack(acts)  # (N, d)


def compute_steering_vector(
    tok, model, pos_prompts: List[str], neg_prompts: List[str], layer: int
) -> torch.Tensor:
    """Diff-of-means steering vector, unit-normalized."""
    pos_acts = get_residual_activations(tok, model, pos_prompts, layer)
    neg_acts = get_residual_activations(tok, model, neg_prompts, layer)
    vec = pos_acts.mean(0) - neg_acts.mean(0)
    vec = vec / (vec.norm() + 1e-8)
    return vec  # (d,)


# ---------------------------------------------------------------------------
# Gram-Schmidt orthogonalization
# ---------------------------------------------------------------------------
def gram_schmidt(vectors: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Orthonormalize a list of vectors. Returns list of same length; vectors
    that are linearly dependent get projected residual (may be very small).
    """
    ortho = []
    for v in vectors:
        v = v.clone().float()
        for u in ortho:
            v = v - (v @ u) * u
        norm = v.norm()
        if norm > 1e-8:
            v = v / norm
        ortho.append(v)
    return ortho


# ---------------------------------------------------------------------------
# Steering hook
# ---------------------------------------------------------------------------
class SteeringHook:
    """Adds a linear combination of steering vectors to the residual at a layer."""

    def __init__(self, vectors: List[torch.Tensor], coefficients: List[float]):
        # vectors: list of (d,) unit tensors; coefficients: scaling per vector
        self.vectors = [v.to(DEVICE) for v in vectors]
        self.coefficients = coefficients

    def __call__(self, module, input, output):
        # Qwen2 layers return a plain Tensor (hidden_states), not a tuple
        if isinstance(output, torch.Tensor):
            h = output
            for vec, coef in zip(self.vectors, self.coefficients):
                h = h + coef * vec.unsqueeze(0).unsqueeze(0)
            return h
        else:
            # Fallback: some layer returns tuple (hidden_states, ...)
            h = output[0]
            for vec, coef in zip(self.vectors, self.coefficients):
                h = h + coef * vec.unsqueeze(0).unsqueeze(0)
            return (h,) + output[1:]


# ---------------------------------------------------------------------------
# Text scoring
# ---------------------------------------------------------------------------
def score_output(tokens_text: str, behavior: str) -> float:
    """
    Returns a score in [-1, +1] for the behavior based on token-level heuristics.
    Positive = steered in "positive" direction.
    """
    tokens_lower = tokens_text.lower().split()
    word_set = set(tokens_lower)

    if behavior == "sentiment":
        pos_hits = len(word_set & SENTIMENT_POS_WORDS)
        neg_hits = len(word_set & SENTIMENT_NEG_WORDS)
        total = pos_hits + neg_hits + 1e-6
        return float((pos_hits - neg_hits) / total)

    elif behavior == "language":
        # Positive = Spanish
        es_hits = len(word_set & SPANISH_WORDS)
        # Also check character-level for Spanish chars
        es_char = sum(1 for c in tokens_text if c in "áéíóúñüÁÉÍÓÚÑÜ¿¡")
        score = min(1.0, (es_hits * 0.1 + es_char * 0.15))
        return float(score * 2 - 1)  # map [0,1] → [-1,+1] (negative = English)

    elif behavior == "verbosity":
        # Positive = verbose (long)
        n_words = len(tokens_lower)
        # Normalize: expected range ~5-40 words
        score = (n_words - 10) / 20.0  # center ~10 words
        return float(max(-1.0, min(1.0, score)))

    elif behavior == "topic_food":
        food_hits = len(word_set & FOOD_WORDS)
        score = min(1.0, food_hits * 0.25)
        return float(score * 2 - 1)

    return 0.0


def generate_with_hooks(
    tok, model, prompt: str,
    hook_fn=None, layer: int = LAYER,
    max_new_tokens: int = MAX_NEW_TOKENS
) -> str:
    """Generate continuation, optionally with a residual-stream hook at `layer`."""
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    handle = None
    if hook_fn is not None:
        # Hook into the output of transformer layer `layer`
        target = model.model.layers[layer]
        handle = target.register_forward_hook(hook_fn)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tok.eos_token_id,
        )
    if handle is not None:
        handle.remove()
    # Decode only newly generated tokens
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Evaluate all behaviors for a given steering configuration
# ---------------------------------------------------------------------------
BEHAVIOR_NAMES = ["sentiment", "language", "verbosity", "topic_food"]


def evaluate_condition(
    tok, model,
    steer_vectors: List[torch.Tensor],  # K vectors (unit-norm), or None = no steer
    eval_prompts: List[str],
    alpha: float = ALPHA,
    layer: int = LAYER,
) -> np.ndarray:
    """
    Returns score matrix (len(eval_prompts), K) where entry [p, k] is the
    behavior-k score for prompt p under this steering condition.
    """
    K = len(BEHAVIOR_NAMES)
    scores = np.zeros((len(eval_prompts), K))
    hook_fn = None
    if steer_vectors is not None:
        coefficients = [alpha] * len(steer_vectors)
        hook_fn = SteeringHook(steer_vectors, coefficients)

    for pi, prompt in enumerate(eval_prompts):
        text = generate_with_hooks(tok, model, prompt, hook_fn, layer)
        for ki, bname in enumerate(BEHAVIOR_NAMES):
            scores[pi, ki] = score_output(text, bname)

    return scores  # (N_eval, K)


# ---------------------------------------------------------------------------
# Build interference matrix
# ---------------------------------------------------------------------------
def build_interference_matrix(
    tok, model,
    steering_vectors: List[torch.Tensor],  # K unit-norm vectors (for SINGLE)
    multi_vectors: List[torch.Tensor],      # K vectors for simultaneous (NAIVE or ORTHO)
    eval_prompts: List[str],
    alpha: float = ALPHA,
    layer: int = LAYER,
    baseline_scores: np.ndarray = None,    # unsteered scores (N_eval, K)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      single_matrix[i,j]: mean score on behavior j when steering ONLY behavior i
      multi_matrix[i,j]: mean score on behavior j when steering ALL simultaneously
                         (using multi_vectors)
    """
    K = len(BEHAVIOR_NAMES)
    single_matrix = np.zeros((K, K))
    multi_matrix = np.zeros((K, K))

    # SINGLE: steer each behavior alone
    print("  [single] evaluating K individual steers...")
    for i in range(K):
        print(f"    steer behavior {i} ({BEHAVIOR_NAMES[i]})...")
        scores = evaluate_condition(
            tok, model, [steering_vectors[i]], eval_prompts, alpha, layer
        )
        single_matrix[i, :] = scores.mean(0)
        torch.cuda.empty_cache()

    # MULTI: steer all K simultaneously
    print("  [multi] evaluating all-K simultaneous steer...")
    scores_multi = evaluate_condition(
        tok, model, multi_vectors, eval_prompts, alpha, layer
    )
    # For the multi-matrix, row i = "we steered behavior i" — but we steer ALL,
    # so the matrix is the same for all rows (all behaviors steered at once).
    # We report it as the shared outcome row repeated.
    mean_multi = scores_multi.mean(0)
    for i in range(K):
        multi_matrix[i, :] = mean_multi
    torch.cuda.empty_cache()

    return single_matrix, multi_matrix


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def main():
    tok, model = load_model()
    torch.cuda.empty_cache()

    # ---- Phase 1: Extract steering vectors ----
    print("\n=== Phase 1: Extract steering vectors ===")
    raw_vectors = []
    for bname in BEHAVIOR_NAMES:
        print(f"  extracting vector for '{bname}'...")
        v = compute_steering_vector(
            tok, model,
            STEER_PROMPTS[bname]["pos"],
            STEER_PROMPTS[bname]["neg"],
            LAYER,
        )
        raw_vectors.append(v)
        torch.cuda.empty_cache()
    print("  Raw vectors extracted. Cosine similarities between pairs:")
    K = len(raw_vectors)
    cosine_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cosine_matrix[i, j] = float(
                (raw_vectors[i] @ raw_vectors[j]).item()
            )
    for i in range(K):
        print(f"    {BEHAVIOR_NAMES[i]:12s}: " + " ".join(
            f"{cosine_matrix[i,j]:.3f}" for j in range(K)
        ))

    # ---- Phase 2: Orthogonalize ----
    print("\n=== Phase 2: Gram-Schmidt orthogonalization ===")
    ortho_vectors = gram_schmidt(raw_vectors)
    print("  Cosine similarities post-GS:")
    for i in range(K):
        for j in range(K):
            c = float((ortho_vectors[i] @ ortho_vectors[j]).item())
            if abs(c) > 0.01 and i != j:
                print(f"    WARNING: ortho[{i}]·ortho[{j}] = {c:.4f}")
    print("  OK (off-diagonal ≈ 0)")

    # ---- Phase 3: Random control vectors ----
    print("\n=== Phase 3: Random orthogonal control vectors ===")
    d = raw_vectors[0].shape[0]
    rand_basis = torch.linalg.qr(
        torch.randn(d, K, generator=torch.manual_seed(SEED + 1))
    ).Q  # (d, K)
    random_vectors = [rand_basis[:, i] for i in range(K)]

    # ---- Phase 4: Baseline (no steering) ----
    print("\n=== Phase 4: Baseline scores (no steer) ===")
    baseline_scores = evaluate_condition(tok, model, None, EVAL_PROMPTS, ALPHA, LAYER)
    baseline_mean = baseline_scores.mean(0)
    print("  Baseline mean scores per behavior:")
    for k, bname in enumerate(BEHAVIOR_NAMES):
        print(f"    {bname:12s}: {baseline_mean[k]:.4f}")
    torch.cuda.empty_cache()

    # ---- Phase 5: Single steers ----
    print("\n=== Phase 5: Single steers (baseline effect per behavior) ===")
    single_scores = np.zeros((K, K))
    for i in range(K):
        print(f"  Single steer: {BEHAVIOR_NAMES[i]}...")
        sc = evaluate_condition(tok, model, [raw_vectors[i]], EVAL_PROMPTS, ALPHA, LAYER)
        single_scores[i, :] = sc.mean(0)
        torch.cuda.empty_cache()
    print("  Single steer scores matrix [steer_i x measure_j]:")
    print("  " + " ".join(f"{b[:6]:>8}" for b in BEHAVIOR_NAMES))
    for i in range(K):
        row = " ".join(f"{single_scores[i,j]:8.4f}" for j in range(K))
        print(f"  {BEHAVIOR_NAMES[i][:10]:10s}: {row}")

    # ---- Phase 6: NAIVE multi steer ----
    print("\n=== Phase 6: NAIVE multi steer (all K simultaneously) ===")
    naive_sc = evaluate_condition(tok, model, raw_vectors, EVAL_PROMPTS, ALPHA, LAYER)
    naive_mean = naive_sc.mean(0)
    print("  Naive multi scores (shared across all steer targets):")
    for k, bname in enumerate(BEHAVIOR_NAMES):
        print(f"    {bname:12s}: {naive_mean[k]:.4f}")
    torch.cuda.empty_cache()

    # Build naive interference matrix (row i = steer all, measure j)
    naive_matrix = np.tile(naive_mean, (K, 1))  # all rows identical

    # ---- Phase 7: ORTHO multi steer ----
    print("\n=== Phase 7: ORTHO multi steer (Gram-Schmidt) ===")
    ortho_sc = evaluate_condition(tok, model, ortho_vectors, EVAL_PROMPTS, ALPHA, LAYER)
    ortho_mean = ortho_sc.mean(0)
    print("  Ortho multi scores:")
    for k, bname in enumerate(BEHAVIOR_NAMES):
        print(f"    {bname:12s}: {ortho_mean[k]:.4f}")
    torch.cuda.empty_cache()

    ortho_matrix = np.tile(ortho_mean, (K, 1))

    # ---- Phase 8: RANDOM control ----
    print("\n=== Phase 8: RANDOM control (orthogonal, non-behavior directions) ===")
    random_sc = evaluate_condition(tok, model, random_vectors, EVAL_PROMPTS, ALPHA, LAYER)
    random_mean = random_sc.mean(0)
    print("  Random control scores (should be ~baseline):")
    for k, bname in enumerate(BEHAVIOR_NAMES):
        print(f"    {bname:12s}: {random_mean[k]:.4f} (baseline: {baseline_mean[k]:.4f})")
    torch.cuda.empty_cache()

    random_matrix = np.tile(random_mean, (K, 1))

    # ---- Phase 9: Interference analysis ----
    print("\n=== Phase 9: Interference analysis ===")

    # On-target effect: diagonal of single_scores - baseline
    on_target_single = np.array([
        single_scores[i, i] - baseline_mean[i] for i in range(K)
    ])

    # On-target retention: how much of single effect is preserved in multi
    # For naive: diagonal of naive_matrix - baseline
    on_target_naive = naive_mean - baseline_mean
    on_target_ortho = ortho_mean - baseline_mean
    on_target_random = random_mean - baseline_mean

    # Retention ratio: (multi on-target) / (single on-target)
    # Clamp denominator to avoid div-by-zero on weak steers
    eps = 1e-6
    retention_naive = np.where(
        np.abs(on_target_single) > eps,
        on_target_naive / (on_target_single + eps * np.sign(on_target_single + eps)),
        0.0
    )
    retention_ortho = np.where(
        np.abs(on_target_single) > eps,
        on_target_ortho / (on_target_single + eps * np.sign(on_target_single + eps)),
        0.0
    )

    # Off-target leakage for SINGLE steers: mean |cross-effect|
    # cross_effect[i,j] for j≠i = single_scores[i,j] - baseline[j]
    single_cross = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            single_cross[i, j] = single_scores[i, j] - baseline_mean[j]
    # Zero out diagonal
    single_cross_offdiag = single_cross.copy()
    np.fill_diagonal(single_cross_offdiag, 0.0)
    single_leakage = np.abs(single_cross_offdiag).sum(1) / (K - 1)

    # For NAIVE multi: cross-effect on non-target behaviors relative to single
    naive_cross = naive_mean - baseline_mean  # (K,) — all are "on" target
    # In multi, you steer all behaviors simultaneously; interference is the
    # deviation from the sum of independent effects
    sum_single_cross = np.zeros(K)
    for j in range(K):
        sum_single_cross[j] = sum(
            single_cross[i, j] for i in range(K) if i != j
        )

    print("\n--- On-target effects (behavior_score - baseline) ---")
    print(f"{'behavior':12s} {'single':>8} {'naive':>8} {'ortho':>8} {'random':>8}")
    for i in range(K):
        print(f"{BEHAVIOR_NAMES[i]:12s} {on_target_single[i]:8.4f} "
              f"{on_target_naive[i]:8.4f} {on_target_ortho[i]:8.4f} {on_target_random[i]:8.4f}")

    print("\n--- Retention ratios (multi / single) ---")
    print(f"{'behavior':12s} {'naive_ret':>10} {'ortho_ret':>10}")
    for i in range(K):
        print(f"{BEHAVIOR_NAMES[i]:12s} {retention_naive[i]:10.4f} {retention_ortho[i]:10.4f}")

    print("\n--- Off-target leakage from SINGLE steers ---")
    for i in range(K):
        print(f"  steer {BEHAVIOR_NAMES[i][:10]:10s} → mean |cross|: {single_leakage[i]:.4f}")

    # Summary interference matrices (effect - baseline, normalized by max single effect)
    max_single = np.max(np.abs(on_target_single)) + eps
    norm_single_cross = single_cross / max_single
    norm_naive_cross = (naive_mean - baseline_mean) / max_single
    norm_ortho_cross = (ortho_mean - baseline_mean) / max_single

    # Mean off-target leakage in multi conditions
    # naive: for each intended steer i, the off-target = effects on j≠i
    # In simultaneous steering, all K behaviors are "on" — off-target means the K
    # behaviors you didn't want to change when targeting one specific behavior.
    # But since we steer ALL simultaneously, the off-target for "targeting i" in
    # multi is the effect on ALL other j caused by the combined injection.
    # Simplification: compare naive vs ortho on the cross-behavior deviations.

    naive_offdiag = np.abs(naive_mean - baseline_mean)  # all are "steered" in multi
    ortho_offdiag = np.abs(ortho_mean - baseline_mean)

    # For each behavior, compute: does ortho retain the on-target while reducing
    # the deviation from baseline on the orthogonal dimension?
    # Key metric: for NAIVE vs ORTHO in multi-condition:
    #   - Desired: on-target[i] large (same direction as single)
    #   - Desired: cross-coupling (off-target changes due to other steers) small

    # Cross-coupling in multi: for each pair (i, j), compute how much steering all
    # K simultaneously moves behavior j beyond what steering i alone would do.
    # = multi_j - single_j_from_steer_i
    naive_cross_coupling = np.zeros((K, K))
    ortho_cross_coupling = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i == j:
                naive_cross_coupling[i, j] = (
                    (naive_mean[i] - baseline_mean[i])
                    - (single_scores[i, i] - baseline_mean[i])
                )
                ortho_cross_coupling[i, j] = (
                    (ortho_mean[i] - baseline_mean[i])
                    - (single_scores[i, i] - baseline_mean[i])
                )
            else:
                # Extra displacement of j caused by simultaneously steering all K
                # versus just steering i
                naive_cross_coupling[i, j] = (
                    (naive_mean[j] - baseline_mean[j])
                    - (single_scores[i, j] - baseline_mean[j])
                )
                ortho_cross_coupling[i, j] = (
                    (ortho_mean[j] - baseline_mean[j])
                    - (single_scores[i, j] - baseline_mean[j])
                )

    # Mean absolute off-target cross-coupling
    off_naive = np.zeros(K)
    off_ortho = np.zeros(K)
    for i in range(K):
        off_vals_n = [abs(naive_cross_coupling[i, j]) for j in range(K) if j != i]
        off_vals_o = [abs(ortho_cross_coupling[i, j]) for j in range(K) if j != i]
        off_naive[i] = np.mean(off_vals_n)
        off_ortho[i] = np.mean(off_vals_o)

    print("\n--- Cross-coupling (multi vs single) per behavior ---")
    print(f"{'behavior':12s} {'naive_xcoup':>12} {'ortho_xcoup':>12} {'reduction':>10}")
    for i in range(K):
        red = (off_naive[i] - off_ortho[i]) / (off_naive[i] + eps)
        print(f"{BEHAVIOR_NAMES[i]:12s} {off_naive[i]:12.4f} {off_ortho[i]:12.4f} {red:10.2%}")

    mean_off_naive = float(off_naive.mean())
    mean_off_ortho = float(off_ortho.mean())
    mean_retention_naive = float(retention_naive.mean())
    mean_retention_ortho = float(retention_ortho.mean())

    # ---- Verdict ----
    ortho_better_leakage = mean_off_ortho < mean_off_naive
    ortho_better_retention = mean_retention_ortho > mean_retention_naive - 0.05  # within 5%
    reduction_pct = (mean_off_naive - mean_off_ortho) / (mean_off_naive + eps) * 100

    print("\n=== VERDICT ===")
    print(f"  Mean on-target retention  | NAIVE: {mean_retention_naive:.4f}  ORTHO: {mean_retention_ortho:.4f}")
    print(f"  Mean off-target coupling  | NAIVE: {mean_off_naive:.4f}  ORTHO: {mean_off_ortho:.4f}")
    print(f"  Reduction in cross-coupling: {reduction_pct:.1f}%")
    if ortho_better_leakage and ortho_better_retention:
        verdict = (
            f"ORTHO WINS: orthogonal coding reduces cross-coupling by "
            f"{reduction_pct:.1f}% while retaining {mean_retention_ortho:.3f} of "
            f"single-steer effect (naive: {mean_retention_naive:.3f})."
        )
    elif ortho_better_leakage and not ortho_better_retention:
        verdict = (
            f"PARTIAL ORTHO: cross-coupling down {reduction_pct:.1f}% but on-target "
            f"retention also drops (ortho {mean_retention_ortho:.3f} vs naive "
            f"{mean_retention_naive:.3f}). Trade-off exists."
        )
    else:
        verdict = (
            f"NAIVE IS FINE: orthogonalization does NOT reduce cross-coupling "
            f"(naive {mean_off_naive:.4f} vs ortho {mean_off_ortho:.4f}). "
            f"Raw steering vectors are already near-orthogonal or the channel "
            f"is robust."
        )
    print(f"\n  {verdict}")

    # ---- Save results ----
    result = {
        "model": MODEL_ID,
        "layer": LAYER,
        "alpha": ALPHA,
        "behaviors": BEHAVIOR_NAMES,
        "baseline_scores": baseline_mean.tolist(),
        "raw_cosine_matrix": cosine_matrix.tolist(),
        "conditions": {
            "SINGLE": {
                "description": "Each behavior steered alone. Matrix[i][j] = effect on j when steering i.",
                "interference_matrix": single_scores.tolist(),
                "on_target_effects": on_target_single.tolist(),
                "mean_off_target_leakage": single_leakage.tolist(),
            },
            "NAIVE_MULTI": {
                "description": "All K raw vectors added simultaneously.",
                "mean_scores": naive_mean.tolist(),
                "on_target_effects": on_target_naive.tolist(),
                "on_target_retention": retention_naive.tolist(),
                "cross_coupling_matrix": naive_cross_coupling.tolist(),
                "mean_off_target_cross_coupling": float(mean_off_naive),
                "mean_on_target_retention": float(mean_retention_naive),
            },
            "ORTHO_MULTI": {
                "description": "All K Gram-Schmidt orthogonalized vectors added simultaneously.",
                "mean_scores": ortho_mean.tolist(),
                "on_target_effects": on_target_ortho.tolist(),
                "on_target_retention": retention_ortho.tolist(),
                "cross_coupling_matrix": ortho_cross_coupling.tolist(),
                "mean_off_target_cross_coupling": float(mean_off_ortho),
                "mean_on_target_retention": float(mean_retention_ortho),
            },
            "RANDOM_CONTROL": {
                "description": "K random orthogonal vectors — should show ~0 on-target effect.",
                "mean_scores": random_mean.tolist(),
                "on_target_effects": on_target_random.tolist(),
                "delta_from_baseline": (random_mean - baseline_mean).tolist(),
            },
        },
        "summary": {
            "naive_mean_on_target_retention": float(mean_retention_naive),
            "ortho_mean_on_target_retention": float(mean_retention_ortho),
            "naive_mean_off_target_cross_coupling": float(mean_off_naive),
            "ortho_mean_off_target_cross_coupling": float(mean_off_ortho),
            "cross_coupling_reduction_pct": float(reduction_pct),
            "ortho_better_leakage": bool(ortho_better_leakage),
            "ortho_better_retention": bool(ortho_better_retention),
        },
        "verdict": verdict,
    }

    out_path = "eigen_steering_result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[done] Results saved to {out_path}")
    return result


if __name__ == "__main__":
    main()
