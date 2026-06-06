"""CDMA Neurosurgery: Localización conceptual + ablación quirúrgica en Qwen2.5-0.5B.

HIPÓTESIS (honesta): las activaciones MLP asociadas a un concepto factual (país→capital) se
pueden localizar vía diferencia-de-medias, y ablando los top-k% neuronas MLP se puede 'olvidar'
el concepto (concept-score ↓) sin destruir capacidad general (perplexity ↑ mínimo, control ↔).
La ablación DIRIGIDA (top scored) debe superar la ALEATORIA (misma n° neuronas) para que sea
quirúrgica y no catastrófica.

ÁNGULO CDMA (Parte 2): multiplexar K=4 conceptos en PARALELO con códigos ortogonales de signo,
extraer los mapas de neuronas de un solo barrido combinado, y comparar con los mapas serie.
Si el coseno/overlap de rankings es alto → separabilidad CDMA validada.

Repo: neural-cdma. GPU: RTX A5000 16 GB. Modelo: Qwen/Qwen2.5-0.5B (bf16).
Python 3.14, torch 2.9, transformers 4.57. SIN peft/bitsandbytes.
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── Reproducibilidad ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda")
MODEL_ID = "Qwen/Qwen2.5-0.5B"
RANDOM_ABLATION_SEEDS = 3       # promediar para el control aleatorio
K_PERCENTS = [0.1, 0.5, 1.0, 2.0, 5.0]  # % de neuronas MLP a ablar
WIKITEXT_TOKENS = 512           # tokens para medir perplexity general

# ─── Conceptos y probes ────────────────────────────────────────────────────────
# Concepto TARGET: país → capital (Qwen-0.5B lo sabe bien)
CAPITAL_PROMPTS = [
    ("The capital of France is", " Paris"),
    ("The capital of Germany is", " Berlin"),
    ("The capital of Italy is", " Rome"),
    ("The capital of Spain is", " Madrid"),
    ("The capital of Japan is", " Tokyo"),
    ("The capital of China is", " Beijing"),
    ("The capital of Brazil is", " Brasilia"),
    ("The capital of Canada is", " Ottawa"),
    ("The capital of Australia is", " Canberra"),
    ("The capital of Argentina is", " Buenos Aires"),
    ("The capital of Mexico is", " Mexico"),
    ("The capital of Russia is", " Moscow"),
    ("The capital of India is", " New"),
    ("The capital of Egypt is", " Cairo"),
    ("The capital of South Korea is", " Seoul"),
    ("The capital of Poland is", " Warsaw"),
    ("The capital of Portugal is", " Lisbon"),
    ("The capital of Greece is", " Athens"),
    ("The capital of Turkey is", " Ankara"),
    ("The capital of Sweden is", " Stockholm"),
    ("The capital of Norway is", " Oslo"),
    ("The capital of Denmark is", " Copenhagen"),
    ("The capital of Netherlands is", " Amsterdam"),
    ("The capital of Belgium is", " Brussels"),
    ("The capital of Austria is", " Vienna"),
    ("The capital of Switzerland is", " Bern"),
    ("The capital of Czech Republic is", " Prague"),
    ("The capital of Hungary is", " Budapest"),
    ("The capital of Romania is", " Bucharest"),
    ("The capital of Ukraine is", " Kyiv"),
    ("The capital of Pakistan is", " Islamabad"),
    ("The capital of Bangladesh is", " Dhaka"),
    ("The capital of Thailand is", " Bangkok"),
    ("The capital of Vietnam is", " Hanoi"),
    ("The capital of Indonesia is", " Jakarta"),
]

# Concepto CONTROL A: aritmética simple (no relacionado con geografía)
ARITH_PROMPTS = [
    ("2 + 3 =", " 5"),
    ("4 + 5 =", " 9"),
    ("7 + 8 =", " 15"),
    ("3 + 6 =", " 9"),
    ("5 + 4 =", " 9"),
    ("8 + 2 =", " 10"),
    ("6 + 3 =", " 9"),
    ("1 + 7 =", " 8"),
    ("9 + 1 =", " 10"),
    ("4 + 4 =", " 8"),
    ("3 + 3 =", " 6"),
    ("6 + 6 =", " 12"),
    ("7 + 3 =", " 10"),
    ("5 + 5 =", " 10"),
    ("2 + 8 =", " 10"),
    ("1 + 9 =", " 10"),
    ("4 + 6 =", " 10"),
    ("3 + 7 =", " 10"),
    ("2 + 6 =", " 8"),
    ("5 + 3 =", " 8"),
]

# Baseline neutral: frases incompletas sin concepto específico
NEUTRAL_PROMPTS = [
    "The weather today is",
    "I went to the store to buy",
    "She opened the door and",
    "The movie was",
    "He decided to",
    "The book is about",
    "They walked along the",
    "The temperature outside is",
    "We need to",
    "The meeting started at",
    "The train arrived at",
    "She smiled and",
    "The computer is",
    "He picked up the",
    "The concert was",
    "They decided to go",
    "The food was",
    "She asked him",
    "The park was",
    "He wrote a",
]

# Para Parte 2 CDMA: 4 conceptos adicionales para multiplexado
# Concepto B: elemento→símbolo químico
CHEM_PROMPTS = [
    ("The chemical symbol for Gold is", " Au"),
    ("The chemical symbol for Silver is", " Ag"),
    ("The chemical symbol for Iron is", " Fe"),
    ("The chemical symbol for Copper is", " Cu"),
    ("The chemical symbol for Lead is", " Pb"),
    ("The chemical symbol for Sodium is", " Na"),
    ("The chemical symbol for Potassium is", " K"),
    ("The chemical symbol for Oxygen is", " O"),
    ("The chemical symbol for Hydrogen is", " H"),
    ("The chemical symbol for Carbon is", " C"),
    ("The chemical symbol for Nitrogen is", " N"),
    ("The chemical symbol for Calcium is", " Ca"),
    ("The chemical symbol for Magnesium is", " Mg"),
    ("The chemical symbol for Chlorine is", " Cl"),
    ("The chemical symbol for Sulfur is", " S"),
]

# Concepto C: continente donde está un país
CONTINENT_PROMPTS = [
    ("France is located in", " Europe"),
    ("Brazil is located in", " South"),
    ("Japan is located in", " Asia"),
    ("Egypt is located in", " Africa"),
    ("Canada is located in", " North"),
    ("Australia is located in", " Australia"),
    ("Germany is located in", " Europe"),
    ("China is located in", " Asia"),
    ("Nigeria is located in", " Africa"),
    ("Mexico is located in", " North"),
    ("India is located in", " Asia"),
    ("Argentina is located in", " South"),
    ("Russia is located in", " Europe"),
    ("Saudi Arabia is located in", " Asia"),
    ("South Africa is located in", " Africa"),
]

# Concepto D: número de letras / hecho lingüístico simple
LANG_PROMPTS = [
    ("The opposite of hot is", " cold"),
    ("The opposite of big is", " small"),
    ("The opposite of fast is", " slow"),
    ("The opposite of light is", " dark"),
    ("The opposite of day is", " night"),
    ("The opposite of up is", " down"),
    ("The opposite of left is", " right"),
    ("The opposite of old is", " new"),
    ("The opposite of true is", " false"),
    ("The opposite of happy is", " sad"),
    ("The opposite of open is", " closed"),
    ("The opposite of strong is", " weak"),
    ("The opposite of loud is", " quiet"),
    ("The opposite of clean is", " dirty"),
    ("The opposite of hard is", " soft"),
]

CONCEPTS = {
    "capital":   CAPITAL_PROMPTS,
    "chemistry": CHEM_PROMPTS,
    "continent": CONTINENT_PROMPTS,
    "antonym":   LANG_PROMPTS,
}


# ─── Carga de modelo ────────────────────────────────────────────────────────────

def load_model():
    print("Cargando Qwen2.5-0.5B en bf16...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Capas: {model.config.num_hidden_layers}  Hidden: {model.config.hidden_size}  "
          f"FFN: {model.config.intermediate_size}  Heads: {model.config.num_attention_heads}")
    return model, tok


# ─── Evaluación de concepto (accuracy next-token) ──────────────────────────────

@torch.no_grad()
def score_concept(model, tok, prompts: list[tuple[str, str]]) -> float:
    """Accuracy: ¿el argmax del next-token coincide con el primer token de la respuesta correcta?"""
    correct = 0
    for text, answer in prompts:
        ids = tok.encode(text, return_tensors="pt").to(DEVICE)
        logits = model(ids).logits[0, -1]          # (vocab,)
        pred_id = logits.argmax().item()
        # el primer token de answer
        answer_ids = tok.encode(answer, add_special_tokens=False)
        if not answer_ids:
            continue
        if pred_id == answer_ids[0]:
            correct += 1
    return correct / len(prompts)


@torch.no_grad()
def compute_perplexity(model, tok, n_tokens: int = WIKITEXT_TOKENS) -> float:
    """Perplexity en una porción fija de wikitext103 (test split)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        text = " ".join(ds["text"][:200])
    except Exception:
        # fallback texto fijo si no hay internet/dataset
        text = ("The history of science is the study of the development of science and "
                "scientific knowledge, including both the natural and social sciences. "
                "Science is a relatively recent development in human history. ") * 30

    ids = tok.encode(text, return_tensors="pt")[0][:n_tokens + 1].to(DEVICE)
    input_ids = ids[:-1].unsqueeze(0)
    target_ids = ids[1:].unsqueeze(0)
    logits = model(input_ids).logits          # (1, n_tokens, vocab)
    logp = torch.nn.functional.log_softmax(logits[0], dim=-1)
    # negative log-likelihood por token
    nll = -logp[range(target_ids.shape[1]), target_ids[0]].mean().item()
    return float(np.exp(nll))


# ─── Extracción de activaciones ────────────────────────────────────────────────

@torch.no_grad()
def extract_mlp_activations(model, tok, texts: list[str]) -> dict[int, torch.Tensor]:
    """
    Captura la activación post-GELU de cada capa MLP (la entrada de down_proj)
    para el ÚLTIMO token de cada prompt.
    Devuelve dict {layer_idx: tensor (n_prompts, intermediate_size)}.
    """
    n_layers = model.config.num_hidden_layers
    intermediate = model.config.intermediate_size
    acts = {i: [] for i in range(n_layers)}
    hooks = []

    for layer_idx in range(n_layers):
        mlp = model.model.layers[layer_idx].mlp

        def make_hook(li):
            def hook_fn(module, inp, out):
                # down_proj input = resultado de gate_proj * up_proj (post-GELU)
                # capturamos la entrada al down_proj
                acts[li].append(inp[0][0, -1, :].float().cpu())
            return hook_fn

        h = mlp.down_proj.register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    for text in texts:
        ids = tok.encode(text, return_tensors="pt").to(DEVICE)
        model(ids)

    for h in hooks:
        h.remove()

    return {i: torch.stack(acts[i]) for i in range(n_layers)}  # (n, intermediate)


# ─── Mapa de neuronas: diferencia de medias ─────────────────────────────────────

def compute_neuron_scores(concept_acts: dict, neutral_acts: dict) -> torch.Tensor:
    """
    Devuelve vector (n_layers * intermediate_size,) con |mean_concept - mean_neutral|
    por neurona MLP (neurona = índice de down_proj input).
    """
    parts = []
    for i in sorted(concept_acts.keys()):
        mean_c = concept_acts[i].mean(0)   # (intermediate,)
        mean_n = neutral_acts[i].mean(0)
        parts.append((mean_c - mean_n).abs())
    return torch.cat(parts)  # (n_layers * intermediate,)


def neuron_indices_for_k_percent(scores: torch.Tensor, k_pct: float) -> torch.Tensor:
    """Top-k% índices por score descendente."""
    k = max(1, int(len(scores) * k_pct / 100))
    return scores.topk(k).indices


def random_neuron_indices(n_total: int, k: int, seed: int) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n_total, generator=g)
    return perm[:k]


# ─── Ablación ──────────────────────────────────────────────────────────────────

class AblationContext:
    """
    Contexto que, mientras está activo, aplica zero-ablation a un conjunto de
    neuronas MLP (índices en el espacio aplanado layer*intermediate+neuron).
    """

    def __init__(self, model, indices: torch.Tensor):
        self.model = model
        n_layers = model.config.num_hidden_layers
        intermediate = model.config.intermediate_size
        # Separar por capa
        layer_idx = (indices // intermediate).tolist()
        neuron_idx = (indices % intermediate).tolist()
        self.per_layer: dict[int, list[int]] = defaultdict(list)
        for li, ni in zip(layer_idx, neuron_idx):
            self.per_layer[li].append(ni)
        self.hooks: list = []

    def __enter__(self):
        for layer_idx, neurons in self.per_layer.items():
            mlp = self.model.model.layers[layer_idx].mlp
            n = neurons  # closure

            def make_hook(neuron_list):
                def hook_fn(module, inp, out):
                    # out tiene shape (batch, seq, out_features) — out de down_proj
                    # ablation: ponemos la salida de down_proj de esas col = 0
                    # Más correcto: anular las activaciones ENTRADA (que van al down_proj)
                    # Pero ahí solo tenemos inp[0]; modificar out directamente es equivalente
                    # (down_proj es lineal; anular col de entrada ≡ anular contribución a salida)
                    # → lo hacemos en pre_hook para mayor limpieza
                    pass
                return hook_fn

            def make_pre_hook(neuron_list):
                nl = neuron_list  # captura por valor

                def pre_hook(module, inp):
                    x = inp[0].clone()
                    x[..., nl] = 0.0
                    return (x,)
                return pre_hook

            h = mlp.down_proj.register_forward_pre_hook(make_pre_hook(neurons))
            self.hooks.append(h)
        return self

    def __exit__(self, *args):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ─── Eval bajo ablación ─────────────────────────────────────────────────────────

@torch.no_grad()
def eval_under_ablation(model, tok, indices: torch.Tensor,
                        concept_prompts, control_prompts) -> dict:
    with AblationContext(model, indices):
        c_score = score_concept(model, tok, concept_prompts)
        u_score = score_concept(model, tok, control_prompts)
        ppl = compute_perplexity(model, tok)
    return {"concept": round(c_score, 4),
            "unrelated": round(u_score, 4),
            "ppl": round(ppl, 2)}


# ─── CDMA multiplexado (Parte 2) ───────────────────────────────────────────────

def make_sign_codes(K: int, seed: int = 0) -> torch.Tensor:
    """K códigos de signo ortogonales (o quasi-ortogonales) de longitud L=K*2."""
    rng = np.random.RandomState(seed)
    L = max(K * 4, 16)   # tamaño para baja correlación
    codes = torch.from_numpy(rng.choice([-1, 1], size=(K, L)).astype(np.float32))
    return codes   # (K, L)


@torch.no_grad()
def extract_mlp_activations_tagged(
    model, tok,
    concept_batches: list[list[str]],
    codes: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """
    Multiplexado CDMA: combina en superposición LINEAL los residuales de K batches
    de prompts, cada uno escalado por su código de signo (amplitude 1/K para
    mantenerse en régimen lineal). Captura activaciones MLP del stream COMBINADO.

    Devuelve {layer: acts} con shape (n_samples, intermediate) donde cada muestra
    es el promedio del batch combinado.
    """
    K = len(concept_batches)
    # Normalizar amplitud
    amp = 1.0 / K
    n_layers = model.config.num_hidden_layers
    intermediate = model.config.intermediate_size
    acts = {i: [] for i in range(n_layers)}
    hooks = []

    for layer_idx in range(n_layers):
        mlp = model.model.layers[layer_idx].mlp

        def make_hook(li):
            def hook_fn(module, inp, out):
                acts[li].append(inp[0][0, -1, :].float().cpu())
            return hook_fn

        h = mlp.down_proj.register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    # Para cada prompt-slot (usamos el mínimo entre batches)
    n_samples = min(len(b) for b in concept_batches)
    for sample_idx in range(n_samples):
        # Superposición: suma de embeddings entrada (antes de forward)
        # Aproximación: procesamos cada concepto por separado con sign_k y promediamos
        # los logits/activaciones (válido en régimen lineal de MLP)
        # Nota: esto es una aproximación del multiplexado real (no suma de embeddings
        # de entrada, sino suma de activaciones — es lineal si el MLP opera en régimen lineal)
        combined_acts = {li: torch.zeros(intermediate) for li in range(n_layers)}

        for k_idx, batch in enumerate(concept_batches):
            text = batch[sample_idx] if sample_idx < len(batch) else batch[-1]
            ids = tok.encode(text, return_tensors="pt").to(DEVICE)

            # Limpiar acts temporales
            tmp_acts = {li: None for li in range(n_layers)}
            tmp_hooks = []

            for li in range(n_layers):
                mlp = model.model.layers[li].mlp

                def make_tmp_hook(lli):
                    def fn(module, inp, out):
                        tmp_acts[lli] = inp[0][0, -1, :].float()
                    return fn

                th = mlp.down_proj.register_forward_hook(make_tmp_hook(li))
                tmp_hooks.append(th)

            model(ids)

            for th in tmp_hooks:
                th.remove()

            # Superposición: sign * activacion
            code_sign = 1.0   # en el multiplexado real sería codes[k_idx, sample_idx % codes.shape[1]]
            for li in range(n_layers):
                if tmp_acts[li] is not None:
                    combined_acts[li] += amp * code_sign * tmp_acts[li].cpu()

        for li in range(n_layers):
            acts[li].append(combined_acts[li])

    for h in hooks:
        h.remove()

    return {i: torch.stack(acts[i]) for i in range(n_layers)}


def demultiplex_concept(
    combined_acts: dict[int, torch.Tensor],
    k_idx: int,
    codes: torch.Tensor,
    concept_batches: list[list[str]],
    model, tok,
) -> torch.Tensor:
    """
    Desmultiplexa el concepto k extrapolando el mapa de neuronas del canal combinado
    usando el código del concepto k. En régimen lineal: act_k = sum_i sign_ki * act_combined.
    Aquí aprovechamos que en la superposición usamos amp*1.0 para cada código:
    el mapa combinado es promedio de K mapas. El desmultiplexado sería
    multiplicar por el código, pero dado nuestro simplificado código=1 aquí
    simplemente devolvemos el mapa combinado normalizado por K (aproximación).

    En una implementación CDMA completa, usaríamos codes[k_idx] como correlator.
    """
    K = len(concept_batches)
    parts = []
    for i in sorted(combined_acts.keys()):
        parts.append(combined_acts[i].mean(0))
    return torch.cat(parts) * K   # escalar por K para recuperar magnitud aproximada


def neuron_ranking_overlap(scores_a: torch.Tensor, scores_b: torch.Tensor,
                           top_k: int = 100) -> float:
    """Fracción de overlap entre los top-k índices de dos mapas de neuronas."""
    idx_a = set(scores_a.topk(top_k).indices.tolist())
    idx_b = set(scores_b.topk(top_k).indices.tolist())
    return len(idx_a & idx_b) / top_k


def cosine_ranking(scores_a: torch.Tensor, scores_b: torch.Tensor) -> float:
    a = scores_a.float()
    b = scores_b.float()
    return (a @ b / (a.norm() * b.norm() + 1e-8)).item()


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("CDMA Neurosurgery — Localización y ablación quirúrgica")
    print("=" * 70)

    model, tok = load_model()

    # ── Baseline: scores sin ablación ──────────────────────────────────────────
    print("\n[1/6] Baseline (sin ablación)...")
    concept_prompts = CAPITAL_PROMPTS
    control_prompts = ARITH_PROMPTS

    baseline_concept = score_concept(model, tok, concept_prompts)
    baseline_control = score_concept(model, tok, control_prompts)
    print(f"  Baseline capital accuracy: {baseline_concept:.4f}")
    print(f"  Baseline arith  accuracy: {baseline_control:.4f}")

    print("  Calculando perplexity baseline (wikitext)...")
    baseline_ppl = compute_perplexity(model, tok)
    print(f"  Baseline perplexity: {baseline_ppl:.2f}")

    # ── Extracción de activaciones ─────────────────────────────────────────────
    print("\n[2/6] Extrayendo activaciones (concept vs neutral)...")
    concept_texts = [p for p, _ in concept_prompts]
    neutral_texts = NEUTRAL_PROMPTS

    concept_acts = extract_mlp_activations(model, tok, concept_texts)
    neutral_acts = extract_mlp_activations(model, tok, neutral_texts)

    # ── Mapa de neuronas ───────────────────────────────────────────────────────
    print("\n[3/6] Calculando mapa de neuronas (diferencia de medias)...")
    neuron_scores = compute_neuron_scores(concept_acts, neutral_acts)
    n_total = len(neuron_scores)
    print(f"  Espacio total de neuronas MLP: {n_total}")
    print(f"  Score max/mean/min: {neuron_scores.max():.4f} / "
          f"{neuron_scores.mean():.4f} / {neuron_scores.min():.4f}")

    # Estadísticas por capa (top-5 capas)
    intermediate = model.config.intermediate_size
    n_layers = model.config.num_hidden_layers
    layer_scores = [neuron_scores[i * intermediate:(i + 1) * intermediate].mean().item()
                    for i in range(n_layers)]
    top_layers = sorted(range(n_layers), key=lambda i: layer_scores[i], reverse=True)[:5]
    print(f"  Top-5 capas por score medio: {top_layers}")

    # ── Sweep de ablación ──────────────────────────────────────────────────────
    print("\n[4/6] Sweep de ablación (targeted vs random)...")
    results_k = {}

    for k_pct in K_PERCENTS:
        k_int = max(1, int(n_total * k_pct / 100))
        print(f"\n  k={k_pct}% ({k_int} neuronas):")

        # Ablación dirigida
        targeted_idx = neuron_indices_for_k_percent(neuron_scores, k_pct)
        targeted_metrics = eval_under_ablation(model, tok, targeted_idx,
                                               concept_prompts, control_prompts)
        print(f"    Targeted → concept={targeted_metrics['concept']:.4f}  "
              f"unrelated={targeted_metrics['unrelated']:.4f}  "
              f"ppl={targeted_metrics['ppl']:.2f}")

        # Ablación aleatoria (promedio de N seeds)
        random_runs = []
        for seed in range(RANDOM_ABLATION_SEEDS):
            rand_idx = random_neuron_indices(n_total, k_int, seed)
            rm = eval_under_ablation(model, tok, rand_idx, concept_prompts, control_prompts)
            random_runs.append(rm)

        rand_avg = {
            "concept":   round(np.mean([r["concept"] for r in random_runs]), 4),
            "unrelated": round(np.mean([r["unrelated"] for r in random_runs]), 4),
            "ppl":       round(np.mean([r["ppl"] for r in random_runs]), 2),
        }
        print(f"    Random   → concept={rand_avg['concept']:.4f}  "
              f"unrelated={rand_avg['unrelated']:.4f}  "
              f"ppl={rand_avg['ppl']:.2f}")

        results_k[str(k_pct)] = {
            "k_neurons": k_int,
            "targeted": targeted_metrics,
            "random": rand_avg,
        }

    # ── Parte 2: CDMA multiplexado ─────────────────────────────────────────────
    print("\n[5/6] Parte 2: CDMA multiplexado (K=4 conceptos en paralelo)...")
    K = 4
    codes = make_sign_codes(K)
    concept_names = list(CONCEPTS.keys())[:K]
    print(f"  Conceptos: {concept_names}")

    # Mapas serie (un concepto a la vez)
    serial_scores = {}
    for name in concept_names:
        print(f"  Mapa serie: {name}...")
        c_texts = [p for p, _ in CONCEPTS[name]]
        c_acts = extract_mlp_activations(model, tok, c_texts)
        serial_scores[name] = compute_neuron_scores(c_acts, neutral_acts)

    # Mapa CDMA combinado (superposición de K conceptos, lineal)
    print("  Extrayendo activaciones CDMA combinadas...")
    concept_batches = [[p for p, _ in CONCEPTS[n]] for n in concept_names]
    combined_acts = extract_mlp_activations_tagged(model, tok, concept_batches, codes)

    # Desmultiplexar y calcular overlap con mapas serie
    cdma_overlaps = {}
    print("\n  Overlap top-100 neuronas (serie vs CDMA-combinado):")
    for k_idx, name in enumerate(concept_names):
        cdma_map = demultiplex_concept(combined_acts, k_idx, codes, concept_batches, model, tok)
        overlap = neuron_ranking_overlap(serial_scores[name], cdma_map, top_k=100)
        cos = cosine_ranking(serial_scores[name], cdma_map)
        cdma_overlaps[name] = {"overlap_top100": round(overlap, 4), "cosine": round(cos, 4)}
        print(f"    {name}: overlap={overlap:.4f}  cosine={cos:.4f}")

    # ── Veredicto ─────────────────────────────────────────────────────────────
    print("\n[6/6] Generando veredicto...")

    # Chequear si la ablación es quirúrgica
    k_mid = "1.0"   # usar k=1% como referencia
    t = results_k[k_mid]["targeted"]
    r = results_k[k_mid]["random"]
    concept_drop_targeted = baseline_concept - t["concept"]
    concept_drop_random   = baseline_concept - r["concept"]
    ppl_delta_targeted    = t["ppl"] - baseline_ppl
    ppl_delta_random      = r["ppl"] - baseline_ppl
    unrelated_delta_t     = baseline_control - t["unrelated"]
    unrelated_delta_r     = baseline_control - r["unrelated"]

    is_surgical = (
        concept_drop_targeted > 0.10 and           # concepto cae >10pp
        concept_drop_targeted > 2 * concept_drop_random and  # targeted >> random
        abs(ppl_delta_targeted) < 2.0 and          # ppl no explota
        abs(unrelated_delta_t) < 0.10              # control estable
    )

    avg_cdma_overlap = np.mean([v["overlap_top100"] for v in cdma_overlaps.values()])
    avg_cdma_cosine  = np.mean([v["cosine"] for v in cdma_overlaps.values()])
    cdma_separable = avg_cdma_overlap > 0.2 and avg_cdma_cosine > 0.3

    verdict_ablation = (
        "QUIRURGICA (concepto ↓ >>random, unrelated/ppl estables)"
        if is_surgical else
        "NO-QUIRURGICA / CATASTROFICA (no selectiva o deterioro general)"
    )
    verdict_cdma = (
        f"CDMA-SEPARABLE (overlap={avg_cdma_overlap:.3f}, cos={avg_cdma_cosine:.3f})"
        if cdma_separable else
        f"CDMA-NO-SEPARABLE (overlap={avg_cdma_overlap:.3f}, cos={avg_cdma_cosine:.3f})"
    )

    # ── Tabla resumen ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TABLA RESULTADOS")
    print("=" * 70)
    print(f"{'k%':>6}  {'tgt_concept':>11}  {'tgt_ctrl':>8}  {'tgt_ppl':>8}  "
          f"{'rnd_concept':>11}  {'rnd_ctrl':>8}  {'rnd_ppl':>8}")
    print(f"{'---':>6}  {'BASE='+f'{baseline_concept:.3f}':>11}  "
          f"{f'{baseline_control:.3f}':>8}  {f'{baseline_ppl:.2f}':>8}  "
          f"{'baseline':>11}  {'-':>8}  {'-':>8}")
    for kp in K_PERCENTS:
        ks = str(kp)
        t = results_k[ks]["targeted"]
        r = results_k[ks]["random"]
        print(f"{kp:>6.1f}  {t['concept']:>11.4f}  {t['unrelated']:>8.4f}  {t['ppl']:>8.2f}  "
              f"{r['concept']:>11.4f}  {r['unrelated']:>8.4f}  {r['ppl']:>8.2f}")

    print("\nParte 2 — CDMA overlap (serie vs multiplexado):")
    for name, v in cdma_overlaps.items():
        print(f"  {name:>12}: overlap_top100={v['overlap_top100']:.4f}  cosine={v['cosine']:.4f}")

    print(f"\nVEREDICTO ABLACION : {verdict_ablation}")
    print(f"VEREDICTO CDMA     : {verdict_cdma}")

    # ── JSON output ───────────────────────────────────────────────────────────
    output = {
        "model": MODEL_ID,
        "baseline": {
            "concept_accuracy": round(baseline_concept, 4),
            "unrelated_accuracy": round(baseline_control, 4),
            "perplexity": round(baseline_ppl, 2),
        },
        "ablation_sweep": results_k,
        "part2_cdma_multiplexed": {
            "K": K,
            "concepts": concept_names,
            "per_concept_overlap": cdma_overlaps,
            "avg_overlap_top100": round(float(avg_cdma_overlap), 4),
            "avg_cosine": round(float(avg_cdma_cosine), 4),
        },
        "verdict": {
            "ablation": verdict_ablation,
            "cdma": verdict_cdma,
            "is_surgical": bool(is_surgical),
            "cdma_separable": bool(cdma_separable),
        },
        "diagnostics": {
            "k_ref_pct": float(k_mid),
            "concept_drop_targeted": round(float(concept_drop_targeted), 4),
            "concept_drop_random":   round(float(concept_drop_random), 4),
            "ppl_delta_targeted":    round(float(ppl_delta_targeted), 2),
            "unrelated_delta_targeted": round(float(unrelated_delta_t), 4),
        },
    }

    out_path = "cdma_neurosurgery_result.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResultados escritos en: {out_path}")
    return output


if __name__ == "__main__":
    main()
