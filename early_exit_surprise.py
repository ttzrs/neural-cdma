"""¿La sorpresa-por-token predice early-exit? (resquicio CPU conectado al paper surprise→easiness).

Hipótesis: tokens fáciles (baja sorpresa) saturan su predicción en capas TEMPRANAS → se puede early-exit
sin cambiar el argmax → menos pesos leídos → CPU memory-bound más rápido en la fracción fácil.

Método (logit-lens, sin entrenar exit heads): para cada posición, aplica norm+lm_head al hidden state de
CADA capa → predicción temprana. Capa-de-salida = menor L tal que pred[L:]==pred_final (saturación estable).
Estratifica capa-de-salida por tercil de sorpresa del token predicho. Mide cómputo ahorrable.

CAVEAT honesto: early-exit real necesita KV de capas saltadas (CALM/LayerSkip lo resuelven copiando estado);
esto mide la SEÑAL (¿saturan temprano los fáciles?), no implementa el runtime.
"""
from __future__ import annotations
import itertools, json
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

dev = torch.device("cpu")   # no GPU local (regla Jose); probe pequeño cabe en CPU
torch.set_num_threads(torch.get_num_threads())
torch.manual_seed(0)
MODEL = "Qwen/Qwen2.5-0.5B"
N_TOK = 1024

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(dev).eval()
norm, lm_head = model.model.norm, model.lm_head
L = model.config.num_hidden_layers
print(f"capas={L} dev={dev}")

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation", streaming=True)
buf = [r["text"] for r in itertools.islice(iter(ds), 4000) if r["text"].strip()]
ids = tok(" ".join(buf)).input_ids[:N_TOK]
ids = torch.tensor(ids, device=dev).unsqueeze(0)


@torch.no_grad()
def run():
    out = model(ids, output_hidden_states=True)
    hs = out.hidden_states              # tuple L+1: (1,T,d)
    final = out.logits[0]              # (T,V)
    final_pred = final.argmax(-1)      # (T,)
    # dificultad = -log p_final(token predicho)  (baja = fácil/confiado)
    surprise = -F.log_softmax(final, -1).gather(-1, final_pred.unsqueeze(-1)).squeeze(-1)  # (T,)
    # pred temprana por capa (1..L)
    early = torch.stack([lm_head(norm(hs[l][0])).argmax(-1) for l in range(1, L + 1)])  # (L,T)
    # capa-de-salida: menor l con pred[l:]==final para todas las capas siguientes
    match = (early == final_pred.unsqueeze(0))                    # (L,T)
    # estable desde l: cumsum desde el final
    stable_from = torch.zeros_like(final_pred)                    # exit layer (1-indexed)
    T = final_pred.shape[0]
    # para cada token, recorrer de la última capa hacia atrás
    allmatch = match.flip(0).cumprod(0).flip(0)                  # (L,T) 1 si match desde l hasta el final
    for t in range(T):
        idx = (allmatch[:, t] == 1).nonzero()
        stable_from[t] = (idx[0, 0].item() + 1) if idx.numel() else L
    return surprise.cpu(), stable_from.cpu().float()


def main():
    surp, exitL = run()
    q1, q2 = torch.quantile(surp, torch.tensor([1/3, 2/3]))
    easy, mid, hard = surp <= q1, (surp > q1) & (surp <= q2), surp > q2
    res = {"model": MODEL, "n_layers": L, "n_tokens": int(surp.numel()),
           "surprise_terciles": [round(float(q1), 3), round(float(q2), 3)]}
    print(f"\n  tercil      | capa-salida media | % salible@L/2 | cómputo ahorrado")
    for name, m in (("FÁCIL", easy), ("MEDIO", mid), ("DIFÍCIL", hard), ("TODOS", torch.ones_like(easy))):
        el = exitL[m]
        mean_exit = float(el.mean())
        pct_half = float((el <= L // 2).float().mean()) * 100
        saved = float((1 - el / L).mean()) * 100      # % de capas saltables en promedio
        res[name] = {"mean_exit_layer": round(mean_exit, 2), "pct_exit_by_halfdepth": round(pct_half, 1),
                     "compute_saved_pct": round(saved, 1)}
        print(f"  {name:10s} | {mean_exit:5.2f} / {L}        | {pct_half:5.1f}%        | {saved:5.1f}%")
    json.dump(res, open("early_exit_surprise_result.json", "w"), indent=2)
    e, h = res["FÁCIL"]["mean_exit_layer"], res["DIFÍCIL"]["mean_exit_layer"]
    print(f"\n  Señal: fácil sale en capa {e} vs difícil {h}." +
          (" FUERTE → sorpresa gatea early-exit." if h - e > 2 else " DÉBIL → poca separación."))
    print("→ early_exit_surprise_result.json")


if __name__ == "__main__":
    main()
