"""K_cap del modo image_vqvae (tokens visuales ENTRENADOS) con el MISMO harness que vision_kcap_hardvocab.py.
Compara contra los números ya medidos de image (kmeans crudo) y text_bpe en vision_kcap_hardvocab_result.json.
Test: ¿la baja multiplexabilidad de la imagen bajo CDMA era artefacto de tokenización kmeans, o intrínseca?"""

from __future__ import annotations

import json

import torch

import vision_kcap_hardvocab as H   # reusa Model, mask_batch, evaluate, run_mode, V=512, T=64, etc.

assert H.V == 512, "VQ codebook=512 debe coincidir con H.V"


def make_image_vqvae_source():
    codes = torch.load("vqvae_codes.pt").to(H.dev)      # (N,64) en [0,512)
    assert codes.shape[1] == H.T, f"grid {codes.shape[1]} != T={H.T}"
    N = codes.shape[0]

    def sample(bs, K):
        return codes[torch.randint(0, N, (bs, K), device=H.dev)]
    return sample


H.SOURCES["image_vqvae"] = make_image_vqvae_source


def main():
    res = H.run_mode("image_vqvae")
    prev = json.load(open("vision_kcap_hardvocab_result.json"))
    out = {"meta": prev["meta"], "image_vqvae": res,
           "image_kmeans": prev["image"], "text_bpe": prev["text_bpe"], "random": prev["random"]}
    json.dump(out, open("vqvae_kcap_result.json", "w"), indent=2)
    kv, kk, kt = res["k_cap_signal80"], prev["image"]["k_cap_signal80"], prev["text_bpe"]["k_cap_signal80"]
    tv, tk, tt = res["thru_max"], prev["image"]["thru_max"], prev["text_bpe"]["thru_max"]
    print(f"\nVEREDICTO VQ-VAE vs kmeans vs text (V=512):")
    print(f"  K_cap:      vqvae={kv}  kmeans={kk}  text_bpe={kt}")
    print(f"  throughput: vqvae={tv:.3f}  kmeans={tk:.3f}  text_bpe={tt:.3f}")
    if kv > kk or tv > tk:
        print("  → Tokens ENTRENADOS mejoran multiplexabilidad de imagen: parte era artefacto kmeans.")
    else:
        print("  → VQ-VAE NO mejora: baja multiplexabilidad de imagen es INTRÍNSECA (negativo robusto).")
    if kv >= kt and tv >= tt:
        print("  → Imagen-VQ alcanza o supera a lenguaje: predicción estrella RESUCITA.")
    print("→ vqvae_kcap_result.json")


if __name__ == "__main__":
    main()
