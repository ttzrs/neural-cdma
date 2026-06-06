"""Clase principal Watermark para embed/extract/verify de proveniencia.

WHITE-BOX ONLY: la verificación requiere acceso al residual stream interno
del modelo. NO funciona como detector black-box de texto (la no-linealidad
del lm_head destruye la señal, medido experimentalmente).

Referencia: experimento watermark_provenance.py en este repo.
Punto de operación medido: B=16 bits, α=0.01·‖h‖, KL≈0, token_agreement≈1.0,
bitacc_residual≈0.92, control≈0.50.

FLUJO CORRECTO:
  1. embed: se usa durante la inferencia para inyectar el payload.
  2. verify/extract: re-corre los prompts CON el hook embed activo, también sin
     él para calibración, y decodifica desde la diferencia delta = h_wm - h_clean.
     Esto replica exactamente el protocolo de watermark_provenance.py.
"""
from __future__ import annotations

import contextlib
from typing import Any

import torch
from scipy import stats as scipy_stats

from .codec import make_codes, encode_delta, decode
from .ecc import ecc_encode, ecc_decode


class Watermark:
    """Watermark CDMA de proveniencia en el residual stream.

    Parámetros
    ----------
    key : str
        Clave secreta propietaria. Determina los códigos CDMA.
    n_bits : int
        Bits del payload (16 = punto de operación validado).
    layer : int
        Índice de capa donde se inyecta la perturbación (inject) y se lee (read).
        Para replicar el experimento exacto: inject_layer=8, read_layer=20 en Qwen2.5-0.5B.
    alpha : float
        Amplitud relativa α/‖h‖. 0.01 = punto de operación (KL≈0, token_agreement≈1.0).
    inject_layer : int | None
        Capa de inyección separada. Si None, usa `layer`.
    read_layer : int | None
        Capa de lectura separada. Si None, usa `layer`.
    """

    def __init__(
        self,
        key: str,
        n_bits: int = 16,
        layer: int = 12,
        alpha: float = 0.01,
        inject_layer: int | None = None,
        read_layer: int | None = None,
    ) -> None:
        self.key = key
        self.n_bits = n_bits
        self.alpha = alpha
        self.inject_layer = inject_layer if inject_layer is not None else layer
        self.read_layer = read_layer if read_layer is not None else layer
        # Los códigos se generan lazy (necesitamos d del modelo)
        self._codes: torch.Tensor | None = None

    def _get_layer(self, model: Any, idx: int) -> Any:
        """Accede a model.model.layers[idx] de forma robusta."""
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers[idx]
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h[idx]
        raise ValueError(
            f"No sé cómo acceder a la capa {idx} de {type(model).__name__}. "
            "Implementa _get_layer para tu arquitectura."
        )

    def _get_codes(self, d: int, device: torch.device) -> torch.Tensor:
        """Genera (o reutiliza) los códigos CDMA para esta clave."""
        if self._codes is None or self._codes.shape[0] != d:
            self._codes = make_codes(self.key, self.n_bits, d, device)
        return self._codes.to(device)

    def _compute_delta(
        self,
        model: Any,
        payload: list[int],
        prompt_ids: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Calcula el vector delta CDMA para un prompt dado.

        Hace un forward clean en inject_layer para obtener h_norm, luego
        codifica el payload con alpha_abs = alpha * h_norm.
        """
        h_cap: dict[str, torch.Tensor] = {}

        def norm_hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h_cap["h"] = h[:, -1, :].detach().float().clone()

        inject_layer = self._get_layer(model, self.inject_layer)
        handle = inject_layer.register_forward_hook(norm_hook)
        with torch.no_grad():
            model(input_ids=prompt_ids)
        handle.remove()

        h_val = h_cap["h"][0]   # (d,)
        h_norm = float(h_val.norm())
        d = h_val.shape[0]
        codes = self._get_codes(d, device)
        bits_pm1 = torch.tensor(
            [1.0 if b else -1.0 for b in payload], dtype=torch.float32, device=device
        )
        alpha_abs = self.alpha * h_norm
        return encode_delta(bits_pm1, codes, alpha_abs)

    @contextlib.contextmanager
    def embed(self, model: Any, payload: list[int] | None = None):
        """Context manager que inyecta el payload en el residual stream.

        Uso:
            bits = [1, 0, 1, 1, ...]  # lista de 0/1 de longitud n_bits
            with wm.embed(model, bits):
                outputs = model.generate(...)

        El hook se elimina automáticamente al salir del context manager.

        IMPLEMENTACIÓN: se precalcula delta en el primer forward (lazy) usando un
        pre-hook que corre UNA vez para medir ‖h‖ en inject_layer, luego el
        inject_hook aplica el delta en todos los forwards posteriores.
        """
        if payload is None:
            payload = [1] * self.n_bits

        bits_pm1_list = [1.0 if b else -1.0 for b in payload]

        # delta se calculará lazy en el primer forward
        state: dict[str, Any] = {"delta": None, "initialized": False}

        inject_layer_obj = self._get_layer(model, self.inject_layer)

        def pre_inject_hook(mod, inp):
            """Pre-hook: captura h ANTES de que la capa modifique, calcula delta una vez."""
            if state["initialized"]:
                return None
            # inp[0] es el tensor de entrada a la capa (batch, seq, d)
            x = inp[0] if isinstance(inp, tuple) else inp
            h_val = x[:, -1, :].detach().float()
            h_norm = float(h_val.norm(dim=-1).mean())
            if h_norm < 1e-6:
                return None
            d = h_val.shape[-1]
            device = h_val.device
            codes = self._get_codes(d, device)
            bits_pm1 = torch.tensor(bits_pm1_list, dtype=torch.float32, device=device)
            alpha_abs = self.alpha * h_norm
            state["delta"] = encode_delta(bits_pm1, codes, alpha_abs)
            state["initialized"] = True

        def inject_hook(mod, inp, out):
            """Post-hook: añade delta al residual output de inject_layer."""
            delta = state["delta"]
            if delta is None:
                return None
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone()
            h[:, -1, :] = h[:, -1, :] + delta.to(h.device, h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

        pre_handle = inject_layer_obj.register_forward_pre_hook(pre_inject_hook)
        post_handle = inject_layer_obj.register_forward_hook(inject_hook)

        try:
            yield state
        finally:
            pre_handle.remove()
            post_handle.remove()
            state["delta"] = None
            state["initialized"] = False

    def _run_and_collect(
        self,
        model: Any,
        prompts: list[str],
        tokenizer: Any,
        device: torch.device,
        with_embed: bool,
        payload: list[int] | None = None,
    ) -> torch.Tensor:
        """Corre prompts y recoge activaciones en read_layer, (total_tokens, d).

        Si with_embed=True, activa el hook de inyección durante el forward.
        """
        h_acc: list[torch.Tensor] = []

        read_layer_obj = self._get_layer(model, self.read_layer)

        def read_hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h_acc.append(h.detach().float().clone())

        read_handle = read_layer_obj.register_forward_hook(read_hook)

        try:
            if with_embed and payload is not None:
                with self.embed(model, payload):
                    for prompt in prompts:
                        toks = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                        with torch.no_grad():
                            model(input_ids=toks)
            else:
                for prompt in prompts:
                    toks = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                    with torch.no_grad():
                        model(input_ids=toks)
        finally:
            read_handle.remove()

        if not h_acc:
            raise RuntimeError("No se capturó ninguna activación.")
        return torch.cat([h.view(-1, h.shape[-1]) for h in h_acc], dim=0)

    def extract(
        self,
        model: Any,
        prompts: list[str],
        tokenizer: Any,
        payload: list[int] | None = None,
        calibration: torch.Tensor | None = None,
        device: torch.device | str | None = None,
    ) -> dict[str, Any]:
        """Extrae el payload watermark desde las activaciones del modelo.

        Protocolo (replicando watermark_provenance.py):
          1. Corre prompts CON embed activo → h_wm
          2. Corre prompts SIN embed → h_clean
          3. delta = h_wm - h_clean (promediado sobre tokens)
          4. Decodifica delta con los códigos CDMA de esta clave

        Si payload es None, se inyecta el payload [1]*n_bits para ver si la señal
        es detectable (útil para comprobar la cadena, no para verificar un payload).

        Retorna:
            {
              "bits": list[int],         # bits recuperados (0/1)
              "bits_pm1": Tensor,        # ±1
              "confidence": Tensor,      # por bit
              "raw_score": Tensor,       # proyección media del delta sobre códigos
            }
        """
        if device is None:
            device = next(model.parameters()).device
        if isinstance(device, str):
            device = torch.device(device)
        if payload is None:
            payload = [1] * self.n_bits

        h_wm = self._run_and_collect(model, prompts, tokenizer, device,
                                     with_embed=True, payload=payload)
        h_clean = self._run_and_collect(model, prompts, tokenizer, device,
                                        with_embed=False)

        # delta = diferencia; promediar sobre tokens mejora SNR
        delta = (h_wm - h_clean).mean(dim=0)   # (d,)
        d = delta.shape[0]
        codes = self._get_codes(d, delta.device)

        bits_hat, confidence = decode(delta.unsqueeze(0), codes, calibration=calibration)
        bits_01 = [(1 if b > 0 else 0) for b in bits_hat.tolist()]

        return {
            "bits": bits_01,
            "bits_pm1": bits_hat,
            "confidence": confidence,
            "raw_score": codes.float().T @ delta.float(),
        }

    def verify(
        self,
        model: Any,
        prompts: list[str],
        tokenizer: Any,
        expected_payload: list[int],
        calibration: torch.Tensor | None = None,
        device: torch.device | str | None = None,
        p_threshold: float = 1e-3,
    ) -> dict[str, Any]:
        """Verifica proveniencia comparando payload esperado vs recuperado.

        Inyecta el payload esperado, lee las activaciones watermarked y clean,
        decodifica el delta y compara con expected_payload.

        Usa la cola binomial unilateral P(≥k matches | chance 0.5) como p_value.
        Con bit_acc≈0.92 en 16 bits y múltiples prompts la señal es definitiva.

        Retorna:
            {
              "match_bits": int,           # bits que coinciden
              "n_bits": int,               # total de bits
              "bit_accuracy": float,
              "p_value": float,            # cola binomial unilateral
              "provenance_confirmed": bool,
              "recovered_bits": list[int],
            }
        """
        result = self.extract(
            model, prompts, tokenizer,
            payload=expected_payload,
            calibration=calibration,
            device=device,
        )
        recovered = result["bits"]
        expected_pm1 = torch.tensor(
            [1.0 if b else -1.0 for b in expected_payload], dtype=torch.float32
        )
        recovered_pm1 = result["bits_pm1"]

        n = self.n_bits
        matches = int((expected_pm1.to(recovered_pm1.device) == recovered_pm1).sum().item())
        bit_acc = matches / n

        # p_value: cola unilateral — P(≥matches | n, 0.5)
        # scipy.stats.binom.sf(k-1, n, 0.5) = P(X >= k)
        p_value = float(scipy_stats.binom.sf(matches - 1, n, 0.5))

        return {
            "match_bits": matches,
            "n_bits": n,
            "bit_accuracy": bit_acc,
            "p_value": p_value,
            "provenance_confirmed": p_value < p_threshold,
            "recovered_bits": recovered,
        }
