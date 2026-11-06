"""Protocolos por sesión con modelo mínimo; no requiere GPU ni descargas."""
from types import SimpleNamespace

import pytest
import torch

from cdma_watermark.cli import build_parser
from cdma_watermark import cli
from cdma_watermark.codec import make_codes, encode_delta, session_pvalue
from cdma_watermark.watermark import Watermark


BITS = [1, 0] * 8


class ModeloMinimo(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Identity()])
        codes = make_codes("modo-test", 16, 64)
        self.register_buffer("bias", codes @ torch.tensor([-20.0, 20.0] * 8))
        self.register_buffer("slope", codes @ torch.arange(1.0, 17.0))
        self.calls = []
        self.fail = False

    def forward(self, input_ids):
        self.calls.append(input_ids.clone())
        if self.fail:
            raise RuntimeError("fallo simulado")
        h = self.bias + input_ids.float().unsqueeze(-1) * self.slope
        return self.model.layers[0](h)


def tokenizer(prompt, return_tensors):
    return SimpleNamespace(input_ids=torch.tensor([[int(x) for x in prompt.split()]]))


@pytest.fixture
def setup():
    model = ModeloMinimo()
    wm = Watermark("modo-test", n_bits=16, layer=0)
    return model, wm


def test_calibrate_weighted_by_tokens_and_clean(setup):
    model, wm = setup
    calibration = wm.calibrate(model, ["1", "2 3 4"], tokenizer)
    clean = model.bias + torch.arange(1.0, 5.0).unsqueeze(-1) * model.slope
    expected = (clean @ wm._get_codes(64, torch.device("cpu"))).mean(0)
    assert calibration.shape == (16,)
    assert not calibration.requires_grad
    assert torch.allclose(calibration, expected, atol=1e-5)
    assert len(model.calls) == 2
    assert not model.model.layers[0]._forward_hooks
    assert not model.model.layers[0]._forward_pre_hooks


def test_blind_log_without_any_model_pass(setup):
    model, wm = setup
    calibration = wm.calibrate(model, ["1 2", "3"], tokenizer)
    codes = wm._get_codes(64, torch.device("cpu"))
    bits_pm1 = torch.tensor([1.0, -1.0] * 8)
    delta = encode_delta(bits_pm1, codes, 1.0)
    logged = (model.bias + 2 * model.slope + delta).repeat(5, 1)
    model.calls.clear()
    result = wm.verify(None, [], None, BITS, mode="blind",
                       calibration=calibration, logged_activations=logged)
    assert result["recovered_bits"] == BITS
    assert result["match_bits"] == 16
    assert result["p_value"] == session_pvalue(16, 16)
    assert not model.calls
    extracted = wm.extract(None, [], None, mode="blind", calibration=calibration,
                           logged_activations=logged)
    assert torch.allclose(extracted["raw_score"], delta @ codes, atol=1e-5)
    # La ruta de demo también funciona en blind, pero sólo hace el pase marcado.
    wm.extract(model, ["1 2 3"], tokenizer, mode="blind", calibration=calibration)
    assert len(model.calls) == 1


@pytest.mark.parametrize("generation_log", [False, True])
def test_paired_log_only_clean_pass_and_predictor_alignment(setup, generation_log):
    model, wm = setup
    codes = wm._get_codes(64, torch.device("cpu"))
    delta = encode_delta(torch.tensor([1.0, -1.0] * 8), codes, 1.0)
    ids = torch.arange(1.0, 6.0) if generation_log else torch.arange(1.0, 7.0)
    clean = model.bias + ids.unsqueeze(-1) * model.slope
    if generation_log:
        clean = clean[-3:]
    logged = clean + delta
    result = wm.verify(model, ["1 2 3 4 5 6"], tokenizer, BITS,
                       logged_activations=logged)
    assert result["recovered_bits"] == BITS
    assert result["p_value"] == 2.0 ** -16
    assert len(model.calls) == 1
    assert not model.model.layers[0]._forward_pre_hooks
    assert not model.model.layers[0]._forward_hooks


def test_paired_demo_reads_after_same_layer_injection(setup):
    model, wm = setup
    result = wm.verify(model, ["1 2 3"], tokenizer, BITS)
    assert result["recovered_bits"] == BITS
    assert len(model.calls) == 2
    assert not model.model.layers[0]._forward_hooks
    assert not model.model.layers[0]._forward_pre_hooks


@pytest.mark.parametrize("kwargs", [
    {"mode": "otro"},
    {"mode": "blind"},
    {"mode": "blind", "calibration": torch.zeros(15)},
    {"mode": "blind", "calibration": torch.full((16,), float("nan"))},
    {"logged_activations": torch.zeros(0, 64)},
    {"logged_activations": torch.zeros(64)},
    {"logged_activations": torch.full((3, 64), float("nan"))},
])
def test_invalid_inputs_fail_before_model_pass(setup, kwargs):
    model, wm = setup
    with pytest.raises(ValueError):
        wm.extract(model, ["1 2 3"], tokenizer, **kwargs)
    assert not model.calls


@pytest.mark.parametrize("prompts", [[], ["1", "2"]])
def test_reject_multiple_sessions_or_missing_text(setup, prompts):
    model, wm = setup
    with pytest.raises(ValueError, match="sesión"):
        wm.verify(model, prompts, tokenizer, BITS)
    assert not model.calls


def test_reject_misaligned_log_and_empty_calibration_corpus(setup):
    model, wm = setup
    for logged in (torch.zeros(4, 64), torch.zeros(2, 65)):
        with pytest.raises(ValueError, match="posiciones"):
            wm.extract(model, ["1 2 3"], tokenizer, logged_activations=logged)
    with pytest.raises(ValueError, match="corpus limpio"):
        wm.calibrate(model, [], tokenizer)


def test_hooks_removed_after_failure(setup):
    model, wm = setup
    model.fail = True
    with pytest.raises(RuntimeError, match="simulado"):
        wm.extract(model, ["1 2 3"], tokenizer)
    assert not model.model.layers[0]._forward_hooks
    assert not model.model.layers[0]._forward_pre_hooks


@pytest.mark.parametrize("command", ["verify", "extract", "audit"])
@pytest.mark.parametrize("mode", ["paired", "blind"])
def test_cli_sessions_and_calibration(setup, monkeypatch, capsys, command, mode):
    model, wm = setup
    monkeypatch.setattr(cli, "_load_model_tok", lambda _: (model, tokenizer, "cpu"))
    model.config = SimpleNamespace(hidden_size=64)
    argv = ["--layer", "0", "--key", "modo-test", command]
    if command != "extract":
        argv += ["--payload", "1,0," * 7 + "1,0"]
    if command != "audit":
        argv += ["--mode", mode]
        if mode == "blind":
            argv += ["--calibration-prompts", "2 3", "4"]
    argv += ["--prompts", "1", "5 6"]
    args = build_parser().parse_args(argv)
    args.func(args)
    output = capsys.readouterr().out
    if command == "verify":
        assert output.count('"p_value"') == 2
    elif command == "extract":
        assert output.count("Recovered bits:") == 2
    else:
        assert output.count("Proveniencia desacoplada (sesión") == 2
    assert len(model.calls) == 4


@pytest.mark.parametrize("calib_prompts", [None, ["1"]])
def test_cli_blind_requires_disjoint_calibration(setup, calib_prompts):
    model, wm = setup
    args = SimpleNamespace(mode="blind", prompts=["1"], calibration_prompts=calib_prompts)
    with pytest.raises(SystemExit, match="calibración|calibration-prompts"):
        cli._calibration(wm, model, tokenizer, "cpu", args)
    assert not model.calls
