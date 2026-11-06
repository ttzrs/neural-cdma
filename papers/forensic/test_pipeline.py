"""Regression tests of Paper A reporting; synthetic fixtures stay in .trabajo/.
Run: python -m pytest papers/forensic/test_pipeline.py -q
"""
import copy
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_tables import THRESHOLDS, build, load_runs, mode_summary, tail, LEGACY_REVISIONS

ROOT = HERE.parent.parent


@pytest.fixture
def work():
    parent = ROOT / '.trabajo' / 'forensic-tests'
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as directory:
        yield Path(directory)


@pytest.fixture
def data():
    """Small synthetic schema fixture, never a published measurement."""
    def decoded(m, n=16):
        return {'n_phys': n, 'm_phys': m, 'p_phys': tail(m, n),
                **({'m_log': min(m, 16), 'p_log': tail(min(m, 16), 16)} if n == 64 else {})}

    sessions = []
    for i in range(4):
        session = {'session': i}
        for B in (16, 64):
            paired = decoded(15 if i == 3 else 16, B)
            blind = decoded(8 + i % 2, B)
            wrong = decoded(8, B)
            session[f'B{B}'] = {
                'paired': paired, 'blind': blind,
                'paired_T': {str(t): paired for t in (8, 16)},
                'blind_T': {str(t): blind for t in (8, 16)},
                'wrong': {k: {'paired': wrong, 'blind': wrong} for k in ('wrong-a', 'wrong-b')},
                'nomark_blind': blind, 'substitution_paired': decoded(B, B),
                'substitution_blind': blind,
            }
        sessions.append(session)

    def aggregate(records, logical=False):
        import statistics
        ps = [r['p_log' if logical else 'p_phys'] for r in records]
        ms = [r['m_log' if logical else 'm_phys'] for r in records]
        n = 16 if logical else records[0]['n_phys']
        return {'n': n, 'bitacc_mean': statistics.mean(ms) / n,
                'matches_median': statistics.median(ms), 'p_median': statistics.median(ps),
                'frac_p_below': {f'{t:g}': sum(p < t for p in ps) / len(ps) for t in THRESHOLDS}}

    summary = {}
    for B in (16, 64):
        rows = [s[f'B{B}'] for s in sessions]
        s = {'fidelity': {'frac_identical': .75, 'first_div_median': 16,
                           'kl_mean_pre_div': 2e-5, 'agree_frac_mean': .95}}
        for mode in ('paired', 'blind', 'nomark_blind', 'substitution_paired', 'substitution_blind'):
            s[mode] = aggregate([r[mode] for r in rows])
        for mode in ('paired', 'blind'):
            s[mode + '_T'] = {str(t): aggregate([r[mode + '_T'][str(t)] for r in rows]) for t in (8, 16)}
        s['wrong'] = {k: {m: aggregate([r['wrong'][k][m] for r in rows]) for m in ('paired', 'blind')}
                      for k in ('wrong-a', 'wrong-b')}
        if B == 64:
            for mode in ('paired', 'blind', 'substitution_paired', 'substitution_blind', 'nomark_blind'):
                s[mode + '_logical'] = aggregate([r[mode] for r in rows], logical=True)
        summary[f'B{B}'] = s
    return {'config': {'model': 'Qwen/Qwen2.5-0.5B', 'dtype': 'fp32', 'd': 896, 'n_layers': 24,
                       'inject_layer': 8, 'read_layer': 20, 'alpha': .01, 'n_sessions': 4,
                       'n_calib_prompts': 4, 'gen_len': 16, 'B_logical': 16, 'wrong_keys': ['wrong-a', 'wrong-b'],
                       'pval_examples': {f'{m}/16': tail(m, 16) for m in range(12, 17)}},
            'sessions': sessions, 'summary': summary}


def test_binomial_floor_and_strict_thresholds():
    assert tail(16, 16) == 2**-16
    assert math.isclose(tail(14, 16), 0.0020904541015625)
    assert tail(16, 16) > 1e-5
    assert tail(12, 16) == 0.0384063720703125


def test_logical_control_uses_logical_not_physical(data):
    r = data['sessions'][0]['B64']['substitution_paired']
    assert r['n_phys'] == 64 and r['p_phys'] < 1e-5
    s = mode_summary(data, 'B64', 'substitution_paired', logical=True)
    assert s['n'] == 16
    assert s['p_median'] >= 2**-16
    assert s['frac_p_below']['1e-05'] == 0
    assert mode_summary(data, 'B64', 'wrong', logical=True,
                        key=data['config']['wrong_keys'][0], T='paired')['n'] == 16
    assert mode_summary(data, 'B64', 'paired_T', logical=True, T=8)['n'] == 16


def test_physical_detection_is_separate_from_payload(data, work):
    results, out = work / 'results', work / 'tables'
    results.mkdir()
    (results / 'watermark_scale_fixture.json').write_text(json.dumps(data))
    build(load_runs(results), out)
    s = mode_summary(data, 'B64', 'substitution_paired')
    assert s['n'] == 64 and s['p_median'] == 2**-64
    assert s['frac_p_below']['1e-05'] == 1
    logical = mode_summary(data, 'B64', 'substitution_paired', logical=True)
    assert logical['n'] == 16 and logical['p_median'] == 2**-16
    assert logical['frac_p_below']['1e-05'] == 0
    verification = (out / 'verification.tex').read_text()
    controls = (out / 'controls.tex').read_text()
    curve = (out / 'p_vs_T.tex').read_text()
    assert '$64$ phys.' in verification and '$16$ log.' in verification
    assert 'R0B64Physsubstitution_pairedp' in controls
    assert 'R0B64PhyspairedTp8' in curve
    assert 'R0B64pairedTp8' in curve


def test_zero_one_then_four_models(data, work):
    results, out = work / 'results', work / 'tables'
    results.mkdir()
    build(load_runs(results), out)
    assert 'No completed scale JSON' in (out / 'observations.tex').read_text()
    assert json.loads((out / 'source_manifest.json').read_text())['sources'] == []
    p = results / 'watermark_scale_one.json'
    p.write_text(json.dumps(data))
    runs = load_runs(results)
    assert len(runs) == 1
    build(runs, out)
    assert 'No completed' not in (out / 'observations.tex').read_text()
    assert '0.984' in (out / 'macros.tex').read_text()
    for i, (model, dtype) in enumerate((('Qwen/Qwen2.5-0.5B', 'bf16'),
                                       ('Qwen/Qwen2.5-1.5B', 'bf16'),
                                       ('Qwen/Qwen2.5-7B', 'bf16'))):
        synthetic = copy.deepcopy(data)
        synthetic['config'].update(model=model, dtype=dtype)
        (results / f'watermark_scale_extra{i}.json').write_text(json.dumps(synthetic))
    runs = load_runs(results)
    build(runs, out)
    assert len(runs) == 4
    fidelity_text = (out / 'fidelity.tex').read_text()
    verification_text = (out / 'verification.tex').read_text()
    assert fidelity_text.index(r'\textbf{fp32}') < fidelity_text.index(r'\textbf{bf16}')
    assert verification_text.index(r'\textbf{fp32}') < verification_text.index(r'\textbf{bf16}')
    assert 'PrecisionAbstract' in (out / 'macros.tex').read_text()
    assert (out / 'controls_part0.tex').exists()
    assert (out / 'p_vs_T.tex').exists()
    assert 'MissingRuns\\endcsname{none}' in (out / 'macros.tex').read_text()
    manifest = json.loads((out / 'source_manifest.json').read_text())
    assert len(manifest['sources']) == 4
    assert all(hashlib.sha256(Path(s['path']).read_bytes()).hexdigest() == s['sha256']
               for s in manifest['sources'])


def test_incomplete_and_duplicate(data, work):
    (work / 'watermark_scale_partial.json').write_text('{"config":')
    assert load_runs(work) == []
    for name in ('one', 'duplicate'):
        (work / f'watermark_scale_{name}.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Duplicate'):
        load_runs(work)


def test_config_examples_must_match(data, work):
    data['_source'] = 'schema-fixture'
    data['_sha256'] = 'fixture'
    data['config']['pval_examples']['14/16'] = .5
    with pytest.raises(ValueError, match='binomial'):
        build([data], work)


def test_reject_pooled_or_contradictory_summary(data, work):
    data['summary']['B16']['paired']['p_median'] = 0.5
    (work / 'watermark_scale_bad.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Contradictory session summary'):
        load_runs(work)


def test_single_model_figures_and_manifest_snapshot(data, work):
    results = work / 'results'
    results.mkdir()
    source = results / 'watermark_scale_fixture.json'
    source.write_text(json.dumps(data))
    out = work / 'tables'
    build(load_runs(results), out)
    command = [sys.executable, str(HERE / 'make_figures.py'), '--tables', str(out),
               '--output', str(work / 'figures')]
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert len(list((work / 'figures').glob('*.pdf'))) == 3
    assert all(p.read_bytes().startswith(b'%PDF-') for p in (work / 'figures').glob('*.pdf'))
    source.write_text(json.dumps(data) + '\n')
    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode != 0
    assert 'Result changed after tables' in failed.stderr


def nomark_fixture(data):
    fixture = copy.deepcopy(data)
    fixture['config']['model_revision'] = LEGACY_REVISIONS[fixture['config']['model']]
    for block in ('B16', 'B64'):
        summary = fixture['summary'][block]
        new = {'nomark_paired': summary['paired'], 'nomark_paired_T': summary['paired_T'],
               'nomark_delta_norm_mean': 0.0001}
        if block == 'B64':
            new['nomark_paired_logical'] = summary['paired_logical']
        fixture['summary'][block] = new
        for session in fixture['sessions']:
            r = session[block]
            session[block] = {'nomark_paired': r['paired'], 'nomark_paired_T': r['paired_T'],
                              'nomark_delta_norm_mean': .0001}
    return fixture


def test_conditional_paired_nomark_and_full_payload(data, work):
    results, out = work / 'results', work / 'tables'
    results.mkdir()
    (results / 'watermark_scale_fixture.json').write_text(json.dumps(data))
    assert '_nomark' not in load_runs(results)[0]
    null = nomark_fixture(data)
    source = results / 'nomark_paired_fixture.json'
    source.write_text(json.dumps(null))
    runs = load_runs(results)
    assert '_nomark' in runs[0]
    build(runs, out)
    assert 'no-mark, paired' in (out / 'controls.tex').read_text()
    assert 'Full payload' in (out / 'verification.tex').read_text()
    assert 'R0B16pairedfull\\endcsname{0.750}' in (out / 'macros.tex').read_text()
    manifest = json.loads((out / 'source_manifest.json').read_text())
    assert manifest['nomark_sources'][0]['path'] == str(source)
    subprocess.run([sys.executable, str(HERE / 'make_figures.py'), '--tables', str(out),
                    '--output', str(work / 'figures')], check=True, capture_output=True)
    source.write_text(json.dumps(null) + '\n')
    result = subprocess.run([sys.executable, str(HERE / 'make_figures.py'), '--tables', str(out),
                             '--output', str(work / 'figures')], capture_output=True, text=True)
    assert result.returncode and 'No-mark result changed' in result.stderr
    null['config']['model_revision'] = 'invalid-revision'
    source.write_text(json.dumps(null))
    with pytest.raises(ValueError, match='model revision mismatch'):
        load_runs(results)
