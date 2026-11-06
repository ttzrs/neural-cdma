#!/usr/bin/env python
"""Generate Paper A tables/macros from completed scale JSONs, never from the smoke run.

Defaults are repository-relative, independent of cwd. --results/--output allow isolated
schema tests. Missing model runs are omitted; incomplete JSON writes are deferred.
Every numeric table cell has a macro. B64 logical controls/curves absent from summary
are reconstructed from the recorded per-session m_log/p_log, not physical statistics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
THRESHOLDS = (1e-2, 1e-3, 1e-4, 1e-5)
EXPECTED = (("Qwen/Qwen2.5-0.5B", "fp32"), ("Qwen/Qwen2.5-0.5B", "bf16"),
            ("Qwen/Qwen2.5-1.5B", "bf16"), ("Qwen/Qwen2.5-7B", "bf16"))


def tail(m, n):
    return sum(math.comb(n, k) for k in range(m, n + 1)) / 2**n


def escape(text):
    return "".join({"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#",
                    "{": r"\{", "}": r"\}"}.get(c, c) for c in str(text))


def number(x, kind="decimal"):
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"Nonfinite table value: {x}")
    if kind == "sci":
        if x == 0:
            return "0"
        a, b = f"{x:.2e}".split("e")
        return rf"{a}\times 10^{{{int(b)}}}"
    if kind == "count":
        return f"{x:g}"
    return f"{x:.3f}"


# Legacy runs omitted config.model_revision; these revisions were confirmed from
# the supplied 0.5B/7B provenance and the local 1.5B snapshot directory.
LEGACY_REVISIONS = {
    "Qwen/Qwen2.5-0.5B": "060db6499f32faf8b98477b0a26969ef7d8b9987",
    "Qwen/Qwen2.5-1.5B": "8faed761d45a263340a0528343f099c05c9a4323",
    "Qwen/Qwen2.5-7B": "d149729398750b98c0af14eb82c78cfe92750796",
}


def attach_nomark(runs, paths):
    """Attach compatible completed no-mark-only runs; preserve their source hashes."""
    by_id = {(r["config"]["model"], r["config"]["dtype"]): r for r in runs}
    seen = set()
    for path in paths:
        raw = Path(path).read_bytes()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            print(f"Deferred incomplete JSON: {path}", file=sys.stderr)
            continue
        c = data["config"]
        ident = (c["model"], c["dtype"])
        if ident in seen:
            raise ValueError(f"Duplicate no-mark model/dtype: {ident}")
        seen.add(ident)
        if ident not in by_id:
            continue
        base = by_id[ident]
        bc = base["config"]
        for key in ("d", "n_layers", "inject_layer", "read_layer", "alpha", "n_sessions",
                    "prompt_len", "gen_len", "B_logical", "owner_key", "seed", "dataset"):
            if key in c and key in bc and c[key] != bc[key]:
                raise ValueError(f"No-mark configuration mismatch: {ident}/{key}")
        base_rev = bc.get("model_revision") or LEGACY_REVISIONS.get(c["model"])
        if c.get("model_revision") and base_rev and c["model_revision"] != base_rev:
            raise ValueError(f"No-mark model revision mismatch: {ident}")
        if c["n_sessions"] != len(data["sessions"]) or not data["sessions"]:
            raise ValueError(f"Invalid no-mark session count: {path}")
        for block in ("B16", "B64"):
            if "nomark_paired" not in data["summary"][block]:
                raise ValueError(f"Missing no-mark paired summary: {path}/{block}")
        validate_session_statistics(data)
        data["_source"], data["_sha256"] = str(Path(path).resolve()), hashlib.sha256(raw).hexdigest()
        base["_nomark"] = data
    return runs


def nomark_run(run):
    if "nomark_paired" in run["summary"]["B16"]:
        return run
    return run.get("_nomark")


def load_runs(results):
    runs = []
    seen = set()
    for path in sorted(Path(results).glob("watermark_scale_*.json")):
        raw = path.read_bytes()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            print(f"Deferred incomplete JSON: {path}", file=sys.stderr)
            continue
        cfg = data["config"]
        ident = (cfg["model"], cfg["dtype"])
        if ident in seen:
            raise ValueError(f"Duplicate model/dtype {ident}: {path}")
        seen.add(ident)
        if cfg["n_sessions"] != len(data["sessions"]) or not data["sessions"]:
            raise ValueError(f"Invalid session count: {path}")
        for block in ("B16", "B64"):
            if block not in data["summary"]:
                raise ValueError(f"Missing {block}: {path}")
        validate_session_statistics(data)
        data["_source"] = str(path.resolve())
        data["_sha256"] = hashlib.sha256(raw).hexdigest()
        runs.append(data)
    attach_nomark(runs, sorted(Path(results).glob("nomark_paired_*.json")))
    return sorted(runs, key=lambda r: EXPECTED.index((r["config"]["model"], r["config"]["dtype"]))
                  if (r["config"]["model"], r["config"]["dtype"]) in EXPECTED else len(EXPECTED))


def validate_session_statistics(run):
    """Fail on a contradictory summary instead of typesetting a pooled or stale tail."""
    for block in ("B16", "B64"):
        for mode in ("paired", "blind", "nomark_blind", "nomark_paired", "substitution_paired", "substitution_blind"):
            if mode not in run["summary"][block]:
                continue
            for logical in (False, True) if block == "B64" else (False,):
                rows = [s[block][mode] for s in run["sessions"]]
                mk, pk = ("m_log", "p_log") if logical else ("m_phys", "p_phys")
                n = run["config"]["B_logical"] if logical else rows[0]["n_phys"]
                for row in rows:
                    if not math.isclose(row[pk], tail(row[mk], n), rel_tol=1e-10, abs_tol=1e-20):
                        raise ValueError(f"Invalid session tail: {block}/{mode}/{pk}")
                summary = mode_summary(run, block, mode, logical)
                expected = {"bitacc_mean": statistics.mean(r[mk] for r in rows) / n,
                            "matches_median": statistics.median(r[mk] for r in rows),
                            "p_median": statistics.median(r[pk] for r in rows)}
                for key, value in expected.items():
                    if not math.isclose(summary[key], value, rel_tol=1e-8, abs_tol=1e-12):
                        raise ValueError(f"Contradictory session summary: {block}/{mode}/{key}")
                fractions = {float(k): v for k, v in summary["frac_p_below"].items()}
                for threshold in THRESHOLDS:
                    observed = sum(r[pk] < threshold for r in rows) / len(rows)
                    if not math.isclose(fractions[threshold], observed, abs_tol=1e-12):
                        raise ValueError(f"Contradictory threshold fraction: {block}/{mode}")


def mode_summary(run, block, mode, logical=False, key=None, T=None):
    """Prefer supplied aggregates; reconstruct only schema's missing logical modes."""
    s = run["summary"][block]
    if not logical:
        if mode == "wrong":
            return s[mode][key][T]
        return s[mode][str(T)] if T is not None else s[mode]
    logical_mode = mode + "_logical"
    if mode != "wrong" and logical_mode in s:
        return s[logical_mode][str(T)] if T is not None else s[logical_mode]
    records = [v[block][mode] for v in run["sessions"]]
    if mode == "wrong":
        records = [v[key][T] for v in records]
    elif T is not None:
        records = [v[str(T)] for v in records]
    n = run["config"]["B_logical"]
    ms, ps = [v["m_log"] for v in records], [v["p_log"] for v in records]
    return {"n": n, "bitacc_mean": statistics.mean(ms) / n,
            "matches_median": statistics.median(ms), "p_median": statistics.median(ps),
            "frac_p_below": {f"{t:g}": sum(p < t for p in ps) / len(ps) for t in THRESHOLDS}}


class Writer:
    def __init__(self, out):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.macros = {}

    def macro(self, name, value, kind=None):
        rendered = number(value, kind) if kind else str(value)
        if name in self.macros and self.macros[name] != rendered:
            raise ValueError(f"Conflicting macro: {name}")
        self.macros[name] = rendered
        return rf"\csname {name}\endcsname"

    def cell(self, name, value, kind="decimal"):
        return "$" + self.macro(name, value, kind) + "$"

    def table(self, filename, header, rows, caption, label, spec=None, wide=False):
        # Chunk before scaling: a long monolithic table can overflow page height.
        chunk_size = 32 if filename.startswith("controls") else 24
        if len(rows) > chunk_size:
            chunks = []
            for start in range(0, len(rows), chunk_size):
                part = filename.removesuffix(".tex") + f"_part{start // chunk_size}.tex"
                self.table(part, header, rows[start:start + chunk_size],
                           caption + (" Continued." if start else ""),
                           label + (f"-part{start // chunk_size}" if start else ""), spec, wide)
                chunks.append(r"\input{tables/" + part + "}")
            (self.out / filename).write_text("\n".join(chunks) + "\n")
            return
        width = len(header)
        spec = spec or "l" + "r" * (width - 1)
        if not rows:
            rows = [[rf"\multicolumn{{{width}}}{{c}}{{No completed scale run available.}}"]]
        body_rows, previous_dtype = [], None
        for row in rows:
            if filename.startswith(("fidelity", "verification")) and len(row) == width:
                dtype = row[0].rsplit(" ", 1)[-1]
                if dtype != previous_dtype:
                    if previous_dtype is not None:
                        body_rows.append(r"\midrule")
                    body_rows.append(rf"\multicolumn{{{width}}}{{l}}{{\textbf{{{escape(dtype)}}}}} \\")
                    previous_dtype = dtype
            body_rows.append(" & ".join(row) + r" \\")
        body = "\n".join(body_rows)
        content = ("% Generated by make_tables.py; do not edit.\n"
                   + r"\begin{table}[htbp]" + "\n\\centering\n\\small\n"
                   + (r"\resizebox{\textwidth}{!}{%" + "\n" if wide else "")
                   + rf"\begin{{tabular}}{{{spec}}}" + "\n\\toprule\n"
                   + " & ".join(header) + " \\\\\n\\midrule\n" + body
                   + "\n\\bottomrule\n\\end{tabular}\n"
                   + ("}\n" if wide else "")
                   + rf"\caption{{{caption}}}\label{{{label}}}" + "\n\\end{table}\n")
        (self.out / filename).write_text(content)


def build(runs, output, v1_path=ROOT / "watermark_provenance_result.json"):
    w = Writer(output)
    # Remove only generated chunk files from older snapshots; otherwise absent runs
    # could leave stale, apparently current tables in the output directory.
    for stale in w.out.glob("*_part[0-9]*.tex"):
        stale.unlink()
    w.macro("RunCount", len(runs), "count")
    w.macro("RunList", ", ".join(escape(r["config"]["model"].removeprefix("Qwen/"))
                                + " (" + escape(r["config"]["dtype"]) + ")" for r in runs) or "none")
    missing = [f"{m.removeprefix('Qwen/')} ({dt})" for m, dt in EXPECTED
               if (m, dt) not in {(r['config']['model'], r['config']['dtype']) for r in runs}]
    w.macro("MissingRuns", escape(", ".join(missing)) if missing else "none")
    w.macro("RunStatus", "Completed runs only; unavailable models are not extrapolated." if runs
            else "No completed scale runs: all scale results remain pending.")
    abstract, observations, null_observations = [], [], []
    w.macro("ProtocolFinding", "Blind accuracy remains near chance across the measured configurations; the audit verifier therefore uses paired replay."
            if runs and all(abs(r["summary"]["B16"]["blind"]["bitacc_mean"] - .5) <= .05 for r in runs)
            else "Blind verification requires condition-specific validation; paired replay remains the specified audit mode.")

    v1 = json.loads(Path(v1_path).read_text())["16"]
    for a, metrics in v1.items():
        tag = a.replace(".", "x")
        for name, value in metrics.items():
            w.macro(f"Vone{tag}{name}", value, "sci" if name == "kl" else "decimal")
    for name, key in (("VoneResidual", "bitacc_residual"), ("VoneLogits", "bitacc_logits"),
                      ("VoneKL", "kl"), ("VoneAgreement", "token_agreement")):
        w.macro(name, v1["0.01"][key], "sci" if key == "kl" else "decimal")
    logits = [v["bitacc_logits"] for v in v1.values()]
    w.macro("VoneLogitsMin", min(logits), "decimal")
    w.macro("VoneLogitsMax", max(logits), "decimal")
    examples = runs[0]["config"]["pval_examples"] if runs else {f"{m}/16": tail(m, 16) for m in range(12, 17)}
    p_rows = []
    for m in range(12, 17):
        p = examples[f"{m}/16"]
        if not math.isclose(p, tail(m, 16), rel_tol=1e-12):
            raise ValueError("pval_examples differs from exact binomial tail")
        p_rows.append([str(m) + "/16", w.cell(f"Pvalue{m}", p, "sci")])
    w.table("v1_pvalues.tex", ["Matches", "Nominal one-session tail"], p_rows,
            "Exact one-sided binomial examples (not empirical scale results). For $B=16$, the minimum tail exceeds $10^{-5}$; $B=64$ physical detection is not subject to this floor.", "tab:pvalues")
    fidelity, verify, controls, curves, configs, hashes, revisions = [], [], [], [], [], [], []
    paired_observations, paired_models = [], []
    for idx, run in enumerate(runs):
        c = run["config"]
        pre = f"R{idx}"
        model = escape(c["model"].removeprefix("Qwen/Qwen2.5-")) + " " + escape(c["dtype"])
        for key, value in c.items():
            if isinstance(value, (float, int)):
                w.macro(pre + key, value, "count")
        configs.append([model] + [w.cell(pre + k, c[k], "count") for k in
                                  ("d", "n_layers", "inject_layer", "read_layer", "n_sessions", "n_calib_prompts", "gen_len")])
        revision = c.get("model_revision") or LEGACY_REVISIONS.get(c["model"])
        revisions.append([model, r"\path{" + revision + "}" if revision else "not recorded",
                          "JSON" if c.get("model_revision") else "legacy snapshot"])
        nr = nomark_run(run)
        if nr:
            paired_models.append(model)
            ns = mode_summary(nr, "B64", "nomark_paired")
            detail = []
            for block in ("B16", "B64"):
                records = [r[block]["nomark_paired"] for r in nr["sessions"]]
                count = sum(r["p_phys"] < 1e-3 for r in records)
                count_loose = sum(r["p_phys"] < 1e-2 for r in records)
                detail.append(block + " physical: " + w.cell(pre + block + "NoMarkCount", count, "count")
                              + "/" + w.cell(pre + block + "NoMarkN", len(records), "count") + " below $10^{-3}$ ("
                              + w.cell(pre + block + "NoMarkLooseCount", count_loose, "count") + " below $10^{-2}$)")
            paired_observations.append(model + ": " + "; ".join(detail) + ".")
        hashes.append([model, r"\texttt{" + run["_sha256"][:16] + "}"])
        for block, logical, label in (("B16", False, "$16$ phys."), ("B64", False, "$64$ phys."), ("B64", True, "$16$ log. ($64$)")):
            f = run["summary"][block]["fidelity"]
            if not logical:
                fidelity.append([model, label] + [w.cell(pre + block + k, f[k], kind) for k, kind in
                             (("frac_identical", "decimal"), ("first_div_median", "count"),
                              ("kl_mean_pre_div", "sci"), ("agree_frac_mean", "decimal"))])
            for mode in ("paired", "blind"):
                s = mode_summary(run, block, mode, logical)
                name = pre + block + ("Phys" if block == "B64" and not logical else "") + mode
                records = [record[block][mode] for record in run["sessions"]]
                full = sum(r["m_log" if block == "B64" else "m_phys"] == 16 for r in records) / len(records)
                verify.append([model, label, mode] + stat_cells(w, name, s) + [w.cell(name + "full", full)])
                curve = [model, label, mode]
                for T in (8, 16, 32, 64, 128):
                    if str(T) in run["summary"][block][mode + "_T"]:
                        ts = mode_summary(run, block, mode + "_T", logical, T=T)
                        curve.append(w.cell(name + "Tp" + str(T), ts["p_median"], "sci"))
                    else:
                        curve.append("--")
                curves.append(curve)
            for key_idx, key in enumerate(c["wrong_keys"], 1):
                for mode in ("paired", "blind"):
                    s = mode_summary(run, block, "wrong", logical, key=key, T=mode)
                    controls.append([model, label, f"wrong {key_idx}, {mode}"]
                                    + stat_cells(w, pre + block + ("Phys" if block == "B64" and not logical else "") + f"wrong{key_idx}" + mode, s))
            for mode in ("nomark_blind", "substitution_paired", "substitution_blind"):
                s = mode_summary(run, block, mode, logical)
                controls.append([model, label, escape(mode).replace(r"\_", " ")]
                                + stat_cells(w, pre + block + ("Phys" if block == "B64" and not logical else "") + mode, s))
            if nr:
                s = mode_summary(nr, block, "nomark_paired", logical)
                controls.append([model, label, "no-mark, paired"] + stat_cells(
                    w, pre + block + ("Phys" if block == "B64" and not logical else "") + "nomark_paired", s))
        def val(tag, value, kind="decimal"):
            return "$" + w.macro(pre + tag, value, kind) + "$"
        f = run["summary"]["B16"]["fidelity"]
        p16 = run["summary"]["B16"]["paired"]
        p64 = run["summary"]["B64"]["paired"]
        l64 = mode_summary(run, "B64", "paired", logical=True)
        abstract.append(model + ": " + val("AbsIdentical", f["frac_identical"]) + " identical, "
                        + val("AbsAccuracy", p16["bitacc_mean"]) + " paired accuracy")
        observations.append(model + ": identical continuations " + val("B16frac_identical", f["frac_identical"])
                            + "; median first divergence " + val("B16first_div_median", f["first_div_median"], "count")
                            + "; conditional KL " + val("B16kl_mean_pre_div", f["kl_mean_pre_div"], "sci")
                            + ". Paired accuracy is " + val("B16pairedacc", p16["bitacc_mean"])
                            + ". At sixty-four physical bits, median detection $p$ is "
                            + val("B64Physpairedp", p64["p_median"], "sci") + ", with "
                            + val("B64Physpairedfrac3", p64["frac_p_below"]["1e-05"])
                            + " of sessions below $10^{-5}$. Logical payload accuracy after repetition is "
                            + val("B64pairedacc", l64["bitacc_mean"]) + ".")
        rates = [x["frac_p_below"] for v in run["summary"]["B64"]["wrong"].values() for x in v.values()]
        rates.append(run["summary"]["B64"]["nomark_blind"]["frac_p_below"])
        null_observations.append(model + ": the five physical null-control series have rates from "
                                 + val("NullMin", min(x["0.01"] for x in rates)) + " to "
                                 + val("NullMax", max(x["0.01"] for x in rates)) + " at $10^{-2}$, and from "
                                 + val("NullMinTight", min(x["0.001"] for x in rates)) + " to "
                                 + val("NullMaxTight", max(x["0.001"] for x in rates)) + " at $10^{-3}$. "
                                 + ("All have zero exceedances at $10^{-4}$ and $10^{-5}$." if all(x[t] == 0 for x in rates for t in ("0.0001", "1e-05"))
                                    else "The smaller-threshold rates are listed in the controls table."))
    by_condition = {(r["config"]["model"], r["config"]["dtype"]): (i, r) for i, r in enumerate(runs)}
    low = by_condition.get(("Qwen/Qwen2.5-0.5B", "fp32"))
    reduced = by_condition.get(("Qwen/Qwen2.5-0.5B", "bf16"))
    if low and reduced:
        def pair_metric(condition, key, field):
            i, r = condition
            return w.cell(f"PrecisionR{i}{key}{field}", r["summary"]["B16"][key][field])
        w.macro("PrecisionAbstract", "At fixed 0.5B, fp32 to bf16 lowers identical continuations from "
                + pair_metric(low, "fidelity", "frac_identical") + " to " + pair_metric(reduced, "fidelity", "frac_identical")
                + " and paired accuracy from " + pair_metric(low, "paired", "bitacc_mean") + " to "
                + pair_metric(reduced, "paired", "bitacc_mean") + ". Precision dominates this measured contrast; bf16 size effects are smaller and nonmonotonic.")
    else:
        w.macro("PrecisionAbstract", " ".join(abstract) + ".")
    w.macro("AbstractMetrics", "; ".join(abstract) + "." if abstract else "Measurements pending.")
    w.macro("ControlResults", " ".join(null_observations))
    w.macro("PairedNoMarkResults", " ".join(paired_observations) if paired_observations else
            "Paired no-mark results are pending; their rows and empirical FPRs are included when available.")
    w.macro("PairedNoMarkPending", ", ".join(escape(r["config"]["model"].removeprefix("Qwen/")) + " "
                                          + escape(r["config"]["dtype"]) for r in runs if not nomark_run(r)) or "none")
    w.table("model_revisions.tex", ["Model/dtype", "Hugging Face model revision", "Provenance"], revisions,
            "Model revisions. Legacy snapshots identify the earlier runs whose JSONs omitted this field; new JSON revisions take precedence.", "tab:revisions", spec="lp{.58\\linewidth}l")
    w.table("setup.tex", ["Model/dtype", "$d$", "$L$", "$\\ell_i$", "$\\ell_r$", "$N$", "$N_c$", "$T$"], configs,
            "Measured configurations. Layer indices refer to block outputs.", "tab:setup", wide=True)
    w.table("fidelity.tex", ["Model/dtype", "$B$", "Identical", "First div.", "$D_{\\rm KL}$", "Agreement"], fidelity,
            "Autoregressive fidelity grouped by dtype, separately for each physical payload size. First divergence is zero-indexed and right-censored at the run length; KL is marked-to-clean, through the first differing prediction (inclusive).", "tab:fidelity", wide=True)
    stat_headers = ["Bit acc.", "Matches", "Nominal median $p$", "$<10^{-2}$", "$<10^{-3}$", "$<10^{-4}$", "$<10^{-5}$"]
    w.table("verification.tex", ["Model/dtype", "$B$", "Mode"] + stat_headers + ["Full payload"], verify,
            "Rows are grouped by dtype. Physical rows report detection over distinct code coordinates; logical rows report sixteen-bit payload recovery after majority vote. Threshold columns are per-session fractions. Full payload is the fraction with sixteen correct logical bits. Logical tails are nominal (payload-randomised): a null majority vote favours $+1$ with probability $11/16$.", "tab:verification", wide=True)
    w.table("controls.tex", ["Model/dtype", "$B$", "Control/mode"] + stat_headers, controls,
            "Null and substitution controls, including sixty-four-physical-bit detection and sixteen-logical-bit recovery. No-mark threshold columns are empirical FPRs; paired rows appear when measured. All tails are nominal; logical tails are nominal (payload-randomised). Wrong keys are fixed controls.", "tab:controls", wide=True)
    w.table("p_vs_T.tex", ["Model/dtype", "Coordinates", "Mode"] + [f"$T={t}$" for t in (8,16,32,64,128)], curves,
            "Median session-level nominal tails versus prefix length. Physical and logical statistics are separated; logical prefix tails are reconstructed from session records.", "tab:curve", wide=True)
    w.table("sources.tex", ["Model/dtype", "SHA-256 prefix (source JSON)"], hashes,
            r"Source revisions for the tables and figures; full hashes and absolute source paths are recorded in tables/source\_manifest.json.", "tab:sources")
    (w.out / "observations.tex").write_text("\n\n".join(observations) if observations else
                                            "No completed scale JSON is available.")
    (w.out / "macros.tex").write_text("% Generated numeric values; csname permits schema keys and run indices.\n" + "\n".join(
        rf"\expandafter\def\csname {k}\endcsname{{{v}}}" for k, v in w.macros.items()) + "\n")
    (w.out / "source_manifest.json").write_text(json.dumps({"sources": [
        {"path": r["_source"], "sha256": r["_sha256"]} for r in runs],
        "nomark_sources": [{"path": r["_nomark"]["_source"], "sha256": r["_nomark"]["_sha256"]} for r in runs if r.get("_nomark")],
        "v1": {"path": str(Path(v1_path).resolve()), "sha256": hashlib.sha256(Path(v1_path).read_bytes()).hexdigest()}}, indent=2) + "\n")
    print(f"Generated {len(list(w.out.glob('*.tex')))} TeX files from {len(runs)} completed scale run(s).")


def stat_cells(w, name, s):
    cells = [w.cell(name + "acc", s["bitacc_mean"]), w.cell(name + "matches", s["matches_median"], "count"),
             w.cell(name + "p", s["p_median"], "sci")]
    for i, threshold in enumerate(THRESHOLDS):
        fractions = {float(k): v for k, v in s["frac_p_below"].items()}
        cells.append(w.cell(name + f"frac{i}", fractions[threshold]))
    return cells


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=HERE / "tables")
    args = parser.parse_args()
    build(load_runs(args.results), args.output)
