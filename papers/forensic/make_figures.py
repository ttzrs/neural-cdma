#!/usr/bin/env python
"""Paper A figures from the same completed JSON snapshot used by make_tables.py."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

# Keep font caches with other intermediates, not in an unwritable HOME.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".trabajo" / "forensic-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_tables import attach_nomark, mode_summary, nomark_run

plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.constrained_layout.use": True})


def save(fig, output, name):
    fig.savefig(output / name, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def empty(ax):
    ax.text(.5, .5, "No completed scale JSON available\n(smoke data are not published)",
            transform=ax.transAxes, ha="center", va="center")
    ax.set_axis_off()


def build(results, output, tables):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(tables) / "source_manifest.json").read_text())
    # Do not mix a table snapshot with results that arrive between generation steps.
    sources = manifest["sources"]
    runs = []
    for source in sources:
        path = Path(source["path"])
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != source["sha256"]:
            raise ValueError(f"Result changed after tables were generated: {path}; rerun build.sh")
        runs.append(json.loads(raw))
    supplemental = manifest.get("nomark_sources", [])
    for source in supplemental:
        if hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("No-mark result changed after tables were generated; rerun build.sh")
    attach_nomark(runs, [source["path"] for source in supplemental])
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), squeeze=False)
    colors = plt.get_cmap("tab10")
    for ax, block, logical in zip(axes[0], ("B16", "B64"), (False, False)):
        if not runs:
            empty(ax)
            continue
        for i, run in enumerate(runs):
            label = run["config"]["model"].removeprefix("Qwen/") + " " + run["config"]["dtype"]
            for mode, style in (("paired", "-"), ("blind", "--")):
                times = sorted(map(int, run["summary"][block][mode + "_T"]))
                ps = [mode_summary(run, block, mode + "_T", logical, T=t)["p_median"] for t in times]
                ax.plot(times, ps, marker="o", markersize=3, linestyle=style, color=colors(i),
                        label=label + ", " + mode)
        if block == "B16":
            ax.axhline(2**-16, color="gray", lw=.7, ls=":", label="16-coordinate floor")
        ax.set(xlabel="Averaged prefix length T", ylabel="Median session-level nominal p",
               yscale="log", title="B = 16" if block == "B16" else "B = 64 physical (16-bit payload)")
        ax.legend(fontsize=5.7, loc="best")
        ax.grid(alpha=.2)
    save(fig, output, "p_vs_T.pdf")

    cols = min(2, max(1, len(runs)))
    rows = math.ceil(max(1, len(runs)) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 2.6 * rows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i >= len(runs):
            empty(ax) if not runs else ax.set_axis_off()
            continue
        run = runs[i]
        series = [("owner paired", [s["B16"]["paired"]["p_phys"] for s in run["sessions"]], "-")]
        series.append(("owner blind", [s["B16"]["blind"]["p_phys"] for s in run["sessions"]], "--"))
        for j, key in enumerate(run["config"]["wrong_keys"], 1):
            for mode, style in (("paired", "-"), ("blind", "--")):
                series.append((f"wrong {j} {mode}", [s["B16"]["wrong"][key][mode]["p_phys"]
                                                     for s in run["sessions"]], style))
        series.append(("no mark (blind)", [s["B16"]["nomark_blind"]["p_phys"]
                                           for s in run["sessions"]], ":"))
        nr = nomark_run(run)
        if nr:
            series.append(("no mark (paired)", [s["B16"]["nomark_paired"]["p_phys"]
                                                for s in nr["sessions"]], "-."))
        for label, ps, style in series:
            ps = sorted(ps)
            ax.step([min(ps)] + ps, [0] + [(j + 1) / len(ps) for j in range(len(ps))],
                    where="post", label=label, linestyle=style)
        ax.set(xscale="symlog", xlim=(0, 1), xlabel="Session-level nominal p (B = 16)", ylabel="Empirical CDF",
               ylim=(0, 1.03), title=run["config"]["model"].removeprefix("Qwen/") + " " + run["config"]["dtype"])
        ax.set_xscale("symlog", linthresh=1e-6)
        ax.legend(fontsize=6, loc="best")
        ax.grid(alpha=.2)
    save(fig, output, "session_p_ecdf.pdf")

    v1_source = manifest["v1"]
    raw = Path(v1_source["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != v1_source["sha256"]:
        raise ValueError("v1 result changed between tables and figures")
    v1 = json.loads(raw)["16"]
    alphas = sorted(map(float, v1))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for name, label, marker in (("bitacc_residual", "Residual: paired linear correlation", "o"),
                                ("bitacc_logits", "Logits: tested projected decoder", "s"),
                                ("bitacc_control", "Wrong-code control", "^")):
        ax.plot(alphas, [v1[str(a)][name] for a in alphas], marker=marker, label=label)
    ax.axhline(.5, color="gray", lw=.8, ls=":", label="Chance reference")
    ax.set(xscale="log", xlabel="Injection fraction α", ylabel="Mean bit accuracy",
           ylim=(0.35, 1.03), title="Legacy v1 probe: Qwen2.5-0.5B fp32, B = 16")
    ax.legend(fontsize=7)
    ax.grid(alpha=.2)
    save(fig, output, "logit_wall.pdf")
    print(f"Generated 3 PDF figures from {len(runs)} completed scale run(s) and the v1 probe.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results", help="Reserved for schema-test parity; the table manifest selects the snapshot")
    parser.add_argument("--output", type=Path, default=HERE / "figures")
    parser.add_argument("--tables", type=Path, default=HERE / "tables")
    args = parser.parse_args()
    build(args.results, args.output, args.tables)
