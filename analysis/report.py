"""Build charts, tables and REPORT.md from the raw result files.

    python -m analysis.report                                   # real results -> ./REPORT.md
    python -m analysis.report --results results/sample --out build/sample

Needs no GPU. Every number in the report comes from files under --results.
"""

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from analysis.load import load_benchmarks, load_eval, load_quantization  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VARIANTS = ["base", "w8", "w4"]
LABELS = {"base": "base (FP16)", "w8": "w8 (INT8 W8A8)", "w4": "w4 (W4A16)"}
# Fixed categorical order, so a variant keeps its color in every chart.
COLORS = {"base": "#2a78d6", "w8": "#eb6834", "w4": "#1baf7a"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MAX_ERROR_RATE = 0.01


# ---------------------------------------------------------------- numbers


def workload_of(cell: str) -> str:
    return cell.split("_cache-")[0]


def aggregate(bench: pd.DataFrame) -> pd.DataFrame:
    """Median over runs for each load point, plus the min/max spread (FR-3.3)."""
    keys = ["variant", "cell", "concurrency"]
    metrics = [c for c in bench.columns if c not in keys + ["run"]]
    med = bench.groupby(keys)[metrics].median()
    spread = bench.groupby(keys).agg(
        runs=("run", "nunique"),
        ttft_p95_min=("ttft_p95_ms", "min"),
        ttft_p95_max=("ttft_p95_ms", "max"),
        tput_min=("output_tok_per_s", "min"),
        tput_max=("output_tok_per_s", "max"),
    )
    return med.join(spread).reset_index()


@dataclass
class Capacity:
    concurrency: float  # highest load that still meets the SLO (interpolated)
    tput: float  # output tokens/s at that load
    ttft_p95: float
    capped: bool  # True if even the highest tested load met the SLO


def capacity(points: pd.DataFrame, slo_ms: float) -> Capacity:
    """Walk up the load curve until p95 TTFT crosses the SLO, then interpolate
    linearly between the last passing and first failing point (FR-5.3)."""
    pts = points.sort_values("concurrency").reset_index(drop=True)
    ok = (pts["ttft_p95_ms"] <= slo_ms) & (pts["error_rate"] <= MAX_ERROR_RATE)
    if not ok.iloc[0]:
        return Capacity(0.0, math.nan, pts["ttft_p95_ms"].iloc[0], False)
    if ok.all():
        last = pts.iloc[-1]
        return Capacity(last["concurrency"], last["output_tok_per_s"], last["ttft_p95_ms"], True)
    i = int(np.argmin(ok.to_numpy()))
    a, b = pts.iloc[i - 1], pts.iloc[i]
    if b["ttft_p95_ms"] <= slo_ms:  # b fails on errors, not latency: stop at a
        return Capacity(a["concurrency"], a["output_tok_per_s"], a["ttft_p95_ms"], False)
    span = b["ttft_p95_ms"] - a["ttft_p95_ms"]
    f = (slo_ms - a["ttft_p95_ms"]) / span if span > 0 else 0.0
    f = min(max(f, 0.0), 1.0)
    return Capacity(
        a["concurrency"] + f * (b["concurrency"] - a["concurrency"]),
        a["output_tok_per_s"] + f * (b["output_tok_per_s"] - a["output_tok_per_s"]),
        slo_ms,
        False,
    )


def cost_per_million(tput: float, price_per_hour: float) -> float:
    """FR-5.1: (price per hour / (tokens per second * 3600)) * 1,000,000"""
    if not tput or math.isnan(tput):
        return math.nan
    return price_per_hour / (tput * 3600) * 1_000_000


def accuracy_table(ev: pd.DataFrame) -> pd.DataFrame:
    """Per task and variant: score, delta vs base in points, and whether the
    delta is within noise (smaller than 2x the combined standard error)."""
    if ev.empty or "base" not in set(ev["variant"]):
        return pd.DataFrame()
    base = ev[ev["variant"] == "base"].set_index("task")
    rows = []
    for _, r in ev.iterrows():
        b = base.loc[r["task"]]
        delta = (r["value"] - b["value"]) * 100
        se = math.hypot(r["stderr"] or 0, b["stderr"] or 0) * 100
        rows.append(
            {
                "variant": r["variant"],
                "task": r["task"],
                "score": r["value"] * 100,
                "stderr": (r["stderr"] or 0) * 100,
                "delta": delta,
                "noise": se * 2,
                "within_noise": r["variant"] == "base" or abs(delta) < 2 * se,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- charts


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def _figure(ncols: int = 1) -> tuple[plt.Figure, list[plt.Axes]]:
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 3.8), facecolor=SURFACE)
    axes = list(np.atleast_1d(axes))
    for ax in axes:
        _style(ax)
    return fig, axes


def _log2_x(ax: plt.Axes, ticks: list[int]) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlabel("Concurrent requests")


def latency_chart(agg: pd.DataFrame, cell: str, slo_ms: float, out: Path) -> None:
    data = agg[agg["cell"] == cell]
    fig, (ax_l, ax_t) = _figure(2)
    for v in VARIANTS:
        d = data[data["variant"] == v].sort_values("concurrency")
        if d.empty:
            continue
        c = COLORS[v]
        ax_l.fill_between(d["concurrency"], d["ttft_p95_min"], d["ttft_p95_max"], color=c, alpha=0.15, lw=0)
        ax_l.plot(d["concurrency"], d["ttft_p95_ms"], color=c, lw=2, marker="o", ms=4, label=LABELS[v])
        ax_t.fill_between(d["concurrency"], d["tput_min"], d["tput_max"], color=c, alpha=0.15, lw=0)
        ax_t.plot(d["concurrency"], d["output_tok_per_s"], color=c, lw=2, marker="o", ms=4, label=LABELS[v])
    ax_l.axhline(slo_ms, color=INK_2, lw=1, ls="--")
    ax_l.annotate(f"SLO {slo_ms:g} ms", (0.02, slo_ms), xycoords=("axes fraction", "data"),
                  xytext=(0, 4), textcoords="offset points", color=INK_2, fontsize=8)  # fmt: skip
    ticks = sorted(data["concurrency"].unique().tolist())
    for ax in (ax_l, ax_t):
        _log2_x(ax, ticks)
    ax_l.set_ylabel("TTFT p95 (ms)")
    ax_t.set_ylabel("Output tokens/s")
    ax_l.set_title("Time to first token, p95", color=INK, fontsize=10, loc="left")
    ax_t.set_title("Throughput", color=INK, fontsize=10, loc="left")
    ax_t.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    fig.suptitle(cell, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def prefix_cache_chart(agg: pd.DataFrame, workload: str, out: Path) -> bool:
    on, off = f"{workload}_cache-on", f"{workload}_cache-off"
    if not {on, off} <= set(agg["cell"]):
        return False
    fig, (ax,) = _figure(1)
    for v in VARIANTS:
        for cell, ls, tag in ((off, "--", "cache off"), (on, "-", "cache on")):
            d = agg[(agg["cell"] == cell) & (agg["variant"] == v)].sort_values("concurrency")
            if not d.empty:
                ax.plot(d["concurrency"], d["ttft_p95_ms"], color=COLORS[v], lw=2, ls=ls,
                        marker="o", ms=4, label=f"{v}, {tag}")  # fmt: skip
    _log2_x(ax, sorted(agg[agg["cell"] == on]["concurrency"].unique().tolist()))
    ax.set_yscale("log")
    ax.set_ylabel("TTFT p95 (ms, log scale)")
    ax.set_title(f"Prefix caching on {workload}", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7, labelcolor=INK_2, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


def pareto_chart(summary: pd.DataFrame, workload: str, out: Path) -> bool:
    d = summary.dropna(subset=["acc_delta", "tput_at_slo"])
    if d.empty:
        return False
    fig, (ax,) = _figure(1)
    for v, r in d.iterrows():
        ax.scatter(r["tput_at_slo"], r["acc_delta"], s=70, color=COLORS[v], edgecolor=SURFACE, lw=2, zorder=3)
        ax.annotate(LABELS[v], (r["tput_at_slo"], r["acc_delta"]), xytext=(7, 4),
                    textcoords="offset points", color=INK, fontsize=9)  # fmt: skip
    ax.axhline(0, color=INK_2, lw=1)
    ax.set_xlabel(f"Output tokens/s at the SLO ({workload})")
    ax.set_ylabel("Accuracy vs base (points)")
    ax.set_title("Speed vs quality", color=INK, fontsize=10, loc="left")
    ax.margins(x=0.25, y=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


# ---------------------------------------------------------------- report


def fmt(x: float, digits: int = 0, prefix: str = "", suffix: str = "") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{prefix}{x:,.{digits}f}{suffix}"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def pick(table: pd.DataFrame, column: str, eligible: list[str]) -> str | None:
    """Best eligible variant on `column`. Ties (e.g. several variants capped at the
    highest tested load) are broken by throughput at the SLO."""
    t = table.loc[[v for v in eligible if v in table.index]].dropna(subset=[column])
    if t.empty:
        return None
    return str(t.sort_values([column, "tput_at_slo"], ascending=False).index[0])


def build(results: Path, out: Path) -> Path:
    bench = load_benchmarks(results)
    if bench.empty:
        sys.exit(f"No benchmark files under {results / 'benchmarks'}")
    quant = load_quantization(results)
    acc = accuracy_table(load_eval(results))
    cost_cfg = yaml.safe_load((ROOT / "configs/cost.yaml").read_text())
    price = cost_cfg["price_per_hour_usd"]
    workloads = {p.stem: yaml.safe_load(p.read_text()) for p in sorted((ROOT / "workloads").glob("*.yaml"))}

    figs = out / "analysis" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    agg = aggregate(bench)
    variants = [v for v in VARIANTS if v in set(agg["variant"])]
    cells = sorted(agg["cell"].unique())

    # Capacity and cost at the SLO for every (variant, cell).
    cap_rows = []
    for cell in cells:
        slo = workloads[workload_of(cell)]["slo"]["ttft_p95_ms"]
        latency_chart(agg, cell, slo, figs / f"latency_{cell}.png")
        for v in variants:
            pts = agg[(agg["cell"] == cell) & (agg["variant"] == v)]
            if pts.empty:
                continue
            c = capacity(pts, slo)
            peak = pts.loc[pts["error_rate"] <= MAX_ERROR_RATE, "output_tok_per_s"].max()
            cap_rows.append(
                {
                    "cell": cell,
                    "variant": v,
                    "slo_ms": slo,
                    "cap": c.concurrency,
                    "capped": c.capped,
                    "tput_at_slo": c.tput,
                    "cost_at_slo": cost_per_million(c.tput, price),
                    "peak_tput": peak,
                    "cost_at_peak": cost_per_million(peak, price),
                }  # fmt: skip
            )
    caps = pd.DataFrame(cap_rows)

    # One row per variant for the headline table; chat_short is the reference workload.
    ref = "chat_short" if "chat_short" in cells else cells[0]
    summary = caps[caps["cell"] == ref].set_index("variant")
    mean_delta = acc.groupby("variant")["delta"].mean() if not acc.empty else pd.Series(dtype=float)
    summary["acc_delta"] = mean_delta.reindex(summary.index)
    for col in ("disk_gib", "weights_gib", "kv_cache_tokens", "perplexity"):
        summary[col] = quant[col].reindex(summary.index) if col in quant else math.nan

    has_pareto = pareto_chart(summary, ref, figs / "pareto.png")
    rag_cells = [w for w in workloads if f"{w}_cache-on" in cells]
    has_cache_chart = [w for w in rag_cells if prefix_cache_chart(agg, w, figs / f"prefix_cache_{w}.png")]

    # Quality gate for recommendations: drop a variant whose average accuracy
    # loss is over 1 point AND larger than the noise on that average. Looking at
    # the average matters: a small drop on every task can be real even when no
    # single task is outside its own error bars.
    eligible = []
    for v in variants:
        rows = acc[acc["variant"] == v] if not acc.empty else acc
        if rows.empty or v == "base":
            eligible.append(v)
            continue
        mean = rows["delta"].mean()
        noise = math.sqrt(((rows["noise"] / 2) ** 2).sum()) / len(rows) * 2
        if mean > -1.0 or abs(mean) < noise:
            eligible.append(v)

    report = render(results, out, agg, caps, summary, acc, quant, workloads, cost_cfg,
                    variants, eligible, ref, has_pareto, has_cache_chart)  # fmt: skip
    path = out / "REPORT.md"
    path.write_text(report, encoding="utf-8")
    return path


def render(results, out, agg, caps, summary, acc, quant, workloads, cost_cfg,
           variants, eligible, ref, has_pareto, cache_workloads) -> str:  # fmt: skip
    L: list[str] = []
    fig = "analysis/figures"
    price = cost_cfg["price_per_hour_usd"]
    checked = cost_cfg["checked_on"] or "not verified yet"
    sample = "sample" in results.parts

    L.append("# Decision report\n")
    if sample:
        L.append("> **Synthetic sample data.** These numbers are made up to test the pipeline. "
                 "They say nothing about real performance.\n")  # fmt: skip
    L.append(f"Generated on {date.today()} from `{results.as_posix()}` by `python -m analysis.report`. "
             "Don't edit by hand; change the code or the data and regenerate.\n")  # fmt: skip

    # ---- recommendation
    L.append("## Recommendation\n")
    by_cell = {c: caps[caps["cell"] == c].set_index("variant") for c in caps["cell"].unique()}
    uses = [
        ("Latency-sensitive chat", ref, "cap", "serves the most concurrent users while keeping p95 TTFT under the SLO"),
        ("High-volume batch jobs", ref, "peak_tput", "has the highest raw throughput, which is what matters when nobody waits on the answer"),
        ("RAG with a shared context", next((f"{w}_cache-on" for w in cache_workloads), None), "cap",
         "handles the most users on the shared-prefix workload with prefix caching on"),
        ("Long documents", "long_context" if "long_context" in by_cell else None, "cap",
         "keeps the most long-prompt requests under the SLO"),
    ]  # fmt: skip
    for use, cell, col, why in uses:
        if cell is None or cell not in by_cell:
            continue
        v = pick(by_cell[cell], col, eligible)
        if v is None:
            continue
        r = by_cell[cell].loc[v]
        value = f"{fmt(r['cap'], 1)}{'+' if r['capped'] else ''} concurrent requests" if col == "cap" \
            else f"{fmt(r['peak_tput'])} output tokens/s"  # fmt: skip
        L.append(f"- **{use}: ship `{v}`.** It {why} ({value} on `{cell}`).")
    dropped = [v for v in variants if v not in eligible]
    if dropped:
        L.append(f"- Not recommended for anything: {', '.join(f'`{v}`' for v in dropped)}. "
                 "Its average accuracy drop is over one point and bigger than the noise on that average.")  # fmt: skip
    if not acc.empty:
        best_acc = acc.groupby("variant")["score"].mean().idxmax()
        L.append(f"- If quality is all that matters, `{best_acc}` has the best average score.")
    L.append("")

    # ---- summary table
    L.append(
        f"## Summary (`{ref}`, SLO p95 TTFT ≤ {workloads[workload_of(ref)]['slo']['ttft_p95_ms']:g} ms)\n"
    )
    rows = []
    for v, r in summary.iterrows():
        rows.append({
            "Variant": LABELS[v],
            "Disk (GiB)": fmt(r["disk_gib"], 2),
            "Weights in GPU (GiB)": fmt(r["weights_gib"], 2),
            "KV cache (tokens)": fmt(r["kv_cache_tokens"]),
            "Perplexity": fmt(r["perplexity"], 2),
            "Accuracy vs base": fmt(r["acc_delta"], 1, suffix=" pt") if v != "base" else "0",
            "Users at SLO": fmt(r["cap"], 1) + ("+" if r["capped"] else ""),
            "Tokens/s at SLO": fmt(r["tput_at_slo"]),
            "$ per 1M tokens": fmt(r["cost_at_slo"], 3, prefix="$"),
        })  # fmt: skip
    L.append(md_table(pd.DataFrame(rows)) + "\n")
    L.append('"+" means the SLO held even at the highest load tested, so the real limit is higher.\n')
    if has_pareto:
        L.append(f"![Speed vs quality]({fig}/pareto.png)\n")

    # ---- per workload
    L.append("## Latency under load\n")
    L.append("Lines are the median of all runs; the shaded band is the min–max across runs.\n")
    for cell in sorted(caps["cell"].unique()):
        wl = workloads[workload_of(cell)]
        L.append(f"### {cell}\n\n{wl['description']}\n")
        L.append(f"![{cell}]({fig}/latency_{cell}.png)\n")
        t = caps[caps["cell"] == cell]
        L.append(md_table(pd.DataFrame({
            "Variant": t["variant"],
            "Users at SLO": [fmt(c, 1) + ("+" if k else "") for c, k in zip(t["cap"], t["capped"], strict=True)],
            "Tokens/s at SLO": [fmt(x) for x in t["tput_at_slo"]],
            "$ / 1M at SLO": [fmt(x, 3, prefix="$") for x in t["cost_at_slo"]],
            "Peak tokens/s": [fmt(x) for x in t["peak_tput"]],
            "$ / 1M at peak": [fmt(x, 3, prefix="$") for x in t["cost_at_peak"]],
        })) + "\n")  # fmt: skip

    # ---- prefix caching
    for w in cache_workloads:
        on, off = agg[agg["cell"] == f"{w}_cache-on"], agg[agg["cell"] == f"{w}_cache-off"]
        L.append(f"## Prefix caching ({w})\n")
        L.append(f"![Prefix caching]({fig}/prefix_cache_{w}.png)\n")
        lines = []
        for v in variants:
            a = on[on["variant"] == v].set_index("concurrency")["ttft_p95_ms"]
            b = off[off["variant"] == v].set_index("concurrency")["ttft_p95_ms"]
            common = a.index.intersection(b.index)
            if len(common):
                c = common[len(common) // 2]
                lines.append(f"- `{v}` at {c} concurrent requests: p95 TTFT {fmt(b[c])} ms without the "
                             f"cache, {fmt(a[c])} ms with it ({fmt(b[c] / a[c], 1)}x faster).")  # fmt: skip
        L.append("\n".join(lines) + "\n")

    # ---- accuracy
    if not acc.empty:
        L.append("## Accuracy\n")
        L.append("Scores in points (0–100). A delta is marked *noise* when it is smaller than twice the "
                 "combined standard error of the two scores.\n")  # fmt: skip
        tbl = acc.pivot(index="task", columns="variant")
        rows = []
        for task in tbl.index:
            row = {"Task": task}
            for v in variants:
                if ("score", v) not in tbl.columns:
                    continue
                s, se, d, ok = (tbl.loc[task, (k, v)] for k in ("score", "stderr", "delta", "within_noise"))
                row[v] = f"{s:.1f} ± {se:.1f}" + (
                    "" if v == "base" else f" ({d:+.1f}{', noise' if ok else ''})"
                )
            rows.append(row)
        L.append(md_table(pd.DataFrame(rows)) + "\n")

    # ---- method
    L.append("## How the numbers were made\n")
    L.append(f"- **Cost:** `cost per 1M output tokens = price per hour / (tokens per second × 3600) × 1,000,000`, "
             f"with {cost_cfg['gpu']} at ${price}/h ({cost_cfg['provider']}; price checked: {checked}). "
             "Cost is taken at the SLO, not at peak throughput, because peak throughput usually breaks any "
             "latency target.")  # fmt: skip
    L.append("- **Users at SLO:** the load where p95 TTFT crosses the SLO, interpolated linearly between the "
             f"two tested points around it. Points with more than {MAX_ERROR_RATE:.0%} errors count as failing.")  # fmt: skip
    runs = int(agg["runs"].min())
    L.append(f"- **Repeats:** every load point ran {runs} time(s); tables use the median.")
    L.append("- Raw files, server launch configs and versions are under `results/`.\n")
    return "\n".join(L)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--out", type=Path, default=ROOT)
    args = parser.parse_args()
    path = build(args.results, args.out)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
