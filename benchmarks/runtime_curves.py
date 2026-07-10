#!/usr/bin/env python
"""Measure average decision time as a function of graph size, to exhibit the
linear scaling of query evaluation on a uniformly automatic class.

For each feasible query (connectedness, 2-colourability) and a geometric sweep
of vertex counts, this times deciding a single graph — with the plain NumPy
automaton loop and, when JAX is installed, the compiled scan — averaged over a
few repetitions. It writes a CSV, reports the per-vertex time (constant iff
evaluation is linear) with the R^2 of a through-the-origin linear fit, and
draws a publication-quality plot (SVG + PDF) with matplotlib.

Usage:
    pip install "autstr[jax,benchmarks]"
    python runtime_curves.py [--reps 5] [--max-exp 7] [--out-dir DIR]

Requires the query automata to be cached (run uniform_graph_benchmark.py first,
or set $AUTSTR_BENCH_CACHE to a directory containing the .sdfa files).
"""
import argparse
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bench_common as bench

import numpy as np

from autstr.graphs import TreeDepthClass
import autstr.sparse_automata as _sa
from uniform_graph_benchmark import (
    CACHE, QUERIES, build_or_load, encode_letters, symbol_codes,
)

SIZES = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000,
         200_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000]

# NumPy's Python transition loop is ~0.3 Mvert/s; cap it so the sweep stays
# quick. JAX's compiled scan runs the whole range.
NUMPY_MAX = 500_000


def time_decide_numpy(dfa, arr, reps):
    best = float("inf")
    for _ in range(reps):
        t0 = time.time()
        dfa.is_accepting[dfa.compute(arr)]
        best = min(best, time.time() - t0)
    return best


def time_decide_jax(dfa, arr, reps):
    word = arr[None, :]
    dfa.accepts_batch(word)               # warm up JIT for this length
    best = float("inf")
    for _ in range(reps):
        t0 = time.time()
        dfa.accepts_batch(word)
        best = min(best, time.time() - t0)
    return best


def linear_fit_r2(xs, ys):
    """Slope and R^2 of the best through-the-origin line y = s*x."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    s = (xs * ys).sum() / (xs * xs).sum()
    ss_res = ((ys - s * xs) ** 2).sum()
    ss_tot = ((ys - ys.mean()) ** 2).sum()
    return s, 1 - ss_res / ss_tot


def run_curve(cls, dfa, spec, sizes, reps):
    codes = symbol_codes(cls, dfa)
    print(f"\n=== {spec['label']} (automaton: {dfa.num_states} states) ===")
    print(f"{'vertices':>12} {'numpy(ms)':>11} {'ns/vert':>9}"
          f" {'jax(ms)':>10} {'ns/vert':>9}")
    data = {"np_n": [], "np_ms": [], "jx_n": [], "jx_ms": []}
    for n in sizes:
        arr = encode_letters(codes, spec["scale"](n))
        if n <= NUMPY_MAX:
            t = time_decide_numpy(dfa, arr, reps) * 1000
            data["np_n"].append(n); data["np_ms"].append(t)
            np_cell, np_ns = f"{t:11.3f}", f"{t*1e6/n:9.1f}"
        else:
            np_cell, np_ns = f"{'--':>11}", f"{'--':>9}"
        if _sa._HAS_JAX:
            t = time_decide_jax(dfa, arr, reps) * 1000
            data["jx_n"].append(n); data["jx_ms"].append(t)
            jx_cell, jx_ns = f"{t:10.3f}", f"{t*1e6/n:9.2f}"
        else:
            jx_cell, jx_ns = f"{'--':>10}", f"{'--':>9}"
        print(f"{n:>12,} {np_cell} {np_ns} {jx_cell} {jx_ns}")

    if len(data["np_n"]) >= 2:
        s, r2 = linear_fit_r2(data["np_n"], data["np_ms"])
        print(f"  NumPy linear fit: {s*1e6:7.1f} ns/vertex, R^2 = {r2:.5f}")
    if len(data["jx_n"]) >= 2:
        s, r2 = linear_fit_r2(data["jx_n"], data["jx_ms"])
        print(f"  JAX   linear fit: {s*1e6:7.2f} ns/vertex, R^2 = {r2:.5f}")
    return data


def draw(series, out_dir, formats):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed — skipping the plot.")
        print("Install it with:  pip install \"autstr[benchmarks]\"")
        return

    plt.rcParams.update({"font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "figure.dpi": 120})
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    all_n, all_ms = [], []
    for i, (label, d) in enumerate(series.items()):
        c = colours[i % len(colours)]
        if d["np_n"]:
            ax.plot(d["np_n"], d["np_ms"], "o-", color=c, lw=1.8, ms=5,
                    label=f"{label} — NumPy loop")
            all_n += d["np_n"]; all_ms += d["np_ms"]
        if d["jx_n"]:
            ax.plot(d["jx_n"], d["jx_ms"], "s--", color=c, lw=1.6, ms=4,
                    markerfacecolor="none",
                    label=f"{label} — JAX scan")
            all_n += d["jx_n"]; all_ms += d["jx_ms"]

    # A slope-1 reference line: on log-log, linear O(n) is a straight 45-degree
    # line, so parallel data confirms linear scaling.
    xr = np.array([min(all_n), max(all_n)], float)
    anchor = min(all_n)
    ref_rate = min(m for n, m in zip(all_n, all_ms) if n == anchor) / anchor
    ax.plot(xr, ref_rate * xr, ":", color="0.5", lw=1.4,
            label="linear reference (slope 1)")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("number of vertices")
    ax.set_ylabel("decision time (ms)")
    ax.set_title("Linear-time MSO query evaluation on TreeDepthClass(4)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = out_dir / f"runtime_curves.{fmt}"
        fig.savefig(path)
        written.append(str(path))
    print("\nWrote plot -> " + ", ".join(written))


def main():
    ap = bench.parser(__doc__)
    args = ap.parse_args()
    cfg = bench.settings(args)
    sizes = [n for n in SIZES if n <= 10 ** cfg["max_exp"]]

    cls = TreeDepthClass(4)
    series, rows = {}, []
    for key in ("connected", "2col"):
        dfa = build_or_load(cls, QUERIES[key], allow_compile=False)
        if dfa is None:
            print(f"  {QUERIES[key]['label']}: not cached — run "
                  f"uniform_graph_benchmark.py first.")
            continue
        d = run_curve(cls, dfa, QUERIES[key], sizes, cfg["reps"])
        series[QUERIES[key]["label"]] = d
        for n, ms in zip(d["np_n"], d["np_ms"]):
            rows.append((QUERIES[key]["label"], n, "numpy", ms))
        for n, ms in zip(d["jx_n"], d["jx_ms"]):
            rows.append((QUERIES[key]["label"], n, "jax", ms))

    csv_path = CACHE / "runtime_curves.csv"
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("query,vertices,backend,decide_ms\n")
        for label, n, backend, ms in rows:
            f.write(f"{label},{n},{backend},{ms}\n")
    print(f"\nWrote CSV -> {csv_path}")

    if series:
        if cfg["plot"]:
            draw(series, Path(args.out_dir), args.formats.split(","))


if __name__ == "__main__":
    main()
