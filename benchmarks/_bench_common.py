"""Shared timing, throughput and plotting helpers for the uniformly-automatic
class benchmarks.

A benchmark provides, for a compiled query automaton:

* ``scale(n)``  -> a list of advice-tape *symbols* for a structure whose advice
  word has length ~n (a sentence query reads only the advice tape);
* ``batch(block, rng)`` -> ``(symbols, answer)`` for a random structure with a
  known ground-truth answer, at constant length for fixed ``block``.

These helpers then measure single-structure scaling, batched throughput
(NumPy loop vs. NumPy batch vs. JAX batch) and draw a vector runtime curve.
"""
import argparse
import time
from pathlib import Path

import numpy as np

from autstr.utils.misc import encode_symbol
import autstr.sparse_automata as _sa


# ---------------------------------------------------------------- profiles
#
# Every benchmark runs in one of two sizes. `light` is the default: it takes
# seconds, fits in a laptop, and is meant to be run on every checkout. `heavy`
# takes minutes and more memory, and additionally unlocks the queries a light
# run skips. Heavy is deliberately not the largest thing we can compile --
# it is the largest thing that still finishes unattended.

LIGHT = dict(max_exp=4, batch=2_000, reps=3)
HEAVY = dict(max_exp=6, batch=50_000, reps=5)


def parser(description=None):
    """An argument parser with the options every benchmark shares."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--profile", choices=("light", "heavy"), default="light",
                    help="light (default): seconds, laptop-safe. "
                         "heavy: minutes, more RAM, extra queries.")
    ap.add_argument("--max-exp", type=int, help="override: largest 10**e size")
    ap.add_argument("--batch", type=int, help="override: words per batch")
    ap.add_argument("--reps", type=int, help="override: timing repetitions")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--formats", default="svg,pdf")
    ap.add_argument("--no-plot", action="store_true", help="skip the figures")
    return ap


def settings(args):
    """Profile defaults, with any explicit flag winning."""
    chosen = dict(HEAVY if args.profile == "heavy" else LIGHT)
    for key in ("max_exp", "batch", "reps"):
        value = getattr(args, key, None)
        if value is not None:
            chosen[key] = value
    chosen["heavy"] = args.profile == "heavy"
    chosen["plot"] = not args.no_plot
    print(f"[profile: {args.profile}] max_exp={chosen['max_exp']} "
          f"batch={chosen['batch']:,} reps={chosen['reps']}")
    return chosen


def encoder(dfa):
    """Return a function mapping an advice-symbol list to an int array, using
    the automaton's own alphabet encoding (single tape)."""
    base = dfa.base_alphabet_frozen
    table = {s: encode_symbol((s,), base) for s in base}
    return lambda symbols: np.fromiter((table[s] for s in symbols),
                                       dtype=np.int64, count=len(symbols))


def decide(dfa, arr):
    return bool(dfa.is_accepting[dfa.compute(arr)])


def run_scaling(dfa, enc, scale, max_exp, label, min_exp=3):
    print(f"\n== Single-structure scaling: {label} (NumPy) ==")
    print(f"{'advice len':>12} {'build(s)':>10} {'decide(s)':>10} "
          f"{'Msym/s':>10}  answer")
    for exp in range(min_exp, max_exp + 1):
        n = 10 ** exp
        t0 = time.time(); arr = enc(scale(n)); t_build = time.time() - t0
        t0 = time.time(); ans = decide(dfa, arr); t_dec = time.time() - t0
        print(f"{len(arr):>12,} {t_build:>10.3f} {t_dec:>10.3f} "
              f"{len(arr)/t_dec/1e6:>10.2f}  {ans}")
    if _sa._HAS_JAX:
        arr = enc(scale(10 ** max_exp))
        dfa.accepts_batch(arr[None, :])
        t0 = time.time(); r = bool(dfa.accepts_batch(arr[None, :])[0])
        t_jx = time.time() - t0
        print(f"  same {len(arr):,}-symbol structure via JAX scan: "
              f"{t_jx*1000:.1f} ms ({len(arr)/t_jx/1e6:.1f} Msym/s), answer={r}")


def run_batch(dfa, enc, batch_gen, batch, label, block=200):
    print(f"\n== Batched throughput: {batch:,} structures, {label} ==")
    rng = __import__("random").Random(0)
    rows, truth = [], []
    for _ in range(batch):
        item, ans = batch_gen(block, rng)
        rows.append(enc(item)); truth.append(ans)
    # Generators keep a batch equal length (single- or multi-tape), so the
    # encoded rows stack directly into a (batch, length) int array.
    lengths = {len(r) for r in rows}
    assert len(lengths) == 1, f"batch words must be equal length, got {sorted(lengths)}"
    words = np.stack(rows)
    length = words.shape[1]
    truth = np.array(truth)
    steps = words.size

    t0 = time.time()
    loop = np.array([decide(dfa, w) for w in words])
    t_loop = time.time() - t0

    saved = _sa._HAS_JAX
    _sa._HAS_JAX = False
    np_batch = dfa.accepts_batch(words)
    t0 = time.time(); np_batch = dfa.accepts_batch(words); t_np = time.time() - t0
    _sa._HAS_JAX = saved

    print(f"  {batch:,} structures x {length} symbols "
          f"({int(truth.sum())} satisfy the query)")
    print(f"  NumPy loop : {t_loop*1000:8.1f} ms  ({steps/t_loop/1e6:7.2f} Msym/s)")
    print(f"  NumPy batch: {t_np*1000:8.1f} ms  ({steps/t_np/1e6:7.2f} Msym/s)"
          f"  speedup {t_loop/t_np:5.1f}x")
    if saved:
        dfa.accepts_batch(words)
        t0 = time.time(); jx = dfa.accepts_batch(words); t_jx = time.time() - t0
        assert np.array_equal(np_batch, jx)
        print(f"  JAX batch  : {t_jx*1000:8.1f} ms  ({steps/t_jx/1e6:7.2f} Msym/s)"
              f"  speedup {t_loop/t_jx:5.1f}x")
    else:
        print("  JAX batch  : (jax not installed — pip install autstr[jax])")
    assert np.array_equal(loop, np_batch)
    assert int((loop == truth).sum()) == batch, "disagreement with ground truth"
    print(f"  correctness: all {batch:,} match ground truth and each other")


def _fit(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    s = (xs * ys).sum() / (xs * xs).sum()
    ss_res = ((ys - s * xs) ** 2).sum()
    ss_tot = ((ys - ys.mean()) ** 2).sum()
    return s, 1 - ss_res / ss_tot


def run_curve(dfa, enc, scale, sizes, reps, label, numpy_max=500_000):
    print(f"\n=== {label} (automaton: {dfa.num_states} states) ===")
    print(f"{'advice len':>12} {'numpy(ms)':>11} {'ns/sym':>9}"
          f" {'jax(ms)':>10} {'ns/sym':>9}")
    d = {"np_n": [], "np_ms": [], "jx_n": [], "jx_ms": []}
    for n in sizes:
        arr = enc(scale(n))
        m = len(arr)
        if n <= numpy_max:
            best = min(_timed(lambda: dfa.compute(arr)) for _ in range(reps)) * 1000
            d["np_n"].append(m); d["np_ms"].append(best)
            nc, nn = f"{best:11.3f}", f"{best*1e6/m:9.1f}"
        else:
            nc, nn = f"{'--':>11}", f"{'--':>9}"
        if _sa._HAS_JAX:
            w = arr[None, :]; dfa.accepts_batch(w)
            best = min(_timed(lambda: dfa.accepts_batch(w)) for _ in range(reps)) * 1000
            d["jx_n"].append(m); d["jx_ms"].append(best)
            jc, jn = f"{best:10.3f}", f"{best*1e6/m:9.2f}"
        else:
            jc, jn = f"{'--':>10}", f"{'--':>9}"
        print(f"{m:>12,} {nc} {nn} {jc} {jn}")
    if len(d["np_n"]) >= 2:
        s, r2 = _fit(d["np_n"], d["np_ms"])
        print(f"  NumPy linear fit: {s*1e6:7.1f} ns/symbol, R^2 = {r2:.5f}")
    if len(d["jx_n"]) >= 2:
        s, r2 = _fit(d["jx_n"], d["jx_ms"])
        print(f"  JAX   linear fit: {s*1e6:7.2f} ns/symbol, R^2 = {r2:.5f}")
    return d


def _timed(fn):
    t0 = time.time(); fn(); return time.time() - t0


def draw(series, out_dir, formats, title, stem):
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
    for i, (lbl, d) in enumerate(series.items()):
        c = colours[i % len(colours)]
        if d["np_n"]:
            ax.plot(d["np_n"], d["np_ms"], "o-", color=c, lw=1.8, ms=5,
                    label=f"{lbl} — NumPy loop")
            all_n += d["np_n"]; all_ms += d["np_ms"]
        if d["jx_n"]:
            ax.plot(d["jx_n"], d["jx_ms"], "s--", color=c, lw=1.6, ms=4,
                    markerfacecolor="none", label=f"{lbl} — JAX scan")
            all_n += d["jx_n"]; all_ms += d["jx_ms"]
    xr = np.array([min(all_n), max(all_n)], float)
    anchor = min(all_n)
    rate = min(m for n, m in zip(all_n, all_ms) if n == anchor) / anchor
    ax.plot(xr, rate * xr, ":", color="0.5", lw=1.4, label="linear reference")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("advice length (symbols)")
    ax.set_ylabel("decision time (ms)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        p = out_dir / f"{stem}.{fmt}"
        fig.savefig(p); written.append(str(p))
    print("\nWrote plot -> " + ", ".join(written))


# ---------------------------------------------------------------- tree automata
#
# The tree classes read a convolution of trees rather than a word, so they
# need their own encoder and timer.

def tree_arrays(sta, tree):
    """Pre-encode a tree into the post-order arrays `run` consumes, so the
    timing measures the automaton and not the encoder."""
    from autstr.sparse_tree_automata import tree_to_arrays
    return tree_to_arrays(tree, sta.base_alphabet_frozen, sta.symbol_arity)


def tree_decide(sta, arrays):
    return bool(sta.is_accepting[sta.run(*arrays)])


def run_tree_scaling(sta, build, sizes, label, reps=3):
    """Time a compiled tree automaton on advice trees of growing size."""
    print(f"\n=== {label} (automaton: {sta.num_states} states) ===")
    print(f"{'nodes':>12} {'build(s)':>10} {'decide(ms)':>12} "
          f"{'ns/node':>9}  answer")
    data = {"n": [], "ms": []}
    for size in sizes:
        t0 = time.time()
        arrays = tree_arrays(sta, build(size))
        build_s = time.time() - t0
        nodes = len(arrays[0])
        best = min(_timed(lambda: sta.run(*arrays)) for _ in range(reps))
        answer = tree_decide(sta, arrays)
        data["n"].append(nodes)
        data["ms"].append(best * 1000)
        print(f"{nodes:>12,} {build_s:>10.3f} {best*1000:>12.3f} "
              f"{best*1e9/nodes:>9.1f}  {answer}")
    if len(data["n"]) >= 2:
        slope, r2 = _fit(data["n"], data["ms"])
        print(f"  linear fit: {slope*1e6:7.1f} ns/node, R^2 = {r2:.5f}")
    return data
