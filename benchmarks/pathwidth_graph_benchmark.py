#!/usr/bin/env python
"""Benchmark MSO queries on the uniformly automatic class of graphs of
pathwidth at most 2 — the linear-layout companion of the tree-depth benchmark.

Graphs of pathwidth w have chromatic number at most w+1, so on `PathWidthClass(2)`
3-colourability is trivially always true; the non-trivial P-time invariants are
**connectedness** and **2-colourability** (bipartiteness), decided by one
automaton each for every pathwidth-2 graph via its advice word (a linear layout
recording, per vertex, its register and the registers of its earlier
neighbours).

Usage:
    pip install "autstr[jax,benchmarks,graphs]"
    python pathwidth_graph_benchmark.py [--max-exp 6] [--batch 50000]
"""
import argparse
import os
import time
from pathlib import Path

from autstr.graphs import PathWidthClass
from autstr.sparse_automata import SparseDFA
import _bench_common as bench

CACHE = Path(os.environ.get("AUTSTR_BENCH_CACHE", "/tmp/autstr-bench-cache"))

CONNECTED = (
    'not (exists s.('
    '(exists x.(Sing(x) and Subset(x,s))) and '
    '(exists y.(Sing(y) and (not Subset(y,s)))) and '
    '(all x.(all y.((not E(x,y)) or ('
    '(Subset(x,s) and Subset(y,s)) or ((not Subset(x,s)) and (not Subset(y,s)))'
    '))))))'
)
TWO_COL = (
    'exists c.(all x.(all y.((not E(x,y)) or '
    '((Subset(x,c) and (not Subset(y,c))) or '
    '((not Subset(x,c)) and Subset(y,c))))))'
)

SIZES = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000,
         200_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000]


# ---- direct pathwidth-2 advice generators, as letter lists ----
def pw_star(n):
    """Star: centre at register 0, leaves at register 1. Bipartite, connected."""
    return [(0, ())] + (n - 1) * [(1, (0,))]


def pw_book(k):
    """k triangles sharing the edge (u, v): non-bipartite, connected. N = k+2."""
    return [(0, ()), (1, (0,))] + k * [(2, (0, 1))]


def col_mixture(k, rng):
    """Edge (u,v) plus k vertices each adjacent to both endpoints (a triangle)
    or to u only (a tree edge); non-bipartite iff at least one triangle."""
    bip = rng.random() < 0.5
    letters = [(0, ()), (1, (0,))]
    for i in range(k):
        tri = (not bip) and (i == 0 or rng.random() < 0.5)
        letters.append((2, (0, 1) if tri else (0,)))
    return letters, bip


def conn_mixture(k, rng):
    """One book (connected) or two disjoint books (disconnected), both on k+2
    vertices."""
    if rng.random() < 0.5:
        return pw_book(k), True
    a = (k - 2) // 2
    return pw_book(a) + pw_book(k - 2 - a), False


QUERIES = {
    "connected": dict(name="pw2_connected", formula=CONNECTED,
                      label="connectedness",
                      scale=lambda n: pw_book(max(1, n - 2)), batch=conn_mixture),
    "2col": dict(name="pw2_2col", formula=TWO_COL, label="2-colourability",
                 scale=lambda n: pw_book(max(1, n - 2)), batch=col_mixture),
}


def build_or_load(pw, spec):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{spec['name']}.sdfa"
    if path.exists():
        dfa = SparseDFA.sparse_dfa_from_file(str(path))
        print(f"  {spec['label']}: {dfa.num_states}-state automaton (cached)")
        return dfa
    print(f"  {spec['label']}: compiling (one-time)...", flush=True)
    t0 = time.time()
    dfa, _ = pw.cls.evaluate(spec["formula"])
    dfa.sparse_dfa_to_file(str(path))
    print(f"  {spec['label']}: {dfa.num_states} states in {time.time()-t0:.1f}s")
    return dfa


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-exp", type=int, default=6)
    ap.add_argument("--batch", type=int, default=50000)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--formats", default="svg,pdf")
    args = ap.parse_args()

    print("Graphs of pathwidth <= 2 — properties: connectedness, 2-colourability")
    pw = PathWidthClass(2)
    S = pw.symbol_of
    sizes = [n for n in SIZES if n <= 10 ** args.max_exp]
    series = {}

    for key in ("connected", "2col"):
        spec = QUERIES[key]
        dfa = build_or_load(pw, spec)
        enc = bench.encoder(dfa)
        scale = lambda n, spec=spec: [S[l] for l in spec["scale"](n)]
        batch = lambda b, rng, spec=spec: (
            lambda letters, ans: ([S[l] for l in letters], ans))(*spec["batch"](b, rng))
        bench.run_scaling(dfa, enc, scale, args.max_exp, spec["label"])
        bench.run_batch(dfa, enc, batch, args.batch, spec["label"])
        series[spec["label"]] = bench.run_curve(dfa, enc, scale, sizes,
                                                args.reps, spec["label"])

    bench.draw(series, Path(args.out_dir), args.formats.split(","),
               "MSO query evaluation on PathWidthClass(2)", "pathwidth_curves")


if __name__ == "__main__":
    main()
