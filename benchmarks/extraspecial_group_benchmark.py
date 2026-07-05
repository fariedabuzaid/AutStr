#!/usr/bin/env python
"""Benchmark a relational query on the uniformly automatic class of extraspecial
p-groups: **do two group elements commute?**

For a fixed prime p the Heisenberg-type group of order p^(1+2n) is nilpotent of
class 2; two elements x = (c, a, b) and y = (c', a', b') commute iff the
symplectic form <a, b'> - <a', b> vanishes mod p. A single 6-state automaton
decides this for every rank n, reading the convolution of the advice (1^(n+1))
with the two element encodings — so it computes in groups of astronomically
large order (p^(1+2n)) in time linear in the rank.

Unlike the sentence benchmarks (which decide a property of a whole structure),
this is a multi-tape query: the automaton reads three tapes (the two elements
and the advice) at once.

Usage:
    pip install "autstr[jax,benchmarks]"
    python extraspecial_group_benchmark.py [--max-exp 6] [--batch 50000] [-p 3]
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np

from autstr.groups import ExtraspecialGroups
from autstr.sparse_automata import SparseDFA
import _bench_common as bench

CACHE = Path(os.environ.get("AUTSTR_BENCH_CACHE", "/tmp/autstr-bench-cache"))

COMMUTE = 'exists z.(M(x,y,z) and M(y,x,z))'   # xy = z = yx

SIZES = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000,
         200_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000]


def build_or_load(heis, p):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"extra{p}_commute.sdfa"
    if path.exists():
        dfa = SparseDFA.sparse_dfa_from_file(str(path))
        _, variables = heis.evaluate(COMMUTE)   # cheap; recovers tape order
        print(f"  commute automaton: {dfa.num_states} states, tapes={variables} (cached)")
        return dfa, variables
    print("  commute: compiling (one-time)...", flush=True)
    t0 = time.time()
    dfa, variables = heis.evaluate(COMMUTE)
    dfa.sparse_dfa_to_file(str(path))
    print(f"  commute automaton: {dfa.num_states} states, tapes={variables} "
          f"in {time.time()-t0:.1f}s")
    return dfa, variables


def multitape_encoder(dfa, variables):
    """Vectorized convolution encoder: a dict {tape -> symbol list} (all equal
    length) to the automaton's internal per-position symbol codes."""
    base = sorted(dfa.base_alphabet_frozen)
    m = len(base)
    idx = {s: i for i, s in enumerate(base)}
    k = len(variables)
    powers = [m ** (k - 1 - j) for j in range(k)]

    def enc(cols):
        length = len(cols[variables[0]])
        total = np.zeros(length, dtype=np.int64)
        for j, v in enumerate(variables):
            digits = np.fromiter((idx[s] for s in cols[v]), np.int64, length)
            total += digits * powers[j]
        return total
    return enc


def commutes(p, x, y):
    _, ax, bx = x
    _, ay, by = y
    return (sum(ax[i] * by[i] for i in range(len(ax))) % p ==
            sum(ay[i] * bx[i] for i in range(len(ax))) % p)


def scale(heis, n):
    """A rank-(n-1) group with a fixed non-commuting pair (x has a = e1, y has
    b = e1), so the advice word has length ~n and the answer is False."""
    r = max(1, n - 1)
    e1 = (1,) + (0,) * (r - 1)
    zero = (0,) * r
    x = (0, e1, zero)
    y = (0, zero, e1)
    return {"advice": heis.advice(r),
            "x": heis.encode(x, r), "y": heis.encode(y, r)}


def batch_gen(heis, p, block, rng):
    """Random element pair in a rank-`block` group; answer = do they commute."""
    r = block
    def elem():
        return (rng.randrange(p),
                tuple(rng.randrange(p) for _ in range(r)),
                tuple(rng.randrange(p) for _ in range(r)))
    x, y = elem(), elem()
    return ({"advice": heis.advice(r), "x": heis.encode(x, r),
             "y": heis.encode(y, r)}, commutes(p, x, y))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-exp", type=int, default=6)
    ap.add_argument("--batch", type=int, default=50000)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("-p", type=int, default=3, help="the prime (default 3)")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--formats", default="svg,pdf")
    args = ap.parse_args()

    print(f"Extraspecial {args.p}-groups — property: do two elements commute?")
    heis = ExtraspecialGroups(args.p)
    dfa, variables = build_or_load(heis, args.p)
    enc = multitape_encoder(dfa, variables)

    label = "commutativity"
    bench.run_scaling(dfa, enc, lambda n: scale(heis, n), args.max_exp, label)
    bench.run_batch(dfa, enc, lambda b, rng: batch_gen(heis, args.p, b, rng),
                    args.batch, label)
    sizes = [n for n in SIZES if n <= 10 ** args.max_exp]
    data = bench.run_curve(dfa, enc, lambda n: scale(heis, n), sizes,
                           args.reps, label)
    bench.draw({label: data}, Path(args.out_dir), args.formats.split(","),
               f"Deciding commutativity in extraspecial {args.p}-groups",
               "extraspecial_commute_curve")


if __name__ == "__main__":
    main()
