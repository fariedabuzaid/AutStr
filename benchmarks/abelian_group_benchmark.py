#!/usr/bin/env python
"""Benchmark a first-order query on the uniformly automatic class of all finite
abelian groups.

Property: **the group has an element of order 2** (equivalently, its order is
even, i.e. some cyclic factor has even order). This is the algebraic analogue of
bipartiteness for graphs — a clean structural yes/no invariant that a *single*
automaton decides for every finite abelian group by reading the group's advice
word (its cyclic decomposition, as LSB-first binary blocks separated by '#').

    even_order := exists x. (x != 0  and  x + x = 0)

Because the advice alphabet has only four symbols, this compiles quickly; a
group with k cyclic factors has an advice word of length ~3k, so "large" here
means a decomposition into millions of factors (the group order itself is
astronomically large — 2^a * 3^b — yet the invariant is decided in time linear
in the number of factors).

Usage:
    pip install "autstr[jax,benchmarks]"
    python abelian_group_benchmark.py [--max-exp 6] [--batch 50000]
"""
import argparse
import os
import time
from pathlib import Path

from autstr.algebra import FiniteAbelianGroups
from autstr.sparse_automata import SparseDFA
import _bench_common as bench

CACHE = Path(os.environ.get("AUTSTR_BENCH_CACHE", "/tmp/autstr-bench-cache"))

# x != 0  and  exists z. (x + x = z and z = 0);  0 is the unique w with w+w=w.
EVEN_ORDER = ('exists x.((not A(x,x,x)) and '
              '(exists z.(A(x,x,z) and A(z,z,z))))')

SIZES = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000,
         200_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000]


def build_or_load(ab):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "abelian_even_order.sdfa"
    if path.exists():
        dfa = SparseDFA.sparse_dfa_from_file(str(path))
        print(f"  even-order automaton: {dfa.num_states} states (cached)")
        return dfa
    print("  even-order: compiling (one-time)...", flush=True)
    t0 = time.time()
    dfa, _ = ab.cls.evaluate(EVEN_ORDER)
    dfa.sparse_dfa_to_file(str(path))
    print(f"  even-order automaton: {dfa.num_states} states in {time.time()-t0:.1f}s")
    return dfa


# ---- advice generators (return the group's advice symbols) ----

def scale(ab, n):
    """A group Z_2^k with an advice word of length ~n (k ~ n/3 factors);
    it has even order, so the query holds."""
    k = max(1, n // 3)
    return ab.advice([2] * k)


def batch_gen(ab, block, rng):
    """`block` cyclic factors of order 2 or 3 (both 3 advice symbols, so the
    length is constant). A coin decides the answer: odd-order groups use only
    Z_3, even-order ones plant one Z_2. ~50/50 yes/no."""
    even = rng.random() < 0.5
    orders = [3] * block
    if even:
        orders[rng.randrange(block)] = 2
    return ab.advice(orders), even


def main():
    ap = bench.parser(__doc__)
    args = ap.parse_args()
    cfg = bench.settings(args)

    print("Finite abelian groups — property: even order (has an element of order 2)")
    ab = FiniteAbelianGroups()
    dfa = build_or_load(ab)
    enc = bench.encoder(dfa)

    bench.run_scaling(dfa, enc, lambda n: scale(ab, n), cfg['max_exp'],
                      "even order")
    bench.run_batch(dfa, enc, lambda b, rng: batch_gen(ab, b, rng),
                    cfg['batch'], "even order")

    sizes = [n for n in SIZES if n <= 10 ** cfg['max_exp']]
    data = bench.run_curve(dfa, enc, lambda n: scale(ab, n), sizes,
                           cfg['reps'], "even order")
    if cfg["plot"]:
        bench.draw({"even order": data}, Path(args.out_dir), args.formats.split(","),
                   "Deciding even order across all finite abelian groups",
                   "abelian_even_order_curve")


if __name__ == "__main__":
    main()
