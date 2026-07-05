#!/usr/bin/env python
"""Benchmark MSO queries on very large tree-depth-4 graphs via a uniformly
automatic class.

The point of a uniformly automatic class is that a monadic second-order query
is compiled into a finite automaton **once** for the whole class; deciding the
query on a concrete graph is then a single linear pass of its advice word
through that automaton — independent of how the automaton was built and cheap
enough to run on graphs with millions of vertices, and trivially batchable.

Queries on `TreeDepthClass(4)` (all graphs of tree-depth at most 4):

* **connectedness** (P-time) — a width-4 formula; ~14 s to an 11-state
  automaton.
* **2-colourability** / bipartiteness (P-time) — width 4; ~17 s to 16 states.
* **3-colourability** (NP-complete) — the minimal NP-hard MSO query (two
  colour-set tapes + two edge-endpoint tapes = width 5) and non-trivial only
  at tree-depth >= 4, since the chromatic number is bounded by the tree-depth
  and K4 is the smallest obstruction. Its automaton is a large one-time
  compile (> 6 GB peak); see *Compiling 3-colourability* in the README.

Usage:
    python uniform_graph_benchmark.py [--max-exp 6] [--batch 50000]

Compiled automata are serialized under $AUTSTR_BENCH_CACHE (default a temp
directory) so repeated runs skip compilation.
"""
import argparse
import os
import time
from pathlib import Path

from autstr.graphs import TreeDepthClass
from autstr.sparse_automata import SparseDFA
from autstr.utils.misc import encode_symbol
import autstr.sparse_automata as _sa

CACHE = Path(os.environ.get("AUTSTR_BENCH_CACHE", "/tmp/autstr-bench-cache"))


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

CONNECTED = (
    'not (exists s.('
    '(exists x.(Sing(x) and Subset(x,s))) and '
    '(exists y.(Sing(y) and (not Subset(y,s)))) and '
    '(all x.(all y.((not E(x,y)) or ('
    '(Subset(x,s) and Subset(y,s)) or ((not Subset(x,s)) and (not Subset(y,s)))'
    '))))'
    '))'
)

TWO_COL = (
    'exists c.(all x.(all y.((not E(x,y)) or '
    '((Subset(x,c) and (not Subset(y,c))) or '
    '((not Subset(x,c)) and Subset(y,c))))))'
)

# 3-colouring with two set variables: colour(v) = (v in a, v in b) drawn from
# {(0,0),(0,1),(1,0)} — (1,1) is forbidden, giving three colours.
_XA = '((Subset(x,a) and (not Subset(y,a))) or ((not Subset(x,a)) and Subset(y,a)))'
_XB = '((Subset(x,b) and (not Subset(y,b))) or ((not Subset(x,b)) and Subset(y,b)))'
THREE_COL = (
    'exists a.(exists b.('
    f'(all x.((not Sing(x)) or (not (Subset(x,a) and Subset(x,b))))) and '
    f'(all x.(all y.((not E(x,y)) or ({_XA} or {_XB}))))'
    '))'
)


# ---------------------------------------------------------------------------
# Advice generators — emitted directly as (depth, profile) letters so we never
# build a networkx graph and can scale to millions of vertices instantly.
# Concatenating per-component letter lists is a valid forest traversal.
# ---------------------------------------------------------------------------

def windmill(k):
    """k triangles sharing a centre: non-bipartite, connected. N = 2k+1."""
    return [(1, ())] + k * [(2, (1,)), (3, (1, 1))]


def star(n):
    """A star on n vertices: bipartite, connected, tree-depth 2."""
    return [(1, ())] + (n - 1) * [(2, (1,))]


def k4():
    """A single K4 component: tree-depth 4, chromatic number 4."""
    return [(1, ()), (2, (1,)), (3, (1, 1)), (4, (1, 1, 1))]


def mixture(k, rng):
    """Centre with k two-vertex blocks (triangle or path each), N = 2k+1. A
    coin decides the answer: bipartite graphs use only paths, non-bipartite
    ones plant at least one triangle. ~50/50 yes/no at constant length."""
    bipartite = rng.random() < 0.5
    letters = [(1, ())]
    for i in range(k):
        triangle = (not bipartite) and (i == 0 or rng.random() < 0.5)
        letters += [(2, (1,)), (3, (1, 1) if triangle else (0, 1))]
    return letters, bipartite


def conn_mixture(k, rng):
    """Either one star (connected) or two stars (disconnected) on the same
    N = 2k+1 vertices. ~50/50 yes/no at constant length."""
    n = 2 * k + 1
    if rng.random() < 0.5:
        return star(n), True
    a = n // 2
    return star(a) + star(n - a), False


def three_col_mixture(k, rng):
    """A K4 or a bipartite P4 prefix (both 4 vertices) plus a windmill tail;
    3-colourable iff the prefix is the P4. Constant length for fixed k."""
    if rng.random() < 0.5:
        return k4() + windmill(k), False
    p4 = [(1, ()), (2, (1,)), (3, (0, 1)), (4, (0, 0, 1))]
    return p4 + windmill(k), True


QUERIES = {
    "connected": dict(name="connected_td4", formula=CONNECTED,
                      label="connectedness",
                      scale=lambda n: star(n), batch=conn_mixture),
    "2col": dict(name="2col_td4", formula=TWO_COL,
                 label="2-colourability",
                 scale=lambda n: windmill((n - 1) // 2), batch=mixture),
    "3col": dict(name="3col_td4", formula=THREE_COL,
                 label="3-colourability",
                 scale=lambda n: windmill((n - 1) // 2), batch=three_col_mixture),
}


# ---------------------------------------------------------------------------

def build_or_load(cls, spec, allow_compile=True):
    """Load a query automaton from cache, or compile and cache it."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{spec['name']}.sdfa"
    if path.exists():
        dfa = SparseDFA.sparse_dfa_from_file(str(path))
        print(f"  {spec['label']}: {dfa.num_states}-state automaton (cached)")
        return dfa
    if not allow_compile:
        return None
    print(f"  {spec['label']}: compiling (one-time)...", flush=True)
    t0 = time.time()
    dfa, _ = cls.evaluate(spec["formula"])
    dfa.sparse_dfa_to_file(str(path))
    print(f"  {spec['label']}: {dfa.num_states} states in {time.time()-t0:.1f}s")
    return dfa


def symbol_codes(cls, dfa):
    """Map each advice letter to the automaton's internal symbol code, once."""
    return {letter: encode_symbol((sym,), dfa.base_alphabet_frozen)
            for letter, sym in cls.symbol_of.items()}


def encode_letters(codes, letters):
    """Pre-encode a letter list to an int array (the timed batch API path)."""
    import numpy as np
    return np.fromiter((codes[l] for l in letters), dtype=np.int64,
                       count=len(letters))


def advice_word(cls, letters):
    """Single-tape word (list of 1-tuples) for a sentence query."""
    return [(cls.symbol_of[l],) for l in letters]


def run_single_scaling(cls, dfa, spec, max_exp):
    print(f"\n== Single-graph scaling: {spec['label']} (one huge graph, NumPy) ==")
    print(f"{'vertices':>12} {'build(s)':>10} {'decide(s)':>10} {'Mverts/s':>10}  answer")
    codes = symbol_codes(cls, dfa)
    for exp in range(3, max_exp + 1):
        n = 10 ** exp
        t0 = time.time()
        arr = encode_letters(codes, spec["scale"](n))
        t_build = time.time() - t0
        t0 = time.time()
        ans = bool(dfa.is_accepting[dfa.compute(arr)])
        t_decide = time.time() - t0
        print(f"{len(arr):>12,} {t_build:>10.3f} {t_decide:>10.3f} "
              f"{len(arr)/t_decide/1e6:>10.2f}  {ans}")

    if _sa._HAS_JAX:
        n = 10 ** max_exp
        arr = encode_letters(codes, spec["scale"](n))
        dfa.accepts_batch(arr[None, :])           # warm up JIT
        t0 = time.time()
        r = bool(dfa.accepts_batch(arr[None, :])[0])
        t_jx = time.time() - t0
        print(f"  same {len(arr):,}-vertex graph via JAX scan: "
              f"{t_jx*1000:.1f} ms ({len(arr)/t_jx/1e6:.1f} Mverts/s), answer={r}")


def run_batch(cls, dfa, spec, batch, block=200):
    import numpy as np
    print(f"\n== Batched throughput: {batch:,} graphs, {spec['label']} ==")
    codes = symbol_codes(cls, dfa)
    rng = __import__("random").Random(0)
    rows, truth = [], []
    for _ in range(batch):
        letters, ans = spec["batch"](block, rng)
        rows.append(encode_letters(codes, letters))
        truth.append(ans)
    words = np.stack(rows)
    length, steps = words.shape[1], words.size
    truth = np.array(truth)

    t0 = time.time()
    loop = np.array([bool(dfa.is_accepting[dfa.compute(w)]) for w in words])
    t_loop = time.time() - t0

    saved = _sa._HAS_JAX
    _sa._HAS_JAX = False
    np_batch = dfa.accepts_batch(words)
    t0 = time.time()
    np_batch = dfa.accepts_batch(words)
    t_np = time.time() - t0
    _sa._HAS_JAX = saved

    print(f"  {batch:,} graphs x {length} vertices "
          f"({int(truth.sum())} satisfy the query)")
    print(f"  NumPy loop : {t_loop*1000:8.1f} ms  ({steps/t_loop/1e6:7.2f} Mverts/s)")
    print(f"  NumPy batch: {t_np*1000:8.1f} ms  ({steps/t_np/1e6:7.2f} Mverts/s)"
          f"  speedup {t_loop/t_np:5.1f}x")
    if saved:
        dfa.accepts_batch(words)
        t0 = time.time()
        jx = dfa.accepts_batch(words)
        t_jx = time.time() - t0
        assert np.array_equal(np_batch, jx)
        print(f"  JAX batch  : {t_jx*1000:8.1f} ms  ({steps/t_jx/1e6:7.2f} Mverts/s)"
              f"  speedup {t_loop/t_jx:5.1f}x")
    else:
        print("  JAX batch  : (jax not installed — pip install autstr[jax])")
    assert np.array_equal(loop, np_batch)
    assert int((loop == truth).sum()) == batch, "disagreement with ground truth"
    print(f"  correctness: all {batch:,} match ground truth and each other")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-exp", type=int, default=6)
    ap.add_argument("--batch", type=int, default=50000)
    args = ap.parse_args()

    print("Query automata for TreeDepthClass(4):")
    cls = TreeDepthClass(4)
    feasible = [build_or_load(cls, QUERIES[q]) for q in ("connected", "2col")]

    for dfa, key in zip(feasible, ("connected", "2col")):
        spec = QUERIES[key]
        run_single_scaling(cls, dfa, spec, args.max_exp)
        run_batch(cls, dfa, spec, args.batch)

    three = build_or_load(cls, QUERIES["3col"], allow_compile=False)
    print("\n== 3-colourability ==")
    if three is not None:
        run_single_scaling(cls, three, QUERIES["3col"], args.max_exp)
        run_batch(cls, three, QUERIES["3col"], args.batch)
    else:
        print("  The 3-col automaton is a large one-time compile (>6 GB peak) and")
        print(f"  is not cached at {CACHE/'3col_td4.sdfa'}. Build it once on a")
        print("  larger machine (see benchmarks/README.md) and drop the file in;")
        print("  deciding 3-colourability on huge graphs is then as cheap as above.")


if __name__ == "__main__":
    main()
