"""Uniformly *tree*-automatic classes: bounded clique-width and bounded
tree-width.

An MSO query is compiled once for the whole class, and then decides the
property on any member graph in time linear in its advice tree. This is
Courcelle's theorem with the automaton actually built.

The two classes trade off in opposite directions. Clique-width has a small
advice alphabet and a cheap edge automaton, so its queries compile in seconds.
Tree-width has a wider alphabet and an edge automaton that carries pending
registers, so the same query costs far more to compile — but decides just as
fast afterwards.

    light (default)   clique-width 2: two-colourability
                      tree-width 1:   connectedness
    heavy             adds tree-width 1: two-colourability (a ~1 min compile)

Advice trees are generated directly, never through networkx, so the sizes here
are limited by memory rather than by graph construction.
"""
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bench_common as bench

from autstr.sparse_tree_automata import Tree
from autstr.tree_graphs import CliqueWidthClass, TreeWidthClass

CW_TWO_COL = ('exists c.(all x.(all y.(E(x,y) -> '
              '(not ((Subset(x,c) and Subset(y,c)) or '
              '((not Subset(x,c)) and (not Subset(y,c))))))))')
TW_CONNECTED = ('all c.(((exists x.(Sing(x) and Subset(x,c))) and '
                '(all x.(all y.((Subset(x,c) and E(x,y)) -> Subset(y,c))))) '
                '-> (all x.(Sing(x) -> Subset(x,c))))')
TW_TWO_COL = CW_TWO_COL

SIZES = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]


# ------------------------------------------------------------------ advice
#
# The k-expression of K_n and of K_{a,b}, built straight as a tree. K_n is
# 2-colourable only for n <= 2; K_{a,b} always is.

def clique_expression(n: int) -> Tree:
    tree = Tree('v0')
    for _ in range(n - 1):
        tree = Tree('r10', Tree('e01', Tree('u', tree, Tree('v1'))))
    return tree


def bipartite_expression(left: int, right: int) -> Tree:
    tree = Tree('v0')
    for _ in range(left - 1):
        tree = Tree('u', tree, Tree('v0'))
    for _ in range(right):
        tree = Tree('u', tree, Tree('v1'))
    return Tree('e01', tree)


def tw_path_expression(n: int, disconnect_at: int = -1) -> Tree:
    """A width-1 layout of the path on n vertices: each vertex's profile names
    the register of its predecessor. A vertex with an empty profile splits the
    path into two components."""
    letters = []
    for i in range(n):
        register = i % 2
        if i == 0 or i == disconnect_at:
            letters.append(f"r{register}s")           # no earlier neighbour
        else:
            letters.append(f"r{register}s{1 - register}")
    tree = Tree(letters[-1])
    for letter in reversed(letters[:-1]):
        tree = Tree(letter, tree, None)               # a left-deep chain
    return tree


def cw_scale(n: int) -> Tree:
    return clique_expression(max(n // 3, 2))          # ~3 nodes per vertex


def tw_scale(n: int) -> Tree:
    return tw_path_expression(n)


def cw_batch(rng: random.Random, vertices: int):
    """A random member with a known answer: a clique (not 2-colourable) or a
    complete bipartite graph (always 2-colourable)."""
    if rng.random() < 0.5:
        return clique_expression(vertices), False
    half = max(vertices // 2, 1)
    return bipartite_expression(half, vertices - half), True


# ------------------------------------------------------------------ running

def compile_query(cls, phi, label):
    t0 = time.time()
    sta, tapes = cls.evaluate(phi)
    print(f"  {label}: {sta.num_states} states, {sta.num_nodes} diagram nodes, "
          f"compiled in {time.time() - t0:.1f}s (tapes={tapes})")
    return sta


def check_batch(sta, rng, count, vertices, label):
    print(f"\n== Correctness on {count} random members ({label}) ==")
    wrong = 0
    for _ in range(count):
        advice, truth = cw_batch(rng, vertices)
        if bench.tree_decide(sta, bench.tree_arrays(sta, advice)) != truth:
            wrong += 1
    print(f"  {count - wrong}/{count} agree with the ground truth")
    assert wrong == 0, "the compiled automaton disagrees with the ground truth"


def main():
    ap = bench.parser(__doc__)
    args = ap.parse_args()
    cfg = bench.settings(args)
    sizes = [n for n in SIZES if n <= 10 ** cfg["max_exp"]]
    rng = random.Random(20260710)

    print("\nClique-width <= 2")
    cw = CliqueWidthClass(2)
    two_col = compile_query(cw, CW_TWO_COL, "two-colourability")
    bench.run_tree_scaling(two_col, cw_scale, sizes,
                           "two-colourability over clique-width 2",
                           reps=cfg["reps"])
    check_batch(two_col, rng, min(cfg["batch"] // 20 or 1, 200), 40,
                "clique-width 2")

    print("\nTree-width <= 1")
    tw = TreeWidthClass(1)
    connected = compile_query(tw, TW_CONNECTED, "connectedness")
    bench.run_tree_scaling(connected, tw_scale, sizes,
                           "connectedness over tree-width 1",
                           reps=cfg["reps"])
    for n, cut in ((501, -1), (501, 250)):
        arrays = bench.tree_arrays(connected, tw_path_expression(n, cut))
        answer = bench.tree_decide(connected, arrays)
        expected = cut < 0
        print(f"  path of {n} vertices, split at {cut}: connected={answer} "
              f"(expected {expected})")
        assert answer == expected

    if not cfg["heavy"]:
        print("\n== Tree-width 1: two-colourability ==")
        print("  Skipped: about a minute to compile. --profile heavy runs it.")
        return

    two_col_tw = compile_query(tw, TW_TWO_COL, "two-colourability")
    bench.run_tree_scaling(two_col_tw, tw_scale, sizes,
                           "two-colourability over tree-width 1",
                           reps=cfg["reps"])
    arrays = bench.tree_arrays(two_col_tw, tw_path_expression(101))
    print(f"  a path is bipartite: {bench.tree_decide(two_col_tw, arrays)}")


if __name__ == "__main__":
    main()
