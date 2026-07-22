import itertools as it
import random

import numpy as np
import pytest

from autstr.utils.tree_automata_tools import _shortlex_key, iterate_trees
from autstr.sparse_tree_automata import (
    SparseTreeAutomaton, Tree, convolve_trees, tree_to_arrays,
)


# ====================================================================
# Reference implementation (exhaustive dict-based DTA)
# ====================================================================

class RefDTA:
    def __init__(self, sta: SparseTreeAutomaton):
        self.n = sta.num_states
        self.BOT = sta.num_states
        self.default = sta.default_state
        self.acc = sta.is_accepting.copy()
        self.num_symbols = sta.num_symbols
        self.table = sta.dense_delta()

    def delta(self, l, r, s):
        return int(self.table[l, r, s])

    def run(self, tree: Tree) -> int:
        # iterative post-order
        stack, out = [(tree, False)], {}
        while stack:
            node, expanded = stack.pop()
            if node is None:
                continue
            if not expanded:
                stack.append((node, True))
                stack.append((node.right, False))
                stack.append((node.left, False))
            else:
                l = out[id(node.left)] if node.left is not None else self.BOT
                r = out[id(node.right)] if node.right is not None else self.BOT
                out[id(node)] = self.delta(l, r, node.label)
        return out[id(tree)]

    def accepts(self, tree: Tree) -> bool:
        return bool(self.acc[self.run(tree)])

    def reachable(self):
        """Exact reachability fixpoint by exhaustive enumeration."""
        avail = {self.BOT}
        changed = True
        while changed:
            changed = False
            for l in list(avail):
                for r in list(avail):
                    for s in range(self.num_symbols):
                        t = self.delta(l, r, s)
                        if t not in avail:
                            avail.add(t)
                            changed = True
        avail.discard(self.BOT)
        return avail

    def is_empty(self):
        return not any(self.acc[q] for q in self.reachable())


# ====================================================================
# Generators
# ====================================================================

def random_spec(rng: random.Random, max_states=5, max_symbols=4,
                max_exc=12, max_pd=6):
    """A random transition function in the flat three-tier form: a global
    default, pair defaults, and (left, right, symbol) exceptions."""
    n = rng.randint(1, max_states)
    m = rng.randint(2, max_symbols)
    k = rng.randint(0, min(max_exc, (n + 1) * (n + 1) * m))
    triples = set()
    while len(triples) < k:
        triples.add((rng.randint(0, n), rng.randint(0, n), rng.randrange(m)))
    kp = rng.randint(0, min(max_pd, (n + 1) * (n + 1)))
    pairs = set()
    while len(pairs) < kp:
        pairs.add((rng.randint(0, n), rng.randint(0, n)))
    return {
        'n': n, 'm': m,
        'default': rng.randrange(n),
        'exc': {t: rng.randrange(n) for t in sorted(triples)},
        'pd': {p: rng.randrange(n) for p in sorted(pairs)},
        'acc': [rng.random() < 0.5 for _ in range(n)],
    }


def sta_of_spec(spec) -> SparseTreeAutomaton:
    exc = sorted(spec['exc'])
    pd = sorted(spec['pd'])
    return SparseTreeAutomaton(
        num_states=spec['n'], default_state=spec['default'],
        exc_left=[t[0] for t in exc], exc_right=[t[1] for t in exc],
        exc_symbol=[t[2] for t in exc],
        exc_target=[spec['exc'][t] for t in exc],
        is_accepting=spec['acc'], symbol_arity=1,
        base_alphabet=set(range(spec['m'])),
        pd_left=[p[0] for p in pd], pd_right=[p[1] for p in pd],
        pd_target=[spec['pd'][p] for p in pd])


def random_sta(rng: random.Random, max_states=5, max_symbols=4,
               max_exc=12, max_pd=6) -> SparseTreeAutomaton:
    n = rng.randint(1, max_states)
    m = rng.randint(2, max_symbols)
    k = rng.randint(0, min(max_exc, (n + 1) * (n + 1) * m))
    triples = set()
    while len(triples) < k:
        triples.add((rng.randint(0, n), rng.randint(0, n), rng.randrange(m)))
    triples = sorted(triples)
    kp = rng.randint(0, min(max_pd, (n + 1) * (n + 1)))
    pairs = set()
    while len(pairs) < kp:
        pairs.add((rng.randint(0, n), rng.randint(0, n)))
    pairs = sorted(pairs)
    return SparseTreeAutomaton(
        num_states=n,
        default_state=rng.randrange(n),
        exc_left=[t[0] for t in triples],
        exc_right=[t[1] for t in triples],
        exc_symbol=[t[2] for t in triples],
        exc_target=[rng.randrange(n) for _ in triples],
        is_accepting=[rng.random() < 0.5 for _ in range(n)],
        symbol_arity=1,
        base_alphabet=set(range(m)),
        pd_left=[p[0] for p in pairs],
        pd_right=[p[1] for p in pairs],
        pd_target=[rng.randrange(n) for _ in pairs],
    )


def random_tree(rng: random.Random, m: int, max_size=12) -> Tree:
    size = rng.randint(1, max_size)

    def build(budget):
        label = rng.randrange(m)
        if budget <= 1:
            return Tree(label), 1
        shape = rng.random()
        used = 1
        left = right = None
        if shape < 0.8:                      # left child
            left, u = build(budget - 1)
            used += u
        if shape > 0.2 and budget - used >= 1:   # right child
            right, u = build(budget - used)
            used += u
        return Tree(label, left, right), used

    return build(size)[0]


# ====================================================================
# Tests
# ====================================================================

class TestCompilation:
    def test_diagrams_agree_with_the_flat_spec(self):
        """Compiling the three-tier flat form into one diagram per child pair
        preserves delta pointwise."""
        rng = random.Random(17)
        for trial in range(60):
            spec = random_spec(rng)
            sta = sta_of_spec(spec)
            table = sta.dense_delta()
            n, m = spec['n'], spec['m']
            for l in range(n + 1):
                for r in range(n + 1):
                    for s in range(m):
                        want = spec['exc'].get((l, r, s))
                        if want is None:
                            want = spec['pd'].get((l, r), spec['default'])
                        assert table[l, r, s] == want, (trial, l, r, s)

    def test_equal_transition_functions_share_a_diagram(self):
        """Hash-consing: two states with the same behavior on a child pair
        resolve to the same node."""
        alpha = {0, 1, 2}
        sta = SparseTreeAutomaton(
            3, 0,
            exc_left=[3, 3, 3, 3], exc_right=[3, 3, 3, 3],
            exc_symbol=[0, 1, 0, 1], exc_target=[1, 2, 1, 2],
            is_accepting=[False, True, True], base_alphabet=alpha,
            pd_left=[3], pd_right=[3], pd_target=[0])
        two = SparseTreeAutomaton(
            3, 0,
            exc_left=[3, 3], exc_right=[3, 3],
            exc_symbol=[0, 1], exc_target=[1, 2],
            is_accepting=[False, True, True], base_alphabet=alpha)
        assert list(sta.pair_nodes) == list(two.pair_nodes)


class TestRunAgainstReference:
    def test_random_trees(self):
        rng = random.Random(0)
        for trial in range(40):
            sta = random_sta(rng)
            ref = RefDTA(sta)
            m = len(sta.base_alphabet)
            for _ in range(25):
                t = random_tree(rng, m)
                arrays = tree_to_arrays(t, sta.base_alphabet_frozen)
                assert sta.run(*arrays) == ref.run(t), (trial, t)
                assert sta.accepts(t) == ref.accepts(t)

    def test_single_node_trees(self):
        rng = random.Random(1)
        sta = random_sta(rng)
        ref = RefDTA(sta)
        for s in range(len(sta.base_alphabet)):
            assert sta.accepts(Tree(s)) == ref.accepts(Tree(s))


class TestBooleanOperations:
    def test_product_and_complement_semantics(self):
        rng = random.Random(2)
        for trial in range(25):
            m = rng.randint(2, 4)
            alpha = set(range(m))

            def gen():
                while True:
                    a = random_sta(rng, max_symbols=m)
                    if len(a.base_alphabet) == m:
                        return a

            A, B = gen(), gen()
            inter = A.intersection(B)
            uni = A.union(B)
            comp = A.complement()
            for _ in range(25):
                t = random_tree(rng, m)
                ra, rb = A.accepts(t), B.accepts(t)
                assert inter.accepts(t) == (ra and rb), (trial, t)
                assert uni.accepts(t) == (ra or rb), (trial, t)
                assert comp.accepts(t) == (not ra), (trial, t)

    def test_product_is_sparse(self):
        """Default x default composes: only child pairs whose factors both
        deviate from their defaults enter the product's pair table, and the
        product's diagrams stay within the apply bound."""
        rng = random.Random(3)
        A = random_sta(rng, max_states=4, max_symbols=3, max_exc=6, max_pd=0)
        B = random_sta(rng, max_states=4, max_symbols=3, max_exc=6, max_pd=0)
        while len(B.base_alphabet) != len(A.base_alphabet):
            B = random_sta(rng, max_states=4, max_symbols=3, max_exc=6,
                           max_pd=0)
        prod = A.intersection(B)
        assert prod.num_nodes <= A.num_nodes * B.num_nodes
        assert len(prod.pair_keys) <= (prod.num_states + 1) ** 2


class TestEmptiness:
    def test_against_exhaustive_fixpoint(self):
        rng = random.Random(4)
        for trial in range(60):
            sta = random_sta(rng)
            ref = RefDTA(sta)
            assert sta.is_empty() == ref.is_empty(), trial
            assert set(np.flatnonzero(sta.reachable_states())) == \
                ref.reachable(), trial

    def test_empty_and_universal(self):
        alpha = {0, 1}
        never = SparseTreeAutomaton(1, 0, [], [], [], [], [False],
                                    base_alphabet=alpha)
        always = SparseTreeAutomaton(1, 0, [], [], [], [], [True],
                                     base_alphabet=alpha)
        assert never.is_empty()
        assert not always.is_empty()
        assert not never.complement().is_empty()

    def test_emptiness_through_products(self):
        """A n ~A is empty; A u ~A is universal (non-empty)."""
        rng = random.Random(5)
        for _ in range(15):
            A = random_sta(rng)
            assert A.intersection(A.complement()).is_empty()
            assert not A.union(A.complement()).is_empty()


class TestConvolution:
    def test_convolve_and_padding(self):
        alpha = {'*', 'a', 'b'}
        t1 = Tree('a', Tree('b'), None)
        t2 = Tree('b', Tree('a', Tree('b')), Tree('a'))
        conv = convolve_trees([t1, t2], alpha, '*')
        assert conv.label == ('a', 'b')
        assert conv.left.label == ('b', 'a')
        assert conv.left.left.label == ('*', 'b')   # t1 absent below
        assert conv.right.label == ('*', 'a')

    def test_arrays_roundtrip_postorder(self):
        alpha = {0, 1}
        t = Tree(1, Tree(0, None, Tree(1)), Tree(0))
        labels, lefts, rights = tree_to_arrays(t, alpha)
        assert len(labels) == 4
        # post-order: children indices precede parents
        for i, (l, r) in enumerate(zip(lefts, rights)):
            assert l < i and r < i


# ====================================================================
# Finiteness against an exhaustive oracle
# ====================================================================

def _trees_up_to_height(m: int, height: int):
    """Every tree of height <= `height` over `m` labels, where the height is
    the number of nodes on the longest root-to-leaf path. `None` (the absent
    tree) is included so it can serve as a child."""
    level = [None]
    for _ in range(height):
        level = [None] + [Tree(label, left, right)
                          for label in range(m)
                          for left in level
                          for right in level]
    return level


def _height(tree) -> int:
    if tree is None:
        return 0
    best, stack = 0, [(tree, 1)]
    while stack:
        node, depth = stack.pop()
        best = max(best, depth)
        for child in (node.left, node.right):
            if child is not None:
                stack.append((child, depth + 1))
    return best


def _accepts_a_tall_tree(sta: SparseTreeAutomaton, m: int) -> bool:
    """Whether some accepted tree is taller than the state count.

    Such a tree repeats a state along one root-to-leaf path, so the context
    between the two occurrences pumps and the language is infinite.
    Conversely an infinite language has trees of unbounded size, and a binary
    tree's size is bounded by its height -- so this is exact, not a heuristic.
    """
    n = sta.num_states
    return any(tree is not None and _height(tree) > n and sta.accepts(tree)
               for tree in _trees_up_to_height(m, n + 1))


class TestFiniteness:
    def test_matches_the_exhaustive_oracle(self):
        rng = random.Random(20260721)
        for _ in range(60):
            sta = random_sta(rng, max_states=2, max_symbols=2,
                             max_exc=8, max_pd=4)
            m = len(sta.base_alphabet)
            assert sta.is_finite() is not _accepts_a_tall_tree(sta, m), sta

    def test_empty_language_is_finite(self):
        sta = SparseTreeAutomaton(1, 0, is_accepting=[False], symbol_arity=1,
                                  base_alphabet={0, 1})
        assert sta.is_finite()

    def test_every_tree_is_infinite(self):
        sta = SparseTreeAutomaton(1, 0, is_accepting=[True], symbol_arity=1,
                                  base_alphabet={0, 1})
        assert not sta.is_finite()


# ====================================================================
# Enumeration against an exhaustive oracle
# ====================================================================

def _trees_up_to_size(m: int, max_size: int):
    """All trees of each size up to `max_size` over `m` labels."""
    by_size = {0: [None]}
    for size in range(1, max_size + 1):
        by_size[size] = [
            Tree(label, left, right)
            for left_size in range(size)
            for label in range(m)
            for left in by_size[left_size]
            for right in by_size[size - 1 - left_size]
        ]
    return by_size


class TestEnumeration:
    def test_matches_brute_force(self):
        rng = random.Random(4242)
        limit = 4
        for _ in range(25):
            sta = random_sta(rng, max_states=3, max_symbols=2,
                             max_exc=8, max_pd=4)
            m = len(sta.base_alphabet)
            by_size = _trees_up_to_size(m, limit)
            expected = [t for size in range(1, limit + 1)
                        for t in by_size[size] if sta.accepts(t)]

            got = []
            for tree in iterate_trees(sta):
                if tree.size() > limit:
                    break
                got.append(tree)

            assert (sorted(map(_shortlex_key, got))
                    == sorted(map(_shortlex_key, expected))), sta

    def test_yields_sizes_in_order(self):
        rng = random.Random(99)
        for _ in range(10):
            sta = random_sta(rng, max_states=3, max_symbols=2,
                             max_exc=8, max_pd=4)
            sizes = []
            for tree in iterate_trees(sta):
                sizes.append(tree.size())
                if len(sizes) > 30:
                    break
            assert sizes == sorted(sizes), sta

    def test_finite_language_terminates(self):
        """Only the single leaf labelled 0 is accepted, so the generator must
        stop rather than climb sizes forever looking for more."""
        BOT = 2                                   # = num_states
        sta = SparseTreeAutomaton(
            2, 1,
            exc_left=[BOT], exc_right=[BOT], exc_symbol=[0], exc_target=[0],
            is_accepting=[True, False], symbol_arity=1, base_alphabet={0, 1})
        assert [t.label for t in iterate_trees(sta)] == [0]
