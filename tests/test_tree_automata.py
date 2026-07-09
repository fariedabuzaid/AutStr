import itertools as it
import random

import numpy as np
import pytest

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
        self.table = {
            (int(l), int(r), int(s)): int(t)
            for l, r, s, t in zip(sta.exc_left, sta.exc_right,
                                  sta.exc_symbol, sta.exc_target)
        }
        self.pd = {
            (int(l), int(r)): int(t)
            for l, r, t in zip(sta.pd_left, sta.pd_right, sta.pd_target)
        }
        self.acc = sta.is_accepting.copy()
        self.num_symbols = sta.num_symbols

    def delta(self, l, r, s):
        key = (l, r, s)
        if key in self.table:
            return self.table[key]
        return self.pd.get((l, r), self.default)

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
        """Default x default composes: the product of two automata with few
        exceptions has few exceptions."""
        rng = random.Random(3)
        A = random_sta(rng, max_states=4, max_symbols=3, max_exc=6, max_pd=0)
        B = random_sta(rng, max_states=4, max_symbols=3, max_exc=6, max_pd=0)
        while len(B.base_alphabet) != len(A.base_alphabet):
            B = random_sta(rng, max_states=4, max_symbols=3, max_exc=6,
                           max_pd=0)
        prod = A.intersection(B)
        assert len(prod.exc_target) <= 4 * (len(A.exc_target) + len(B.exc_target)) + 16


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
