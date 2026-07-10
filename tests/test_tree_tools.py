import random

import numpy as np
import pytest

from autstr.sparse_tree_automata import SparseTreeAutomaton, Tree, tree_to_arrays
from autstr.utils.misc import encode_symbol
from autstr.utils.tree_automata_tools import (
    attach_padding, equivalent, expand, minimize, project,
)
from test_tree_automata import RefDTA, random_sta, random_tree


# ====================================================================
# Dense reference determinization (independent implementation)
# ====================================================================

def dense_determinize(ref: RefDTA, preimages, num_new_symbols, new_arity,
                      base_alphabet, pad_symbol):
    """Exhaustive dict-based subset construction over the NTA with symbol
    preimages and the padding-closure absent-child set."""
    d = ref.default
    # padding closure
    avail = {ref.BOT}
    changed = True
    while changed:
        changed = False
        for l in list(avail):
            for r in list(avail):
                for s in preimages[pad_symbol]:
                    t = ref.delta(l, r, s)
                    if t not in avail:
                        avail.add(t)
                        changed = True
    s_bot = frozenset(avail)

    subsets = {frozenset({d}): 0}
    order = [frozenset({d})]
    table = {}
    frontier = [None, 0]            # None encodes the absent-child set
    done = set()

    def sub(option):
        return s_bot if option is None else order[option]

    while frontier:
        # exhaustive: (re)visit all combos over all known options
        options = list(done) + frontier
        done.update(frontier)
        frontier = []
        for x in options:
            for y in options:
                for sym in range(num_new_symbols):
                    if (x, y, sym) in table:
                        continue
                    T = frozenset(ref.delta(a, b, s)
                                  for a in sub(x) for b in sub(y)
                                  for s in preimages[sym])
                    if T not in subsets:
                        subsets[T] = len(order)
                        order.append(T)
                        frontier.append(subsets[T])
                    table[(x, y, sym)] = subsets[T]

    # convert to a SparseTreeAutomaton: default = {d}, everything else listed
    n = len(order)
    BOT = n
    exc = []
    for (x, y, sym), t in table.items():
        if t == 0:
            continue
        exc.append((BOT if x is None else x, BOT if y is None else y, sym, t))
    accepting = [bool(set(np.flatnonzero(ref.acc)) & s) for s in order]
    exc.sort()
    return SparseTreeAutomaton(
        n, 0,
        [e[0] for e in exc], [e[1] for e in exc],
        [e[2] for e in exc], [e[3] for e in exc],
        accepting, new_arity, base_alphabet)


def random_sta_arity(rng, arity, m_base, max_states=4, max_exc=10, max_pd=5):
    """Random automaton over base_alphabet range(m_base) with given arity."""
    n = rng.randint(1, max_states)
    S = m_base ** arity
    k = rng.randint(0, min(max_exc, (n + 1) * (n + 1) * S))
    triples = set()
    while len(triples) < k:
        triples.add((rng.randint(0, n), rng.randint(0, n), rng.randrange(S)))
    triples = sorted(triples)
    kp = rng.randint(0, min(max_pd, (n + 1) * (n + 1)))
    pairs = set()
    while len(pairs) < kp:
        pairs.add((rng.randint(0, n), rng.randint(0, n)))
    pairs = sorted(pairs)
    return SparseTreeAutomaton(
        n, rng.randrange(n),
        [t[0] for t in triples], [t[1] for t in triples],
        [t[2] for t in triples], [rng.randrange(n) for _ in triples],
        [rng.random() < 0.5 for _ in range(n)],
        symbol_arity=arity, base_alphabet=set(range(m_base)),
        pd_left=[p[0] for p in pairs], pd_right=[p[1] for p in pairs],
        pd_target=[rng.randrange(n) for _ in pairs])


# ====================================================================
# expand
# ====================================================================

class TestExpand:
    def test_relabelling_semantics(self):
        """expand(A, 2, [0]) accepts a 2-tape tree iff A accepts the same
        tree with labels projected to tape 0 (same domain)."""
        rng = random.Random(0)
        for trial in range(20):
            m = rng.randint(2, 3)
            A = random_sta_arity(rng, 1, m)
            E = expand(A, 2, [0])
            for _ in range(20):
                t2 = random_tree(rng, m * m)     # random arity-2 labels
                arrays2 = tree_to_arrays_encoded(t2)
                labels0 = arrays2[0] // m        # project labels to tape 0
                got = bool(E.is_accepting[E.run(*arrays2)])
                ref = bool(A.is_accepting[A.run(labels0, arrays2[1], arrays2[2])])
                assert got == ref, trial

    def test_duplicate_positions(self):
        """Mapping both original tapes to one position restricts the
        transition function to the diagonal of the two tapes."""
        rng = random.Random(1)
        m = 2
        A = random_sta_arity(rng, 2, m, max_exc=8)
        E = expand(A, 1, [0, 0])
        source, image = A.dense_delta(), E.dense_delta()
        for l in range(A.num_states + 1):
            for r in range(A.num_states + 1):
                for a in range(m):
                    assert image[l, r, a] == source[l, r, a * m + a]


def tree_to_arrays_encoded(t: Tree):
    """Random trees carry already-encoded integer labels; pack directly."""
    labels, lefts, rights = [], [], []
    stack = [(t, False)]
    index = {}
    while stack:
        node, expanded = stack.pop()
        if node is None:
            continue
        if not expanded:
            stack.append((node, True))
            stack.append((node.right, False))
            stack.append((node.left, False))
        else:
            labels.append(node.label)
            lefts.append(index[id(node.left)] if node.left is not None else -1)
            rights.append(index[id(node.right)] if node.right is not None else -1)
            index[id(node)] = len(labels) - 1
    return (np.array(labels, dtype=np.int64),
            np.array(lefts, dtype=np.int64),
            np.array(rights, dtype=np.int64))


# ====================================================================
# projection / padding closure vs the dense reference
# ====================================================================

class TestProjection:
    def test_against_dense_reference(self):
        rng = random.Random(2)
        for trial in range(15):
            m = 2
            A = random_sta_arity(rng, 2, m, max_states=3, max_exc=8)
            got = project(A, 1, padding_symbol=0)

            ref = RefDTA(A)
            S_new = m
            preimages = {s: [] for s in range(S_new)}
            for sig in range(m * m):
                preimages[sig // m].append(sig)     # drop tape 1
            pad_new = encode_symbol((0,), frozenset(range(m)))
            want = dense_determinize(ref, preimages, S_new, 1,
                                     set(range(m)), pad_new)
            assert equivalent(got, want), trial

    def test_projection_of_diagonal(self):
        """Project tape 1 of the diagonal relation {(t, t)}: the result must
        accept every tree (over unary labels {0,1})."""
        m = 2
        # diagonal over 2 tapes: accept iff every label has equal components;
        # states: 0 = ok (accepting); default 1 = dead
        eq_syms = [s for s in range(m * m) if s // m == s % m]
        exc = []
        for l in (0, 2):            # 2 = BOT for num_states=2
            for r in (0, 2):
                for s in eq_syms:
                    exc.append((l, r, s, 0))
        exc.sort()
        diag = SparseTreeAutomaton(
            2, 1,
            [e[0] for e in exc], [e[1] for e in exc],
            [e[2] for e in exc], [e[3] for e in exc],
            [True, False], 2, set(range(m)))
        proj = project(diag, 1, padding_symbol=0)
        rng = random.Random(3)
        for _ in range(30):
            t = random_tree(rng, m)
            assert proj.accepts(tree_to_arrays_encoded(t)), t

    def test_padding_closure_matters(self):
        """A relation whose witness extends below the remaining tape: R =
        {(t, w)} where w has a node below t's domain. After projecting w,
        every tree should be accepted iff such a witness exists."""
        m = 2
        # accept iff somewhere a label is (pad, x) with x real: witness leaks
        # below tape 0. Build: state 0 = seen-leak (accepting), default 1 dead;
        # any node keeps 0 alive; leak symbols (pad=0 on tape 0, 1 on tape 1)
        # -> 0 from anything.
        leak = encode_symbol((0, 1), frozenset(range(m)))
        exc = []
        for l in (0, 1, 2):
            for r in (0, 1, 2):
                exc.append((l, r, leak, 0))
                if 0 in (l, r):
                    for s in range(m * m):
                        exc.append((l, r, s, 0))
        exc = sorted(set(exc))
        A = SparseTreeAutomaton(
            2, 1,
            [e[0] for e in exc], [e[1] for e in exc],
            [e[2] for e in exc], [e[3] for e in exc],
            [True, False], 2, set(range(m)))
        proj = project(A, 1, padding_symbol=0)
        # the witness can always hang a leak node below any tree
        rng = random.Random(4)
        for _ in range(20):
            t = random_tree(rng, m)
            assert proj.accepts(tree_to_arrays_encoded(t))


class TestAttachPadding:
    def test_trim_semantics(self):
        """The saturated automaton accepts a tree with pure-padding regions
        attached below iff the source accepts the tree with those regions
        trimmed away (elements labelled over {0, 1}, padding letter 2) —
        regardless of any native transitions the source has on the padding
        letter."""
        rng = random.Random(5)
        m = 3

        def pad_tree(depth=2):
            if depth == 0 or rng.random() < 0.4:
                return Tree(2)
            return Tree(2,
                        pad_tree(depth - 1) if rng.random() < 0.7 else None,
                        pad_tree(depth - 1) if rng.random() < 0.7 else None)

        def attach(t):
            if t is None:
                return pad_tree() if rng.random() < 0.35 else None
            return Tree(t.label, attach(t.left), attach(t.right))

        for trial in range(20):
            A = random_sta_arity(rng, 1, m, max_states=4, max_exc=12)
            P = attach_padding(A, padding_symbol=2)
            for _ in range(25):
                t = random_tree(rng, 2)          # labels 0/1: never padding
                want = bool(A.is_accepting[A.run(*tree_to_arrays_encoded(t))])
                assert bool(P.is_accepting[P.run(
                    *tree_to_arrays_encoded(t))]) == want, trial
                padded = attach(t)
                assert bool(P.is_accepting[P.run(
                    *tree_to_arrays_encoded(padded))]) == want, trial


# ====================================================================
# minimization
# ====================================================================

class TestMinimize:
    def test_language_preserved(self):
        rng = random.Random(7)
        for trial in range(25):
            A = random_sta(rng)
            M = minimize(A)
            assert M.num_states <= max(A.num_states, 1), trial
            assert equivalent(A, M), trial

    def test_idempotent(self):
        rng = random.Random(8)
        for _ in range(10):
            A = random_sta(rng)
            M = minimize(A)
            MM = minimize(M)
            assert MM.num_states == M.num_states
            assert equivalent(M, MM)

    def test_merges_twins(self):
        """Two states with identical behavior collapse."""
        m = 2
        # states 0,1 behave identically (both accepting, same exceptions),
        # state 2 = default dead
        exc = [(3, 3, 0, 0), (3, 3, 1, 1)]      # leaves reach 0 or 1
        A = SparseTreeAutomaton(
            3, 2,
            [e[0] for e in exc], [e[1] for e in exc],
            [e[2] for e in exc], [e[3] for e in exc],
            [True, True, False], 1, set(range(m)))
        M = minimize(A)
        assert M.num_states < A.num_states
        assert equivalent(A, M)

    def test_canonical_size_agreement(self):
        """minimize(sparse projection) and minimize(dense projection) have the
        same state count — both are the canonical minimal automaton."""
        rng = random.Random(9)
        for trial in range(10):
            m = 2
            A = random_sta_arity(rng, 2, m, max_states=3, max_exc=8)
            got = minimize(project(A, 1, padding_symbol=0))
            ref = RefDTA(A)
            preimages = {s: [] for s in range(m)}
            for sig in range(m * m):
                preimages[sig // m].append(sig)
            want = minimize(dense_determinize(
                ref, preimages, m, 1, set(range(m)),
                encode_symbol((0,), frozenset(range(m)))))
            assert equivalent(got, want), trial
            assert got.num_states == want.num_states, trial
