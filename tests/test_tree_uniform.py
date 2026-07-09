import random

import pytest

from autstr.sparse_tree_automata import SparseTreeAutomaton, Tree
from autstr.tree_uniform import UniformlyTreeAutomaticClass
from autstr.uniform import UniformlyAutomaticClass, dfa_from_delta
from autstr.utils.misc import encode_symbol
from autstr.utils.tree_automata_tools import from_string_dfa, string_chain


# ====================================================================
# A genuinely tree-shaped class: the structure of the nodes of the
# advice tree, with left/right-child, leaf and equality relations.
# Elements are root paths inside the advice: the path to node v, all
# nodes labelled 'x'.
# ====================================================================

B = {'*', 'a', 'x'}


def enc(letters):
    return encode_symbol(tuple(letters), frozenset(B))


def _sta(num_states, default, exc, acc, arity):
    return SparseTreeAutomaton(
        num_states, default,
        [e[0] for e in exc], [e[1] for e in exc],
        [e[2] for e in exc], [e[3] for e in exc], acc, arity, set(B))


def node_class() -> UniformlyTreeAutomaticClass:
    # Dom(advice, x): advice all-'a', element a root path within it
    A, P, D, BOT = 0, 1, 2, 3
    exc = []
    for l in (A, BOT):
        for r in (A, BOT):
            exc.append((l, r, enc(('a', '*')), A))
            exc.append((l, r, enc(('a', 'x')), P))      # path endpoint
    for c in (A, BOT):
        exc += [(P, c, enc(('a', 'x')), P), (c, P, enc(('a', 'x')), P)]
    dom = _sta(3, D, exc, [False, True, False], 2)

    # E(x, y): the two paths coincide
    exc = []
    for l in (A, BOT):
        for r in (A, BOT):
            exc.append((l, r, enc(('a', '*', '*')), A))
            exc.append((l, r, enc(('a', 'x', 'x')), P))
    for c in (A, BOT):
        exc += [(P, c, enc(('a', 'x', 'x')), P),
                (c, P, enc(('a', 'x', 'x')), P)]
    eq = _sta(3, D, exc, [False, True, False], 3)

    # L(x, y) / R(x, y): y extends x by one left / right step
    def child_relation(left_step: bool):
        A_, Y, P_, D_, BOT_ = 0, 1, 2, 3, 4
        exc = []
        for l in (A_, BOT_):
            for r in (A_, BOT_):
                exc.append((l, r, enc(('a', '*', '*')), A_))
                exc.append((l, r, enc(('a', '*', 'x')), Y))  # y's extra node
        for c in (A_, BOT_):
            if left_step:
                exc.append((Y, c, enc(('a', 'x', 'x')), P_))  # x's endpoint
            else:
                exc.append((c, Y, enc(('a', 'x', 'x')), P_))
            exc += [(P_, c, enc(('a', 'x', 'x')), P_),
                    (c, P_, enc(('a', 'x', 'x')), P_)]
        return _sta(4, D_, exc, [False, False, True, False], 3)

    # Leaf(x): the endpoint has no advice below it
    exc = []
    for l in (A, BOT):
        for r in (A, BOT):
            exc.append((l, r, enc(('a', '*')), A))
    exc.append((BOT, BOT, enc(('a', 'x')), P))
    for c in (A, BOT):
        exc += [(P, c, enc(('a', 'x')), P), (c, P, enc(('a', 'x')), P)]
    leaf = _sta(3, D, exc, [False, True, False], 2)

    return UniformlyTreeAutomaticClass(
        {'U': dom, 'E': eq, 'L': child_relation(True),
         'R': child_relation(False), 'Leaf': leaf},
        padding_symbol='*', max_states=100_000)


def random_advice(rng: random.Random, max_size=7) -> Tree:
    def build(budget):
        if budget <= 1 or rng.random() < 0.3:
            return Tree('a'), 1
        used = 1
        left = right = None
        if rng.random() < 0.7:
            left, u = build(budget - used)
            used += u
        if rng.random() < 0.7 and budget - used >= 1:
            right, u = build(budget - used)
            used += u
        return Tree('a', left, right), used
    return build(rng.randint(1, max_size))[0]


def addresses(t: Tree):
    """All node addresses of t as strings over {'0' (left), '1' (right)}."""
    out, stack = [], [(t, '')]
    while stack:
        node, addr = stack.pop()
        out.append(addr)
        if node.left is not None:
            stack.append((node.left, addr + '0'))
        if node.right is not None:
            stack.append((node.right, addr + '1'))
    return out


def subtree(t: Tree, addr: str) -> Tree:
    for mv in addr:
        t = t.left if mv == '0' else t.right
    return t


def path_tree(addr: str) -> Tree:
    node = Tree('x')
    for mv in reversed(addr):
        node = Tree('x', node, None) if mv == '0' else Tree('x', None, node)
    return node


@pytest.fixture(scope="module")
def nodes():
    return node_class()


class TestNodeClass:
    def test_membership_and_relations_pointwise(self, nodes):
        rng = random.Random(0)
        for _ in range(6):
            t = random_advice(rng)
            addrs = addresses(t)
            for a in addrs:
                sub = subtree(t, a)
                want_leaf = sub.left is None and sub.right is None
                assert nodes.check('Leaf(x)', t, x=path_tree(a)) == want_leaf
                assert nodes.check('L(x,y)', t, x=path_tree(a),
                                   y=path_tree(a + '0')) == \
                    (sub.left is not None)
                assert nodes.check('R(x,y)', t, x=path_tree(a),
                                   y=path_tree(a + '1')) == \
                    (sub.right is not None)
            # a path outside the advice is not in the domain
            outside = max(addrs, key=len) + '0'
            if outside not in addrs:
                assert not nodes.check('E(x,x)', t, x=path_tree(outside))

    def test_sentences_against_direct_inspection(self, nodes):
        rng = random.Random(1)
        for _ in range(8):
            t = random_advice(rng)
            addrs = addresses(t)
            leaves = [a for a in addrs
                      if subtree(t, a).left is None
                      and subtree(t, a).right is None]
            binary = any(subtree(t, a).left is not None
                         and subtree(t, a).right is not None for a in addrs)
            assert nodes.check('exists x.(Leaf(x))', t)
            assert nodes.check(
                'exists x.(exists y.(exists z.(L(x,y) and R(x,z))))',
                t) == binary
            assert nodes.check(
                'exists x.(exists y.(Leaf(x) and Leaf(y) and (not E(x,y))))',
                t) == (len(leaves) >= 2)
            assert nodes.check(
                'all x.((not Leaf(x)) -> (exists y.(L(x,y) or R(x,y))))', t)

    def test_invalid_advice_rejected(self, nodes):
        assert not nodes.check('exists x.(Leaf(x))', Tree('x'))
        assert nodes.check('exists x.(Leaf(x))', Tree('a'))

    def test_define(self, nodes):
        rng = random.Random(2)
        nodes.define('Child', 'L(x,y) or R(x,y)')
        for _ in range(5):
            t = random_advice(rng)
            addrs = addresses(t)
            binary = any(subtree(t, a).left is not None
                         and subtree(t, a).right is not None for a in addrs)
            assert nodes.check(
                'all x.(Leaf(x) or exists y.(Child(x,y)))', t)
            assert nodes.check(
                'exists x.(exists y.(exists z.'
                '(Child(x,y) and Child(x,z) and (not E(y,z)))))',
                t) == binary

    def test_get_structure(self, nodes):
        rng = random.Random(3)
        for _ in range(4):
            t = random_advice(rng)
            addrs = addresses(t)
            S = nodes.get_structure(t)
            for a in addrs:
                assert S.automata['U'].accepts(path_tree(a)), a
            outside = max(addrs, key=len) + '1'
            if outside not in addrs:
                assert not S.automata['U'].accepts(path_tree(outside))
            has_left = any(subtree(t, a).left is not None for a in addrs)
            assert S.check('exists x.(exists y.(L(x,y)))') == has_left


# ====================================================================
# Cross-validation: finite linear orders through both uniform engines
# (advice 1^n presents ({1..n}, <); the tree engine reads chain trees)
# ====================================================================

SB = {'*', '1'}


def make_string_class() -> UniformlyAutomaticClass:
    def u_delta(q, sym):
        if q == 'i' and sym == ('1', '1'):
            return 's0'
        if q == 's0' and sym == ('1', '1'):
            return 's0'
        if q in ('s0', 's1') and sym == ('1', '*'):
            return 's1'
        return 'd'

    def lt_delta(q, sym):
        if q == 'q0' and sym == ('1', '1', '1'):
            return 'q0'
        if q in ('q0', 'q1') and sym == ('1', '*', '1'):
            return 'q1'
        if q in ('q1', 'q2') and sym == ('1', '*', '*'):
            return 'q2'
        return 'd'

    U = dfa_from_delta(SB, ['i', 's0', 's1', 'd'], 2, u_delta, 'i',
                       {'s0', 's1'})
    Lt = dfa_from_delta(SB, ['q0', 'q1', 'q2', 'd'], 3, lt_delta, 'q0',
                        {'q1', 'q2'})
    return UniformlyAutomaticClass({'U': U, 'Lt': Lt}, padding_symbol='*')


def make_tree_class(sc: UniformlyAutomaticClass
                    ) -> UniformlyTreeAutomaticClass:
    return UniformlyTreeAutomaticClass(
        {'U': from_string_dfa(sc.class_automata['U']),
         'Lt': from_string_dfa(sc.class_automata['Lt'])},
        padding_symbol='*', max_states=100_000)


SENTENCES = [
    ('exists x.(all y.(not Lt(y,x)))', lambda n: True),    # minimum exists
    ('exists x.(Lt(x,x))', lambda n: False),               # irreflexive
    ('all x.(exists y.(Lt(x,y)))', lambda n: False),       # maximum fails it
    ('exists x.(exists y.(Lt(x,y) and (exists z.(Lt(y,z)))))',
     lambda n: n >= 3),                                    # >= 3 elements
]


@pytest.fixture(scope="module")
def engines():
    sc = make_string_class()
    return sc, make_tree_class(sc)


class TestLinearOrderCrossValidation:
    def test_sentences_agree(self, engines):
        sc, tc = engines
        for phi, truth in SENTENCES:
            for n in range(1, 6):
                want = truth(n)
                assert sc.check(phi, ['1'] * n) == want, (phi, n)
                assert tc.check(phi, string_chain('1' * n)) == want, (phi, n)

    def test_element_assignments_agree(self, engines):
        sc, tc = engines
        n = 4
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                want = i < j
                s_got = sc.check('Lt(x,y)', ['1'] * n,
                                 x=['1'] * i + ['*'] * (n - i),
                                 y=['1'] * j + ['*'] * (n - j))
                t_got = tc.check('Lt(x,y)', string_chain('1' * n),
                                 x=string_chain('1' * i),
                                 y=string_chain('1' * j))
                assert s_got == want and t_got == want, (i, j)

    def test_define_agrees(self):
        sc = make_string_class()
        tc = make_tree_class(sc)
        sc.define('Gt', 'Lt(y,x)')
        tc.define('Gt', 'Lt(y,x)')
        for phi, truth in [
            ('exists x.(all y.(not Gt(y,x)))', lambda n: True),  # maximum
            ('exists x.(exists y.(Gt(x,y) and exists z.(Gt(y,z))))',
             lambda n: n >= 3),
        ]:
            for n in range(1, 5):
                want = truth(n)
                assert sc.check(phi, ['1'] * n) == want, (phi, n)
                assert tc.check(phi, string_chain('1' * n)) == want, (phi, n)

    def test_get_structure_agrees(self, engines):
        sc, tc = engines
        for n in (1, 3):
            Ss = sc.get_structure(['1'] * n)
            St = tc.get_structure(string_chain('1' * n))
            for phi, truth in SENTENCES:
                want = truth(n)
                assert Ss.check(phi) == want, (phi, n)
                assert St.check(phi) == want, (phi, n)
