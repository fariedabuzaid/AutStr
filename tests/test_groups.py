import itertools as it

import pytest

from autstr.groups import (CutRankGroups, ExtraspecialGroups,
                           FiniteAbelianGroups, IndexTwoCyclicGroups)


@pytest.fixture(scope="module")
def idx2():
    return IndexTwoCyclicGroups()


@pytest.fixture(scope="module")
def idx2_mult(idx2):
    dfa, variables = idx2.evaluate('M(x,y,z)')
    assert set(variables) == {'advice', 'x', 'y', 'z'}
    return dfa, variables


@pytest.fixture(scope="module")
def heis3():
    return ExtraspecialGroups(3)


def convolve(variables, columns):
    length = len(columns['advice'])
    return [tuple(columns[v][i] for v in variables) for i in range(length)]


class TestIndexTwoCyclicGroups:
    def elements(self, idx2, advice):
        _, n = idx2.parameters(advice)
        return [(e, a) for e in (0, 1) for a in range(n)]

    def test_reference_law_is_a_group(self, idx2):
        """Sanity of the reference multiplication itself: associativity,
        identity and inverses, exhaustively, for one group per family."""
        advices = [idx2.abelian(3), idx2.cyclic(3), idx2.dihedral(4),
                   idx2.dicyclic(4), idx2.semidihedral(8), idx2.modular(8)]
        for advice in advices:
            elems = self.elements(idx2, advice)
            mul = lambda g, h: idx2.multiply(advice, g, h)
            for g, h, k in it.product(elems, repeat=3):
                assert mul(mul(g, h), k) == mul(g, mul(h, k)), (advice, g, h, k)
            identity = (0, 0)
            for g in elems:
                assert mul(identity, g) == g and mul(g, identity) == g
                assert any(mul(g, h) == identity for h in elems)

    def test_multiplication_automaton_exhaustive(self, idx2, idx2_mult):
        """The defined relation M agrees with the reference group law on
        every product of every family's sample group."""
        dfa, variables = idx2_mult
        advices = [idx2.abelian(3), idx2.cyclic(3), idx2.dihedral(4),
                   idx2.dicyclic(4), idx2.semidihedral(8), idx2.modular(8)]
        for advice in advices:
            elems = self.elements(idx2, advice)
            words = {g: idx2.encode(g, advice) for g in elems}
            for g, h in it.product(elems, repeat=2):
                expected = idx2.multiply(advice, g, h)
                for z in elems:
                    word = convolve(variables, {
                        'advice': advice, 'x': words[g], 'y': words[h], 'z': words[z]})
                    assert dfa.accepts(word) == (z == expected), (advice, g, h, z)

    def test_known_group_facts(self, idx2):
        # Q8: unique element of order 2 (the central -1); D_4: several
        order_two = ('exists x.((not Eq(x,o)) and M(x,x,o))')
        # ... with o = identity, assigned explicitly
        assert idx2.check(order_two, idx2.dicyclic(4), o=(0, 0))
        involutions_distinct = (
            'exists x.(exists y.((not Eq(x,y)) and (not Eq(x,o)) and (not Eq(y,o)) '
            'and M(x,x,o) and M(y,y,o)))')
        assert not idx2.check(involutions_distinct, idx2.dicyclic(4), o=(0, 0))
        assert idx2.check(involutions_distinct, idx2.dihedral(4), o=(0, 0))

    def test_nonabelian_uniform(self, idx2):
        """One automaton decides commutativity for the whole class."""
        phi = 'exists x.(exists y.(exists z.(M(x,y,z) and (not M(y,x,z)))))'
        dfa, variables = idx2.evaluate(phi)
        assert variables == ['advice']
        cases = [
            (idx2.abelian(5), False), (idx2.cyclic(6), False),
            (idx2.dihedral(2), False),  # D_2 = Klein four group
            (idx2.dihedral(3), True), (idx2.dihedral(8), True),
            (idx2.dicyclic(4), True), (idx2.semidihedral(8), True),
            (idx2.modular(8), True),
        ]
        for advice, expected in cases:
            assert dfa.accepts([(s,) for s in advice]) == expected, advice

    def test_advice_validation(self, idx2):
        with pytest.raises(ValueError):
            idx2.dicyclic(3)  # odd cyclic part
        with pytest.raises(ValueError):
            idx2.semidihedral(6)  # not a power of two
        with pytest.raises(ValueError):
            idx2.modular(2)  # too small


class TestExtraspecialGroups:
    def test_multiplication_exhaustive_rank1(self, heis3):
        """All 27^2 products of the extraspecial group of order 27."""
        p, n = 3, 1
        elems = [(c, (a,), (b,)) for c in range(p) for a in range(p) for b in range(p)]
        dfa, variables = heis3.evaluate('M(x,y,z)')
        assert set(variables) == {'advice', 'x', 'y', 'z'}
        advice = heis3.advice(n)
        words = {g: heis3.encode(g, n) for g in elems}
        for g, h in it.product(elems, repeat=2):
            expected = heis3.multiply(g, h)
            columns = {'advice': advice, 'x': words[g], 'y': words[h]}
            word = convolve(variables, dict(columns, z=words[expected]))
            assert dfa.accepts(word), (g, h)
            wrong = ((expected[0] + 1) % p, expected[1], expected[2])
            word = convolve(variables, dict(columns, z=words[wrong]))
            assert not dfa.accepts(word), (g, h)

    def test_center(self, heis3):
        assert heis3.check('Cen(x)', 2, x=(2, (0, 0), (0, 0)))
        assert not heis3.check('Cen(x)', 2, x=(0, (1, 0), (0, 0)))
        # Cen coincides with "commutes with everything"
        phi = ('all y.(all z.((not M(x,y,z)) or M(y,x,z)))')
        assert heis3.check(phi, 1, x=(1, (0,), (0,)))
        assert not heis3.check(phi, 1, x=(0, (1,), (0,)))

    def test_nonabelian_uniform(self, heis3):
        phi = 'exists x.(exists y.(exists z.(M(x,y,z) and (not M(y,x,z)))))'
        dfa, variables = heis3.evaluate(phi)
        assert variables == ['advice']
        for n, expected in [(0, False), (1, True), (2, True), (3, True)]:
            assert dfa.accepts([(s,) for s in heis3.advice(n)]) == expected, n

    def test_exponent_p(self, heis3):
        """For odd p every element satisfies x^p = 1 (x*x*x is idempotent
        only for the identity, so M(b,b,b) says b = 1)."""
        phi = 'all x.(exists a.(exists b.(M(x,x,a) and M(a,x,b) and M(b,b,b))))'
        assert heis3.check(phi, 1)
        assert heis3.check(phi, 2)
        # p = 2 extraspecial groups have elements of order 4 instead
        heis2 = ExtraspecialGroups(2)
        square_trivial = 'all x.(exists a.(M(x,x,a) and M(a,a,a)))'
        assert heis2.check(square_trivial, 0)  # Z_2
        assert not heis2.check(square_trivial, 1)  # D_4: order-4 elements

    def test_prime_validation(self):
        with pytest.raises(ValueError):
            ExtraspecialGroups(4)


@pytest.fixture(scope="module")
def ab():
    return FiniteAbelianGroups()


class TestFiniteAbelianGroups:
    def test_addition_cyclic(self, ab):
        for x, y in [(0, 0), (1, 5), (4, 5), (3, 3), (5, 5)]:
            for z in range(6):
                expected = (x + y) % 6 == z
                assert ab.check('A(x,y,z)', [6], x=x, y=y, z=z) == expected, (x, y, z)

    def test_addition_direct_sum(self, ab):
        orders = [2, 3]
        assert ab.check('A(x,y,z)', orders, x=(1, 1), y=(1, 2), z=(0, 0))
        assert ab.check('A(x,y,z)', orders, x=(0, 2), y=(1, 2), z=(1, 1))
        assert not ab.check('A(x,y,z)', orders, x=(1, 1), y=(1, 2), z=(1, 0))

    def test_identity_definable(self, ab):
        # A(x,x,x) holds iff x = 0
        assert ab.check('A(x,x,x)', [4], x=0)
        for x in (1, 2, 3):
            assert not ab.check('A(x,x,x)', [4], x=x)

    def test_group_axioms(self, ab):
        inverses = 'all x.(exists y.(exists z.(A(x,y,z) and A(z,z,z))))'
        commutative = 'all x.(all y.(all z.((not A(x,y,z)) or A(y,x,z))))'
        for orders in ([1], [5], [2, 2]):
            assert ab.check(inverses, orders)
            assert ab.check(commutative, orders)

    def test_two_torsion_uniform(self, ab):
        """exists x != 0 with x + x = 0 — one automaton decides it for every
        finite abelian group: true iff some cyclic factor has even order."""
        phi = ('exists x.((not A(x,x,x)) and '
               '(exists z.(A(x,x,z) and A(z,z,z))))')
        dfa, variables = ab.evaluate(phi)
        assert variables == ['advice']
        cases = [([4], True), ([3], False), ([2, 3], True),
                 ([9], False), ([1], False), ([5, 7], False), ([5, 6], True)]
        for orders, expected in cases:
            advice = ab.advice(orders)
            assert dfa.accepts([(s,) for s in advice]) == expected, orders

    def test_encoding_validation(self, ab):
        with pytest.raises(ValueError):
            ab.encode(6, [6])  # out of range
        with pytest.raises(ValueError):
            ab.encode((1,), [2, 3])  # wrong number of components


@pytest.fixture(scope="module")
def crg2():
    return CutRankGroups(2)


@pytest.fixture(scope="module")
def crg2_mult(crg2):
    dfa, variables = crg2.evaluate('M(x,y,z)')
    assert set(variables) == {'advice', 'x', 'y', 'z'}
    return dfa, variables


@pytest.fixture(scope="module")
def crg3():
    return CutRankGroups(3)


class TestCutRankGroups:
    """The bounded linear cut-rank class: reference law, layout compiler and
    multiplication automaton, on layouts of very different pathwidth."""

    def elements(self, crg, n):
        return [(b, a)
                for b in it.product(range(crg.p), repeat=crg.k)
                for a in it.product(range(crg.p), repeat=n)]

    def forms(self, crg, n):
        forms = {'zero': {}, 'clique': crg.clique_form(n),
                 'matching': crg.matching_form(n)}
        if n >= 4:
            # vertex cover {1}: x_1 commutes with nothing, cut-rank 1
            forms['star'] = {(j, 1): (1,) for j in range(2, n + 1)}
        return forms

    def test_reference_law_is_a_group(self, crg2, crg3):
        """The cocycle really is one: associativity, identity, inverses."""
        for crg, n in [(crg2, 3), (crg3, 2)]:
            for name, form in self.forms(crg, n).items():
                elems = self.elements(crg, n)
                mul = lambda g, h: crg.multiply(n, form, g, h)
                for g, h, f in it.product(elems, repeat=3):
                    assert mul(mul(g, h), f) == mul(g, mul(h, f)), (name, g, h, f)
                one = crg.identity(n)
                for g in elems:
                    assert mul(one, g) == g and mul(g, one) == g
                    assert any(mul(g, h) == one for h in elems)

    def test_multiplication_automaton_exhaustive(self, crg2, crg2_mult):
        """M agrees with the reference law on every product, for layouts
        including the clique (pathwidth n-1, cut-rank 1)."""
        dfa, variables = crg2_mult
        n = 4
        for name, form in self.forms(crg2, n).items():
            advice = crg2.advice(n, form)
            elems = self.elements(crg2, n)
            words = {g: crg2.encode(g, n) for g in elems}
            for g, h in it.product(elems, repeat=2):
                expected = crg2.multiply(n, form, g, h)
                columns = {'advice': advice, 'x': words[g], 'y': words[h]}
                word = convolve(variables, dict(columns, z=words[expected]))
                assert dfa.accepts(word), (name, g, h)
                wrong = (((expected[0][0] + 1) % 2,), expected[1])
                word = convolve(variables, dict(columns, z=words[wrong]))
                assert not dfa.accepts(word), (name, g, h)

    def test_clique_scales(self, crg2, crg2_mult):
        """n = 12: no automaton that stores digits of active vertices could
        do this with p^(k+r) states — the clique keeps every vertex active."""
        import random
        rng = random.Random(7)
        dfa, variables = crg2_mult
        n = 12
        form = crg2.clique_form(n)
        advice = crg2.advice(n, form)
        for _ in range(100):
            g = ((rng.randrange(2),), tuple(rng.randrange(2) for _ in range(n)))
            h = ((rng.randrange(2),), tuple(rng.randrange(2) for _ in range(n)))
            expected = crg2.multiply(n, form, g, h)
            columns = {'advice': advice,
                       'x': crg2.encode(g, n), 'y': crg2.encode(h, n)}
            word = convolve(variables, dict(columns, z=crg2.encode(expected, n)))
            assert dfa.accepts(word), (g, h)
            wrong = (((expected[0][0] + 1) % 2,), expected[1])
            word = convolve(variables, dict(columns, z=crg2.encode(wrong, n)))
            assert not dfa.accepts(word), (g, h)

    def test_matching_is_extraspecial(self, crg3, heis3):
        """The matching layout reproduces the Heisenberg-type law of
        ExtraspecialGroups under digit interleaving."""
        import random
        rng = random.Random(11)
        m = 3
        n = 2 * m
        form = crg3.matching_form(n)

        def embed(c, a, b):
            digits = [0] * n
            for t in range(m):
                digits[2 * t] = b[t]      # position 2t+1 (odd, 1-indexed)
                digits[2 * t + 1] = a[t]  # position 2t+2 (even, 1-indexed)
            return ((c,), tuple(digits))

        for _ in range(200):
            g = (rng.randrange(3), tuple(rng.randrange(3) for _ in range(m)),
                 tuple(rng.randrange(3) for _ in range(m)))
            h = (rng.randrange(3), tuple(rng.randrange(3) for _ in range(m)),
                 tuple(rng.randrange(3) for _ in range(m)))
            expected = heis3.multiply(g, h)
            assert crg3.multiply(n, form, embed(*g), embed(*h)) == embed(*expected)

    def test_pauli_triangle(self, crg2):
        """Triangle commutation graph at p = 2 = the 1-qubit Pauli group:
        x1 x2 x3 is central of order 4, its square is the central y."""
        n = 3
        form = crg2.clique_form(n)
        advice = crg2.advice(n, form)
        x1 = ((0,), (1, 0, 0))
        x2 = ((0,), (0, 1, 0))
        x3 = ((0,), (0, 0, 1))
        g = crg2.multiply(n, form, crg2.multiply(n, form, x1, x2), x3)
        y = ((1,), (0, 0, 0))
        assert crg2.multiply(n, form, g, g) == y
        assert crg2.multiply(n, form, y, y) == crg2.identity(n)
        assert crg2.check('M(x,x,z)', advice, x=g, z=y)
        assert not crg2.check('M(x,x,z)', advice, x=g, z=crg2.identity(n))
        # centrality is first-order definable from M
        central = 'all u.(all v.(all w.((not M(x,u,v)) or (not M(u,x,w)) or Eq(v,w))))'
        assert crg2.check(central, advice, x=g)
        assert crg2.check(central, advice, x=y)
        assert not crg2.check(central, advice, x=x1)

    def test_width_guard_and_r2(self):
        """A form of cut-rank 2 is rejected at r = 1 and handled at r = 2."""
        crg = CutRankGroups(2)
        n = 4
        form = {(3, 1): (1,), (4, 2): (1,), (2, 1): (1,)}
        assert crg.linear_cut_rank(n, form) == 2
        with pytest.raises(ValueError):
            crg.advice(n, form)
        wide = CutRankGroups(2, r=2)
        advice = wide.advice(n, form)
        dfa, variables = wide.evaluate('M(x,y,z)')
        elems = [(b, a) for b in it.product(range(2), repeat=1)
                 for a in it.product(range(2), repeat=n)]
        words = {g: wide.encode(g, n) for g in elems}
        for g, h in it.product(elems, repeat=2):
            expected = wide.multiply(n, form, g, h)
            columns = {'advice': advice, 'x': words[g], 'y': words[h]}
            word = convolve(variables, dict(columns, z=words[expected]))
            assert dfa.accepts(word), (g, h)
            wrong = (((expected[0][0] + 1) % 2,), expected[1])
            word = convolve(variables, dict(columns, z=words[wrong]))
            assert not dfa.accepts(word), (g, h)

    def test_linear_cut_rank(self, crg2):
        assert crg2.linear_cut_rank(6, crg2.clique_form(6)) == 1
        assert crg2.linear_cut_rank(6, crg2.matching_form(6)) == 1
        assert crg2.linear_cut_rank(3, {}) == 0
        cover = {(2, 1): (1,), (5, 1): (1,), (4, 2): (1,), (6, 2): (1,)}
        assert crg2.linear_cut_rank(6, cover) <= 2

    def test_nonabelian_uniform(self, crg2):
        """One automaton decides commutativity across all layouts."""
        phi = 'exists x.(exists y.(exists z.(M(x,y,z) and (not M(y,x,z)))))'
        dfa, variables = crg2.evaluate(phi)
        assert variables == ['advice']
        cases = [(crg2.advice(3, {}), False),
                 (crg2.advice(3, crg2.clique_form(3)), True),
                 (crg2.advice(4, crg2.matching_form(4)), True)]
        for advice, expected in cases:
            assert dfa.accepts([(s,) for s in advice]) == expected

    def test_exponent_p(self, crg3):
        """Exponent p for odd p, order-4 elements at p = 2 (Pauli)."""
        n = 2
        form = crg3.clique_form(n)
        elems = [(b, a) for b in it.product(range(3), repeat=1)
                 for a in it.product(range(3), repeat=n)]
        one = crg3.identity(n)
        for g in elems:
            cube = crg3.multiply(n, form, crg3.multiply(n, form, g, g), g)
            assert cube == one, g

    def test_validation(self, crg2):
        with pytest.raises(ValueError):
            CutRankGroups(4)
        with pytest.raises(ValueError):
            CutRankGroups(3, k=2, r=2)  # 59049 advice letters
        with pytest.raises(ValueError):
            crg2.encode(((0,), (2, 0)), 2)  # digit out of range
        with pytest.raises(ValueError):
            crg2.encode(((0,), (0, 0, 0)), 2)  # wrong length
        with pytest.raises(ValueError):
            crg2.advice(3, {(1, 2): (1,)})  # needs i < j
        with pytest.raises(ValueError):
            crg2.advice(3, {(2, 1): (1, 0)})  # label length != k
