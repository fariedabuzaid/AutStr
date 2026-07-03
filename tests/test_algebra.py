import pytest

from autstr.algebra import (
    FiniteBooleanAlgebras,
    FiniteAbelianGroups,
    Z1pElement,
    z1p_localization,
)


@pytest.fixture(scope="module")
def ba():
    return FiniteBooleanAlgebras()


@pytest.fixture(scope="module")
def ab():
    return FiniteAbelianGroups()


class TestFiniteBooleanAlgebras:
    def test_operations(self, ba):
        n = 4
        assert ba.check('Meet(x,y,z)', n, x={0, 1}, y={1, 2}, z={1})
        assert not ba.check('Meet(x,y,z)', n, x={0, 1}, y={1, 2}, z={0, 1})
        assert ba.check('Join(x,y,z)', n, x={0, 1}, y={1, 2}, z={0, 1, 2})
        assert ba.check('Compl(x,y)', n, x={0, 1}, y={2, 3})
        assert not ba.check('Compl(x,y)', n, x={0, 1}, y={1, 2, 3})
        assert ba.check('Leq(x,y)', n, x={2}, y={1, 2})
        assert not ba.check('Leq(x,y)', n, x={0, 2}, y={1, 2})
        assert ba.check('Atom(x)', n, x={3})
        assert not ba.check('Atom(x)', n, x={2, 3})
        assert not ba.check('Atom(x)', n, x=set())

    def test_complementation_law(self, ba):
        phi = 'all x.(exists y.(Compl(x,y)))'
        for n in (1, 3, 5):
            assert ba.check(phi, n)

    def test_two_atoms_iff_n_at_least_2(self, ba):
        phi = 'exists x.(exists y.(Atom(x) and Atom(y) and (not Leq(x,y))))'
        assert not ba.check(phi, 1)
        assert ba.check(phi, 2)
        assert ba.check(phi, 4)

    def test_uniformity(self, ba):
        """One automaton decides nontriviality for every finite boolean
        algebra: some x is not below its own complement iff n >= 1."""
        proper = 'exists x.(exists y.(Compl(x,y) and (not Leq(x,y))))'
        dfa, variables = ba.evaluate(proper)
        assert variables == ['advice']
        assert not dfa.accepts([(s,) for s in ba.advice(0)])
        assert dfa.accepts([(s,) for s in ba.advice(1)])
        assert dfa.accepts([(s,) for s in ba.advice(5)])


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


class TestZ1pLocalization:
    def test_prime_validation(self):
        with pytest.raises(ValueError):
            z1p_localization(1)
        with pytest.raises(ValueError):
            z1p_localization(12)

    def test_normalization_and_fraction_input(self):
        loc = z1p_localization(3)
        assert loc.element(27, 5) == Z1pElement(1, 2)
        assert loc.element(0, 7) == Z1pElement(0, 0)
        assert loc.from_fraction(18, 9) == Z1pElement(2, 0)
        with pytest.raises(ValueError):
            loc.from_fraction(1, 10)

    def test_addition_and_subtraction(self):
        loc = z1p_localization(2)
        x = loc.from_fraction(1, 2)
        y = loc.from_fraction(3, 4)
        z = loc.add(x, y)
        assert z == Z1pElement(5, 2)  # 1/2 + 3/4 = 5/4
        assert loc.sub(z, y) == x
        assert loc.neg(x) == Z1pElement(-1, 1)

    def test_equality_via_canonical_form(self):
        loc = z1p_localization(5)
        a = loc.element(25, 2)
        b = loc.element(1, 0)
        c = loc.element(5, 1)
        assert loc.equals(a, b)
        assert loc.equals(b, c)


@pytest.fixture(scope="module")
def z2():
    return z1p_localization(2)


@pytest.fixture(scope="module")
def z3():
    return z1p_localization(3)


class TestZ1pPresentation:
    def test_addition_against_arithmetic_model(self, z2):
        elements = [z2.element(n, k) for n in (-5, -3, -1, 0, 1, 2, 3, 7)
                    for k in (0, 1, 2)]
        pairs = [(elements[i], elements[j])
                 for i in range(0, len(elements), 3)
                 for j in range(1, len(elements), 5)]
        for x, y in pairs:
            z = z2.add(x, y)
            assert z2.check('A(x,y,z)', x=x, y=y, z=z), (x, y, z)
            wrong = z2.add(z, z2.element(1))
            assert not z2.check('A(x,y,z)', x=x, y=y, z=wrong), (x, y, wrong)

    def test_fractional_addition(self, z2):
        half = z2.from_fraction(1, 2)
        three_quarters = z2.from_fraction(3, 4)
        assert z2.check('A(x,y,z)', x=half, y=three_quarters,
                        z=z2.from_fraction(5, 4))
        assert z2.check('A(x,y,z)', x=z2.neg(half), y=three_quarters,
                        z=z2.from_fraction(1, 4))
        assert z2.check('A(x,y,z)', x=z2.neg(three_quarters),
                        y=z2.from_fraction(1, 4), z=z2.neg(half))

    def test_sign_zero_equality(self, z2):
        assert z2.check('N0(x)', x=0)
        assert z2.check('N0(x)', x=z2.from_fraction(3, 4))
        assert not z2.check('N0(x)', x=-1)
        assert z2.check('Z(x)', x=0)
        assert not z2.check('Z(x)', x=z2.from_fraction(1, 2))
        assert z2.check('Eq(x,y)', x=z2.element(2, 1), y=z2.element(1, 0))
        assert not z2.check('Eq(x,y)', x=1, y=z2.from_fraction(1, 2))

    def test_group_axioms(self, z2):
        assert z2.check('all x.(exists y.(exists z.(A(x,y,z) and Z(z))))')  # inverses
        assert z2.check('all x.(all y.(all z.((not A(x,y,z)) or A(y,x,z))))')  # commutativity

    def test_p_divisibility_distinguishes_localizations(self, z2, z3):
        """The classic FO difference: every element of Z[1/p] is p-divisible,
        but not q-divisible for primes q != p."""
        div2 = 'all x.(exists y.(A(y,y,x)))'
        div3 = 'all x.(exists y.(exists w.(A(y,y,w) and A(w,y,x))))'
        assert z2.check(div2)
        assert not z2.check(div3)
        assert not z3.check(div2)
        assert z3.check(div3)
