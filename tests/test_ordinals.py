"""The ordinals below ω^ω, as n-dimensional interpretations of ℕ.

The oracle is Python itself: an ordinal below ω^n is its tuple of Cantor
coefficients in descending exponent order, and ordinal comparison is that
tuple's lexicographic order. So every order fact checked here has an
independent ground truth that needs no automaton.

ω³ is exercised sparingly — the interpretation is three-dimensional and its
sentences are correspondingly expensive.
"""
import itertools

import pytest

from autstr.ordinals import Ordinal


@pytest.fixture(scope="module")
def omega():
    return Ordinal(1)


@pytest.fixture(scope="module")
def omega_squared():
    return Ordinal(2)


@pytest.fixture(scope="module")
def below(omega_squared):
    """``x < y`` over ω², as a relation that takes ordinals as Python
    values."""
    x, y = omega_squared.symbolic().vars('x y')
    return x.lt(y).evaluate()


@pytest.fixture(scope="module")
def successor(omega_squared):
    """``y = x + 1`` over ω², likewise."""
    x, y = omega_squared.symbolic().vars('x y')
    return x.succ(y).evaluate()


class TestCodec:
    def test_roundtrip(self, omega_squared):
        for value in [(0, 0), (0, 7), (1, 0), (3, 5), (12, 9)]:
            assert omega_squared.decode(omega_squared.encode(value)) == value

    def test_an_integer_is_the_finite_ordinal(self, omega_squared):
        assert omega_squared.encode(5) == omega_squared.encode((0, 5))

    def test_short_tuples_are_padded_with_leading_zeros(self):
        w3 = Ordinal(3)
        assert w3.encode((2, 5)) == w3.encode((0, 2, 5))

    def test_too_many_coefficients_are_rejected(self, omega_squared):
        with pytest.raises(ValueError, match="more than"):
            omega_squared.encode((1, 2, 3))

    def test_negative_coefficients_are_rejected(self, omega_squared):
        with pytest.raises(ValueError, match="natural"):
            omega_squared.encode((1, -2))


class TestOrder:
    """The order is reverse-lexicographic on the coefficients — checked
    exhaustively against the Python tuple order on a finite window of ω²."""

    def test_agrees_with_the_tuple_order(self, below):
        window = list(itertools.product(range(3), range(4)))
        for a, b in itertools.product(window, repeat=2):
            assert below.contains(x=a, y=b) == (a < b), (a, b)

    def test_the_finite_ordinals_sit_below_omega(self, below):
        for finite in (0, 1, 7, 1000):
            assert below.contains(x=finite, y=(1, 0))
            assert not below.contains(x=(1, 0), y=finite)

    def test_is_a_strict_total_order(self, omega, omega_squared):
        assert omega.is_total_order()
        assert omega_squared.is_total_order()


class TestSuccessor:
    def test_adds_one_to_the_constant_coefficient(self, successor):
        assert successor.contains(x=(1, 3), y=(1, 4))
        assert successor.contains(x=0, y=1)
        assert not successor.contains(x=(1, 3), y=(1, 5))
        # ω is a limit: it is nobody's successor, in particular not ω·0 + n
        assert not successor.contains(x=(0, 7), y=(1, 0))

    def test_symbolic_surface(self, omega_squared):
        S = omega_squared.symbolic()
        a, b = S.vars('a b')
        assert a.succ(b).evaluate().contains(a=(2, 0), b=(2, 1))
        assert a.lt(b).evaluate().contains(a=(0, 7), b=(1, 0))
        assert a.eq(b).evaluate().contains(a=(1, 1), b=(1, 1))


class TestOrdinalFacts:
    """First-order facts that hold of ω^n and distinguish it from ℕ."""

    def test_has_a_least_element_and_no_greatest(self, omega_squared):
        assert omega_squared.check('exists x.(all y.(Lt(x,y) | Eq(x,y)))')
        assert omega_squared.check('all x.(exists y.(Lt(x,y)))')

    def test_omega_squared_has_limit_ordinals_but_omega_does_not(
            self, omega, omega_squared):
        # a limit is neither zero nor a successor
        limit = ('exists x.((exists z.(Lt(z,x))) & '
                 '(not exists y.(Succ(y,x))))')
        assert not omega.check(limit)        # ℕ: every non-zero is a successor
        assert omega_squared.check(limit)    # ω, ω·2, … are limits

    def test_every_element_has_a_successor(self, omega_squared):
        assert omega_squared.check('all x.(exists y.(Succ(x,y)))')

    def test_the_order_is_discrete_upwards(self, omega_squared):
        # the successor is the immediate one: nothing sits strictly between
        assert omega_squared.check(
            'all x.(all y.(Succ(x,y) -> '
            '(not exists z.(Lt(x,z) & Lt(z,y)))))')


class TestConstruction:
    def test_exponent_must_be_positive(self):
        with pytest.raises(ValueError, match=">= 1"):
            Ordinal(0)

    def test_repr(self, omega_squared):
        assert repr(omega_squared) == "<Ordinal omega^2>"

    def test_omega_is_the_naturals(self, omega):
        # dimension 1: the interpretation is a definable reduct of ℕ, so the
        # order relation is the source's own, minimized
        from autstr.arithmetic import BuechiArithmetic
        assert omega.presentation.automata['Lt'].num_states == \
            BuechiArithmetic().automata['Lt'].num_states


# ======================================================================
# TreeOrdinal: the ordinals below omega^(omega^n), on the tree engine
# ======================================================================

from autstr.ordinals import TreeOrdinal                        # noqa: E402
from autstr.sparse_tree_automata import Tree                   # noqa: E402


@pytest.fixture(scope="module")
def omega_to_the_omega():
    return TreeOrdinal(1)


@pytest.fixture(scope="module")
def big():
    return TreeOrdinal(2)


def ordinal_key(structure, cantor):
    """The oracle: a Cantor map as the list of its terms in descending
    exponent order. Python compares those lists exactly as the ordinals they
    denote compare -- a larger leading exponent wins, then a larger leading
    coefficient, and a longer list wins over a prefix of itself since every
    further term is positive."""
    return sorted(structure._normalize(cantor).items(), reverse=True)


#: ordinals below omega^omega, written as Cantor maps
SMALL_SPACE = [0, 1, 2, 5, {1: 1}, {1: 2}, {1: 1, 0: 1}, {2: 1}, {2: 5},
               {2: 1, 1: 3, 0: 7}, {4: 1}, {4: 1, 0: 1}, {3: 2, 1: 1}]

#: ordinals below omega^(omega^2); an exponent is now a pair
BIG_SPACE = [0, 1, 7, {(0, 1): 1}, {(0, 2): 3}, {(1, 0): 1}, {(1, 0): 2},
             {(1, 0): 1, (0, 5): 4}, {(2, 0): 1}, {(1, 3): 1},
             {(1, 3): 1, (0, 0): 9}, {3: 2}]


class TestTreeCodec:
    def test_roundtrip(self, omega_to_the_omega):
        W = omega_to_the_omega
        for value in SMALL_SPACE:
            assert W.decode(W.encode(value)) == W._normalize(value)

    def test_zero_is_the_empty_map(self, omega_to_the_omega):
        W = omega_to_the_omega
        assert W.decode(W.encode(0)) == {}
        assert W.encode(0) == Tree('o')

    def test_an_integer_is_the_finite_ordinal(self, omega_to_the_omega):
        W = omega_to_the_omega
        assert W.encode(5) == W.encode({0: 5})

    def test_an_exponent_may_be_an_integer_or_a_short_tuple(self, big):
        assert big.encode({3: 2}) == big.encode({(0, 3): 2})
        assert big.decode(big.encode({3: 2})) == {(0, 3): 2}

    def test_a_zero_coefficient_is_no_term(self, omega_to_the_omega):
        W = omega_to_the_omega
        assert W.encode({2: 1, 5: 0}) == W.encode({2: 1})

    def test_a_negative_coefficient_is_rejected(self, omega_to_the_omega):
        with pytest.raises(ValueError, match="natural"):
            omega_to_the_omega.encode({1: -2})

    def test_too_long_an_exponent_is_rejected(self, omega_to_the_omega):
        with pytest.raises(ValueError, match="more than"):
            omega_to_the_omega.encode({(1, 2): 1})


class TestTreeUniverse:
    """One tree per ordinal: the universe automaton rejects every other tree,
    which is what makes the encoding a bijection and the order automaton's
    position-wise alignment sound."""

    def test_accepts_the_canonical_trees(self, omega_to_the_omega):
        W = omega_to_the_omega
        for value in SMALL_SPACE:
            assert W.automata['U'].accepts(W.encode(value)), value

    def test_rejects_a_trailing_zero_block(self, omega_to_the_omega):
        # a spine node with no payload is a zero block; as the deepest node it
        # would give the ordinal zero a second encoding
        assert not omega_to_the_omega.automata['U'].accepts(
            Tree('o', Tree('s', None, None), None))

    def test_rejects_a_leading_zero_bit(self, omega_to_the_omega):
        # the chain '01' is the coefficient 1 written with a leading zero:
        # the deepest bit is the most significant and must be a one
        chain = Tree('1', Tree('0', None, None), None)
        assert not omega_to_the_omega.automata['U'].accepts(
            Tree('o', Tree('s', None, chain), None))

    def test_rejects_the_wrong_nesting_depth(self, big):
        # at n = 2 a block is itself a spine, never a bare coefficient chain
        flat = Tree('o', Tree('s', None, Tree('1', None, None)), None)
        assert not big.automata['U'].accepts(flat)
        assert big.automata['U'].accepts(big.encode({(0, 0): 1}))


class TestTreeOrder:
    @pytest.mark.parametrize("structure_name,space",
                             [('omega_to_the_omega', SMALL_SPACE),
                              ('big', BIG_SPACE)])
    def test_agrees_with_the_cantor_oracle(self, structure_name, space,
                                           request):
        structure = request.getfixturevalue(structure_name)
        less, equal = structure.automata['Lt'], structure.automata['Eq']
        for a, b in itertools.product(space, repeat=2):
            x, y = structure.encode(a), structure.encode(b)
            key_a, key_b = (ordinal_key(structure, a),
                            ordinal_key(structure, b))
            assert less.accepts(x, y) == (key_a < key_b), (a, b)
            assert equal.accepts(x, y) == (key_a == key_b), (a, b)

    def test_is_a_strict_total_order(self, omega_to_the_omega, big):
        assert omega_to_the_omega.is_total_order()
        assert big.is_total_order()

    def test_the_symbolic_surface(self, omega_to_the_omega):
        a, b = omega_to_the_omega.symbolic().vars('a b')
        assert a.lt(b).evaluate().contains(a={2: 5}, b={3: 1})   # ω²·5 < ω³
        assert not a.lt(b).evaluate().contains(a={3: 1}, b={2: 5})

    def test_agrees_with_the_word_engine_where_they_overlap(
            self, omega_to_the_omega):
        """Two independent constructions of the same order: ω² sits inside
        ω^ω, and `Ordinal(2)` builds it by interpreting Büchi arithmetic while
        `TreeOrdinal(1)` builds it from hand-written tree automata."""
        word = Ordinal(2)
        x, y = word.symbolic().vars('x y')
        word_less = x.lt(y).evaluate()
        tree_less = omega_to_the_omega.automata['Lt']

        window = list(itertools.product(range(3), range(3)))
        for (a1, a0), (b1, b0) in itertools.product(window, repeat=2):
            tree = tree_less.accepts(
                omega_to_the_omega.encode({1: a1, 0: a0}),
                omega_to_the_omega.encode({1: b1, 0: b0}))
            assert word_less.contains(x=(a1, a0), y=(b1, b0)) == tree


class TestTreeSuccessor:
    """`Succ` is defined from the order rather than authored -- every ordinal
    has an immediate successor, so `y = x + 1` is first-order."""

    def test_semantics(self, omega_to_the_omega):
        # deferred: it is built the first time a query mentions it
        x, y = omega_to_the_omega.symbolic().vars('x y')
        successor = x.succ(y).evaluate()
        for a, b, want in [(3, 4, True), (3, 5, False),
                           ({1: 1}, {1: 1, 0: 1}, True),
                           ({1: 1, 0: 2}, {1: 1, 0: 3}, True),
                           (7, {1: 1}, False)]:        # ω is a limit
            assert successor.contains(x=a, y=b) == want, (a, b)

    def test_every_ordinal_has_one(self, omega_to_the_omega):
        assert omega_to_the_omega.check('all x.(exists y.(Succ(x,y)))')

    def test_the_order_is_discrete_upwards(self, omega_to_the_omega):
        assert omega_to_the_omega.check(
            'all x.(all y.(Succ(x,y) -> (not exists z.(Lt(x,z) & Lt(z,y)))))')


def _zero(v):
    return f'all q.(Lt({v},q) | Eq({v},q))'


def _limit(v):
    return f'((not ({_zero(v)})) & (not exists p.(Succ(p,{v}))))'


def _limit_of_limits(v):
    return (f'({_limit(v)} & all w.(Lt(w,{v}) -> exists u.('
            f'Lt(w,u) & Lt(u,{v}) & {_limit("u")})))')


class TestPastTheWordBarrier:
    """What the tree engine buys. In ω^n the limit ordinals stack only n-1
    deep -- the limits of ω² are the ω·k, and no ω·k is a limit of those. In
    ω^ω they stack arbitrarily, and ω² itself is a limit of limits. So one
    first-order sentence separates the two structures, and only the tree
    engine can decide it on ω^ω.
    """

    def test_both_have_limit_ordinals(self, omega_to_the_omega):
        assert omega_to_the_omega.check(f'exists x.({_limit("x")})')
        assert Ordinal(2).check(f'exists x.({_limit("x")})')

    def test_only_omega_to_the_omega_has_a_limit_of_limits(
            self, omega_to_the_omega):
        assert omega_to_the_omega.check(f'exists x.({_limit_of_limits("x")})')
        assert not Ordinal(2).check(f'exists x.({_limit_of_limits("x")})')

    def test_it_has_a_least_element_and_no_greatest(self, omega_to_the_omega):
        assert omega_to_the_omega.check('exists x.(all y.(Lt(x,y) | Eq(x,y)))')
        assert omega_to_the_omega.check('all x.(exists y.(Lt(x,y)))')


class TestTreeConstruction:
    def test_exponent_must_be_positive(self):
        with pytest.raises(ValueError, match=">= 1"):
            TreeOrdinal(0)

    def test_repr(self, omega_to_the_omega):
        assert repr(omega_to_the_omega) == "<TreeOrdinal omega^(omega^1)>"

    def test_the_relations(self, omega_to_the_omega):
        assert sorted(omega_to_the_omega.get_relation_symbols()) == \
            ['Eq', 'Lt', 'Succ', 'U']
