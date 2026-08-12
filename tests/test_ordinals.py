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
        from autstr.buildin.presentations import BuechiArithmetic
        assert omega.presentation.automata['Lt'].num_states == \
            BuechiArithmetic().automata['Lt'].num_states
