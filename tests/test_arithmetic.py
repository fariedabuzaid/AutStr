"""Büchi arithmetic through the symbolic interface."""
from itertools import islice

import pytest

from autstr.arithmetic import decode, encode, integers


@pytest.fixture(scope='module')
def Z():
    return integers()


def bounded(formula, limit=6):
    """Solutions whose entries all stay within +/- limit, taken from a prefix
    of the length-lexicographic enumeration."""
    return [t for t in islice(iter(formula), 400)
            if all(abs(v) <= limit for v in t)]


# ----------------------------------------------------------------------
# encoding
# ----------------------------------------------------------------------

@pytest.mark.parametrize('n', [0, 1, 2, 7, 8, 255, -1, -2, -9, -256])
def test_encode_decode_roundtrip(n):
    assert decode(encode(n)) == n


def test_decode_ignores_padding():
    assert decode(encode(13) + ['*', '*']) == 13


# ----------------------------------------------------------------------
# terms
# ----------------------------------------------------------------------

def test_addition(Z):
    x, y, z = Z.vars('x y z')
    formula = (x + y).eq(z)
    assert not formula.is_empty()
    assert not formula.is_finite()
    for a, b, c in bounded(formula):
        assert a + b == c


def test_negation(Z):
    x, y = Z.vars('x y')
    formula = (-x).eq(y)
    for a, b in bounded(formula):
        assert b == -a


def test_subtraction(Z):
    x, y, z = Z.vars('x y z')
    for a, b, c in bounded((x - y).eq(z)):
        assert a - b == c


def test_integer_multiple(Z):
    x, y = Z.vars('x y')
    for a, b in bounded((2 * x).eq(y)):
        assert b == 2 * a


def test_larger_integer_multiple(Z):
    x, y = Z.vars('x y')
    relation = x.times(11).eq(y).evaluate()
    assert relation.contains(x=3, y=33)
    assert not relation.contains(x=3, y=32)


def test_python_integers_are_constants(Z):
    x, = Z.vars('x')
    formula = x.lt(10)
    assert not formula.is_empty()
    assert not formula.is_finite()
    for (a,) in bounded(formula):
        assert a < 10


# ----------------------------------------------------------------------
# relations
# ----------------------------------------------------------------------

def test_order(Z):
    x, y = Z.vars('x y')
    relation = x.lt(y).evaluate()
    assert relation.contains(x=-3, y=2)
    assert not relation.contains(x=2, y=-3)
    assert not relation.contains(x=2, y=2)


def test_power_of_two_divisibility(Z):
    x, y = Z.vars('x y')
    relation = x.divided_by_power(y).evaluate()
    assert relation.contains(x=12, y=4)     # 4 = 2^2 divides 12
    assert not relation.contains(x=12, y=8)  # 8 does not divide 12
    assert not relation.contains(x=12, y=3)  # 3 is not a power of two


def test_relations_reachable_by_name(Z):
    x, = Z.vars('x')
    N0 = Z.rel('N0')
    relation = N0(x).evaluate()
    assert relation.contains(x=5)
    assert not relation.contains(x=-5)


# ----------------------------------------------------------------------
# relational algebra
# ----------------------------------------------------------------------

def test_conjunction_and_projection(Z):
    x, y, z = Z.vars('x y z')
    constraints = (x + y).eq(z) & z.gt(0) & z.lt(3)
    for a, b, c in bounded(constraints):
        assert a + b == c and 0 < c < 3

    projected = constraints.drop(['z'])
    assert projected.variables() == ['x', 'y']
    for a, b in bounded(projected):
        assert 0 < a + b < 3

    for a, b in bounded(~projected):
        assert not 0 < a + b < 3


def test_membership(Z):
    x, y, z = Z.vars('x y z')
    formula = (x + y).eq(z)
    assert (1, 1, 2) in formula
    assert (10, 2, 13) not in formula
    assert formula.contains(x=10, y=2, z=12)


def test_exinf(Z):
    x, y = Z.vars('x y')
    # exactly one y equals each x
    assert x.eq(y).exinf('y').is_empty()
    # unboundedly many exceed it
    assert not x.lt(y).exinf('y').is_empty()


def test_universal_quantification(Z):
    x, y = Z.vars('x y')
    assert (x + y).eq(0).drop(y).all(x).check()      # every integer has an inverse
    assert x.lt(y).implies(~y.lt(x)).all([x, y]).check()


def test_finiteness_is_about_tuples(Z):
    x, y = Z.vars('x y')
    assert (x.eq(3) & y.eq(4)).is_finite()
    assert (x.gt(0) & x.lt(5)).is_finite()
    assert not x.lt(y).is_finite()


def test_enumeration_yields_integers(Z):
    x, y = Z.vars('x y')
    formula = (x + y).eq(4) & x.gt(0) & x.lt(y)
    assert sorted(islice(iter(formula), 5)) == [(1, 3)]
