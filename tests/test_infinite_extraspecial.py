"""The infinite extraspecial p-group.

A single infinite structure rather than a class of finite ones, so its
elements have an advice-free encoding and can be written as constants. That
combination -- a non-commutative operator plus a codec -- is what makes it the
structure that can tell `a * x` from `x * a`.
"""
import random

import pytest

from autstr.groups import InfiniteExtraspecialGroup

U = ((1,), (0,), 0)      # a = (1), b = (0)
V = ((0,), (1,), 0)      # a = (0), b = (1)


@pytest.fixture(scope="module")
def group():
    return InfiniteExtraspecialGroup(3)


@pytest.fixture(scope="module")
def S(group):
    return group.symbolic()


class TestPresentation:
    def test_encoding_roundtrips(self, group):
        rng = random.Random(3)
        for _ in range(30):
            n = rng.randint(0, 4)
            a = tuple(rng.randrange(3) for _ in range(n))
            b = tuple(rng.randrange(3) for _ in range(n))
            element = (a, b, rng.randrange(3))
            assert group.decode(group.encode(element)) == element

    def test_product_matches_the_reference(self, group, S):
        x, y, z = S.vars('x y z')
        product = (x * y).eq(z).evaluate()
        rng = random.Random(11)
        for _ in range(40):
            def sample():
                n = rng.randint(0, 3)
                return (tuple(rng.randrange(3) for _ in range(n)),
                        tuple(rng.randrange(3) for _ in range(n)),
                        rng.randrange(3))
            a, b = sample(), sample()
            want = group.multiply(a, b)
            assert product.contains(x=a, y=b, z=want), (a, b)
            other = (want[0], want[1], (want[2] + 1) % 3)
            assert not product.contains(x=a, y=b, z=other), (a, b)

    def test_it_really_is_non_abelian(self, group, S):
        assert group.multiply(U, V) != group.multiply(V, U)
        x, y = S.vars('x y')
        assert not (x * y).eq(y * x).all('x y').check()


class TestOperandOrder:
    """`a * x` must compile as the product in that order. When the left
    operand is a plain Python value, Python routes it to `Term.__rmul__`,
    which has to swap -- without the swap the query silently answers about
    `x * a`, and only a non-commutative structure with a codec can see it."""

    def test_reflected_multiplication_keeps_the_order(self, group, S):
        x, w = S.vars('x w')
        # U * x, with U a Python value: this is Term.__rmul__
        relation = (U * x).eq(w).evaluate()
        assert relation.contains(x=V, w=group.multiply(U, V))
        assert not relation.contains(x=V, w=group.multiply(V, U))

    def test_it_differs_from_the_forward_direction(self, group, S):
        x, w = S.vars('x w')
        forward = (x * U).eq(w).evaluate()
        assert forward.contains(x=V, w=group.multiply(V, U))
        assert not forward.contains(x=V, w=group.multiply(U, V))

    def test_reflected_and_explicit_constant_agree(self, S):
        x, w = S.vars('x w')
        assert (U * x).eq(w) == (S.const(U) * x).eq(w)
