"""The symbolic layer over the tree engine.

Skolem arithmetic (N_{>0}, ·) is the oracle: every query has an obvious
reference answer in Python, so the compiled tree automaton can be checked
against it directly.
"""
import pytest

from autstr.buildin.tree_presentations import SkolemArithmetic
from autstr.symbolic import (
    CompileError, FunctionCodec, Signature, SymbolicSymbolError,
)


@pytest.fixture(scope="module")
def skolem():
    return SkolemArithmetic(max_states=500_000)


@pytest.fixture(scope="module")
def S(skolem):
    signature = Signature(codec=FunctionCodec(SkolemArithmetic.encode,
                                              SkolemArithmetic.decode))
    signature.function('*', graph='M', out=2)
    signature.operator('*', '*')
    signature.operator('eq', 'E')
    return skolem.symbolic(signature)


class TestSignature:
    def test_relation_symbols_exclude_the_domain(self, S):
        assert 'U' not in S.backend.relation_symbols()
        assert set(S.backend.relation_symbols()) == {'M', 'E'}

    def test_arities_come_from_the_automata(self, S):
        assert S.backend.arity('M') == 3
        assert S.backend.arity('E') == 2

    def test_wrong_relation_arity_is_reported(self, S):
        x, y = S.vars('x y')
        with pytest.raises(CompileError, match='arity 3'):
            S.atom('M', [x, y]).evaluate()

    def test_wrong_function_arity_is_reported(self, S):
        x, = S.vars('x')
        with pytest.raises(SymbolicSymbolError, match='arity 2'):
            S.get_symbolic_func('*')(x)


class TestProduct:
    def test_membership_matches_multiplication(self, S):
        x, y, z = S.vars('x y z')
        relation = (x * y).eq(z).evaluate()
        for a, b in [(1, 1), (2, 3), (6, 2), (4, 5), (7, 3), (12, 1)]:
            assert relation.contains(x=a, y=b, z=a * b), (a, b)
            assert not relation.contains(x=a, y=b, z=a * b + 1), (a, b)

    def test_nested_terms(self, S):
        x, y, z, w = S.vars('x y z w')
        relation = ((x * y) * z).eq(w).evaluate()
        assert relation.contains(x=2, y=3, z=5, w=30)
        assert not relation.contains(x=2, y=3, z=5, w=31)


class TestSentences:
    def test_multiplication_is_commutative(self, S):
        x, y, z = S.vars('x y z')
        assert (x * y).eq(z).implies((y * x).eq(z)).all('x y z').check()

    def test_not_every_element_is_a_square(self, S):
        x, y = S.vars('x y')
        # exists x. not exists y. y * y = x  -- 2 is a witness.
        assert (~((y * y).eq(x).drop('y'))).drop('x').check()

    def test_divisibility_agrees_with_python(self, S):
        x, y, z = S.vars('x y z')
        divides = (x * z).eq(y).drop('z')
        relation = divides.evaluate()
        for a, b in [(2, 6), (3, 12), (4, 6), (5, 5), (7, 30), (6, 24)]:
            assert relation.contains(x=a, y=b) == (b % a == 0), (a, b)


class TestUnsupportedOperations:
    """The tree engine has no counterpart for these yet; each must say so
    rather than answer a different question."""

    def test_enumeration_explains_itself(self, S):
        x, y, z = S.vars('x y z')
        relation = (x * y).eq(z).evaluate()
        with pytest.raises(NotImplementedError, match="enumeration"):
            next(iter(relation))

    def test_finiteness_explains_itself(self, S):
        x, y, z = S.vars('x y z')
        relation = (x * y).eq(z).evaluate()
        with pytest.raises(NotImplementedError, match="canonical"):
            relation.is_finite()

    def test_constants_explain_themselves(self, S):
        x, y = S.vars('x y')
        with pytest.raises(NotImplementedError, match="assignment"):
            (x * y).eq(S.const(12)).evaluate()
