"""The symbolic layer over the tree engine.

Skolem arithmetic (N_{>0}, ·) is the oracle: every query has an obvious
reference answer in Python, so the compiled tree automaton can be checked
against it directly.
"""
import pytest

from autstr.buildin.tree_presentations import SkolemArithmetic
from autstr.utils.tree_automata_tools import canonical
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


class TestFiniteness:
    """Finiteness is a question about tuples, so the canonical convolutions
    have to be taken first."""

    def test_divisor_pairs_are_finite(self, S):
        x, y = S.vars('x y')
        assert (x * y).eq(S.const(12)).evaluate().is_finite()

    def test_multiples_are_infinite(self, S):
        y, z = S.vars('y z')
        assert not (S.const(3) * z).eq(y).drop('z').evaluate().is_finite()

    def test_the_whole_product_relation_is_infinite(self, S):
        x, y, z = S.vars('x y z')
        assert not (x * y).eq(z).evaluate().is_finite()

    def test_a_singleton_is_finite(self, S):
        x = S.vars('x')[0]
        assert (x * S.const(1)).eq(S.const(7)).evaluate().is_finite()

    def test_canonical_is_what_makes_it_a_question_about_tuples(self, S):
        """Without trimming the attached all-padding regions, a saturated
        automaton accepts each tuple in infinitely many spellings -- so the
        finite relation would look infinite. This is the tree counterpart of
        the string bug that `automata_tools.canonical` fixed."""
        x, y = S.vars('x y')
        sta = (x * y).eq(S.const(12)).evaluate().dfa
        assert not sta.is_finite()                      # infinitely many trees
        assert canonical(sta, '*').is_finite()          # finitely many tuples


class TestUnsupportedOperations:
    """The tree engine has no counterpart for these yet; each must say so
    rather than answer a different question."""

    def test_enumeration_explains_itself(self, S):
        x, y, z = S.vars('x y z')
        relation = (x * y).eq(z).evaluate()
        with pytest.raises(NotImplementedError, match="enumeration"):
            next(iter(relation))


class TestConstants:
    """A constant is spliced into the query as a temporary relation, prepared
    exactly as at construction time."""

    def test_divisor_pairs_of_a_constant(self, S):
        x, y = S.vars('x y')
        relation = (x * y).eq(S.const(12)).evaluate()
        for a in range(1, 14):
            for b in range(1, 14):
                assert relation.contains(x=a, y=b) == (a * b == 12), (a, b)

    def test_constant_on_both_sides(self, S):
        x = S.vars('x')[0]
        relation = (x * S.const(3)).eq(S.const(12)).evaluate()
        assert relation.contains(x=4)
        assert not relation.contains(x=3)
        assert not relation.contains(x=12)

    def test_splicing_leaves_the_presentation_unchanged(self, S, skolem):
        before = dict(skolem.automata)
        x, y = S.vars('x y')
        (x * y).eq(S.const(12)).evaluate()
        assert dict(skolem.automata) == before
