"""The symbolic layer over the tree engine.

Skolem arithmetic (N_{>0}, ·) is the oracle: every query has an obvious
reference answer in Python, so the compiled tree automaton can be checked
against it directly.
"""
from itertools import islice

import pytest

from autstr.buildin.tree_presentations import SkolemArithmetic
from autstr.sparse_tree_automata import convolve_trees
from autstr.symbolic import (
    CompileError, FunctionCodec, Signature, SymbolicSymbolError,
)
from autstr.utils.tree_automata_tools import canonical

enc = SkolemArithmetic.encode


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


class TestEnumeration:
    """Shortlex on the convolution: by node count, then by labels. That is the
    faithful analogue of the string engine's length-lexicographic order, but
    it orders by *encoding size*, which for Skolem is factorization complexity
    rather than magnitude."""

    def test_enumerates_exactly_the_divisor_pairs(self, S):
        x, y = S.vars('x y')
        got = list((x * y).eq(S.const(12)).evaluate())
        assert sorted(got) == sorted((a, 12 // a)
                                     for a in range(1, 13) if 12 % a == 0)
        assert len(got) == len(set(got))          # no tuple twice

    def test_sizes_are_non_decreasing(self, S):
        x, y = S.vars('x y')
        sizes = [convolve_trees([enc(a), enc(b)], S.backend.presentation
                                .base_alphabet, '*').size()
                 for a, b in (x * y).eq(S.const(12)).evaluate()]
        assert sizes == sorted(sizes)

    def test_an_infinite_relation_streams(self, S):
        x, y, z = S.vars('x y z')
        got = list(islice(iter((x * y).eq(z).evaluate()), 8))
        assert len(got) == 8
        for a, b, c in got:
            assert a * b == c, (a, b, c)

    def test_complexity_order_is_not_magnitude_order(self, S):
        """128 = 2^7 encodes smaller than the prime 7, because exponents are
        stored in binary while primes cost spine length. So the enumeration is
        by factorization complexity, not by magnitude -- which is why
        `iterate` stays structural and value order is left to the codec."""
        x, y = S.vars('x y')
        elements = [a for a, _ in islice(
            iter((x * S.const(1)).eq(y).evaluate()), 60)]
        assert 128 in elements and 7 in elements
        assert elements.index(128) < elements.index(7)


class TestExistsInfinitely:
    """`exinf` rewrites to "some witness runs k nodes deeper than every
    reference" -- the tree form of the string engine's "k letters longer",
    with k the body's state count, so the pigeonhole bites along a path."""

    def test_every_element_has_infinitely_many_multiples(self, S):
        x, y, z = S.vars('x y z')
        divides = (x * z).eq(y).drop('z')
        relation = divides.exinf('y').evaluate()
        for n in (1, 2, 5, 12):
            assert relation.contains(x=n), n

    def test_no_element_has_infinitely_many_divisors(self, S):
        x, y, z = S.vars('x y z')
        divides = (x * z).eq(y).drop('z')
        assert not divides.exinf('x').check()

    def test_agrees_with_finiteness_of_the_fibre(self, S):
        """The two answers must line up: the divisors of 12 are a finite
        fibre, so no y has infinitely many x with x * y = 12."""
        x, y = S.vars('x y')
        assert (x * y).eq(S.const(12)).evaluate().is_finite()
        assert not (x * y).eq(S.const(12)).exinf('x').check()


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
