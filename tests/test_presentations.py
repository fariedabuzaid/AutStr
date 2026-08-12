"""Invariants of `AutomaticPresentation` itself.

A presentation's relations are relations *over the universe*: a tuple whose
coordinates are not well-formed encodings must not satisfy any of them. The
engine relies on that — quantifiers restrict a bound variable to `U`, so an
unrestricted relation makes a universal sentence look false on encodings that
are not elements at all.
"""
import itertools

import pytest

from autstr.buildin.automata import zero
from autstr.buildin.presentations import (
    BuechiArithmetic, BuechiArithmeticZ, MSO0,
)
from autstr.interpretations import interpret
from autstr.presentations import AutomaticPresentation


def _leaks(presentation, name) -> bool:
    """Does the relation accept a tuple outside the universe?"""
    dfa = presentation.automata[name]
    domain = presentation._domain_product(dfa.symbol_arity)
    return not dfa.intersection(domain.complement()).minimize().is_empty()


class TestRelationsLiveInTheUniverse:
    """Regression: the restriction skipped the *last* tape, so unary relations
    were never restricted at all. `N0` then accepted the single letter '0',
    which is not a well-formed integer, and every universal sentence over an
    interpretation built from `N0` came out false."""

    def test_builtin_relations_are_restricted(self):
        for presentation in (BuechiArithmetic(), BuechiArithmeticZ(), MSO0()):
            for name in presentation.automata:
                if name == 'U':
                    continue
                assert not _leaks(presentation, name), name

    def test_unary_relation_rejects_a_malformed_encoding(self):
        z = BuechiArithmeticZ()
        assert not z.automata['U'].accepts([('0',)])
        assert not z.automata['N0'].accepts([('0',)])

    def test_a_universal_sentence_survives_an_interpretation(self):
        # δ = N0, Id = equality: every element of the new universe satisfies
        # Id(x, x), so `all x. Id(x,x)` must hold.
        z = BuechiArithmeticZ()
        naturals = interpret(z, domain=('N0(x)', ['x']),
                             relations={'Id': ('Eq(x,y)', ['x', 'y'])})
        assert naturals.check('all x. Id(x,x)')
        assert not naturals.check('exists x.(not Id(x,x))')


def _same_language(one, other) -> bool:
    """Language equality: the symmetric difference is empty."""
    return (one.intersection(other.complement()).is_empty() and
            other.intersection(one.complement()).is_empty())


#: a true and a false sentence over Buechi arithmetic
TRUE, FALSE = 'all x.(Eq(x,x))', 'all x.(Lt(x,x))'

#: the binary connectives, with their Python truth tables
CONNECTIVES = {
    '&': lambda a, b: a and b,
    '|': lambda a, b: a or b,
    '->': lambda a, b: (not a) or b,
    '<->': lambda a, b: a == b,
}


def _convolution(a: int, b: int):
    """The two-tape encoding of a pair of naturals, padded to equal length."""
    wa, wb = list(format(a, 'b')[::-1]), list(format(b, 'b')[::-1])
    n = max(len(wa), len(wb))
    return list(zip(wa + ['*'] * (n - len(wa)), wb + ['*'] * (n - len(wb))))


class TestSentencesUnderConnectives:
    """A sentence has no free variables, so it evaluates to the all/none
    marker rather than to a relation with tapes: an arity-1 automaton that is
    non-empty exactly when the sentence is true.

    The marker is deliberately not the universe -- over an empty structure a
    universal sentence is still vacuously true -- so placing it into an
    enclosing formula is not a tape renaming. Attempting one raised IndexError,
    and for two sentences there was no tape to rename to at all.
    """

    @pytest.mark.parametrize("connective", list(CONNECTIVES))
    @pytest.mark.parametrize("left,right", list(itertools.product([True, False],
                                                                 repeat=2)))
    def test_two_sentences(self, connective, left, right):
        arithmetic = BuechiArithmetic()
        phi = (f'({TRUE if left else FALSE}) {connective} '
               f'({TRUE if right else FALSE})')
        expected = CONNECTIVES[connective](left, right)
        assert arithmetic.check(phi) == expected
        assert arithmetic.check(f'not ({phi})') == (not expected)

    def test_sentences_nest(self):
        arithmetic = BuechiArithmetic()
        assert arithmetic.check(f'(({TRUE}) | ({FALSE})) & (not ({FALSE}))')
        assert arithmetic.check(
            f'all y.((({TRUE}) & ({TRUE})) & Eq(y,y))')
        assert not arithmetic.check(f'(all w.({TRUE})) & (exists w.({FALSE}))')

    @pytest.mark.parametrize("connective", list(CONNECTIVES))
    @pytest.mark.parametrize("truth", [True, False])
    @pytest.mark.parametrize("first", [True, False])
    def test_a_sentence_beside_an_open_formula(self, connective, truth, first):
        """The sentence contributes its truth value and the open formula its
        relation, on either side of the connective."""
        arithmetic = BuechiArithmetic()
        sentence = TRUE if truth else FALSE
        phi = (f'({sentence}) {connective} Lt(y,z)' if first
               else f'Lt(y,z) {connective} ({sentence})')
        relation = arithmetic.evaluate(phi)

        for a, b in itertools.product(range(4), repeat=2):
            expected = (CONNECTIVES[connective](truth, a < b) if first
                        else CONNECTIVES[connective](a < b, truth))
            assert relation.accepts(_convolution(a, b)) == expected, (a, b)

    def test_a_true_operand_is_the_universe_not_every_word(self):
        """A subformula's automaton is a relation over the universe. The
        marker is not -- it accepts every word -- so a true sentence beside an
        open formula must contribute the product of universes instead, or the
        result admits encodings that are not elements."""
        arithmetic = BuechiArithmetic()
        filled = arithmetic.evaluate(f'({TRUE}) | Lt(y,z)')
        assert not filled.accepts([('0', '0'), ('0', '1')])   # '00' is not a
        assert filled.accepts([('0', '0')])                   # well-formed 0
        assert _same_language(filled, arithmetic._domain_product(2))

        kept = arithmetic.evaluate(f'({TRUE}) & Lt(y,z)')
        assert _same_language(kept, arithmetic.evaluate('Lt(y,z)'))
        assert arithmetic.evaluate(f'({FALSE}) & Lt(y,z)').is_empty()


class TestSentencesOverAnEmptyUniverse:
    """The degenerate case the marker convention exists for: over an empty
    structure every universal sentence is vacuously true, which is why a true
    sentence is the all-marker and not the universe. An open formula beside it
    still collapses, since every relation over an empty universe is empty.
    """

    @staticmethod
    def _empty():
        sigma = {'0', '1', '*'}
        return AutomaticPresentation(
            {'U': zero(1, sigma), 'Eq': zero(2, sigma), 'Lt': zero(2, sigma)},
            padding_symbol='*')

    def test_a_universal_sentence_is_vacuously_true(self):
        assert self._empty().check('all x.(Lt(x,x))')

    def test_an_existential_sentence_is_false(self):
        assert not self._empty().check('exists x.(Eq(x,x))')

    def test_vacuous_truth_survives_a_connective(self):
        empty = self._empty()
        assert empty.check('(all x.(Lt(x,x))) & (all x.(Lt(x,x)))')
        assert empty.check('(all x.(Lt(x,x))) | (exists x.(Eq(x,x)))')
        assert not empty.check('(all x.(Lt(x,x))) & (exists x.(Eq(x,x)))')

    def test_an_open_formula_beside_it_is_still_empty(self):
        empty = self._empty()
        assert empty.evaluate('(all x.(Lt(x,x))) | Lt(y,z)').is_empty()
