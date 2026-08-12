"""Invariants of `AutomaticPresentation` itself.

A presentation's relations are relations *over the universe*: a tuple whose
coordinates are not well-formed encodings must not satisfy any of them. The
engine relies on that — quantifiers restrict a bound variable to `U`, so an
unrestricted relation makes a universal sentence look false on encodings that
are not elements at all.
"""
from autstr.buildin.presentations import (
    BuechiArithmetic, BuechiArithmeticZ, MSO0,
)
from autstr.interpretations import interpret


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


class TestSentencesUnderConnectives:
    """A sentence has no free variables, so it evaluates to the all/none
    marker rather than to a relation with tapes. Placing that marker into an
    enclosing conjunction or disjunction used to be attempted as a tape
    renaming, which raised `IndexError` — for two sentences there is no tape to
    rename to at all.
    """

    def test_two_sentences_conjoined(self):
        arithmetic = BuechiArithmetic()
        true, false = 'all x.(Eq(x,x))', 'all x.(Lt(x,x))'
        assert arithmetic.check(f'({true}) & ({true})')
        assert not arithmetic.check(f'({true}) & ({false})')
        assert not arithmetic.check(f'({false}) & ({true})')

    def test_two_sentences_disjoined(self):
        arithmetic = BuechiArithmetic()
        true, false = 'all x.(Eq(x,x))', 'all x.(Lt(x,x))'
        assert arithmetic.check(f'({false}) | ({true})')
        assert not arithmetic.check(f'({false}) | ({false})')

    def test_a_sentence_beside_an_open_formula(self):
        """The sentence contributes its truth value and the open formula its
        relation: a true conjunct must leave the relation alone, a false one
        must empty it, and a true disjunct must fill it with the whole
        universe."""
        arithmetic = BuechiArithmetic()
        true, false = 'all x.(Eq(x,x))', 'all x.(Lt(x,x))'

        less = arithmetic.evaluate('Lt(y,z)')
        assert _same_language(arithmetic.evaluate(f'({true}) & Lt(y,z)'), less)
        assert arithmetic.evaluate(f'({false}) & Lt(y,z)').is_empty()

        assert _same_language(arithmetic.evaluate(f'({true}) | Lt(y,z)'),
                              arithmetic._domain_product(2))
        assert _same_language(arithmetic.evaluate(f'({false}) | Lt(y,z)'), less)

    def test_a_true_disjunct_does_not_admit_non_elements(self):
        """The full relation of a structure is the product of its universes,
        not every word: a true sentence disjoined with an open formula must
        still reject encodings that are not elements."""
        arithmetic = BuechiArithmetic()
        filled = arithmetic.evaluate('(all x.(Eq(x,x))) | Lt(y,z)')
        assert not filled.accepts([('0', '0'), ('0', '1')])   # '00' is not a
        assert filled.accepts([('0', '0')])                   # well-formed 0
