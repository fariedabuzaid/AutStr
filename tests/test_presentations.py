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
