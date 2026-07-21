"""Evaluation targets for compiled symbolic queries.

A backend knows how to answer a first-order query, what a relation's arity is,
and how to move between element encodings and automaton tapes. Everything
above this module is backend-agnostic, which is what lets one expression
language serve both a single structure and a whole uniformly automatic class.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from autstr.buildin.automata import k_longer_automaton
from autstr.utils.automata_tools import (
    canonical, iterate_language, word_automaton,
)
from autstr.utils.logic import get_free_elementary_vars


class Backend:
    """The interface a `SymbolicContext` evaluates against."""

    def relation_symbols(self) -> List[str]:
        raise NotImplementedError

    def arity(self, symbol: str) -> Optional[int]:
        """Arity of a relation symbol, or None if it is unknown here."""
        raise NotImplementedError

    def arity_of(self, dfa) -> int:
        """Relation arity of an automaton produced by this backend."""
        raise NotImplementedError

    def evaluate(self, expression, updates: Dict, prepared: Dict):
        """Answer a query. Returns ``(dfa, tape names)``."""
        raise NotImplementedError

    def constant_automaton(self, word: Sequence):
        raise NotImplementedError

    def longer_witness_automaton(self, k: int, references: int):
        raise NotImplementedError

    def accepts(self, dfa, values: Sequence, codec) -> bool:
        raise NotImplementedError

    def iterate(self, dfa, codec, arity: int):
        raise NotImplementedError

    def is_finite(self, dfa) -> bool:
        """Whether the relation holds of finitely many *tuples*."""
        raise NotImplementedError

    def reserved_variable_names(self) -> set:
        """Variable names this backend cannot represent."""
        return set()

    # -- member-level evaluation, for classes only ---------------------
    def check_member(self, expression, advice, assignments, implicit):
        raise NotImplementedError(
            "member checking applies to a uniformly automatic class; a single "
            "structure has no members, so use `check` instead")

    def evaluate_member(self, expression, advice, assignments):
        raise NotImplementedError(
            "member evaluation applies to a uniformly automatic class; a "
            "single structure has no members, so use `evaluate` instead")

    def get_structure(self, advice):
        raise NotImplementedError(
            "only a uniformly automatic class instantiates member structures")

    def describe(self) -> str:
        raise NotImplementedError


def _convolve(words: Sequence[Sequence], padding_symbol) -> List[tuple]:
    """Pad a tuple of words to equal length and interleave them into the
    sequence of symbol tuples an automaton reads."""
    words = [list(w) for w in words]
    length = max((len(w) for w in words), default=0)
    for word in words:
        word.extend([padding_symbol] * (length - len(word)))
    return [tuple(w[i] for w in words) for i in range(length)]


class StructureBackend(Backend):
    """Queries answered by an `AutomaticPresentation`."""

    def __init__(self, presentation):
        self.presentation = presentation

    def relation_symbols(self):
        return [s for s in self.presentation.get_relation_symbols() if s != 'U']

    def arity(self, symbol):
        dfa = self.presentation.automata.get(symbol)
        return None if dfa is None else dfa.symbol_arity

    def arity_of(self, dfa):
        return dfa.symbol_arity

    def evaluate(self, expression, updates, prepared):
        dfa = self.presentation.evaluate(
            expression,
            updates=dict(updates) if updates else None,
            prepared_updates=dict(prepared) if prepared else None)
        return dfa, get_free_elementary_vars(expression)

    def constant_automaton(self, word):
        return word_automaton(word, self.presentation.sigma,
                              self.presentation.padding_symbol)

    def longer_witness_automaton(self, k, references):
        return k_longer_automaton(k, references, self.presentation.sigma,
                                  self.presentation.padding_symbol)

    def accepts(self, dfa, values, codec):
        if codec is None:
            words = values
        else:
            words = [codec.encode(v) for v in values]
        return dfa.accepts(_convolve(words, self.presentation.padding_symbol))

    def iterate(self, dfa, codec, arity):
        padding = self.presentation.padding_symbol
        for tapes in iterate_language(dfa, backward=True, padding_symbol=padding):
            # `iterate_language` builds its words right-to-left, so each tape
            # comes back reversed with respect to the automaton's reading order.
            words = [tape[::-1] for tape in tapes]
            if codec is None:
                yield tuple(words)
            else:
                yield tuple(codec.decode(w) for w in words)

    def is_finite(self, dfa):
        # The tuple set is finite iff the canonical convolutions are; the
        # automaton's own language never is, thanks to the all-padding loops.
        return canonical(dfa, self.presentation.padding_symbol).is_finite()

    def describe(self):
        return f"structure with relations {sorted(self.relation_symbols())}"


class ClassBackend(Backend):
    """Queries answered by a `UniformlyAutomaticClass`.

    Formulas are written over the class signature exactly as for a single
    structure; the advice tape is added and quantifiers are relativized to the
    member domain by the class's own evaluator. The advice appears in results
    under the reserved tape name ``'advice'``.
    """

    ADVICE = 'advice'

    def __init__(self, klass):
        self.klass = klass

    def relation_symbols(self):
        return [s for s in self.klass.get_relation_symbols() if s != 'U']

    def arity(self, symbol):
        dfa = self.klass.class_automata.get(symbol)
        # Tape 0 carries the advice, so the relation arity is one less.
        return None if dfa is None else dfa.symbol_arity - 1

    def arity_of(self, dfa):
        return dfa.symbol_arity - 1

    def evaluate(self, expression, updates, prepared):
        if updates or prepared:
            raise NotImplementedError(
                "constants and spliced automata are not yet supported over a "
                "uniformly automatic class; use `define` to name a derived "
                "class relation instead")
        dfa, variables = self.klass.evaluate(expression)
        return dfa, variables

    def reserved_variable_names(self):
        # A user variable called 'advice' would be indistinguishable from the
        # advice tape in the result's tape list.
        return {self.ADVICE}

    def constant_automaton(self, word):
        raise NotImplementedError(
            "a class element's encoding depends on the advice, so constants "
            "have no advice-free meaning")

    def longer_witness_automaton(self, k, references):
        raise NotImplementedError("exinf is not yet supported over a class")

    def accepts(self, dfa, values, codec):
        raise NotImplementedError(
            "membership over a class needs an advice string; instantiate a "
            "member with `get_structure(advice)` first")

    def iterate(self, dfa, codec, arity):
        raise NotImplementedError(
            "enumeration over a class needs an advice string; instantiate a "
            "member with `get_structure(advice)` first")

    def is_finite(self, dfa):
        raise NotImplementedError(
            "finiteness over a class is a question about a member; instantiate "
            "one with `get_structure(advice)` first")

    # -- member-level evaluation --------------------------------------
    def check_member(self, expression, advice, assignments, implicit):
        check = self.klass.check_implicit if implicit else self.klass.check
        return check(expression, advice, **assignments)

    def evaluate_member(self, expression, advice, assignments):
        return self.klass.evaluate_implicit(expression, advice, **assignments)

    def get_structure(self, advice):
        return self.klass.get_structure(advice)

    def describe(self):
        return f"class with relations {sorted(self.relation_symbols())}"
