"""First-order interpretations of automatic structures.

Automatic structures are closed under first-order interpretations, and AutStr
can *compute* the interpreted presentation: `presentation.evaluate(φ)` already
returns the automaton of an FO formula's satisfying assignments, so an
interpretation is orchestration, not a new engine. It is the FO counterpart of
the set / MSO interpretations that build the Caucal hierarchy.

**This module is Phase 1: one-dimensional, no quotient.** An element of the
interpreted structure is a single element of the source satisfying the domain
formula; each relation is defined by an FO formula over the source's signature.
Because elements keep the source's own encoding, the interpreted automata are
the canonical minimal DFAs of those formulas — bit-for-bit what a hand-built
presentation of the same relations would produce, with no encoding overhead.

Higher-dimensional interpretations (elements as k-tuples, encoded as k-tape
convolutions) and quotient interpretations are deliberately not here yet: the
product encoding can be wasteful when the domain realizes only a sparse subset
of the tuples, so they want an alphabet-pruning pass and a measurement against
a bespoke encoding first.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

from nltk.sem import logic

from autstr.presentations import AutomaticPresentation
from autstr.sparse_automata import SparseDFA
from autstr.utils.automata_tools import permute_tapes
from autstr.utils.logic import get_free_elementary_vars

Formula = Union[str, logic.Expression]
#: a relation is a formula plus the order its free variables play as arguments;
#: the order may be omitted when the sorted free variables already match
RelationSpec = Union[Formula, Tuple[Formula, Sequence[str]]]


def interpret(source: AutomaticPresentation, domain: Formula,
              relations: Dict[str, RelationSpec]) -> AutomaticPresentation:
    """The one-dimensional first-order interpretation of a structure in
    `source`.

    :param source: the structure to interpret in.
    :param domain: an FO formula δ(x) with exactly one free variable; the new
        universe is the source elements satisfying it.
    :param relations: ``{name: spec}`` where ``spec`` is a formula, or a
        ``(formula, argument-order)`` pair. The formula's free variables are
        the relation's arguments; give the order explicitly whenever their
        sorted order is not the intended argument order.
    :return: a fresh `AutomaticPresentation` over the source's alphabet, whose
        universe and relations are the (minimized) automata of those formulas.
    """
    universe = _evaluate(source, domain)
    if universe.symbol_arity != 1:
        raise ValueError(
            f"the domain formula must have exactly one free variable, "
            f"got arity {universe.symbol_arity}")

    automata: Dict[str, SparseDFA] = {'U': universe}
    for name, spec in relations.items():
        if name == 'U':
            raise ValueError("'U' is the reserved universe symbol")
        formula, order = spec if isinstance(spec, tuple) else (spec, None)
        automata[name] = _evaluate(source, formula, order)

    return AutomaticPresentation(automata, padding_symbol=source.padding_symbol)


def _evaluate(source: AutomaticPresentation, formula: Formula,
              order: Optional[Sequence[str]] = None) -> SparseDFA:
    """The automaton of `formula` over `source`, with its tapes put into
    `order` when one is given."""
    if isinstance(formula, str):
        formula = logic.Expression.fromstring(formula)
    dfa = source.evaluate(formula)

    free = get_free_elementary_vars(formula)     # the tape order evaluate used
    if order is None:
        return dfa
    order = list(order)
    if sorted(order) != sorted(free):
        raise ValueError(
            f"argument order {order} does not match the formula's free "
            f"variables {free}")
    permutation = [free.index(v) for v in order]
    if permutation == list(range(len(order))):
        return dfa
    return permute_tapes(dfa, permutation)
