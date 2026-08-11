"""First-order interpretations of automatic structures.

Automatic structures are closed under first-order interpretations, and AutStr
can *compute* the interpreted presentation: `presentation.evaluate(φ)` already
returns the automaton of an FO formula's satisfying assignments, so an
interpretation is orchestration, not a new engine. It is the FO counterpart of
the set / MSO interpretations that build the Caucal hierarchy.

An element of the interpreted structure is a **k-tuple** of source elements
satisfying the domain formula; each relation is defined by an FO formula over
the source's signature, with its free variables grouped k-per-argument. The
formula automata are folded — every k consecutive coordinate tapes become one
element tape over the product alphabet — so the result is a standard
presentation whose elements are k-tuples.

For ``dimension == 1`` (the default) this is just a definable reduct/expansion
with a restricted domain: elements keep the source's encoding, and the
interpreted automata are the canonical minimal DFAs of the formulas — bit-for-
bit what a hand-built presentation would produce. For ``k > 1`` elements are
encoded as k-tape convolutions; measurement shows this adds no states (the
engine minimizes away a sparse domain's redundancy) but a bounded ×k
symbol-width overhead, intrinsic to representing tuples.

Quotient interpretations (elements as classes of a definable equivalence) are
not here yet.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

from nltk.sem import logic

from autstr.presentations import AutomaticPresentation
from autstr.sparse_automata import SparseDFA
from autstr.utils.automata_tools import fold_tapes, permute_tapes
from autstr.utils.logic import get_free_elementary_vars

Formula = Union[str, logic.Expression]
#: a relation is a formula, or a ``(formula, coordinate-order)`` pair; the
#: coordinate order lists the free variables element-major (each element's k
#: coordinates together), and may be omitted when the sorted order already is
RelationSpec = Union[Formula, Tuple[Formula, Sequence[str]]]


def interpret(source: AutomaticPresentation, domain: RelationSpec,
              relations: Dict[str, RelationSpec],
              dimension: int = 1) -> AutomaticPresentation:
    """The first-order interpretation of a structure in `source`.

    :param source: the structure to interpret in.
    :param domain: the domain formula δ(x̄) with ``dimension`` free variables —
        the coordinates of one element; the new universe is the tuples
        satisfying it. A ``(formula, coordinate-order)`` pair fixes which free
        variable is which coordinate when the sorted order will not do.
    :param relations: ``{name: spec}`` where ``spec`` is a formula or a
        ``(formula, coordinate-order)`` pair. A relation of arity r has
        ``dimension · r`` free variables; the coordinate order lists them
        element-major, so folding groups each argument's coordinates.
    :param dimension: k — elements are k-tuples of source elements.
    :return: a fresh `AutomaticPresentation`. For k > 1 its alphabet is the
        source alphabet's k-fold product.
    """
    if dimension < 1:
        raise ValueError("dimension must be >= 1")
    k = dimension

    universe = _fold_formula(source, domain, k)
    if universe.symbol_arity != 1:
        raise ValueError(
            f"the domain formula must have {k} free variable(s) — the "
            f"coordinates of one element — but defines "
            f"{universe.symbol_arity} element(s)")

    automata: Dict[str, SparseDFA] = {'U': universe}
    for name, spec in relations.items():
        if name == 'U':
            raise ValueError("'U' is the reserved universe symbol")
        automata[name] = _fold_formula(source, spec, k)

    padding = source.padding_symbol if k == 1 \
        else (source.padding_symbol,) * k
    return AutomaticPresentation(automata, padding_symbol=padding)


def _fold_formula(source: AutomaticPresentation, spec: RelationSpec,
                  k: int) -> SparseDFA:
    """Evaluate a formula over `source`, order its tapes coordinate-major, and
    fold every k of them into one element tape."""
    formula, order = spec if isinstance(spec, tuple) else (spec, None)
    if isinstance(formula, str):
        formula = logic.Expression.fromstring(formula)
    dfa = source.evaluate(formula)

    free = get_free_elementary_vars(formula)     # the tape order evaluate used
    order = free if order is None else list(order)
    if sorted(order) != sorted(free):
        raise ValueError(
            f"coordinate order {order} does not match the formula's free "
            f"variables {free}")
    if len(order) % k:
        raise ValueError(
            f"formula has {len(order)} free variables, not a multiple of the "
            f"dimension {k}")

    permutation = [free.index(v) for v in order]
    if permutation != list(range(len(order))):
        dfa = permute_tapes(dfa, permutation)
    return fold_tapes(dfa, k) if k > 1 else dfa
