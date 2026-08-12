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
supported at every dimension: pass ``quotient=ε``, and the universe is
restricted to the shortlex-least representative of each class.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

from nltk.sem import logic

from autstr.presentations import AutomaticPresentation
from autstr.sparse_automata import SparseDFA
from autstr.utils.automata_tools import (
    fold_tapes, permute_tapes, shortlex_order,
)
from autstr.utils.logic import get_free_elementary_vars

#: scaffolding relations a quotient interpretation installs on the intermediate
#: structure and drops from the result
_EQUIV = '_Equiv'
_LE = '_Le'

Formula = Union[str, logic.Expression]
#: a relation is a formula, or a ``(formula, coordinate-order)`` pair; the
#: coordinate order lists the free variables element-major (each element's k
#: coordinates together), and may be omitted when the sorted order already is
RelationSpec = Union[Formula, Tuple[Formula, Sequence[str]]]


def interpret(source: AutomaticPresentation, domain: RelationSpec,
              relations: Dict[str, RelationSpec],
              dimension: int = 1,
              quotient: Optional[RelationSpec] = None) -> AutomaticPresentation:
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
    :param quotient: an equivalence formula ε(x̄, ȳ) (a binary relation over
        elements, so ``2 · dimension`` free variables). When given, elements
        are its equivalence classes: the universe is restricted to the
        shortlex-least representative of each class, and the relations are read
        on those representatives. The caller must ensure ε really is an
        equivalence and that every relation is ε-invariant.
    :return: a fresh `AutomaticPresentation`. For k > 1 its alphabet is the
        source alphabet's k-fold product.
    """
    if dimension < 1:
        raise ValueError("dimension must be >= 1")
    if quotient is not None:
        return _quotient(source, domain, relations, dimension, quotient)
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


def _quotient(source: AutomaticPresentation, domain: RelationSpec,
              relations: Dict[str, RelationSpec], dimension: int,
              equivalence: RelationSpec) -> AutomaticPresentation:
    """Interpret with a quotient, in two stages: first the plain (k-dim)
    interpretation carrying the equivalence as a relation, then a
    one-dimensional interpretation restricting the universe to the
    shortlex-least representative of each class.

    The representative predicate is first-order over the equivalence and a
    shortlex well-order on the element encoding — ``x`` is canonical iff it is
    ``<=`` every element equivalent to it — so the whole thing stays inside the
    engine.
    """
    if _EQUIV in relations or _LE in relations:
        raise ValueError(f"{_EQUIV!r} and {_LE!r} are reserved for the "
                         f"quotient construction")
    raw = interpret(source, domain, {**relations, _EQUIV: equivalence},
                    dimension)
    raw.update(**{_LE: shortlex_order(raw.sigma, raw.padding_symbol)})

    representative = f'all y.({_EQUIV}(x,y) -> {_LE}(x,y))'
    specs: Dict[str, RelationSpec] = {}
    for name in relations:
        arity = raw.relation(name).symbol_arity
        args = [f'v{i}' for i in range(arity)]
        specs[name] = (f'{name}({",".join(args)})', args)
    return interpret(raw, (representative, ['x']), specs)


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
