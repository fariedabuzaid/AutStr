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
supported at every dimension and on either engine: pass ``quotient=ε``, and the
universe is restricted to one representative per class.

**Both engines.** The source may be an `AutomaticPresentation` or a
`TreeAutomaticPresentation`, and the result is a presentation of the same kind;
the orchestration is identical because both engines encode a convolution letter
the same way. Over trees an element of a k-dimensional interpretation is a
k-tuple of trees, which *is* one tree over k-tuples — the same fold, since the
tree convolution already overlays the shapes.

Only the choice of representative differs. Over words it is the shortlex-least
element of the class, and shortlex is a well-order, so that element exists.
Over trees no automatic order is well-founded — growing a tree at the position
where two differ makes it *smaller* — so a class need have no least element at
all, and the representative is instead the least *description*: a member that
reaches only so far past the positions the whole class shares. That is Kuske
and Weidner's construction; `_tree_representatives` gives it in full, including
why the expensive half of their proof is not needed here. It is the one part of
this module that can blow up — provably so — and it is worth passing
``max_states`` to the source presentation before asking for a large one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

from nltk.sem import logic

from autstr.presentations import AutomaticPresentation
from autstr.utils.logic import get_free_elementary_vars

#: scaffolding relations a quotient interpretation installs on the intermediate
#: structure and drops from the result
_EQUIV = '_Equiv'
_LE = '_Le'
_REP = '_Rep'

@dataclass(frozen=True)
class _Engine:
    """The tape machinery of whichever engine the source belongs to.

    Interpreting is the same orchestration over words and over trees — evaluate
    the formulas, order their tapes, fold every k into one element tape — and
    the operations differ only in which automaton type they act on. Both
    engines encode a symbol the same way (an MTBDD over the binary digits of
    the convolution letter, tape-major), which is why the fold is literally the
    same diagram surgery on either side.
    """
    name: str
    permute_tapes: Callable
    fold_tapes: Callable
    presentation: Callable
    #: given the interpreted presentation carrying the equivalence as
    #: `_EQUIV`, the automaton of one representative per class
    representatives: Callable


def _engine_for(source) -> _Engine:
    from autstr.tree_presentations import TreeAutomaticPresentation
    if isinstance(source, TreeAutomaticPresentation):
        from autstr.utils import tree_automata_tools as trees
        return _Engine(
            name='tree',
            permute_tapes=trees.permute_tapes,
            fold_tapes=trees.fold_tapes,
            presentation=lambda automata, padding: TreeAutomaticPresentation(
                automata, padding_symbol=padding,
                max_states=source.max_states),
            representatives=_tree_representatives)

    from autstr.utils import automata_tools as words
    return _Engine(
        name='string',
        permute_tapes=words.permute_tapes,
        fold_tapes=words.fold_tapes,
        presentation=lambda automata, padding: AutomaticPresentation(
            automata, padding_symbol=padding),
        representatives=_string_representatives)


Formula = Union[str, logic.Expression]
#: a relation is a formula, or a ``(formula, coordinate-order)`` pair; the
#: coordinate order lists the free variables element-major (each element's k
#: coordinates together), and may be omitted when the sorted order already is
RelationSpec = Union[Formula, Tuple[Formula, Sequence[str]]]


def interpret(source, domain: RelationSpec,
              relations: Dict[str, RelationSpec],
              dimension: int = 1,
              quotient: Optional[RelationSpec] = None):
    """The first-order interpretation of a structure in `source`.

    :param source: the structure to interpret in — an `AutomaticPresentation`
        or a `TreeAutomaticPresentation`. The result is a presentation of the
        same kind.
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
        are its equivalence classes: the universe is restricted to one
        representative of each class, and the relations are read on those
        representatives. The caller must ensure ε really is an equivalence and
        that every relation is ε-invariant. Over words the representative is
        the shortlex-least member of the class; over trees it is the least
        description, which is the same idea made to work without a well-order,
        and which may cost exponentially many states (`_tree_representatives`).
    :return: a fresh presentation of the same kind as `source`. For k > 1 its
        alphabet is the source alphabet's k-fold product.
    """
    if dimension < 1:
        raise ValueError("dimension must be >= 1")
    engine = _engine_for(source)
    if quotient is not None:
        return _quotient(source, domain, relations, dimension, quotient,
                         engine)
    k = dimension

    universe = _fold_formula(source, domain, k, engine)
    if universe.symbol_arity != 1:
        raise ValueError(
            f"the domain formula must have {k} free variable(s) — the "
            f"coordinates of one element — but defines "
            f"{universe.symbol_arity} element(s)")

    automata = {'U': universe}
    for name, spec in relations.items():
        if name == 'U':
            raise ValueError("'U' is the reserved universe symbol")
        automata[name] = _fold_formula(source, spec, k, engine)

    padding = source.padding_symbol if k == 1 \
        else (source.padding_symbol,) * k
    return engine.presentation(automata, padding)


def _quotient(source, domain: RelationSpec,
              relations: Dict[str, RelationSpec], dimension: int,
              equivalence: RelationSpec, engine: _Engine):
    """Interpret with a quotient, in two stages: first the plain (k-dim)
    interpretation carrying the equivalence as a relation, then a
    one-dimensional interpretation restricting the universe to one
    representative per class.

    Both engines end up here; they differ only in how a class is made to name
    exactly one of its members, which is `_string_representatives` or
    `_tree_representatives`.
    """
    reserved = (_EQUIV, _LE, _REP)
    clash = [name for name in reserved if name in relations]
    if clash:
        raise ValueError(f"{', '.join(map(repr, clash))} "
                         f"{'is' if len(clash) == 1 else 'are'} reserved for "
                         f"the quotient construction")
    raw = interpret(source, domain, {**relations, _EQUIV: equivalence},
                    dimension)
    raw.update(**{_REP: engine.representatives(raw)})

    specs: Dict[str, RelationSpec] = {}
    for name in relations:
        arity = raw.relation(name).symbol_arity
        args = [f'v{i}' for i in range(arity)]
        specs[name] = (f'{name}({",".join(args)})', args)
    return interpret(raw, (f'{_REP}(x)', ['x']), specs)


def _string_representatives(raw):
    """The shortlex-least element of each class.

    Shortlex is a well-order, so a class has exactly one least member, and
    being least is first-order over the equivalence and the order — which
    keeps the whole construction inside the engine.
    """
    from autstr.utils import automata_tools as words
    raw.update(**{_LE: words.shortlex_order(raw.sigma, raw.padding_symbol)})
    return raw.evaluate(f'all y.({_EQUIV}(x,y) -> {_LE}(x,y))')


def _tree_representatives(raw):
    """The least *description* of each class — Kuske and Weidner's
    construction (*Size and computation of injective tree automatic
    presentations*, MFCS 2011, §3).

    Least element of the class will not do here. There is a tree-automatic
    linear order (`autstr.utils.tree_automata_tools.tree_order`: compare at the
    lexicographically least position where two trees differ, an absent
    position counting as larger), but it is not well-founded — growing a tree
    at that position makes it smaller — so a class need have no least member.
    Shortlex escapes this over words only because the convolution aligns
    positions, which makes length the primary key for free; a tree convolution
    aligns shapes instead, and comparing two trees' sizes is not a finite-state
    property at all.

    So minimize over a finite part of the class instead. The *shadow* of a
    class is the set of positions its members all have, and a *description* is
    a member that reaches at most ``|A_∼|`` levels past it. Every class has
    finitely many descriptions and at least one — below the shadow a subtree
    can be pumped down to the height of the equivalence automaton without
    leaving the class — so the least description exists, and picking it is the
    choice this makes.

    The shadow itself never has to be computed. A description is a member `u`
    for which *some* set of positions `n` lies inside every member of the class
    and holds `u` within ``|A_∼|`` levels of itself::

        u is a description of [t]  ≡  u ~ t ∧ ∃n. (∀s. t ~ s → dom(n) ⊆ dom(s))
                                                 ∧ dom(u) ⊆ dom(n)·{1,2}^{≤k}

    The two conditions on `n` sandwich `u` rather than pulling against each
    other. The first makes `n` a *lower* bound on the whole class, so — `u`
    being a member — ``dom(n) ⊆ dom(u)`` already holds; the second is the
    *upper* bound, and is what cuts an infinite class down to a finite set of
    candidates, which is the whole point.

    So the existential settles on the shadow by itself: the first condition
    admits exactly the `n` below the shadow, and the second only gets weaker as
    `n` grows, so the best `n` is the largest admissible one — the shadow. That
    is why the hard half of the paper's argument, recognizing that a set of
    positions *contains* the shadow, is never needed: nothing here ever asks
    `n` to be at least the shadow.

    The formula is read over a scratch presentation whose universe is *every*
    tree, since `n` ranges over sets of positions rather than over elements.
    The exponential blowup Kuske and Weidner prove unavoidable lives in that
    ``∀s`` — it is a projection followed by a complement, and the subset
    construction inside it is the ``2^Q`` of their Lemma 3.4.

    Expect the representatives themselves to be *large* trees, and much larger
    than the smallest member of their class: the order prefers a tree that
    grows, so the least description is as full as the description bound allows.
    The automaton stays small; the elements it accepts do not.
    """
    from autstr.tree_presentations import TreeAutomaticPresentation, tree_one
    from autstr.utils import tree_automata_tools as trees

    equivalence = raw.relation(_EQUIV)
    alphabet, padding = raw.base_alphabet, raw.padding_symbol
    # how far past the shadow a description may reach: the pumping bound is
    # the number of states of the equivalence automaton
    depth = equivalence.num_states

    scratch = TreeAutomaticPresentation(
        {'U': tree_one(1, alphabet),
         # the elements are one sort among the trees, the sets of positions
         # another, so neither is the universe and nothing is restricted
         'Dom': raw.automata['U'],
         'Eqv': equivalence,
         'Sub': trees.domain_within(alphabet, padding),
         'Frg': trees.domain_within(alphabet, padding, depth),
         'Lt': trees.tree_order(alphabet, padding, strict=True)},
        padding_symbol=padding, enforce_consistency=False,
        max_states=raw.max_states)
    # `Desc(x,y)`: x is a description of y's class. The variables are named so
    # that their sorted order — which is the tape order `evaluate` returns, and
    # so the argument order of the installed relation — puts the description
    # first.
    scratch.update(Desc='Eqv(x,y) & exists n.('
                        '(all s.(Eqv(y,s) -> Sub(n,s))) & Frg(x,n))')
    return scratch.evaluate(
        'Dom(x) & Desc(x,x) & (not exists u.(Desc(u,x) & Lt(u,x)))')


def _fold_formula(source, spec: RelationSpec, k: int, engine: _Engine):
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
        dfa = engine.permute_tapes(dfa, permutation)
    return engine.fold_tapes(dfa, k) if k > 1 else dfa
