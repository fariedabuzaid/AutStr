"""Uniformly tree-automatic classes of structures.

The tree analog of `autstr.uniform`: a single tuple of tree automata
presents a whole class of structures. Every automaton carries one additional
tape (tape 0) holding an *advice tree* that is read convolution-synchronously
with the element encodings. Fixing an advice tree t instantiates one member
structure S_t:

    universe(S_t)  = { x  | t ⊗ x ∈ L(U) }
    R^{S_t}        = { x̄ | t ⊗ x̄ ∈ L(R) }

First-order queries are evaluated once for the entire class; model checking
a sentence against a member structure then reduces to running its advice
tree through the query automaton. Because the advice may be a tree, the
member universes can live on tree-shaped skeletons — the setting for bounded
tree-width graph classes and tree-indexed algebraic families.

The formula layer (relativization of quantifiers to Dom(advice, ·), the
advice as one more first-order variable, Adv as the projection of the domain
onto the advice tape) is inherited unchanged from the string implementation;
only the automaton operations differ.
"""
import itertools as it
from typing import Dict, List, Optional, Union

from nltk.sem import logic

from autstr.sparse_tree_automata import (
    SparseTreeAutomaton, Tree, convolve_trees, tree_to_arrays,
)
from autstr.tree_presentations import TreeAutomaticPresentation, tree_one
from autstr.uniform import UniformlyAutomaticClass
from autstr.utils.logic import get_free_elementary_vars
from autstr.utils.misc import encode_symbol
from autstr.utils.tree_automata_tools import (
    attach_padding, expand, minimize, permute_tapes, project, tree_automaton,
)


def sta_from_delta(sigma, states, arity, delta, finals, dead='dead',
                   tapes=None) -> SparseTreeAutomaton:
    """Build a SparseTreeAutomaton from a bottom-up transition function.

    :param delta: delta(left_state, right_state, symbol_tuple) -> state,
        where an absent child is passed as None. `dead` names the sink
        (stored as the sparse global default); dead children are never
        enumerated, so delta need not handle them.
    :param tapes: optional per-tape alphabets. A convolution tape usually
        ranges over a small part of the base alphabet -- the advice tape reads
        advice letters, an element tape reads element letters -- and every
        mixed tuple is dead. Enumerating ``sigma^arity`` therefore spends
        almost all of its time confirming that nonsense is dead: for
        clique-width at k = 4 only 207 of 17576 triples are meaningful.
        Naming the tapes' alphabets restricts the enumeration to their
        product; every other symbol falls to the global default, which is the
        dead sink. Defaults to `sigma` on every tape.

    Each child pair's majority target becomes its pair default (ties prefer
    the dead sink), and only deviations are stored as exceptions — states
    that loop on most symbols stay cheap over large alphabets. With `tapes`
    given, the unenumerated symbols already outnumber the rest, so the dead
    sink is the majority everywhere and no pair defaults are emitted.
    """
    from collections import Counter

    sigma_frozen = frozenset(sigma)
    full = len(sigma_frozen) ** arity
    if tapes is None:
        tapes = [sigma_frozen] * arity
    if len(tapes) != arity:
        raise ValueError(f"expected {arity} tape alphabets, got {len(tapes)}")
    for tape in tapes:
        if not set(tape) <= sigma_frozen:
            raise ValueError("tape alphabets must be subsets of sigma")

    symbols = [(encode_symbol(sym, sigma_frozen), sym)
               for sym in it.product(*[sorted(t) for t in tapes])]
    restricted = len(symbols) < full

    real = [q for q in states if q != dead]
    ids = {q: i for i, q in enumerate(real)}
    ids[dead] = len(real)
    n = len(real) + 1                            # dead is the last real state
    dead_id, BOT = n - 1, n

    exc = []
    pd = []
    options = [(None, BOT)] + [(q, ids[q]) for q in real]
    for lq, lid in options:
        for rq, rid in options:
            row = [delta(lq, rq, sym) for _, sym in symbols]
            row = [dead if t is None else t for t in row]
            if restricted:
                # the symbols we did not enumerate are dead, and they are the
                # majority, so the global default already serves them
                for (code, _), t in zip(symbols, row):
                    if t != dead:
                        exc.append((lid, rid, code, ids[t]))
                continue
            counts = Counter(row)
            majority = max(counts, key=lambda q: (counts[q], q == dead))
            if majority != dead:
                pd.append((lid, rid, ids[majority]))
            for (code, _), t in zip(symbols, row):
                if t != majority:
                    exc.append((lid, rid, code, ids[t]))
    exc.sort()
    acc = [q in finals for q in real] + [dead in finals]
    return SparseTreeAutomaton(
        n, dead_id,
        [e[0] for e in exc], [e[1] for e in exc],
        [e[2] for e in exc], [e[3] for e in exc],
        acc, arity, set(sigma),
        [p[0] for p in pd], [p[1] for p in pd], [p[2] for p in pd])


class UniformlyTreeAutomaticClass(UniformlyAutomaticClass):
    """A uniformly tree-automatic presentation of a class of structures.

    :param automata: dictionary of SparseTreeAutomatons. 'U' is reserved for
        the domain automaton of symbol arity 2 (advice tape, element tape).
        Every other key R presents a relation of arity r with an automaton of
        symbol arity 1 + r (advice tape first, then the element tapes).
    :param padding_symbol: base letter used to pad the convolutions.
    :param max_states: optional cap for the subset determinizations inside
        projections (a clear error instead of an exponential blowup).
    """

    def __init__(self, automata: Dict[str, SparseTreeAutomaton],
                 padding_symbol="*", max_states: Optional[int] = None) -> None:
        if 'U' not in automata:
            raise ValueError(
                "A uniformly tree-automatic class needs a domain automaton 'U'")
        self.padding_symbol = padding_symbol
        self.max_states = max_states
        self.class_automata = dict(automata)
        self.base_alphabet = automata['U'].base_alphabet

        # As in the string case, the advice is one more first-order variable:
        # trivial universe, Dom(advice, element) as the domain relation, and
        # Adv(advice) = the projection of Dom onto the advice tape.
        wrapped = {'U': tree_one(1, self.base_alphabet)}
        wrapped['Dom'] = automata['U']
        wrapped['Adv'] = minimize(project(
            attach_padding(automata['U'], padding_symbol), 1, padding_symbol,
            max_states=max_states))
        for name, sta in automata.items():
            if name == 'U':
                continue
            if name in ('Dom', 'Adv'):
                raise ValueError(f"Relation name {name!r} is reserved")
            wrapped[name] = sta
        self.presentation = TreeAutomaticPresentation(
            wrapped, padding_symbol=padding_symbol, enforce_consistency=False,
            max_states=max_states)

    # evaluate(), _relativize(), _variable_names() and get_relation_symbols()
    # are inherited: they operate on formulas and on self.presentation only.

    def check(self, phi: Union[str, logic.Expression], advice: Tree,
              **assignments) -> bool:
        """Model check a formula against the member structure S_advice.

        :param phi: a formula over the class signature. Free variables can be
            assigned concrete elements via `assignments` (name = element
            tree); unassigned free variables are existentially quantified.
        :param advice: the advice tree.
        """
        if isinstance(phi, str):
            phi = logic.Expression.fromstring(phi)
        free_vars = get_free_elementary_vars(phi)
        unknown = set(assignments) - set(free_vars)
        if unknown:
            raise ValueError(f"assignments for non-free variables: {unknown}")
        for x in free_vars:
            if x not in assignments:
                phi = logic.ExistsExpression(logic.Variable(x), phi)

        sta, variables = self.evaluate(phi)
        trees = [advice if v == 'advice' else assignments[v]
                 for v in variables]
        conv = convolve_trees(trees, sta.base_alphabet_frozen,
                              self.padding_symbol)
        return sta.accepts(tree_to_arrays(conv, sta.base_alphabet_frozen,
                                          sta.symbol_arity))

    def check_implicit(self, phi, advice: Tree, **assignments) -> bool:
        """Model check a formula against the member S_advice without compiling a
        query tree automaton: the formula is evaluated bottom-up on the fly over
        the base tree automata. Scales to classes whose query automaton is
        infeasible. See `autstr.implicit`.

        Unlike `check`, the on-the-fly evaluator is synchronous: it walks the
        advice tree and every assigned element tree in lockstep and does not
        convolve or pad mismatched shapes, so the assigned trees must already be
        padded to the advice's shape."""
        from autstr import implicit
        return implicit.check_class_tree(
            phi, advice, assignments, dict(self.presentation.automata),
            self._implicit_element_alphabet(),
            self._relativize, self._variable_names)

    def evaluate_implicit(self, phi, advice: Tree, **assignments):
        """The satisfying set of phi on the member S_advice, computed
        implicitly (no query tree automaton): unassigned free variables stay
        open and are solved for over the fixed advice. Returns a
        `TreeSolutionSet` of `{var: tree}` assignments (trees of the
        advice's shape) — its `len` is the exact solution count, iterating
        lazily yields the assignments. See `autstr.implicit`."""
        from autstr import implicit
        return implicit.evaluate_class_tree(
            phi, advice, assignments, dict(self.presentation.automata),
            self._implicit_element_alphabet(),
            self._relativize, self._variable_names)

    def define(self, name: str, phi: Union[str, logic.Expression]
               ) -> SparseTreeAutomaton:
        """Define a new class relation by a first-order formula over the
        existing signature. The relation's arguments are the free variables
        of phi in sorted order; the advice moves back to tape 0."""
        if name in ('U', 'Dom', 'Adv'):
            raise ValueError(f"Relation name {name!r} is reserved")
        sta, variables = self.evaluate(phi)
        advice_pos = variables.index('advice')
        perm = [advice_pos] + [i for i in range(len(variables))
                               if i != advice_pos]
        sta = permute_tapes(sta, perm)
        self.class_automata[name] = sta
        self.presentation.update(**{name: sta})
        return sta

    def get_structure(self, advice: Tree) -> TreeAutomaticPresentation:
        """Instantiate the member structure S_advice as an ordinary
        tree-automatic presentation (the advice tape is fixed to the given
        tree and projected out)."""
        anchor = attach_padding(
            tree_automaton(advice, self.base_alphabet), self.padding_symbol)

        automata = {}
        for name, sta in self.class_automata.items():
            fixed = minimize(attach_padding(sta, self.padding_symbol)
                             .intersection(
                                 expand(anchor, sta.symbol_arity, pos=[0])))
            automata[name] = minimize(project(
                fixed, 0, self.padding_symbol, max_states=self.max_states))
        return TreeAutomaticPresentation(
            automata, padding_symbol=self.padding_symbol,
            max_states=self.max_states)
