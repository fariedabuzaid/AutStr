"""Uniformly automatic classes of structures.

A uniformly automatic presentation describes a whole class of structures with
a single tuple of automata: every automaton carries one additional tape (tape
0) holding an *advice* string that is read synchronously with the element
encodings. Fixing an advice string α instantiates one member structure S_α:

    universe(S_α)  = { x  | α ⊗ x ∈ L(U) }
    R^{S_α}        = { x̄ | α ⊗ x̄ ∈ L(R) }

First-order queries are evaluated once for the entire class; model checking a
sentence against a member structure then reduces to running its advice string
through the query automaton.
"""
import itertools as it
from typing import Dict, List, Optional, Tuple, Union

from nltk.sem import logic

from autstr.presentations import AutomaticPresentation
from autstr.sparse_automata import SparseDFA
from autstr.buildin.automata import one
from autstr.buildin.presentations import create_sparse_dfa, encode_symbol
from autstr.utils.automata_tools import expand, pad, permute_tapes, projection, word_automaton
from autstr.utils.logic import get_free_elementary_vars
from autstr.utils.misc import get_unique_id


def dfa_from_delta(sigma, states, arity, delta, initial, finals,
                   tapes=None, dead=None) -> SparseDFA:
    """Build a SparseDFA from a transition function over the full symbol
    space sigma^arity. Convenience for constructing presentation automata.

    :param tapes: optional per-tape alphabets (the string analog of
        `sta_from_delta`'s parameter). A convolution tape usually ranges over
        a small part of the base alphabet — the advice tape reads advice
        letters, an element tape reads element letters — and every mixed
        tuple is dead. Naming the tapes' alphabets restricts the enumeration
        to their product; every other symbol falls to `dead`, which must then
        be named and becomes every state's sparse default.
    :param dead: name of the sink state (required with `tapes`); delta must
        map it to itself on every enumerated symbol.
    """
    import numpy as np

    if tapes is None:
        input_symbols = set(it.product(sorted(sigma), repeat=arity))
        transitions = {q: {sym: delta(q, sym) for sym in input_symbols} for q in states}
        return create_sparse_dfa(list(states), input_symbols, transitions, initial, finals)

    if dead is None:
        raise ValueError("tapes requires a named dead state")
    if len(tapes) != arity:
        raise ValueError(f"expected {arity} tape alphabets, got {len(tapes)}")
    states = list(states)
    state_to_index = {s: i for i, s in enumerate(states)}
    if dead not in state_to_index:
        raise ValueError(f"dead state {dead!r} is not in states")
    if initial not in state_to_index:
        raise ValueError(f"initial state {initial!r} is not in states")
    base_alphabet = set(sigma)
    exceptions = {q: [] for q in states}
    for sym in it.product(*[sorted(t) for t in tapes]):
        enc = None
        for q in states:
            target = delta(q, sym)
            if target != dead:
                if target not in state_to_index:
                    raise ValueError(
                        f"delta({q!r}, {sym!r}) = {target!r} is not in states")
                if enc is None:
                    enc = encode_symbol(sym, base_alphabet)
                exceptions[q].append((enc, state_to_index[target]))
    max_exceptions = max(1, max(len(e) for e in exceptions.values()))
    ex_syms = np.full((len(states), max_exceptions), -1, dtype=np.int64)
    ex_states = np.full((len(states), max_exceptions), -1, dtype=np.int64)
    for q, rows in exceptions.items():
        rows.sort()
        i = state_to_index[q]
        ex_syms[i, :len(rows)] = [s for s, _ in rows]
        ex_states[i, :len(rows)] = [t for _, t in rows]
    return SparseDFA(
        num_states=len(states),
        default_states=np.full(len(states), state_to_index[dead], dtype=np.int64),
        exception_symbols=ex_syms,
        exception_states=ex_states,
        is_accepting=np.array([q in finals for q in states], dtype=bool),
        start_state=state_to_index[initial],
        symbol_arity=arity,
        base_alphabet=base_alphabet,
    ).minimize()


class UniformlyAutomaticClass:
    """A uniformly automatic presentation of a class of structures.

    :param automata: dictionary of SparseDFAs. 'U' is reserved for the domain
        automaton of symbol arity 2 (advice tape, element tape). Every other
        key R presents a relation of arity r with an automaton of symbol
        arity 1 + r (advice tape first, then the element tapes).
    :param padding_symbol: symbol used to pad the shorter tapes of a
        convolution.
    """

    def __init__(self, automata: Dict[str, SparseDFA], padding_symbol="*") -> None:
        if 'U' not in automata:
            raise ValueError("A uniformly automatic class needs a domain automaton 'U'")
        self.padding_symbol = padding_symbol
        self.class_automata = dict(automata)
        self.base_alphabet = automata['U'].base_alphabet

        # Internally the advice is treated as one more first-order variable:
        # the wrapped presentation has a trivial universe, the domain is the
        # binary relation Dom(advice, element), and Adv(advice) recognizes the
        # valid advice strings (the projection of Dom onto the advice tape).
        wrapped = {'U': one(symbol_arity=1, base_alphabet=self.base_alphabet)}
        wrapped['Dom'] = automata['U']
        wrapped['Adv'] = projection(pad(automata['U'], padding_symbol), 1).minimize()
        for name, dfa in automata.items():
            if name == 'U':
                continue
            if name in ('Dom', 'Adv'):
                raise ValueError(f"Relation name {name!r} is reserved")
            wrapped[name] = dfa
        self.presentation = AutomaticPresentation(
            wrapped, padding_symbol=padding_symbol, enforce_consistency=False
        )

    def get_relation_symbols(self) -> List[str]:
        """All relation symbols of the class signature ('U' is the domain)."""
        return list(self.class_automata.keys())

    def symbolic(self, signature=None):
        """A symbolic interface to this class. Expressions are written over
        the class signature exactly as for a single structure; the advice tape
        is added and quantifiers relativized to the member domain during
        compilation, and results carry the advice under the tape name
        ``'advice'``.

        :param signature: declared functions and operators. An element codec
            has no advice-free meaning here and is not used.
        :return: a `autstr.symbolic.SymbolicContext`.
        """
        from autstr.symbolic.backends import ClassBackend
        from autstr.symbolic.context import SymbolicContext
        return SymbolicContext(ClassBackend(self), signature)

    @staticmethod
    def _variable_names(phi: logic.Expression) -> set:
        """All variable names occurring in phi (free and bound)."""
        names = {str(v) for v in phi.free()}
        if isinstance(phi, (logic.AllExpression, logic.ExistsExpression)):
            names.add(str(phi.variable))
            names |= UniformlyAutomaticClass._variable_names(phi.term)
        elif isinstance(phi, logic.NegatedExpression):
            names |= UniformlyAutomaticClass._variable_names(phi.term)
        elif isinstance(phi, logic.BinaryExpression):
            names |= UniformlyAutomaticClass._variable_names(phi.first)
            names |= UniformlyAutomaticClass._variable_names(phi.second)
        return names

    @staticmethod
    def _relativize(phi: logic.Expression, advice_var: str) -> str:
        """Rewrite a class formula into a formula over the wrapped
        presentation: atoms get the advice variable prepended and quantifiers
        are relativized to the domain Dom(advice, ·). Negated quantifiers are
        rewritten to quantified negations on the fly."""
        rec = lambda psi: UniformlyAutomaticClass._relativize(psi, advice_var)
        if isinstance(phi, logic.AllExpression):
            x = str(phi.variable)
            return f"all {x}.((not Dom({advice_var},{x})) or {rec(phi.term)})"
        elif isinstance(phi, logic.ExistsExpression):
            x = str(phi.variable)
            return f"exists {x}.(Dom({advice_var},{x}) and {rec(phi.term)})"
        elif isinstance(phi, logic.AndExpression):
            return f"({rec(phi.first)} and {rec(phi.second)})"
        elif isinstance(phi, logic.OrExpression):
            return f"({rec(phi.first)} or {rec(phi.second)})"
        elif isinstance(phi, logic.ImpExpression):
            return f"((not {rec(phi.first)}) or {rec(phi.second)})"
        elif isinstance(phi, logic.IffExpression):
            a, b = rec(phi.first), rec(phi.second)
            return f"(((not {a}) or {b}) and ((not {b}) or {a}))"
        elif isinstance(phi, logic.NegatedExpression):
            inner = phi.term
            # Push negation over quantifiers so the downstream optimizer never
            # sees a negation directly on a quantifier
            if isinstance(inner, logic.AllExpression):
                return rec(logic.ExistsExpression(inner.variable, logic.NegatedExpression(inner.term)))
            if isinstance(inner, logic.ExistsExpression):
                return rec(logic.AllExpression(inner.variable, logic.NegatedExpression(inner.term)))
            if isinstance(inner, logic.NegatedExpression):
                return rec(inner.term)
            return f"(not {rec(inner)})"
        elif isinstance(phi, logic.ApplicationExpression):
            name = str(phi.pred)
            if name == 'U':
                name = 'Dom'
            args = ",".join(str(a) for a in phi.args)
            return f"{name}({advice_var},{args})"
        else:
            raise ValueError(f"Unsupported expression type: {type(phi)}")

    def evaluate(self, phi: Union[str, logic.Expression]) -> Tuple[SparseDFA, List[str]]:
        """Evaluate a first-order query over the class.

        :param phi: formula over the class signature; quantifiers range over
            the elements of the member structures.
        :returns: (dfa, variables) where dfa presents all satisfying
            assignments — its tapes are the advice followed by the free
            variables of phi — and variables names the tapes in order.
        """
        if isinstance(phi, str):
            phi = logic.Expression.fromstring(phi)
        phi = phi.simplify()

        free_vars = get_free_elementary_vars(phi)
        all_vars = sorted(self._variable_names(phi) | set(free_vars)) or ['p']
        advice_var = get_unique_id(all_vars, 1)

        relativized = self._relativize(phi, advice_var)
        conjuncts = [relativized, f"Adv({advice_var})"]
        conjuncts += [f"Dom({advice_var},{x})" for x in free_vars]
        query = " and ".join(f"({c})" for c in conjuncts)

        dfa = self.presentation.evaluate(query)
        variables = sorted(free_vars + [advice_var])
        # Report the advice tape under a stable name
        return dfa, ['advice' if v == advice_var else v for v in variables]

    def check(self, phi: Union[str, logic.Expression], advice: List, **assignments) -> bool:
        """Model check a formula against the member structure S_advice.

        :param phi: a formula over the class signature. Free variables can be
            assigned concrete elements via `assignments` (name = encoded word
            of the same length as the advice); unassigned free variables are
            existentially quantified.
        :param advice: the advice string (sequence of base alphabet symbols).
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

        dfa, variables = self.evaluate(phi)
        advice = list(advice)
        columns = {'advice': advice}
        for name, word in assignments.items():
            word = list(word)
            if len(word) != len(advice):
                raise ValueError(f"assignment {name!r} has length {len(word)}, "
                                 f"advice has length {len(advice)}")
            columns[name] = word
        return dfa.accepts([
            tuple(columns[v][i] for v in variables) for i in range(len(advice))
        ])

    def _implicit_element_alphabet(self):
        alphabet = getattr(self, 'element_alphabet', None)
        if alphabet is None:
            raise ValueError(
                "check_implicit needs the class's element-tape alphabet; set "
                "`.element_alphabet` (the per-position element symbols).")
        return alphabet

    def check_implicit(self, phi, advice, **assignments) -> bool:
        """Model check a formula against the member S_advice *without* compiling
        a query automaton: the formula is evaluated on the fly over the base
        automata (implicit product / on-the-fly powerset / acceptance flip). Same
        contract as `check`; scales to classes whose query automaton is
        infeasible to build. See `autstr.implicit`."""
        from autstr import implicit
        return implicit.check_class_string(
            phi, advice, assignments, dict(self.presentation.automata),
            self._implicit_element_alphabet(),
            self._relativize, self._variable_names)

    def evaluate_implicit(self, phi, advice, **assignments):
        """The satisfying set of phi on the member S_advice, computed
        implicitly (no query automaton): unassigned free variables stay open
        and are solved for over the fixed advice. Returns a
        `StringSolutionSet` of `{var: word}` assignments — its `len` is the
        exact solution count (no enumeration), iterating lazily yields the
        assignments. See `autstr.implicit`."""
        from autstr import implicit
        return implicit.evaluate_class_string(
            phi, advice, assignments, dict(self.presentation.automata),
            self._implicit_element_alphabet(),
            self._relativize, self._variable_names)

    def define(self, name: str, phi: Union[str, logic.Expression]) -> SparseDFA:
        """Define a new class relation by a first-order formula over the
        existing signature (the uniform analog of a Büchi-style bootstrap).
        The relation's arguments are the free variables of phi in sorted
        order; the advice stays on tape 0.
        """
        if name in ('U', 'Dom', 'Adv'):
            raise ValueError(f"Relation name {name!r} is reserved")
        dfa, variables = self.evaluate(phi)
        advice_pos = variables.index('advice')
        perm = [advice_pos] + [i for i in range(len(variables)) if i != advice_pos]
        dfa = permute_tapes(dfa, perm)
        self.class_automata[name] = dfa
        self.presentation.update(**{name: dfa})
        return dfa

    def get_structure(self, advice: List) -> AutomaticPresentation:
        """Instantiate the member structure S_advice as an ordinary
        automatic presentation (the advice tape is fixed and projected out).

        :param advice: the advice string (sequence of base alphabet symbols).
        """
        anchor = word_automaton(advice, self.base_alphabet, self.padding_symbol)

        automata = {}
        for name, dfa in self.class_automata.items():
            fixed = pad(dfa, self.padding_symbol).intersection(
                expand(anchor, dfa.symbol_arity, pos=[0])
            ).minimize()
            automata[name] = projection(fixed, 0).minimize()
        return AutomaticPresentation(automata, padding_symbol=self.padding_symbol)
