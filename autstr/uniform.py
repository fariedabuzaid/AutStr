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
from autstr.buildin.presentations import create_sparse_dfa
from autstr.utils.automata_tools import expand, pad, permute_tapes, projection, word_automaton
from autstr.utils.logic import get_free_elementary_vars
from autstr.utils.misc import get_unique_id


def dfa_from_delta(sigma, states, arity, delta, initial, finals) -> SparseDFA:
    """Build a SparseDFA from a transition function over the full symbol
    space sigma^arity. Convenience for constructing presentation automata."""
    input_symbols = set(it.product(sorted(sigma), repeat=arity))
    transitions = {q: {sym: delta(q, sym) for sym in input_symbols} for q in states}
    return create_sparse_dfa(list(states), input_symbols, transitions, initial, finals)


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

    def _relativize(self, phi: logic.Expression, advice_var: str) -> str:
        """Rewrite a class formula into a formula over the wrapped
        presentation: atoms get the advice variable prepended and quantifiers
        are relativized to the domain Dom(advice, ·). Negated quantifiers are
        rewritten to quantified negations on the fly."""
        rec = lambda psi: self._relativize(psi, advice_var)
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
