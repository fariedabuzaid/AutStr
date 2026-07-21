"""Tree-automatic presentations: structures whose elements are finite trees
and whose relations are recognized by sparse bottom-up tree automata reading
tree convolutions.

Mirrors `autstr.presentations.AutomaticPresentation` with the tree pipeline
underneath. The pipeline invariant also mirrors the string engine: stored
relation automata are *padding-saturated* (accept their canonical convolutions
with arbitrary all-padding regions attached below, via `attach_padding`), so
`expand` may widen them to more tapes; intersections and unions preserve
saturation; complements are re-intersected with the domain product; and
`project` — which produces the canonical (trimmed) language — is followed by
re-saturation, the tree analog of the string pipeline's pad/unpad dance.
"""
from typing import Dict, Optional

from nltk.sem import logic

from autstr.sparse_tree_automata import SparseTreeAutomaton
from autstr.utils.logic import get_free_elementary_vars, optimize_query
from autstr.utils.tree_automata_tools import (
    attach_padding, expand, minimize, project,
)


def tree_one(symbol_arity: int = 1, base_alphabet=None) -> SparseTreeAutomaton:
    """Automaton accepting every tree."""
    return SparseTreeAutomaton(1, 0, [], [], [], [], [True],
                               symbol_arity, base_alphabet or {0})


def tree_zero(symbol_arity: int = 1, base_alphabet=None) -> SparseTreeAutomaton:
    """Automaton rejecting every tree."""
    return SparseTreeAutomaton(1, 0, [], [], [], [], [False],
                               symbol_arity, base_alphabet or {0})


class TreeAutomaticPresentation:
    """A presentation of a structure by tree automata.

    :param automata: 'U' is the domain automaton (symbol arity 1); every other
        key R presents a relation of arity k by an automaton of symbol arity k
        over convolutions of element trees.
    :param padding_symbol: base letter used to pad convolutions.
    :param max_states: optional cap for the subset determinizations inside
        projection (a clear error instead of an exponential blowup).
    """

    def __init__(self, automata: Dict[str, SparseTreeAutomaton],
                 padding_symbol="*", enforce_consistency: bool = True,
                 max_states: Optional[int] = None) -> None:
        self.padding_symbol = padding_symbol
        self.max_states = max_states
        universe = minimize(attach_padding(automata['U'], padding_symbol))
        self.automata = {'U': universe}
        for name, dfa in automata.items():
            if name == 'U':
                continue
            if enforce_consistency:
                self.automata[name] = self._prepare_automaton(dfa)
            else:
                self.automata[name] = minimize(
                    attach_padding(dfa, padding_symbol))
        self.base_alphabet = universe.base_alphabet

    def _prepare_automaton(self, dfa: SparseTreeAutomaton) -> SparseTreeAutomaton:
        """Saturate with padding and restrict every tape to the domain."""
        arity = dfa.symbol_arity
        result = minimize(attach_padding(dfa, self.padding_symbol))
        for i in range(arity):
            domain_i = minimize(expand(self.automata['U'], arity, [i]))
            result = minimize(result.intersection(domain_i))
        return result

    def _domain_product(self, arity: int) -> SparseTreeAutomaton:
        """The universe automaton over `arity` tapes."""
        if arity <= 1:
            return self.automata['U']
        domain = minimize(expand(self.automata['U'], arity, [0]))
        for i in range(1, arity):
            domain = minimize(domain.intersection(
                minimize(expand(self.automata['U'], arity, [i]))))
        return domain

    def get_relation_symbols(self):
        return list(self.automata.keys())

    def symbolic(self, signature=None):
        """A symbolic interface to this structure: variables, relation and
        function symbols that build first-order expressions with Python
        operators instead of formula strings.

        Elements are trees, so a signature's codec encodes Python values to
        `Tree`s. Enumeration, finiteness and constants are not available over
        the tree engine yet and raise with the reason.

        :param signature: declared functions, operators and element codec.
        :return: a `autstr.symbolic.SymbolicContext`.
        """
        from autstr.symbolic.backends import TreeStructureBackend
        from autstr.symbolic.context import SymbolicContext
        return SymbolicContext(TreeStructureBackend(self), signature)

    def update(self, **automata) -> None:
        """Install or replace relations. Values may be automata (saturated
        and domain-restricted like at construction time) or formula strings
        over the current signature."""
        for name, value in automata.items():
            if name == 'U':
                raise ValueError("cannot replace the domain automaton")
            if isinstance(value, SparseTreeAutomaton):
                self.automata[name] = self._prepare_automaton(value)
            else:
                query = optimize_query(logic.Expression.fromstring(value))
                self.automata[name] = self._prepare_automaton(
                    self._build_automaton(query))

    def check(self, phi) -> bool:
        """Truth of phi (free variables existentially quantified)."""
        if isinstance(phi, str):
            phi = logic.Expression.fromstring(phi)
        phi = phi.simplify()
        return not self._build_automaton(phi).is_empty()

    def evaluate(self, phi) -> SparseTreeAutomaton:
        """Automaton of all satisfying assignments (tapes = sorted free
        variables; padding-saturated form)."""
        if isinstance(phi, str):
            phi = logic.Expression.fromstring(phi)
        return self._build_automaton(optimize_query(phi))

    def _build_automaton(self, phi) -> SparseTreeAutomaton:
        if isinstance(phi, str):
            phi = logic.Expression.fromstring(phi)

        if isinstance(phi, logic.AllExpression):
            variable = str(phi.variable)
            free_vars = get_free_elementary_vars(phi.term)
            if variable not in free_vars:
                return self._build_automaton(phi.term)
            psi = (phi.term.negate()).simplify()
            inner = minimize(self._build_automaton(psi))
            pos = free_vars.index(variable)
            if len(free_vars) > 1:
                domain = self._domain_product(len(free_vars) - 1)
                projected = project(inner, pos, self.padding_symbol,
                                    max_states=self.max_states)
                result = minimize(attach_padding(projected,
                                                 self.padding_symbol))
                result = minimize(result.complement().intersection(domain))
                return result
            return tree_zero(1, self.base_alphabet) if not inner.is_empty() \
                else tree_one(1, self.base_alphabet)

        elif isinstance(phi, logic.ExistsExpression):
            variable = str(phi.variable)
            free_vars = get_free_elementary_vars(phi.term)
            inner = self._build_automaton(phi.term)
            if variable not in free_vars:
                return inner
            pos = free_vars.index(variable)
            if len(free_vars) > 1:
                projected = project(inner, pos, self.padding_symbol,
                                    max_states=self.max_states)
                return minimize(attach_padding(projected,
                                               self.padding_symbol))
            return tree_one(1, self.base_alphabet) if not inner.is_empty() \
                else tree_zero(1, self.base_alphabet)

        elif isinstance(phi, logic.AndExpression) or \
                isinstance(phi, logic.OrExpression):
            free_vars = get_free_elementary_vars(phi)
            free_l = get_free_elementary_vars(phi.first)
            free_r = get_free_elementary_vars(phi.second)
            left = expand(self._build_automaton(phi.first), len(free_vars),
                          [free_vars.index(v) for v in free_l])
            right = expand(self._build_automaton(phi.second), len(free_vars),
                           [free_vars.index(v) for v in free_r])
            if isinstance(phi, logic.AndExpression):
                return minimize(left.intersection(right))
            return minimize(left.union(right))

        elif isinstance(phi, logic.ImpExpression):
            return self._build_automaton(logic.OrExpression(
                logic.NegatedExpression(phi.first), phi.second))

        elif isinstance(phi, logic.IffExpression):
            return self._build_automaton(logic.AndExpression(
                logic.OrExpression(logic.NegatedExpression(phi.first),
                                   phi.second),
                logic.OrExpression(logic.NegatedExpression(phi.second),
                                   phi.first)))

        elif isinstance(phi, logic.NegatedExpression):
            if isinstance(phi.term, logic.NegatedExpression):
                return self._build_automaton(phi.term.term)
            free_vars = get_free_elementary_vars(phi)
            domain = self._domain_product(len(free_vars))
            inner = self._build_automaton(phi.term).complement()
            return minimize(inner.intersection(domain))

        elif isinstance(phi, logic.ApplicationExpression):
            relation = str(phi.pred)
            variables = get_free_elementary_vars(phi)
            return minimize(expand(
                self.automata[relation], len(variables),
                [variables.index(str(v)) for v in phi.args]))

        raise ValueError(f"Unsupported expression type: {type(phi)}")
