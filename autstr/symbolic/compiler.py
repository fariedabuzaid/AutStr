"""Compilation of symbolic expressions into first-order queries.

The output is an `nltk` expression plus a table of automata to splice in, which
is exactly what every evaluation backend in the package already consumes --
`AutomaticPresentation._build_automaton`, the relativizing class evaluator, and
the implicit engine. The symbolic layer is a frontend, not a fourth engine.

Two things happen here that the string-formula approach could not do safely:

**Variable renaming.** `nltk` only recognizes an argument as an individual
variable if its name matches ``[a-df-z][0-9]*``; anything else -- ``foo``,
``x_1``, ``e`` -- is silently reclassified and *drops out of the free-variable
list*, which corrupts tape order rather than raising. User-chosen names are
therefore mangled to legal ones on the way in and restored on the way out.
Free variables are numbered in sorted order with a fixed width, so the
lexicographic order the engine uses to lay out tapes agrees with the sorted
order of the user's names.

**Term flattening.** An atom ``R(f(x), y)`` becomes
``exists w.(Graph_f(x, w) and R(w, y))``. Witnesses are introduced once per
distinct subterm within an atom, and they are quantified at the atom rather
than hoisted, so a partial function's graph keeps its meaning under negation.
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

from nltk.sem.logic import (
    AllExpression, AndExpression, ApplicationExpression, ConstantExpression,
    ExistsExpression, FunctionVariableExpression, IndividualVariableExpression,
    NegatedExpression, OrExpression, Variable, is_indvar,
)

from autstr.symbolic import expr as E

#: Prefixes for the three kinds of generated names. All are legal `nltk`
#: individual-variable initials, and the groups are disjoint by construction.
FREE_PREFIX = 'a'
BOUND_PREFIX = 'b'
WITNESS_PREFIX = 'd'


class CompileError(Exception):
    pass


def _nltk_predicate(name: str):
    """A predicate expression for a relation symbol."""
    variable = Variable(name)
    if is_indvar(name):
        return FunctionVariableExpression(variable)
    return ConstantExpression(variable)


def _nltk_atom(symbol: str, args: Sequence[str]):
    expression = _nltk_predicate(symbol)
    for name in args:
        expression = ApplicationExpression(
            expression, IndividualVariableExpression(Variable(name)))
    return expression


class _Names:
    """Allocates `nltk`-legal names and remembers the mapping back."""

    def __init__(self, free: Sequence[str], bound: Sequence[str]):
        width = max(2, len(str(max(len(free), len(bound), 1))))
        self.to_user: Dict[str, str] = {}
        self.to_internal: Dict[str, str] = {}
        for i, name in enumerate(sorted(free)):
            self._bind(name, f"{FREE_PREFIX}{i:0{width}d}")
        for i, name in enumerate(sorted(bound)):
            self._bind(name, f"{BOUND_PREFIX}{i:0{width}d}")
        self._width = width
        self._witnesses = 0

    def _bind(self, user: str, internal: str) -> None:
        self.to_user[internal] = user
        self.to_internal[user] = internal

    def internal(self, user: str) -> str:
        try:
            return self.to_internal[user]
        except KeyError:
            raise CompileError(f"unbound variable {user!r}") from None

    def witness(self) -> str:
        name = f"{WITNESS_PREFIX}{self._witnesses:0{self._width}d}"
        self._witnesses += 1
        return name


def _collect_variables(formula: E.Formula) -> Tuple[set, set]:
    """(free names, names bound by a quantifier) over the whole formula."""
    bound = set()

    def walk(node):
        if isinstance(node, E._Quantifier):
            bound.update(node.bound)
            walk(node.body)
        elif isinstance(node, E.ExInf):
            bound.add(node.variable)
            walk(node.body)
        elif isinstance(node, E.Not):
            walk(node.body)
        elif isinstance(node, E._Binary):
            walk(node.left)
            walk(node.right)

    walk(formula)
    return set(formula.variables()), bound


class Compiler:
    """Lowers one symbolic formula. Instantiate per compilation -- it carries
    the name allocator and the automata collected along the way."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.updates: Dict[str, object] = {}
        self.prepared: Dict[str, object] = {}
        self._names = None
        self._spliced = 0
        # Arities of the automata spliced in during *this* compilation. Kept
        # here rather than on the context so that concurrent or repeated
        # compilations against one structure cannot see each other's symbols.
        self._spliced_arities: Dict[str, int] = {}

    # -- entry point --------------------------------------------------
    def compile(self, formula: E.Formula):
        """Returns ``(nltk expression, free variable names in tape order)``."""
        free, bound = _collect_variables(formula)
        self._names = _Names(free, bound)
        expression = self._formula(formula)
        return expression, sorted(free)

    @property
    def names(self) -> _Names:
        return self._names

    def arity(self, symbol: str) -> int:
        if symbol in self._spliced_arities:
            return self._spliced_arities[symbol]
        return self.ctx.relation_arity(symbol)

    # -- formulas -----------------------------------------------------
    def _formula(self, node: E.Formula):
        if isinstance(node, E.Atom):
            arity = self.arity(node.symbol)
            if arity != len(node.args):
                raise CompileError(
                    f"relation {node.symbol!r} has arity {arity}, "
                    f"applied to {len(node.args)} arguments")
            return self._flatten(node.symbol, node.args)
        if isinstance(node, E.DfaAtom):
            symbol = self._splice(node)
            return self._flatten(symbol, node.args)
        if isinstance(node, E.Not):
            return NegatedExpression(self._formula(node.body))
        if isinstance(node, E.And):
            return AndExpression(self._formula(node.left),
                                 self._formula(node.right))
        if isinstance(node, E.Or):
            return OrExpression(self._formula(node.left),
                                self._formula(node.right))
        if isinstance(node, (E.Exists, E.Forall)):
            build = (ExistsExpression if isinstance(node, E.Exists)
                     else AllExpression)
            inner = self._formula(node.body)
            for name in reversed(node.bound):
                inner = build(Variable(self._names.internal(name)), inner)
            return inner
        if isinstance(node, E.ExInf):
            return self._formula(self.ctx._expand_exinf(node))
        raise CompileError(f"cannot compile {type(node).__name__}")

    # -- atoms with terms ---------------------------------------------
    def _flatten(self, symbol: str, args: Sequence[E.Term]):
        """``R(t_1, ..., t_n)`` with the non-variable terms replaced by
        existentially quantified witnesses."""
        witnesses: Dict[E.Term, str] = {}
        conjuncts: List = []
        names = [self._term(t, witnesses, conjuncts) for t in args]

        expression = _nltk_atom(symbol, names)
        for conjunct in reversed(conjuncts):
            expression = AndExpression(conjunct, expression)
        for name in reversed(list(witnesses.values())):
            expression = ExistsExpression(Variable(name), expression)
        return expression

    def _term(self, term: E.Term, witnesses: Dict, conjuncts: List) -> str:
        """The variable name standing for ``term``, emitting the defining
        conjunct if the term is not already a variable."""
        if isinstance(term, E.Var):
            return self._names.internal(term.name)
        if term in witnesses:
            return witnesses[term]

        if isinstance(term, E.Const):
            name = self._names.witness()
            symbol = self._splice(self.ctx._constant_atom(term))
            conjuncts.append(_nltk_atom(symbol, [name]))
        elif isinstance(term, E.Apply):
            function = self.ctx.function(term.func)
            graph_arity = self.arity(function.graph)
            inputs, out = function.positions(graph_arity)
            if len(inputs) != len(term.args):
                raise CompileError(
                    f"function {term.func!r} takes {len(inputs)} arguments, "
                    f"applied to {len(term.args)}")
            argument_names = [self._term(a, witnesses, conjuncts)
                              for a in term.args]
            name = self._names.witness()
            slots = [None] * graph_arity
            for position, argument in zip(inputs, argument_names):
                slots[position] = argument
            slots[out] = name
            conjuncts.append(_nltk_atom(function.graph, slots))
        else:
            raise CompileError(f"cannot compile term {type(term).__name__}")

        witnesses[term] = name
        return name

    # -- spliced automata ---------------------------------------------
    def _splice(self, node) -> str:
        """Register an automaton under a fresh relation symbol."""
        symbol = f"Spliced{self._spliced}"
        self._spliced += 1
        if getattr(node, 'prepared', False):
            self.prepared[symbol] = node.dfa
        else:
            self.updates[symbol] = node.dfa
        self._spliced_arities[symbol] = self.ctx._relation_arity_of(node.dfa)
        return symbol


def restore(names: _Names, internal: Sequence[str]) -> List[str]:
    """Map internal tape names back to the user's variable names."""
    return [names.to_user.get(name, name) for name in internal]
