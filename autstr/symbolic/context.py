"""Binding a signature to a structure: the user-facing symbolic interface.

A `SymbolicContext` is what `AutomaticPresentation.symbolic()` and
`UniformlyAutomaticClass.symbolic()` hand back. It mints variables, relation
and function symbols, and evaluates the expressions built from them against
its backend.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Union

from autstr.symbolic import expr as E
from autstr.symbolic.compiler import Compiler, CompileError, restore
from autstr.symbolic.signature import Function, Signature


class SymbolicSymbolError(Exception):
    pass


class RelationSymbol:
    """A relation symbol of the signature, applied to build atoms."""

    def __init__(self, ctx, symbol: str):
        self.ctx = ctx
        self.symbol = symbol

    @property
    def arity(self) -> int:
        return self.ctx.relation_arity(self.symbol)

    def __call__(self, *args) -> E.Formula:
        return self.ctx.atom(self.symbol, args)

    def __repr__(self):
        return f"<relation {self.symbol}/{self.arity}>"


class FunctionSymbol:
    """A function symbol of the signature, applied to build terms."""

    def __init__(self, ctx, name: str, function: Function):
        self.ctx = ctx
        self.name = name
        self.function = function

    @property
    def arity(self) -> int:
        return self.ctx.relation_arity(self.function.graph) - 1

    def __call__(self, *args) -> E.Term:
        if len(args) != self.arity:
            raise SymbolicSymbolError(
                f"function {self.name!r} has arity {self.arity}, "
                f"applied to {len(args)} arguments")
        return E.Apply(self.ctx, self.name, [self.ctx.term(a) for a in args])

    def __repr__(self):
        return f"<function {self.name}/{self.arity}>"


class SymbolicContext:
    """The symbolic interface to one structure or one class of structures.

    :param backend: evaluation target (see `autstr.symbolic.backends`).
    :param signature: declared functions, operators and codec. Relation
        arities are read from the backend's automata.
    """

    def __init__(self, backend, signature: Optional[Signature] = None):
        self.backend = backend
        self.signature = signature if signature is not None else Signature()
        self._materialized = itertools.count()

    # ------------------------------------------------------------------
    # building blocks
    # ------------------------------------------------------------------
    def var(self, name: str) -> E.Var:
        """A single symbolic variable. Any non-empty name works -- names are
        renamed to legal ones during compilation and restored in results."""
        if not isinstance(name, str) or not name:
            raise TypeError("variable names are non-empty strings")
        if name in self.backend.reserved_variable_names():
            raise SymbolicSymbolError(
                f"{name!r} is reserved by {self.describe()} and cannot name a "
                f"variable")
        return E.Var(self, name)

    def vars(self, names: Union[str, Sequence[str]]) -> tuple:
        """Several symbolic variables. Accepts a list of names or a single
        whitespace-separated string."""
        if isinstance(names, str):
            names = names.split()
        return tuple(self.var(n) for n in names)

    # The proposal's spelling, kept as the primary public name.
    get_symbolic_vars = vars

    def const(self, value) -> E.Const:
        """A constant term for a Python value, encoded through the codec."""
        if self.signature.codec is None:
            raise SymbolicSymbolError(
                f"{self.describe()} has no element codec, so Python values "
                f"cannot be used as constants; pass a Signature(codec=...)")
        return E.Const(self, value)

    def term(self, x) -> E.Term:
        """Coerce a variable name, Python value or term into a term."""
        if isinstance(x, E.Term):
            if x.ctx is not self:
                raise ValueError("term belongs to a different structure")
            return x
        if isinstance(x, str):
            return self.var(x)
        return self.const(x)

    def rel(self, symbol: str) -> RelationSymbol:
        """A relation symbol of the signature."""
        self.relation_arity(symbol)  # fail early on unknown symbols
        return RelationSymbol(self, symbol)

    get_symbolic_rel = rel

    def func(self, name: str) -> FunctionSymbol:
        """A function symbol of the signature."""
        return FunctionSymbol(self, name, self.function(name))

    get_symbolic_func = func

    def atom(self, symbol: str, args) -> E.Formula:
        """The atom ``symbol(*args)``."""
        return E.Atom(self, symbol, [self.term(a) for a in args])

    def relation(self, dfa, args, label: str = 'given') -> E.Formula:
        """An atom backed by an automaton built outside the symbolic layer."""
        return E.DfaAtom(self, dfa, [self.term(a) for a in args], label)

    # ------------------------------------------------------------------
    # signature lookup
    # ------------------------------------------------------------------
    def relation_arity(self, symbol: str) -> int:
        arity = self.backend.arity(symbol)
        if arity is None:
            arity = self.signature.relations.get(symbol)
        if arity is None:
            raise SymbolicSymbolError(
                f"{self.describe()} has no relation {symbol!r}; "
                f"available: {sorted(self.backend.relation_symbols())}")
        return arity

    def function(self, name: str) -> Function:
        try:
            return self.signature.functions[name]
        except KeyError:
            raise SymbolicSymbolError(
                f"{self.describe()} declares no function {name!r}; "
                f"available: {sorted(self.signature.functions)}") from None

    def symbols(self) -> Dict[str, List[str]]:
        """A summary of what this context offers."""
        return {
            'relations': sorted(self.backend.relation_symbols()),
            'functions': sorted(self.signature.functions),
            'operators': dict(self.signature.operators),
        }

    def describe(self) -> str:
        return self.backend.describe()

    # ------------------------------------------------------------------
    # operator dispatch
    # ------------------------------------------------------------------
    def _apply_operator(self, op: str, args) -> Union[E.Term, E.Formula]:
        try:
            symbol = self.signature.operators[op]
        except KeyError:
            raise TypeError(
                f"{self.describe()} declares no operator {op!r}; "
                f"available: {sorted(self.signature.operators)}") from None
        args = [self.term(a) for a in args]
        if symbol in self.signature.functions:
            return E.Apply(self, symbol, args)
        return self.atom(symbol, args)

    def _atom_from_operator(self, op: str, args) -> E.Formula:
        result = self._apply_operator(op, args)
        if not isinstance(result, E.Formula):
            raise TypeError(f"operator {op!r} is a function, not a relation")
        return result

    def _is_binary(self, op: str) -> bool:
        """Whether the operator's symbol takes two arguments."""
        symbol = self.signature.operators[op]
        if symbol in self.signature.functions:
            function = self.signature.functions[symbol]
            return self.relation_arity(function.graph) - 1 == 2
        return self.relation_arity(symbol) == 2

    def fresh_name(self, taken) -> str:
        """A user-level variable name not in ``taken``."""
        taken = set(taken)
        for i in itertools.count():
            name = f"_v{i}"
            if name not in taken:
                return name

    # ------------------------------------------------------------------
    # constants and exists-infinity
    # ------------------------------------------------------------------
    def _constant_atom(self, term: E.Const) -> E.DfaAtom:
        dfa = self.backend.constant_automaton(
            self.signature.codec.encode(term.value))
        return E.DfaAtom(self, dfa, [], label=f"const {term.value!r}")

    def _relation_arity_of(self, dfa) -> int:
        return self.backend.arity_of(dfa)

    def _expand_exinf(self, node: E.ExInf) -> E.Formula:
        """Rewrite :math:`\\exists^\\infty x.\\varphi` into an ordinary
        projection against a "witnesses are unboundedly long" automaton, the
        standard automatic-structure encoding.

        The body has to be compiled first -- the witness automaton's parameter
        is the body automaton's state count -- so this is one of the few
        places where a subformula is materialized on the way.
        """
        body = node.body
        variables = body.variables()
        if node.variable not in variables:
            raise SymbolicSymbolError(
                f"exinf variable {node.variable!r} is not free in the body")
        others = [v for v in variables if v != node.variable]

        compiled = self.evaluate(body)
        witness = self.backend.longer_witness_automaton(
            compiled.dfa.num_states + 1, len(others))

        body_atom = E.DfaAtom(self, compiled.dfa,
                              [self.var(v) for v in compiled.variables],
                              label='exinf body', prepared=True)
        witness_atom = E.DfaAtom(
            self, witness,
            [self.var(v) for v in others] + [self.var(node.variable)],
            label='exinf witness')
        return E.Exists(self, [node.variable], body_atom & witness_atom)

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------
    def compile(self, formula: E.Formula):
        """The first-order query this formula lowers to. Returned as
        ``(expression, variables)`` with ``variables`` naming the tapes in the
        user's own vocabulary -- useful for inspection and debugging."""
        compiler = Compiler(self)
        expression, user_variables = compiler.compile(formula)
        return expression, user_variables

    def evaluate(self, formula: E.Formula) -> 'Relation':
        """Compile and evaluate, returning the presentation of the satisfying
        assignments together with its tape order."""
        compiler = Compiler(self)
        expression, _ = compiler.compile(formula)
        dfa, internal = self.backend.evaluate(
            expression, compiler.updates, compiler.prepared)
        variables = restore(compiler.names, internal)
        return Relation(self, dfa, variables)

    def check(self, formula: E.Formula) -> bool:
        """True if the formula is satisfiable over the structure -- free
        variables read as existentially quantified."""
        return not self.evaluate(formula).is_empty()

    # ------------------------------------------------------------------
    # member-level evaluation (uniformly automatic classes)
    # ------------------------------------------------------------------
    def check_member(self, formula: E.Formula, advice, implicit: bool = False,
                     **assignments) -> bool:
        """Model check a formula against the member structure picked out by
        ``advice``.

        :param advice: the advice string identifying the member.
        :param implicit: evaluate on the fly over the base automata instead of
            compiling a query automaton -- the same trade-off as
            `UniformlyAutomaticClass.check_implicit`, and the only route for
            classes whose query automaton is infeasible to build.
        :param assignments: concrete elements for free variables, named by the
            variables of this expression. Unassigned free variables are read as
            existentially quantified.
        """
        expression, internal = self._member_query(formula, assignments)
        return self.backend.check_member(expression, advice, internal, implicit)

    def evaluate_member(self, formula: E.Formula, advice, **assignments):
        """The satisfying set of a formula on one member, computed implicitly.

        Returns a solution set that knows its exact size without enumerating
        and yields assignments lazily; see `autstr.implicit`.
        """
        expression, internal = self._member_query(formula, assignments)
        return self.backend.evaluate_member(expression, advice, internal)

    def get_structure(self, advice):
        """The member structure for ``advice`` as an ordinary automatic
        presentation. Call `symbolic` on it to work inside that one member."""
        return self.backend.get_structure(advice)

    def _member_query(self, formula: E.Formula, assignments: Dict):
        compiler = Compiler(self)
        expression, user_variables = compiler.compile(formula)
        unknown = set(assignments) - set(user_variables)
        if unknown:
            raise SymbolicSymbolError(
                f"assignments for variables that are not free in the formula: "
                f"{sorted(unknown)}")
        internal = {compiler.names.internal(name): value
                    for name, value in assignments.items()}
        return expression, internal

    def materialize(self, formula: E.Formula, name: str = None) -> E.Formula:
        """Evaluate now, and return a formula that splices the result in."""
        compiled = self.evaluate(formula)
        label = name or f"materialized{next(self._materialized)}"
        return E.DfaAtom(self, compiled.dfa,
                         [self.var(v) for v in compiled.variables],
                         label=label, prepared=True)

    def __repr__(self):
        return f"<SymbolicContext {self.describe()}>"


class Relation:
    """The result of evaluating a formula: an automaton plus the tape order.

    The tape order is the sorted list of the formula's free variable names.
    Membership and iteration are keyed by name, never by position, so renaming
    a variable cannot silently change what a query means.
    """

    def __init__(self, ctx: SymbolicContext, dfa, variables: Sequence[str]):
        self.ctx = ctx
        self.dfa = dfa
        self.variables = list(variables)

    @property
    def arity(self) -> int:
        return len(self.variables)

    def is_empty(self) -> bool:
        return self.dfa.is_empty()

    def is_finite(self) -> bool:
        """Whether finitely many tuples satisfy the relation."""
        return self.ctx.backend.is_finite(self.dfa)

    def reorder(self, variables: Sequence[str]) -> 'Relation':
        """The same relation with its tapes permuted into the given order."""
        from autstr.utils.automata_tools import permute_tapes
        variables = list(variables)
        if sorted(variables) != sorted(self.variables):
            raise ValueError(
                f"reorder needs a permutation of {self.variables}, "
                f"got {variables}")
        perm = [self.variables.index(v) for v in variables]
        return Relation(self.ctx, permute_tapes(self.dfa, perm), variables)

    def contains(self, *positional, **assignment) -> bool:
        """Whether a tuple satisfies the relation. Positional arguments follow
        `variables`; keyword arguments name them."""
        if positional and assignment:
            raise TypeError("give either positional or named values")
        if positional:
            if len(positional) != self.arity:
                raise ValueError(
                    f"relation has arity {self.arity}, got {len(positional)}")
            assignment = dict(zip(self.variables, positional))
        missing = set(self.variables) - set(assignment)
        if missing:
            raise ValueError(f"no value given for {sorted(missing)}")
        unknown = set(assignment) - set(self.variables)
        if unknown:
            raise ValueError(f"not variables of this relation: {sorted(unknown)}")
        return self.ctx.backend.accepts(
            self.dfa, [assignment[v] for v in self.variables],
            self.ctx.signature.codec)

    def __contains__(self, item):
        if not isinstance(item, tuple):
            item = (item,)
        return self.contains(*item)

    def __iter__(self):
        """Enumerate the satisfying tuples in length-lexicographic order,
        decoded through the codec when the signature has one."""
        return self.ctx.backend.iterate(self.dfa, self.ctx.signature.codec,
                                        self.arity)

    def __repr__(self):
        return (f"<Relation {tuple(self.variables)} "
                f"{self.dfa.num_states} states>")
