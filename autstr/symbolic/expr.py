"""The symbolic expression AST.

Terms denote elements of a structure, formulas denote relations over it. Nodes
are immutable and compare structurally, so equal subexpressions are
interchangeable and can be used as dictionary keys. Nothing here touches an
automaton: building an expression is pure bookkeeping, and all automata
construction happens when the expression is handed to a backend (see
`autstr.symbolic.compiler`).

Operators are not hardwired. ``x + y`` looks up ``'+'`` in the structure's
`Signature.operators` and builds an application of whatever function symbol it
names; a structure that declares no ``'+'`` raises a clear error instead of
silently meaning addition.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, List, Sequence, Tuple, Union


class Node:
    """Common base: immutable, structurally compared, bound to a context."""

    __slots__ = ('ctx', '_key')

    def __init__(self, ctx, key: tuple):
        object.__setattr__(self, 'ctx', ctx)
        object.__setattr__(self, '_key', key)

    def __setattr__(self, name, value):
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __eq__(self, other):
        return (type(self) is type(other)
                and self.ctx is other.ctx
                and self._key == other._key)

    def __hash__(self):
        return hash((type(self).__name__, id(self.ctx), self._key))


# ======================================================================
# Terms
# ======================================================================

class Term(Node):
    """A term denoting an element of the structure."""

    __slots__ = ()

    # -- variables ----------------------------------------------------
    def variables(self) -> List[str]:
        """Free variable names, sorted."""
        raise NotImplementedError

    # -- relational atoms ---------------------------------------------
    def eq(self, other) -> 'Formula':
        """The relation ``self = other``, via the structure's equality
        relation."""
        return self.ctx._atom_from_operator('eq', [self, other])

    def rel(self, symbol: str, *others) -> 'Formula':
        """The atom ``symbol(self, *others)``."""
        return self.ctx.atom(symbol, [self, *others])

    def __getattr__(self, name: str) -> Any:
        """Method-name operators declared by the signature (``x.lt(y)``)."""
        # `ctx` must never route through here -- it is read below, so a missing
        # slot would recurse instead of raising.
        if name.startswith('_') or name in ('ctx', 'name', 'value', 'args'):
            raise AttributeError(name)
        operators = self.ctx.signature.operators
        if name not in operators:
            raise AttributeError(
                f"{self.ctx.describe()} declares no operator {name!r}; "
                f"available: {sorted(operators)}")

        def bound(*others):
            return self.ctx._apply_operator(name, [self, *others])

        return bound

    # -- arithmetic-style operators -----------------------------------
    def _binop(self, op: str, other, swap: bool = False) -> 'Term':
        args = [self.ctx.term(other), self] if swap else [self, self.ctx.term(other)]
        result = self.ctx._apply_operator(op, args)
        if not isinstance(result, Term):
            raise TypeError(f"operator {op!r} of {self.ctx.describe()} is a "
                            f"relation, not a function; use it as a method")
        return result

    def __add__(self, other): return self._binop('+', other)
    def __radd__(self, other): return self._binop('+', other, swap=True)
    def __matmul__(self, other): return self._binop('@', other)
    def __rmatmul__(self, other): return self._binop('@', other, swap=True)

    def __neg__(self):
        return self.ctx._apply_operator('-', [self])

    def __sub__(self, other):
        other = self.ctx.term(other)
        if '-' in self.ctx.signature.operators and self.ctx._is_binary('-'):
            return self._binop('-', other)
        return self + (-other)

    def __rsub__(self, other):
        return self.ctx.term(other) - self

    def __mul__(self, other):
        if '*' in self.ctx.signature.operators:
            return self._binop('*', other)
        if isinstance(other, int):
            return self.times(other)
        raise TypeError(
            f"{self.ctx.describe()} declares no operator '*'; multiplication "
            f"by an integer is available as .times(n)")

    def __rmul__(self, other):
        return self.__mul__(other)

    def times(self, n: int) -> 'Term':
        """The ``n``-fold sum ``self + ... + self`` under the structure's
        ``'+'``, built by base-2 decomposition so that only
        :math:`O(\\log_2 n)` distinct subterms are created.

        Negative ``n`` requires the structure to declare a ``'-'`` inverse.
        """
        if not isinstance(n, int):
            raise TypeError("times(n) needs an integer")
        negative = n < 0
        n = abs(n)
        if n == 0:
            raise ValueError(
                "times(0) has no generic meaning; use the structure's zero "
                "constant explicitly")

        doubling, total = self, None
        for i in range(math.floor(math.log2(n)) + 1):
            if i > 0:
                doubling = doubling + doubling
            if (n >> i) & 1:
                total = doubling if total is None else total + doubling
        return -total if negative else total

    def substitute(self, **replacements) -> 'Term':
        """Replace free variables. Terms are immutable, so this returns a new
        term and never disturbs expressions that share this one."""
        return self._substitute(
            {k: self.ctx.term(v) for k, v in replacements.items()})

    def _substitute(self, mapping) -> 'Term':
        raise NotImplementedError


class Var(Term):
    """A free variable."""

    __slots__ = ('name',)

    def __init__(self, ctx, name: str):
        super().__init__(ctx, (name,))
        object.__setattr__(self, 'name', name)

    def variables(self):
        return [self.name]

    def _substitute(self, mapping):
        return mapping.get(self.name, self)

    def __str__(self):
        return self.name


class Const(Term):
    """A Python value, encoded through the signature's codec."""

    __slots__ = ('value',)

    def __init__(self, ctx, value):
        super().__init__(ctx, (value,))
        object.__setattr__(self, 'value', value)

    def variables(self):
        return []

    def _substitute(self, mapping):
        return self

    def __str__(self):
        return repr(self.value)


class Apply(Term):
    """An application ``f(t_1, ..., t_n)`` of a declared function symbol."""

    __slots__ = ('func', 'args')

    def __init__(self, ctx, func: str, args: Sequence[Term]):
        args = tuple(args)
        super().__init__(ctx, (func, args))
        object.__setattr__(self, 'func', func)
        object.__setattr__(self, 'args', args)

    def variables(self):
        return sorted({v for a in self.args for v in a.variables()})

    def _substitute(self, mapping):
        return Apply(self.ctx, self.func,
                     [a._substitute(mapping) for a in self.args])

    def __str__(self):
        return f"{self.func}({', '.join(str(a) for a in self.args)})"


# ======================================================================
# Formulas
# ======================================================================

class Formula(Node):
    """A formula denoting a relation over the structure."""

    __slots__ = ()

    def variables(self) -> List[str]:
        """Free variable names, sorted. This is also the tape order of the
        automaton produced by `evaluate`."""
        raise NotImplementedError

    # -- boolean algebra ----------------------------------------------
    def __and__(self, other): return And(self.ctx, self, _formula(self.ctx, other))
    def __or__(self, other): return Or(self.ctx, self, _formula(self.ctx, other))
    def __invert__(self): return Not(self.ctx, self)

    def implies(self, other) -> 'Formula':
        return ~self | _formula(self.ctx, other)

    def iff(self, other) -> 'Formula':
        other = _formula(self.ctx, other)
        return (~self | other) & (~other | self)

    # -- quantification -----------------------------------------------
    def _quantified(self, variables, what: str) -> List[str]:
        """The names to bind, refusing any that is not free in the body.

        Binding a variable that does not occur is logically harmless but is
        almost always a typo, and it fails silently: the intended variables
        stay free and are existentially closed by `check`.
        """
        from autstr.symbolic.context import SymbolicSymbolError
        names = _names(variables)
        free = set(self.variables())
        missing = [n for n in names if n not in free]
        if missing:
            raise SymbolicSymbolError(
                f"cannot {what} {missing}: not free in the body, whose free "
                f"variables are {sorted(free)}")
        return names

    def drop(self, variables) -> 'Formula':
        """Project the relation away from ``variables`` -- equivalently,
        existentially quantify them."""
        return Exists(self.ctx, self._quantified(variables, 'drop'), self)

    ex = drop

    def all(self, variables) -> 'Formula':
        """Universally quantify ``variables``."""
        return Forall(self.ctx, self._quantified(variables, 'quantify'), self)

    def exinf(self, variable) -> 'Formula':
        """:math:`\\exists^\\infty` -- the tuples extended by infinitely many
        witnesses for ``variable``."""
        names = _names(variable)
        if len(names) != 1:
            raise ValueError("exinf quantifies a single variable")
        return ExInf(self.ctx, names[0], self)

    def substitute(self, **replacements) -> 'Formula':
        """Replace free variables, avoiding capture by bound variables."""
        return self._substitute(
            {k: self.ctx.term(v) for k, v in replacements.items()})

    def _substitute(self, mapping) -> 'Formula':
        raise NotImplementedError

    # -- evaluation ---------------------------------------------------
    def evaluate(self):
        """Compile to a `Relation`: the presentation of the satisfying
        assignments together with its tape order."""
        return self.ctx.evaluate(self)

    def check(self) -> bool:
        """True if the relation is non-empty (free variables read as
        existentially quantified)."""
        return self.ctx.check(self)

    def is_empty(self) -> bool:
        return self.evaluate().is_empty()

    def is_finite(self) -> bool:
        return self.evaluate().is_finite()

    def contains(self, *args, **assignment) -> bool:
        return self.evaluate().contains(*args, **assignment)

    def __contains__(self, item):
        return item in self.evaluate()

    def __iter__(self):
        return iter(self.evaluate())

    def materialize(self, name: str = None):
        """Evaluate now and return a formula standing for the resulting
        automaton. Use it to share an expensive subformula across queries;
        the compiled automaton is spliced in instead of being rebuilt."""
        return self.ctx.materialize(self, name)


class Atom(Formula):
    """``R(t_1, ..., t_n)`` for a relation symbol of the signature."""

    __slots__ = ('symbol', 'args')

    def __init__(self, ctx, symbol: str, args: Sequence[Term]):
        args = tuple(args)
        super().__init__(ctx, (symbol, args))
        object.__setattr__(self, 'symbol', symbol)
        object.__setattr__(self, 'args', args)

    def variables(self):
        return sorted({v for a in self.args for v in a.variables()})

    def _substitute(self, mapping):
        return Atom(self.ctx, self.symbol,
                    [a._substitute(mapping) for a in self.args])

    def __str__(self):
        return f"{self.symbol}({', '.join(str(a) for a in self.args)})"


class DfaAtom(Formula):
    """An atom backed by an automaton supplied directly rather than by a
    signature symbol -- the splice point for `Formula.materialize` and for
    automata built outside the symbolic layer."""

    __slots__ = ('dfa', 'args', 'label', 'prepared')

    def __init__(self, ctx, dfa, args: Sequence[Term], label: str = 'anon',
                 prepared: bool = False):
        args = tuple(args)
        super().__init__(ctx, (id(dfa), args, label))
        object.__setattr__(self, 'dfa', dfa)
        object.__setattr__(self, 'args', args)
        object.__setattr__(self, 'label', label)
        object.__setattr__(self, 'prepared', prepared)

    def variables(self):
        return sorted({v for a in self.args for v in a.variables()})

    def _substitute(self, mapping):
        return DfaAtom(self.ctx, self.dfa,
                       [a._substitute(mapping) for a in self.args],
                       self.label, self.prepared)

    def __str__(self):
        return f"<{self.label}>({', '.join(str(a) for a in self.args)})"


class Not(Formula):
    __slots__ = ('body',)

    def __init__(self, ctx, body: Formula):
        super().__init__(ctx, (body,))
        object.__setattr__(self, 'body', body)

    def variables(self):
        return self.body.variables()

    def _substitute(self, mapping):
        return Not(self.ctx, self.body._substitute(mapping))

    def __str__(self):
        return f"not {self.body}"


class _Binary(Formula):
    __slots__ = ('left', 'right')
    connective = '?'

    def __init__(self, ctx, left: Formula, right: Formula):
        super().__init__(ctx, (left, right))
        object.__setattr__(self, 'left', left)
        object.__setattr__(self, 'right', right)

    def variables(self):
        return sorted(set(self.left.variables()) | set(self.right.variables()))

    def _substitute(self, mapping):
        return type(self)(self.ctx, self.left._substitute(mapping),
                          self.right._substitute(mapping))

    def __str__(self):
        return f"({self.left} {self.connective} {self.right})"


class And(_Binary):
    connective = 'and'
    __slots__ = ()


class Or(_Binary):
    connective = 'or'
    __slots__ = ()


class _Quantifier(Formula):
    __slots__ = ('bound', 'body')
    keyword = '?'

    def __init__(self, ctx, bound: Sequence[str], body: Formula):
        bound = tuple(bound)
        super().__init__(ctx, (bound, body))
        object.__setattr__(self, 'bound', bound)
        object.__setattr__(self, 'body', body)

    def variables(self):
        return sorted(set(self.body.variables()) - set(self.bound))

    def _substitute(self, mapping):
        # A replacement for a bound name does not reach into the body; a
        # replacement whose *value* mentions a bound name would be captured, so
        # the binder is renamed first.
        inner = {k: v for k, v in mapping.items() if k not in self.bound}
        if not inner:
            return self
        incoming = {v for t in inner.values() for v in t.variables()}
        body, bound = self.body, list(self.bound)
        for i, name in enumerate(bound):
            if name in incoming:
                # A fresh binder must avoid the body's *bound* names too, or
                # renaming would just push the capture one level down.
                fresh = self.ctx.fresh_name(
                    all_names(body) | incoming | set(bound))
                body = body._substitute({name: Var(self.ctx, fresh)})
                bound[i] = fresh
        return type(self)(self.ctx, bound, body._substitute(inner))

    def __str__(self):
        return f"{self.keyword} {' '.join(self.bound)}. {self.body}"


class Exists(_Quantifier):
    keyword = 'exists'
    __slots__ = ()


class Forall(_Quantifier):
    keyword = 'forall'
    __slots__ = ()


class ExInf(Formula):
    """:math:`\\exists^\\infty x. \\varphi`."""

    __slots__ = ('variable', 'body')

    def __init__(self, ctx, variable: str, body: Formula):
        super().__init__(ctx, (variable, body))
        object.__setattr__(self, 'variable', variable)
        object.__setattr__(self, 'body', body)

    def variables(self):
        return sorted(set(self.body.variables()) - {self.variable})

    def _substitute(self, mapping):
        inner = {k: v for k, v in mapping.items() if k != self.variable}
        if not inner:
            return self
        return ExInf(self.ctx, self.variable, self.body._substitute(inner))

    def __str__(self):
        return f"exists-inf {self.variable}. {self.body}"


# ======================================================================
# helpers
# ======================================================================

def all_names(node) -> set:
    """Every variable name occurring in a formula, free or bound."""
    names = set(node.variables())
    if isinstance(node, _Quantifier):
        names |= set(node.bound) | all_names(node.body)
    elif isinstance(node, ExInf):
        names |= {node.variable} | all_names(node.body)
    elif isinstance(node, Not):
        names |= all_names(node.body)
    elif isinstance(node, _Binary):
        names |= all_names(node.left) | all_names(node.right)
    return names


def _formula(ctx, x) -> Formula:
    if isinstance(x, Formula):
        if x.ctx is not ctx:
            raise ValueError("cannot combine formulas from different structures")
        return x
    raise TypeError(f"expected a formula, got {type(x).__name__}")


def _names(variables) -> List[str]:
    """Accept a variable, a name, a whitespace-separated string of names, or
    an iterable of either.

    Splitting a string matches `SymbolicContext.vars`, which is where these
    names come from. Without it ``.all('x y z')`` bound one variable literally
    called ``'x y z'``, which occurs nowhere -- so x, y and z stayed free and
    were existentially closed, turning a universal into an existential with no
    error anywhere.
    """
    if isinstance(variables, Var):
        variables = [variables]
    elif isinstance(variables, str):
        variables = variables.split()
    names = []
    for v in variables:
        if isinstance(v, Var):
            names.append(v.name)
        elif isinstance(v, str):
            names.append(v)
        else:
            raise TypeError(f"not a variable: {v!r}")
    return names
