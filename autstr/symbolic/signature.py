"""Signatures: what a structure exposes to the symbolic layer.

An automatic presentation is a bag of automata keyed by relation symbol. A
signature adds the information the symbolic layer needs on top of that: which
relations are graphs of functions, which Python operators those functions are
bound to, and how Python values translate to and from element encodings.

Arities are never declared -- they are read off the automata themselves
(`dfa.symbol_arity`, minus the advice tape for a uniformly automatic class).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

# Python operators a signature may bind to a function symbol. The keys are the
# names used in `Signature.operators`; `expr` dispatches through them.
BINARY_OPERATORS = ('+', '-', '*', '@')
UNARY_OPERATORS = ('-',)


@dataclass(frozen=True)
class Function:
    """A function symbol, presented by the automaton of its graph.

    :param graph: relation symbol whose automaton recognizes the graph.
    :param out: position of the output tape in the graph relation. Defaults to
        the last tape; negative values count from the end.
    :param arity: number of inputs. Derived from the graph relation's arity
        when the signature is bound to a structure.
    """
    graph: str
    out: int = -1

    def positions(self, graph_arity: int) -> tuple:
        """(input positions in order, output position) for a graph of the
        given arity."""
        out = self.out % graph_arity
        inputs = tuple(i for i in range(graph_arity) if i != out)
        return inputs, out


#: The standard name for a structure's equality relation. Some structures
#: also answer to 'E', but 'E' is the *edge* relation in every graph class, so
#: equality is always named explicitly rather than guessed from the symbols.
EQUALITY_SYMBOL = 'Eq'


def operation_signature(relations, graph: str, operator: str,
                        equality: str = EQUALITY_SYMBOL,
                        codec=None) -> 'Signature':
    """The signature of a structure whose binary operation is presented by the
    ternary graph relation `graph`, bound to `operator`.

    Equality is bound to ``.eq`` when the structure declares `equality`. The
    name is passed in rather than guessed: 'E' means equality in Skolem
    arithmetic but the *edge* relation in every graph class, so guessing would
    silently answer "are these adjacent?" for "are these equal?".

    A structure without an equality relation still gets the operator, but its
    terms cannot become formulas -- ``(x + y).eq(z)`` is the only way to say
    what a term denotes.

    :param relations: the structure's relation symbols.
    :param graph: the ternary relation R(x, y, z) meaning ``x op y = z``.
    :param operator: the Python operator to bind, ``'*'`` or ``'+'``.
    :param codec: optional element codec; unused over a uniformly automatic
        class, where an element's encoding depends on the advice.
    """
    signature = Signature(codec=codec)
    signature.function(operator, graph=graph, out=2)
    signature.operator(operator, operator)
    if equality in relations:
        signature.operator('eq', equality)
    return signature


def relational_signature(relations, methods: Dict[str, str],
                         equality: str = EQUALITY_SYMBOL,
                         codec=None) -> 'Signature':
    """The signature of a purely relational structure: each method name in
    `methods` bound to the relation symbol it names, plus equality when the
    structure declares it.

    Nothing binds to ``+`` or ``*`` — a relational structure carries no
    operation, so every symbol is reached as a method, exactly like ``.lt`` in
    the arithmetic signature. The requested methods are bound whether or not
    the structure declares them, so a symbol that is missing fails loudly when
    a formula uses it rather than silently going unbound; only equality, which
    a caller asks for generically, is conditional.

    :param relations: the structure's relation symbols.
    :param methods: ``{method name: relation symbol}``.
    :param equality: the equality symbol, bound to ``.eq`` when present.
    :param codec: optional element codec for writing elements as constants.
    """
    signature = Signature(codec=codec)
    for method, symbol in methods.items():
        signature.operator(method, symbol)
    if equality in relations:
        signature.operator('eq', equality)
    return signature


def graph_signature(relations, edge: str = 'E', adjacency: str = 'adj',
                    equality: str = EQUALITY_SYMBOL, codec=None) -> 'Signature':
    """The signature of a graph: the binary edge relation `edge` bound to the
    method ``.{adjacency}(y)`` (default ``.adj``), plus equality when the
    structure declares it.

    A graph carries no operation, so nothing binds to ``+`` or ``*``; adjacency
    is a relation method, exactly like ``.lt`` in the arithmetic signature.
    The edge name is passed in, never guessed — ``E`` means the edge here but
    equality elsewhere, the same hazard `operation_signature` guards against.

    :param relations: the structure's relation symbols.
    :param edge: the binary relation read as adjacency.
    :param adjacency: the method name it binds to.
    :param codec: optional element codec for writing vertices as constants.
    """
    return relational_signature(relations, {adjacency: edge},
                                equality=equality, codec=codec)


def order_signature(relations, less: str = 'Lt', order: str = 'lt',
                    equality: str = EQUALITY_SYMBOL, codec=None,
                    methods: Optional[Dict[str, str]] = None) -> 'Signature':
    """The signature of an ordered structure: the binary relation `less` bound
    to ``.{order}(y)`` (default ``.lt``), plus equality when declared.

    An order is not a graph — ``x.lt(y)`` and ``x.adj(y)`` read differently
    even where both are binary — so orders get their own vocabulary rather than
    being wrapped as graphs. Further relations of the same structure (a
    successor, a limit predicate) go in `methods`.

    :param relations: the structure's relation symbols.
    :param less: the binary relation read as the strict order.
    :param order: the method name it binds to.
    :param codec: optional element codec for writing elements as constants.
    :param methods: further ``{method name: relation symbol}`` bindings.
    """
    return relational_signature(relations, {order: less, **(methods or {})},
                                equality=equality, codec=codec)


class ElementCodec:
    """Translation between Python values and element encodings.

    What an encoding *is* belongs to the backend: a list of base-alphabet
    symbols in the order the automata read them for the string engines, a
    `Tree` for the tree engine. The codec's output is only ever handed back to
    the backend that asked for it, so this layer does not interpret it.

    Supplying a codec is optional: without one the symbolic layer still works,
    but constants cannot be written as Python values and solutions are yielded
    in their raw encoded form.
    """

    def encode(self, value: Any) -> Any:
        raise NotImplementedError

    def decode(self, encoded: Any) -> Any:
        raise NotImplementedError


@dataclass(frozen=True)
class FunctionCodec(ElementCodec):
    """A codec built from two plain functions."""
    encoder: Callable[[Any], Any]
    decoder: Optional[Callable[[Any], Any]] = None

    def encode(self, value):
        # Returned as the encoder produced it: coercing to a list here would
        # bake in the word-shaped engines and break tree encodings.
        return self.encoder(value)

    def decode(self, word):
        if self.decoder is None:
            raise NotImplementedError("this codec cannot decode")
        return self.decoder(word)


@dataclass
class Signature:
    """The symbolic-layer description of a structure's signature.

    :param functions: function symbol -> `Function`. The graph relation must
        exist in the presentation.
    :param operators: Python operator or method name -> symbol it dispatches
        to. Keys may name a function symbol's operator (``'+'``, ``'-'``,
        ``'*'``, ``'@'``) or a relation method (any identifier, e.g. ``'lt'``,
        ``'eq'``), and values are function or relation symbols respectively.
    :param codec: optional `ElementCodec` for constants and decoding.
    :param relations: optional arity overrides. Normally arities come from the
        automata; entries here are only consulted for symbols that are not
        (yet) present in the presentation.
    """
    functions: Dict[str, Function] = field(default_factory=dict)
    operators: Dict[str, str] = field(default_factory=dict)
    codec: Optional[ElementCodec] = None
    relations: Dict[str, int] = field(default_factory=dict)

    def function(self, name: str, graph: str, out: int = -1) -> Signature:
        """Declare a function symbol. Returns self, so declarations chain."""
        self.functions[name] = Function(graph=graph, out=out)
        return self

    def operator(self, op: str, symbol: str) -> Signature:
        """Bind a Python operator or method name to a function or relation
        symbol. Returns self, so declarations chain."""
        self.operators[op] = symbol
        return self
