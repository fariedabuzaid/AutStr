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


class ElementCodec:
    """Translation between Python values and element encodings.

    An encoding is a list of base-alphabet symbols, in the order the automata
    read them. Supplying a codec is optional: without one the symbolic layer
    still works, but constants cannot be written as Python values and solutions
    are yielded as raw words.
    """

    def encode(self, value: Any) -> List:
        raise NotImplementedError

    def decode(self, word: Sequence) -> Any:
        raise NotImplementedError


@dataclass(frozen=True)
class FunctionCodec(ElementCodec):
    """A codec built from two plain functions."""
    encoder: Callable[[Any], List]
    decoder: Optional[Callable[[Sequence], Any]] = None

    def encode(self, value):
        return list(self.encoder(value))

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
