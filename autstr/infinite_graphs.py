"""Infinite graphs as automatic structures.

A thin, engine-agnostic wrapper over an automatic or tree-automatic
presentation whose signature has a domain ``U`` and a binary edge relation. It
adds the graph vocabulary on top of the symbolic layer — ``x.adj(y)`` for
adjacency, plus ``.eq`` — so every infinite-graph factory (ordinals, integer
grids, the level-2 collapsible pushdown graphs, …) plugs into one surface,
exactly as ``InfiniteExtraspecialGroup`` shares a presentation.

The wrapper decides nothing itself: it forwards to the presentation, which is
where FO(∃^∞) is decided by synchronous projection. It works over either the
string or the tree engine, since both presentations expose the same relational
interface (`symbolic`, `check`, `evaluate`, `relation`, `get_relation_symbols`).
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

from autstr.symbolic import FunctionCodec, graph_signature


class InfiniteGraph:
    """A graph presented by automata: a domain and a binary edge relation.

    :param presentation: an `AutomaticPresentation` or `TreeAutomaticPresentation`
        carrying the domain ``U`` and the edge relation.
    :param edge: the binary relation read as adjacency (default ``'E'``).
    :param directed: whether edges are directed. Undirected is the default; the
        edge automaton is expected to be symmetric, which `is_symmetric` checks.
    :param codec: optional element codec, so vertices can be written as Python
        constants and solutions decoded.
    """

    #: the default name of the edge relation
    EDGE = 'E'

    def __init__(self, presentation, edge: Optional[str] = None,
                 directed: bool = False, codec=None) -> None:
        self.presentation = presentation
        self.edge = edge or self.EDGE
        self.directed = directed
        self.codec = codec

        symbols = presentation.get_relation_symbols()
        if self.edge not in symbols:
            raise ValueError(
                f"the presentation has no edge relation {self.edge!r}; "
                f"its relations are {sorted(s for s in symbols if s != 'U')}")
        arity = presentation.relation(self.edge).symbol_arity
        if arity != 2:
            raise ValueError(
                f"the edge relation {self.edge!r} has arity {arity}, not 2; "
                f"a graph edge is binary")

    def default_signature(self):
        """The signature `symbolic()` uses when none is given: ``.adj`` bound to
        the edge relation, and ``.eq`` when the graph declares equality."""
        return graph_signature(self.presentation.get_relation_symbols(),
                               edge=self.edge, codec=self.codec)

    def symbolic(self, signature=None):
        """A symbolic interface to the graph. Build first-order formulas with
        ``x.adj(y)`` and the usual connectives / quantifiers; see
        `autstr.symbolic`."""
        return self.presentation.symbolic(signature or self.default_signature())

    def is_symmetric(self) -> bool:
        """Whether the edge relation is symmetric — decidable here, since it is
        a first-order question over an automatic structure. An undirected graph
        must satisfy it."""
        return self.presentation.check(
            f"all x.(all y.({self.edge}(x,y) -> {self.edge}(y,x)))")

    # -- thin passthroughs to the presentation ------------------------
    def check(self, phi) -> bool:
        """Truth of a formula over the graph (free variables existential)."""
        return self.presentation.check(phi)

    def evaluate(self, phi):
        """The relation of satisfying assignments of a formula."""
        return self.presentation.evaluate(phi)

    def get_relation_symbols(self):
        """All relation symbols of the graph ('U' is the domain)."""
        return self.presentation.get_relation_symbols()

    def __repr__(self):
        kind = 'directed' if self.directed else 'undirected'
        return (f"<InfiniteGraph {kind}, edge={self.edge!r}, "
                f"relations={sorted(s for s in self.get_relation_symbols() if s != 'U')}>")


# ----------------------------------------------------------------------
# Concrete factory: the integer grid
# ----------------------------------------------------------------------

#: least-power-of-two, i.e. the constant 1, over Büchi arithmetic
_ONE = 'Pt(o) & (all p.(Pt(p) -> (not Lt(p,o))))'
_PAD = '*'


class IntegerGrid:
    """The n-dimensional integer grid — the Cayley graph of ℤⁿ with the
    standard generators.

    Vertices are points of ℤⁿ; two are adjacent iff they differ by ±1 in
    exactly one coordinate. That is precisely the *asynchronous* product of n
    copies of the two-way integer path (move one coordinate, hold the rest), so
    the grid is built by folding `autstr.composition.direct_product` over n
    integer paths rather than by authoring an automaton.

    FO is decidable — it is an automatic structure. MSO is *not*: the grid
    interprets the halting problem, the tidiest illustration that FO and MSO
    are different questions.

    :param n: the dimension (n ≥ 1). ``IntegerGrid(1)`` is the two-way path.
    """

    def __init__(self, n: int = 2) -> None:
        if n < 1:
            raise ValueError("dimension must be >= 1")
        self.n = n
        path = self._path()                      # one Büchi build, reused
        presentation = path
        for _ in range(n - 1):
            presentation = _async_product(presentation, path)
        self.presentation = presentation
        self.graph = InfiniteGraph(
            presentation, edge='E',
            codec=FunctionCodec(self.encode, self.decode))

    @staticmethod
    def _path():
        """The two-way integer path as a minimal presentation carrying only
        the domain, the ±1 edge, and equality — so the product folds those and
        nothing else."""
        from autstr.buildin.presentations import BuechiArithmeticZ
        from autstr.presentations import AutomaticPresentation
        z = BuechiArithmeticZ()
        z.update(E=f'exists o.(({_ONE}) & (A(x,o,y) | A(y,o,x)))')
        return AutomaticPresentation(
            {'U': z.automata['U'], 'E': z.automata['E'], 'Eq': z.automata['Eq']},
            padding_symbol=z.padding_symbol)

    # -- element codec: an n-tuple <-> its nested-pair convolution ------
    def encode(self, point: Sequence[int]) -> list:
        from autstr.arithmetic import encode as encode_int
        point = tuple(point)
        if len(point) != self.n:
            raise ValueError(f"expected a {self.n}-tuple, got {point!r}")
        word = encode_int(point[0])
        pad = _PAD
        for coordinate in point[1:]:
            other = encode_int(coordinate)
            length = max(len(word), len(other))
            word = [(word[k] if k < len(word) else pad,
                     other[k] if k < len(other) else _PAD)
                    for k in range(length)]
            pad = (pad, _PAD)
        return word

    def decode(self, word) -> Tuple[int, ...]:
        from autstr.arithmetic import decode as decode_int
        coordinates = []
        current = list(word)
        for _ in range(self.n - 1):
            coordinates.append(decode_int([letter[1] for letter in current]))
            current = [letter[0] for letter in current]
        coordinates.append(decode_int(current))
        return tuple(reversed(coordinates))

    # -- interface -----------------------------------------------------
    def symbolic(self, signature=None):
        """A symbolic interface to the grid; write adjacency as ``x.adj(y)``
        and vertices as Python n-tuples."""
        return self.graph.symbolic(signature)

    def is_symmetric(self) -> bool:
        return self.graph.is_symmetric()

    def check(self, phi) -> bool:
        return self.graph.check(phi)

    def evaluate(self, phi):
        return self.graph.evaluate(phi)

    def __repr__(self):
        return f"<IntegerGrid dimension={self.n}>"


def _async_product(left, right):
    from autstr.composition import direct_product
    return direct_product(left, right, kind='async')
