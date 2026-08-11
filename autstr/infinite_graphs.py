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

from typing import Optional

from autstr.symbolic import graph_signature


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
