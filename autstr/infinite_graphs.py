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


# ----------------------------------------------------------------------
# Concrete factory: the regular tree
# ----------------------------------------------------------------------

class RegularTree:
    """The infinite k-ary tree :math:`T_k`, with its successors and the prefix
    order.

    Vertices are the words over ``{0, …, k-1}``: the root is the empty word,
    and ``x·i`` is the i-th child of ``x``. So the tree *is* its own encoding,
    and the relations are small automata read straight off the word operations
    — appending one letter, and being a prefix — rather than derived from
    another structure.

    The presentation carries ``S0 … S{k-1}`` (``Si(x, y)`` iff ``y = x·i``),
    their union ``Child``, ``Prefix`` (the reflexive prefix order, i.e.
    ancestor-or-self), ``Eq``, and the undirected child edge ``E``, under which
    the tree is a graph: the root has degree k and every other vertex degree
    k+1.

    :math:`T_k` is the Cayley graph of the free monoid on k generators, and the
    2k-regular version is the Cayley graph of the free group — the same object
    the pushdown graphs are unravelled from.

    :param k: the branching degree (k ≥ 1). ``RegularTree(1)`` is a ray.
    """

    def __init__(self, k: int = 2) -> None:
        if k < 1:
            raise ValueError("the branching degree must be >= 1")
        self.k = k
        self.letters = [str(i) for i in range(k)]
        alphabet = self.letters + [_PAD]

        from autstr.presentations import AutomaticPresentation
        automata = {'U': _words(alphabet, self.letters),
                    'Eq': _identity(alphabet, self.letters),
                    'Prefix': _prefix(alphabet, self.letters)}
        automata.update({f'S{i}': _append(alphabet, self.letters, letter)
                         for i, letter in enumerate(self.letters)})
        self.presentation = AutomaticPresentation(automata, padding_symbol=_PAD)
        self.presentation.update(
            Child=' | '.join(f'S{i}(x,y)' for i in range(k)))
        self.presentation.update(E='Child(x,y) | Child(y,x)')

        self.graph = InfiniteGraph(
            self.presentation, edge='E',
            codec=FunctionCodec(self.encode, self.decode))

    # -- element codec: a vertex as its word ---------------------------
    def encode(self, vertex) -> list:
        """The encoding of a vertex, written as a sequence of child indices —
        ``(0, 1, 1)``, or the string ``'011'`` — with the empty sequence for
        the root."""
        word = []
        for step in vertex:
            index = int(step)
            if not 0 <= index < self.k:
                raise ValueError(
                    f"{step!r} is not a child index of a {self.k}-ary tree")
            word.append(self.letters[index])
        return word

    def decode(self, word) -> Tuple[int, ...]:
        """The vertex encoded by a word, as a tuple of child indices."""
        return tuple(int(symbol) for symbol in word if symbol != _PAD)

    # -- interface -----------------------------------------------------
    def symbolic(self, signature=None):
        """A symbolic interface to the tree; write the child edge as
        ``x.adj(y)`` and vertices as sequences of child indices. The
        successors and the prefix order are reached by name, through
        `autstr.symbolic.SymbolicContext.rel`."""
        return self.graph.symbolic(signature)

    def is_symmetric(self) -> bool:
        return self.graph.is_symmetric()

    def check(self, phi) -> bool:
        return self.graph.check(phi)

    def evaluate(self, phi):
        return self.graph.evaluate(phi)

    def get_relation_symbols(self):
        return self.presentation.get_relation_symbols()

    def __repr__(self):
        return f"<RegularTree branching={self.k}>"


def _dfa(alphabet, arity: int, transitions, initial: str, final):
    from autstr.utils.automata_tools import partial_dfa
    return partial_dfa(set(alphabet), arity, transitions, initial, set(final))


def _words(alphabet, letters):
    """The universe: every word over the child letters, the root included."""
    return _dfa(alphabet, 1,
                {'w': {(letter,): 'w' for letter in letters}},
                initial='w', final={'w'})


def _identity(alphabet, letters):
    """``x = y``, on the convolution of two words."""
    return _dfa(alphabet, 2,
                {'s': {(letter, letter): 's' for letter in letters}},
                initial='s', final={'s'})


def _append(alphabet, letters, letter):
    """``y = x·letter``: the tapes agree while x runs, then x pads out and y
    carries the one extra letter."""
    table = {'s': {(a, a): 's' for a in letters}}
    table['s'][(_PAD, letter)] = 'done'
    table['done'] = {}
    return _dfa(alphabet, 2, table, initial='s', final={'done'})


def _prefix(alphabet, letters):
    """``x`` is a prefix of ``y``, reflexively: the tapes agree while x runs,
    and whatever y has left is the descent below it."""
    table = {'s': {(a, a): 's' for a in letters},
             'below': {(_PAD, a): 'below' for a in letters}}
    table['s'].update({(_PAD, a): 'below' for a in letters})
    return _dfa(alphabet, 2, table, initial='s', final={'s', 'below'})
