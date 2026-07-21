"""Graphs of bounded tree-width and bounded clique-width as uniformly
tree-automatic classes.

The class presents graphs over *sets* of vertices (as in MSO0), so
first-order logic over the presentation is monadic second-order logic over
the graph — evaluating an MSO query once yields a tree automaton that decides
it on every member graph in linear time (Courcelle's theorem).

**Practical envelope.** Transitions are decision diagrams over the symbol's
digits (`autstr.mtbdd`), so the *width* of the convolution alphabet no longer
drives the cost: a query with several free tapes is as cheap as the digits its
transitions actually test. What remains is the subset explosion of
determinizing an existential quantifier. Element (path) quantifiers project
onto small subset automata and are cheap; *set* quantifiers (MSO proper)
determinize over subsets of the intermediate automaton's states, and each such
subset carries its own diagram. Two-colourability compiles at w = 1; deeper
set-quantifier nesting is bounded by memory, not by the alphabet. Deciding a
compiled automaton on a graph is always linear and fast — compilation is the
bottleneck.

The signature is shared with the string graph classes:

    Sing(x)      x is a singleton
    Subset(x,y)  x is a subset of y
    E(x,y)       x = {u}, y = {v} and u,v are adjacent

**Tree-width <= w** (`TreeWidthClass`): the advice is a binary tree with one
node per vertex, labelled (register in {0..w}, adjacency profile), plus a
structural letter 'n' that introduces no vertex (used to route branch
points). Introducing a vertex at register r replaces the nearest occupant of
r above; the profile lists registers of the vertex's neighbors among the
current occupants on its root path. The live registers along any path form
bags of size <= w+1 — exactly a tree decomposition of width w — and every
graph of tree-width <= w arises this way from a nice decomposition.

**Clique-width <= k** (`CliqueWidthClass`): the advice is a k-expression --
leaves create a labelled vertex, `u` takes a disjoint union, `r{i}{j}`
relabels, and `e{i}{j}` joins every label-i vertex to every label-j vertex.
The vertices are the leaves. Adjacency is much cheaper to recognize than for
tree-width: two vertices are joined exactly when some `e{i}{j}` node sees one
holding label i and the other label j, so the automaton need only carry each
marked vertex's current label. The advice alphabet is correspondingly small,
and MSO queries compile far faster -- two-colourability is a 9-state automaton
at k = 2, against a minute at tree-width 1.

**Rank-width <= r** (`RankWidthClass`): the advice is a rank decomposition
(a binary tree whose leaves are the vertices) annotated with the GF(2)
factorization data of its cuts -- the graph analog of the bounded-rank-width
group classes, compiled with the same `chain_ring` linear algebra at
p = 2, d = 1. Each node carries a basis-change matrix per child and each
binary node the bilinear form of its sibling block; adjacency of x and y is
w_y^T Q w_x at the node where their subtrees meet, so the E automaton only
carries the marked vertices' r-bit interface vectors. Rank-width lower-bounds
clique-width (rw <= cw <= 2^{rw+1} - 1) and is bounded on dense graphs where
tree-width is not (cliques have rank-width 1).

Vertex sets are encoded synchronously over the advice: the element tree is
the union of the root paths to the set's members, labelled '1' on members
and '0' on the way (the empty set is the single node '0'). For tree-width
every node is a vertex; for clique-width and rank-width only the leaves are.
`TreeWidthGraph`, `CliqueWidthGraph` and `RankWidthGraph` encapsulate single
graphs and convert to networkx.
"""
import itertools as it
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from autstr import chain_ring as cr
from autstr.sparse_tree_automata import SparseTreeAutomaton, Tree
from autstr.tree_presentations import TreeAutomaticPresentation
from autstr.tree_uniform import UniformlyTreeAutomaticClass, sta_from_delta
from autstr.uniform import SymbolicClassWrapper

PAD = '*'
NOP = 'n'


def _register_symbol(register: int, profile: Tuple[int, ...]) -> str:
    """Advice alphabet symbol for a (register, profile) vertex letter."""
    return f"r{register}s" + "".join(str(s) for s in profile)


# ====================================================================
# Tree-encoded graphs
# ====================================================================

class TreeWidthGraph:
    """A graph in its tree representation: a binary tree with one
    (register, profile) letter per vertex (and 'n' for structural nodes).
    A vertex's profile lists the registers of its already-introduced
    neighbors, resolved to the nearest writer above on the root path."""

    def __init__(self, letters: Tree, nodes: Optional[Dict[str, object]] = None):
        self.tree = letters
        vertex_addrs: List[str] = []
        stack = [(letters, '', frozenset())]
        while stack:
            node, addr, occupied = stack.pop()
            label = node.label
            if label != NOP:
                r, profile = label
                profile = tuple(profile)
                if r < 0 or any(s < 0 for s in profile):
                    raise ValueError(f"node {addr!r}: negative register")
                if r in profile:
                    raise ValueError(f"node {addr!r}: profile contains the "
                                     f"vertex's own register {r}")
                if len(set(profile)) != len(profile):
                    raise ValueError(f"node {addr!r}: duplicate registers")
                if not set(profile) <= occupied:
                    raise ValueError(
                        f"node {addr!r}: profile references unoccupied "
                        f"registers {set(profile) - occupied}")
                vertex_addrs.append(addr)
                occupied = occupied | {r}
            if node.left is not None:
                stack.append((node.left, addr + '0', occupied))
            if node.right is not None:
                stack.append((node.right, addr + '1', occupied))
        self.vertex_addrs = sorted(vertex_addrs)
        if nodes is None:
            nodes = {a: a for a in self.vertex_addrs}
        if set(nodes) != set(self.vertex_addrs):
            raise ValueError("nodes must be keyed by the vertex addresses")
        self.name_of = dict(nodes)
        self.addr_of = {name: addr for addr, name in self.name_of.items()}
        self.nodes = [self.name_of[a] for a in self.vertex_addrs]

    @property
    def num_nodes(self) -> int:
        return len(self.vertex_addrs)

    @property
    def width(self) -> int:
        """Maximal register index (>= tree-width of the graph)."""
        best = 0
        stack = [self.tree]
        while stack:
            node = stack.pop()
            if node.label != NOP:
                best = max(best, node.label[0])
            stack.extend(c for c in (node.left, node.right) if c is not None)
        return best

    def edges(self) -> List[Tuple]:
        """Edge list (node names) decoded from the letters."""
        out = []
        stack = [(self.tree, '', {})]
        while stack:
            node, addr, occupant = stack.pop()
            if node.label != NOP:
                r, profile = node.label
                me = self.name_of[addr]
                for s in profile:
                    out.append((occupant[s], me))
                occupant = {**occupant, r: me}
            if node.left is not None:
                stack.append((node.left, addr + '0', occupant))
            if node.right is not None:
                stack.append((node.right, addr + '1', occupant))
        return out

    def encode_set(self, subset) -> Tree:
        """Encode a set of nodes as a marked tree over the advice shape."""
        subset = set(subset)
        unknown = subset - set(self.nodes)
        if unknown:
            raise ValueError(f"not nodes of this graph: {unknown}")
        marked = {self.addr_of[v] for v in subset}
        domain = {''} | {a[:i] for a in marked for i in range(1, len(a) + 1)}

        def build(addr):
            left = build(addr + '0') if addr + '0' in domain else None
            right = build(addr + '1') if addr + '1' in domain else None
            return Tree('1' if addr in marked else '0', left, right)
        return build('')

    def to_networkx(self):
        """Convert back to a networkx graph (node names preserved)."""
        try:
            import networkx as nx
        except ImportError as e:
            raise ImportError("to_networkx requires networkx "
                              "(pip install autstr[graphs])") from e
        graph = nx.Graph()
        graph.add_nodes_from(self.nodes)
        graph.add_edges_from(self.edges())
        return graph

    @classmethod
    def from_networkx(cls, graph, decomposition=None) -> 'TreeWidthGraph':
        """Build the tree representation from a networkx graph.

        :param graph: undirected networkx graph (at least one vertex).
        :param decomposition: optional tree decomposition as a networkx tree
            whose nodes are frozensets of vertices (bags). If omitted, the
            min-fill-in heuristic computes one (valid, possibly wider than
            the tree-width).
        """
        if graph.number_of_nodes() == 0:
            raise ValueError("the empty graph has no tree representation")
        if decomposition is None:
            from networkx.algorithms.approximation import \
                treewidth_min_fill_in
            _, decomposition = treewidth_min_fill_in(graph)
        bags = sorted(decomposition.nodes, key=lambda b: sorted(map(str, b)))
        root = bags[0]

        def fanout(subtrees):
            subtrees = [t for t in subtrees if t is not None]
            if not subtrees:
                return None
            if len(subtrees) == 1:
                return subtrees[0]
            return Tree(NOP, subtrees[0], fanout(subtrees[1:]))

        def build(bag, parent, placed):
            placed = dict(placed)               # name -> register (this bag)
            letters = []
            for v in sorted(bag - set(placed), key=str):
                used = set(placed.values())
                r = next(i for i in it.count() if i not in used)
                profile = tuple(sorted(
                    placed[u] for u in graph.neighbors(v) if u in placed))
                letters.append((r, profile, v))
                placed[v] = r
            children = [b for b in decomposition.neighbors(bag)
                        if b != parent]
            cur = fanout([build(b, bag,
                                {u: placed[u] for u in b if u in placed})
                          for b in children])
            for r, profile, v in reversed(letters):
                cur = Tree((r, profile, v), cur, None)
            return cur

        tagged = build(root, None, {})
        if tagged is None:
            raise ValueError("decomposition covers no vertex")

        names: Dict[str, object] = {}

        def strip(node, addr):
            if node is None:
                return None
            if node.label == NOP:
                label = NOP
            else:
                r, profile, v = node.label
                names[addr] = v
                label = (r, profile)
            return Tree(label, strip(node.left, addr + '0'),
                        strip(node.right, addr + '1'))

        return cls(strip(tagged, ''), nodes=names)

    def __repr__(self) -> str:
        return f"TreeWidthGraph({self.num_nodes} nodes, width {self.width})"


# ====================================================================
# The uniformly tree-automatic class
# ====================================================================

class TreeWidthClass(SymbolicClassWrapper):
    """The uniformly tree-automatic class of graphs of tree-width <= w,
    presented over set-valued elements (MSO0 style)."""

    #: elements are vertex sets and E is the edge relation, not equality
    GRAPH = None
    #: extensional equality of sets
    EQUALITY = 'Subset(x,y) and Subset(y,x)'

    def __init__(self, w: int, max_states: Optional[int] = None):
        if w < 0:
            raise ValueError("width bound must be >= 0")
        self.w = w
        registers = range(w + 1)
        self.letters = [
            (r, profile)
            for r in registers
            for size in range(w + 1)
            for profile in it.combinations(
                [s for s in registers if s != r], size)
        ]
        self.symbol_of = {letter: _register_symbol(*letter)
                          for letter in self.letters}
        self.letter_of = {sym: letter for letter, sym in self.symbol_of.items()}
        self.sigma = {PAD, '0', '1', NOP} | set(self.symbol_of.values())
        self.advice_letters = {NOP} | set(self.symbol_of.values())
        self.marks = {PAD, '0', '1'}

        self.cls = UniformlyTreeAutomaticClass({
            'U': self._universe_automaton(),
            'Sing': self._sing_automaton(),
            'Subset': self._subset_automaton(),
            'E': self._edge_automaton(),
        }, padding_symbol=PAD, max_states=max_states)

        self._declare_equality(getattr(self, '_eager_equality', False))
    # ---------------- the automata ----------------

    def _advice_letter(self, a):
        """(register, profile) for vertex letters, NOP for 'n', None else."""
        if a == NOP:
            return NOP
        return self.letter_of.get(a)

    def _universe_automaton(self) -> SparseTreeAutomaton:
        """U(advice, x): the advice is valid (profiles only reference
        registers written above) and x is a set encoding over its vertices.
        State = (registers required from above, set-encoding phase) with
        phases A (no set domain below), S (valid marked domain), Z (an
        unmarked leaf: only the root may be one, encoding the empty set)."""
        registers = list(range(self.w + 1))
        reqs = [frozenset(c) for size in range(self.w + 2)
                for c in it.combinations(registers, size)]
        states = [(req, xs) for req in reqs for xs in 'ASZ'] + ['dead']

        def delta(lq, rq, sym):
            a, x = sym
            letter = self._advice_letter(a)
            if letter is None:
                return 'dead'
            req = frozenset()
            phases = []
            for q in (lq, rq):
                if q is not None:
                    req |= q[0]
                    phases.append(q[1])
            if 'Z' in phases:
                return 'dead'
            if letter != NOP:
                r, profile = letter
                req = (req - {r}) | set(profile)
            if x == PAD:
                xs = 'A' if 'S' not in phases else None
            elif x == '1':
                xs = 'S' if letter != NOP else None
            elif x == '0':
                xs = 'S' if 'S' in phases else 'Z'
            else:
                xs = None
            return 'dead' if xs is None else (req, xs)

        finals = {(frozenset(), 'S'), (frozenset(), 'Z')}
        return sta_from_delta(self.sigma, states, 2, delta, finals,
                              tapes=[self.advice_letters, self.marks])

    def _sing_automaton(self) -> SparseTreeAutomaton:
        """Sing(advice, X): exactly one mark, at the tip of the path."""
        states = ['A', 'M', 'dead']

        def delta(lq, rq, sym):
            a, x = sym
            if self._advice_letter(a) is None:
                return 'dead'
            marks = sum(1 for q in (lq, rq) if q == 'M')
            if x == PAD:
                return 'A' if marks == 0 else 'dead'
            if x == '1':
                return 'M' if marks == 0 and a != NOP else 'dead'
            if x == '0':
                return 'M' if marks == 1 else 'dead'
            return 'dead'

        return sta_from_delta(self.sigma, states, 2, delta, {'M'},
                              tapes=[self.advice_letters, self.marks])

    def _subset_automaton(self) -> SparseTreeAutomaton:
        """Subset(advice, X, Y): positionwise X <= Y."""
        allowed = {(PAD, PAD), (PAD, '0'), (PAD, '1'),
                   ('0', '0'), ('0', '1'), ('1', '1')}

        def delta(lq, rq, sym):
            a, x, y = sym
            if self._advice_letter(a) is None or (x, y) not in allowed:
                return 'dead'
            return 'ok'

        return sta_from_delta(self.sigma, ['ok', 'dead'], 3, delta, {'ok'},
                              tapes=[self.advice_letters, self.marks,
                                     self.marks])

    def _edge_automaton(self) -> SparseTreeAutomaton:
        """E(advice, X, Y): X = {u}, Y = {v}, u != v adjacent. Edges only
        join nested vertices: the deeper mark's profile becomes the pending
        register set, killed register by register as writers intervene on
        the way up; the higher mark is adjacent iff its own register is
        still pending when reached."""
        registers = list(range(self.w + 1))
        pendings = [frozenset(c) for size in range(self.w + 2)
                    for c in it.combinations(registers, size)]
        states = ['A', 'D'] + [('P', t, S) for t in 'xy' for S in pendings]
        states += ['dead']

        def delta(lq, rq, sym):
            a, x, y = sym
            letter = self._advice_letter(a)
            if letter is None:
                return 'dead'
            kids = [q for q in (lq, rq) if q is not None]
            carriers = [q for q in kids
                        if q == 'D' or (isinstance(q, tuple) and q[0] == 'P')]
            if len(carriers) > 1 or any(q not in ('A',) and q not in carriers
                                        for q in kids):
                return 'dead'
            carrier = carriers[0] if carriers else None
            vertex = letter if letter != NOP else None

            if (x, y) == (PAD, PAD):
                return 'A' if carrier is None else 'dead'
            if (x, y) in (('1', PAD), (PAD, '1')):
                if carrier is not None or vertex is None:
                    return 'dead'
                tape = 'x' if x == '1' else 'y'
                return ('P', tape, frozenset(vertex[1]))
            if (x, y) in (('0', PAD), (PAD, '0')):
                tape = 'x' if x == '0' else 'y'
                if not (isinstance(carrier, tuple) and carrier[1] == tape):
                    return 'dead'
                S = carrier[2]
                return ('P', tape, S - {vertex[0]} if vertex else S)
            if (x, y) in (('0', '1'), ('1', '0')):
                deeper = 'x' if y == '1' else 'y'
                if vertex is None or \
                        not (isinstance(carrier, tuple)
                             and carrier[1] == deeper):
                    return 'dead'
                return 'D' if vertex[0] in carrier[2] else 'dead'
            if (x, y) == ('0', '0'):
                return 'D' if carrier == 'D' else 'dead'
            return 'dead'

        return sta_from_delta(self.sigma, states, 3, delta, {'D'},
                              tapes=[self.advice_letters, self.marks,
                                     self.marks])

    # ---------------- class-level operations ----------------

    def advice(self, graph: Union[TreeWidthGraph, Tree]) -> Tree:
        """The advice tree of a graph (its letters as alphabet symbols)."""
        if not isinstance(graph, TreeWidthGraph):
            return graph
        if graph.width > self.w:
            raise ValueError(f"graph has layout width {graph.width} "
                             f"> w = {self.w}")

        def convert(node):
            if node is None:
                return None
            label = NOP if node.label == NOP else self.symbol_of[node.label]
            return Tree(label, convert(node.left), convert(node.right))
        return convert(graph.tree)

    def evaluate(self, phi):
        """Evaluate an MSO query over the class; see
        UniformlyTreeAutomaticClass.evaluate. Variables range over vertex
        sets."""
        return self.cls.evaluate(phi)

    def check(self, phi, graph: Union[TreeWidthGraph, Tree], **sets) -> bool:
        """Model check an MSO query against a single graph.

        :param phi: formula over Sing/Subset/E; free variables can be
            assigned via `sets` (name = set of nodes), unassigned ones are
            quantified existentially.
        :param graph: a TreeWidthGraph (or a raw advice tree).
        :param sets: assignments for free variables, as sets of nodes.
        """
        advice = self.advice(graph)
        if not sets:
            return self.cls.check(phi, advice)
        if not isinstance(graph, TreeWidthGraph):
            raise ValueError("set assignments require a graph object")
        trees = {name: graph.encode_set(subset)
                 for name, subset in sets.items()}
        return self.cls.check(phi, advice, **trees)

    def get_structure(self, graph) -> TreeAutomaticPresentation:
        """The MSO0-style tree-automatic presentation of a single graph."""
        return self.cls.get_structure(self.advice(graph))


# ====================================================================
# Bounded clique-width
# ====================================================================

def _union_symbol() -> str:
    return 'u'


def _vertex_symbol(label: int) -> str:
    return f"v{label}"


def _relabel_symbol(source: int, target: int) -> str:
    return f"r{source}{target}"


def _join_symbol(first: int, second: int) -> str:
    return f"e{first}{second}"


class CliqueWidthGraph:
    """A graph given by a k-expression, held as a binary tree.

    The expression's leaves create vertices, so the vertices *are* the leaves,
    numbered left to right. Inner nodes are the three clique-width operations:

        u        disjoint union of the two children
        r{i}{j}  relabel every label-i vertex to label j
        e{i}{j}  join: add every edge between label i and label j

    `u` is binary; `r` and `e` are unary (left child only).
    """

    def __init__(self, expression: Tree, k: int):
        if not 2 <= k <= 9:
            raise ValueError("clique-width bound must be in [2, 9]")
        self.k = k
        self.tree = expression
        self.vertices: List[int] = []
        self.edges: Set[frozenset] = set()
        self._labels_of: Dict[int, int] = {}
        self._decode()

    # ---------------- decoding ----------------

    def _leaves(self, node: Tree) -> List[Tree]:
        found, stack = [], [node]
        while stack:
            current = stack.pop()
            if current.left is None and current.right is None:
                found.append(current)
                continue
            if current.right is not None:
                stack.append(current.right)
            if current.left is not None:
                stack.append(current.left)
        return found

    def _decode(self) -> None:
        """Number the leaves left to right, then evaluate the expression
        bottom-up carrying, per node, the vertex set of each label."""
        index = {id(leaf): i for i, leaf in enumerate(self._leaves(self.tree))}
        self.vertices = list(range(len(index)))

        def evaluate(node: Tree) -> Dict[int, Set[int]]:
            label = node.label
            left = node.left
            right = node.right
            if left is None and right is None:
                if not (label.startswith('v') and len(label) == 2):
                    raise ValueError(f"leaf must create a vertex, got {label!r}")
                colour = int(label[1])
                if colour >= self.k:
                    raise ValueError(f"label {colour} exceeds k = {self.k}")
                vertex = index[id(node)]
                self._labels_of[vertex] = colour
                return {colour: {vertex}}

            if label == 'u':
                if left is None or right is None:
                    raise ValueError("union needs two children")
                classes = evaluate(left)
                for colour, group in evaluate(right).items():
                    classes.setdefault(colour, set()).update(group)
                return classes

            if right is not None or left is None:
                raise ValueError(f"{label!r} is unary: left child only")
            classes = evaluate(left)

            if label.startswith('r') and len(label) == 3:
                source, target = int(label[1]), int(label[2])
                if source == target or max(source, target) >= self.k:
                    raise ValueError(f"bad relabel {label!r}")
                moved = classes.pop(source, set())
                classes.setdefault(target, set()).update(moved)
                return classes

            if label.startswith('e') and len(label) == 3:
                first, second = int(label[1]), int(label[2])
                if first >= second or second >= self.k:
                    raise ValueError(f"bad join {label!r}")
                for u in classes.get(first, ()):
                    for v in classes.get(second, ()):
                        self.edges.add(frozenset((u, v)))
                return classes

            raise ValueError(f"unknown operation {label!r}")

        # recursion depth is the expression depth; the builders below are
        # left-deep, so iterate rather than recurse on the spine
        import sys
        limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(limit, 10 * (len(self.vertices) + 10)))
        try:
            evaluate(self.tree)
        finally:
            sys.setrecursionlimit(limit)

    # ---------------- sets ----------------

    def encode_set(self, subset) -> Tree:
        """The set as a tree of marks: '1' at the chosen leaves, '0' on the
        paths above them, absent elsewhere. The empty set is a single '0'."""
        wanted = set(subset)
        if not wanted:
            return Tree('0')
        index = {id(leaf): i for i, leaf in enumerate(self._leaves(self.tree))}

        def build(node):
            if node is None:
                return None, False
            if node.left is None and node.right is None:
                marked = index[id(node)] in wanted
                return (Tree('1') if marked else None), marked
            left, has_left = build(node.left)
            right, has_right = build(node.right)
            if not (has_left or has_right):
                return None, False
            return Tree('0', left, right), True

        tree, _ = build(self.tree)
        return tree

    def to_networkx(self):
        import networkx as nx
        graph = nx.Graph()
        graph.add_nodes_from(self.vertices)
        graph.add_edges_from(tuple(edge) for edge in self.edges)
        return graph

    # ---------------- standard families ----------------

    @classmethod
    def clique(cls, n: int) -> "CliqueWidthGraph":
        """K_n, clique-width 2: absorb each new vertex into label 0."""
        tree = Tree(_vertex_symbol(0))
        for _ in range(n - 1):
            tree = Tree(_union_symbol(), tree, Tree(_vertex_symbol(1)))
            tree = Tree(_join_symbol(0, 1), tree)
            tree = Tree(_relabel_symbol(1, 0), tree)
        return cls(tree, 2)

    @classmethod
    def complete_bipartite(cls, left: int, right: int) -> "CliqueWidthGraph":
        """K_{left,right}, clique-width 2: join the two colour classes once."""
        tree = Tree(_vertex_symbol(0))
        for _ in range(left - 1):
            tree = Tree(_union_symbol(), tree, Tree(_vertex_symbol(0)))
        for _ in range(right):
            tree = Tree(_union_symbol(), tree, Tree(_vertex_symbol(1)))
        return cls(Tree(_join_symbol(0, 1), tree), 2)

    @classmethod
    def path(cls, n: int) -> "CliqueWidthGraph":
        """P_n, clique-width 3: label 0 is the growing end, 1 the new vertex,
        2 the settled interior."""
        tree = Tree(_vertex_symbol(0))
        for _ in range(n - 1):
            tree = Tree(_union_symbol(), tree, Tree(_vertex_symbol(1)))
            tree = Tree(_join_symbol(0, 1), tree)
            tree = Tree(_relabel_symbol(0, 2), tree)
            tree = Tree(_relabel_symbol(1, 0), tree)
        return cls(tree, 3)

    @classmethod
    def cycle(cls, n: int) -> "CliqueWidthGraph":
        """C_n, clique-width 4: as the path, but the first vertex keeps label
        3 so the closing edge can be added at the root."""
        if n < 3:
            raise ValueError("a cycle needs at least three vertices")
        tree = Tree(_vertex_symbol(3))                 # the first vertex
        tree = Tree(_union_symbol(), tree, Tree(_vertex_symbol(0)))
        tree = Tree(_join_symbol(0, 3), tree)          # first -- second
        for _ in range(n - 2):
            tree = Tree(_union_symbol(), tree, Tree(_vertex_symbol(1)))
            tree = Tree(_join_symbol(0, 1), tree)      # end -- new
            tree = Tree(_relabel_symbol(0, 2), tree)   # end settles
            tree = Tree(_relabel_symbol(1, 0), tree)   # new becomes the end
        return cls(Tree(_join_symbol(0, 3), tree), 4)  # close the cycle


class CliqueWidthClass(SymbolicClassWrapper):
    """The uniformly tree-automatic class of graphs of clique-width <= k,
    presented over set-valued elements (MSO0 style).

    The advice is a k-expression (see `CliqueWidthGraph`); the vertices are its
    leaves, and a vertex set is encoded synchronously as the union of the root
    paths to its members, '1' on members and '0' on the way.

    Adjacency is far simpler here than for tree-width: two vertices are joined
    exactly when some `e{i}{j}` node sees one of them holding label i and the
    other label j. A bottom-up automaton therefore only has to remember the
    current label of each marked vertex, and whether the join has happened.
    """

    #: elements are vertex sets and E is the edge relation, not equality
    GRAPH = None
    #: extensional equality of sets
    EQUALITY = 'Subset(x,y) and Subset(y,x)'

    def __init__(self, k: int, max_states: Optional[int] = None):
        if not 2 <= k <= 9:
            raise ValueError("clique-width bound must be in [2, 9]")
        self.k = k
        self.labels = range(k)
        self.sigma = ({PAD, '0', '1', _union_symbol()} |
                      {_vertex_symbol(i) for i in self.labels} |
                      {_relabel_symbol(i, j) for i in self.labels
                       for j in self.labels if i != j} |
                      {_join_symbol(i, j) for i in self.labels
                       for j in self.labels if i < j})
        self.advice_letters = self.sigma - {PAD, '0', '1'}
        self.marks = {PAD, '0', '1'}

        self.cls = UniformlyTreeAutomaticClass({
            'U': self._universe_automaton(),
            'Sing': self._sing_automaton(),
            'Subset': self._subset_automaton(),
            'E': self._edge_automaton(),
        }, padding_symbol=PAD, max_states=max_states)

        self._declare_equality(getattr(self, '_eager_equality', False))
    # ---------------- letters ----------------

    def _kind(self, a):
        """('v', i) | ('u',) | ('r', i, j) | ('e', i, j), or None."""
        if a == _union_symbol():
            return ('u',)
        if len(a) == 2 and a[0] == 'v' and a[1].isdigit():
            i = int(a[1])
            return ('v', i) if i < self.k else None
        if len(a) == 3 and a[0] in 're' and a[1:].isdigit():
            i, j = int(a[1]), int(a[2])
            if max(i, j) >= self.k:
                return None
            if a[0] == 'r':
                return ('r', i, j) if i != j else None
            return ('e', i, j) if i < j else None
        return None

    def _shape_ok(self, kind, left, right) -> bool:
        """`u` is binary, `v` a leaf, `r` and `e` unary (left only)."""
        if kind[0] == 'v':
            return left is None and right is None
        if kind[0] == 'u':
            return left is not None and right is not None
        return left is not None and right is None

    # ---------------- the automata ----------------

    def _universe_automaton(self) -> SparseTreeAutomaton:
        """U(advice, x): the advice is a well-formed k-expression and x is a
        set encoding over its leaves. Phases A (no set domain below), S (a
        marked domain), Z (an unmarked node with nothing below: the empty
        set, and only at the root)."""
        def delta(lq, rq, sym):
            a, x = sym
            kind = self._kind(a)
            if kind is None or not self._shape_ok(kind, lq, rq):
                return 'dead'
            phases = [q for q in (lq, rq) if q is not None]
            if 'Z' in phases:
                return 'dead'
            if x == PAD:
                xs = 'A' if 'S' not in phases else None
            elif x == '1':
                xs = 'S' if kind[0] == 'v' else None    # only leaves are vertices
            elif x == '0':
                xs = 'S' if 'S' in phases else 'Z'
            else:
                xs = None
            return 'dead' if xs is None else xs

        return sta_from_delta(self.sigma, ['A', 'S', 'Z', 'dead'], 2, delta,
                              {'S', 'Z'},
                              tapes=[self.advice_letters, self.marks])

    def _sing_automaton(self) -> SparseTreeAutomaton:
        """Sing(advice, X): exactly one mark, on a leaf."""
        def delta(lq, rq, sym):
            a, x = sym
            kind = self._kind(a)
            if kind is None:
                return 'dead'
            marks = sum(1 for q in (lq, rq) if q == 'M')
            if x == PAD:
                return 'A' if marks == 0 else 'dead'
            if x == '1':
                return 'M' if marks == 0 and kind[0] == 'v' else 'dead'
            if x == '0':
                return 'M' if marks == 1 else 'dead'
            return 'dead'

        return sta_from_delta(self.sigma, ['A', 'M', 'dead'], 2, delta, {'M'},
                              tapes=[self.advice_letters, self.marks])

    def _subset_automaton(self) -> SparseTreeAutomaton:
        """Subset(advice, X, Y): positionwise X <= Y."""
        allowed = {(PAD, PAD), (PAD, '0'), (PAD, '1'),
                   ('0', '0'), ('0', '1'), ('1', '1')}

        def delta(lq, rq, sym):
            a, x, y = sym
            if self._kind(a) is None or (x, y) not in allowed:
                return 'dead'
            return 'ok'

        return sta_from_delta(self.sigma, ['ok', 'dead'], 3, delta, {'ok'},
                              tapes=[self.advice_letters, self.marks,
                                     self.marks])

    def _edge_automaton(self) -> SparseTreeAutomaton:
        """E(advice, X, Y): X = {u}, Y = {v}, u != v adjacent.

        State: which of the two marked vertices lies below, the label each one
        currently carries, and whether a join has already connected them.
        'D' has seen the join, so the labels no longer matter."""
        states = (['A', 'D', 'dead'] +
                  [('X', i) for i in self.labels] +
                  [('Y', i) for i in self.labels] +
                  [('B', i, j) for i in self.labels for j in self.labels])

        def parts(q):
            """(x below, x's label, y below, y's label, joined)"""
            if q == 'A':
                return False, None, False, None, False
            if q == 'D':
                return True, None, True, None, True
            if q[0] == 'X':
                return True, q[1], False, None, False
            if q[0] == 'Y':
                return False, None, True, q[1], False
            return True, q[1], True, q[2], False

        def state(hx, lx, hy, ly, joined):
            if joined:
                return 'D'
            if hx and hy:
                return ('B', lx, ly)
            if hx:
                return ('X', lx)
            if hy:
                return ('Y', ly)
            return 'A'

        def delta(lq, rq, sym):
            a, x, y = sym
            kind = self._kind(a)
            if kind is None or not self._shape_ok(kind, lq, rq):
                return 'dead'

            if kind[0] == 'v':                      # a vertex is created here
                if (x, y) == (PAD, PAD):
                    return 'A'
                if (x, y) == ('1', PAD):
                    return ('X', kind[1])
                if (x, y) == (PAD, '1'):
                    return ('Y', kind[1])
                return 'dead'                       # '0' on a leaf, or u = v

            hx = hy = joined = False
            lx = ly = None
            for q in (lq, rq):
                if q is None:
                    continue
                qhx, qlx, qhy, qly, qjoined = parts(q)
                if (qhx and hx) or (qhy and hy):
                    return 'dead'                   # X or Y marked twice
                if qhx:
                    hx, lx = True, qlx
                if qhy:
                    hy, ly = True, qly
                joined = joined or qjoined

            # an inner node lies in a set's domain exactly when a member is
            # below it
            if x != ('0' if hx else PAD) or y != ('0' if hy else PAD):
                return 'dead'

            if kind[0] == 'r':
                source, target = kind[1], kind[2]
                lx = target if lx == source else lx
                ly = target if ly == source else ly
            elif kind[0] == 'e' and hx and hy and not joined:
                if {lx, ly} == {kind[1], kind[2]}:
                    joined = True
            return state(hx, lx, hy, ly, joined)

        return sta_from_delta(self.sigma, states, 3, delta, {'D'},
                              tapes=[self.advice_letters, self.marks,
                                     self.marks])

    # ---------------- class-level operations ----------------

    def advice(self, graph: Union[CliqueWidthGraph, Tree]) -> Tree:
        """The advice tree of a graph (its k-expression)."""
        if not isinstance(graph, CliqueWidthGraph):
            return graph
        if graph.k > self.k:
            raise ValueError(f"graph needs {graph.k} labels > k = {self.k}")
        return graph.tree

    def evaluate(self, phi):
        """Evaluate an MSO query over the class; variables range over vertex
        sets. See UniformlyTreeAutomaticClass.evaluate."""
        return self.cls.evaluate(phi)

    def check(self, phi, graph: Union[CliqueWidthGraph, Tree], **sets) -> bool:
        """Model check an MSO query against a single graph."""
        advice = self.advice(graph)
        if not sets:
            return self.cls.check(phi, advice)
        if not isinstance(graph, CliqueWidthGraph):
            raise ValueError("set assignments require a graph object")
        trees = {name: graph.encode_set(subset)
                 for name, subset in sets.items()}
        return self.cls.check(phi, advice, **trees)

    def get_structure(self, graph) -> TreeAutomaticPresentation:
        """The MSO0-style tree-automatic presentation of a single graph."""
        return self.cls.get_structure(self.advice(graph))


# ====================================================================
# Graphs of bounded rank-width
# ====================================================================

class RankWidthGraph:
    """A graph with a *rank decomposition*: a binary layout tree whose
    leaves are the vertices (numbered left to right), plus an edge set.

    The width of the decomposition is the maximum, over all subtrees S, of
    the GF(2) rank of the bipartite adjacency matrix between the leaves
    inside S and the leaves outside (the *cut-rank*); the rank-width of the
    graph is the minimum over decompositions. This is the graph analog of
    the module cut-rank of the bounded-rank-width group classes, and the
    class compiler below reuses the same linear algebra
    (`autstr.chain_ring` at p = 2, d = 1). Unary layout nodes are allowed
    (they do not change the cuts).
    """

    def __init__(self, shape: Tree, edges):
        self.shape = shape
        self._leaf_nodes = self._leaves(shape)
        self.n = len(self._leaf_nodes)
        self.vertices = list(range(self.n))
        self.edges: Set[frozenset] = set()
        for e in edges:
            u, v = tuple(e)
            if u == v or not (0 <= u < self.n and 0 <= v < self.n):
                raise ValueError(f"bad edge {(u, v)} over {self.n} vertices")
            self.edges.add(frozenset((u, v)))
        A = np.zeros((self.n, self.n), dtype=np.int64)
        for e in self.edges:
            u, v = tuple(e)
            A[u, v] = A[v, u] = 1
        self.adjacency = A
        # contiguous leaf span [lo, hi) per node, leaves left to right
        self.span: Dict[int, Tuple[int, int]] = {}
        counter = [0]

        def mark(node):
            lo = counter[0]
            if node.left is None and node.right is None:
                counter[0] += 1
            else:
                if node.left is not None:
                    mark(node.left)
                if node.right is not None:
                    mark(node.right)
            self.span[id(node)] = (lo, counter[0])

        mark(shape)

    @staticmethod
    def _leaves(node: Tree) -> List[Tree]:
        found, stack = [], [node]
        while stack:
            current = stack.pop()
            if current.left is None and current.right is None:
                found.append(current)
                continue
            if current.right is not None:
                stack.append(current.right)
            if current.left is not None:
                stack.append(current.left)
        return found

    def cut_matrix(self, node: Tree) -> np.ndarray:
        """The bipartite adjacency block of the node's cut: rows the outside
        vertices, columns the inside leaves (left-to-right)."""
        lo, hi = self.span[id(node)]
        outside = [u for u in self.vertices if not lo <= u < hi]
        return self.adjacency[np.ix_(outside, list(range(lo, hi)))]

    @property
    def width(self) -> int:
        """The rank-width of this decomposition: the maximum GF(2) cut-rank
        over all proper subtrees."""
        best = 0
        stack = [self.shape]
        while stack:
            node = stack.pop()
            lo, hi = self.span[id(node)]
            if hi - lo < self.n:
                best = max(best, cr.module_cut_rank(self.cut_matrix(node),
                                                    2, 1))
            for child in (node.left, node.right):
                if child is not None:
                    stack.append(child)
        return best

    # ---------------- sets ----------------

    def encode_set(self, subset) -> Tree:
        """The set as a tree of marks: '1' at the chosen leaves, '0' on the
        paths above them, absent elsewhere. The empty set is a single '0'."""
        wanted = set(subset)
        if not wanted:
            return Tree('0')
        index = {id(leaf): i for i, leaf in enumerate(self._leaf_nodes)}

        def build(node):
            if node is None:
                return None, False
            if node.left is None and node.right is None:
                marked = index[id(node)] in wanted
                return (Tree('1') if marked else None), marked
            left, has_left = build(node.left)
            right, has_right = build(node.right)
            if not (has_left or has_right):
                return None, False
            return Tree('0', left, right), True

        tree, _ = build(self.shape)
        return tree

    def encode_set_padded(self, subset, pad: str = PAD) -> Tree:
        """The set as a mark tree of the decomposition's *exact* shape,
        `pad` outside the trimmed domain -- what the implicit evaluator
        needs (it runs all tapes synchronously over the advice shape)."""
        trimmed = self.encode_set(subset)

        def build(node, m):
            if node is None:
                return None
            return Tree(m.label if m is not None else pad,
                        build(node.left, m.left if m is not None else None),
                        build(node.right, m.right if m is not None else None))

        return build(self.shape, trimmed)

    def decode_set(self, tree: Tree) -> Set[int]:
        """The vertex set of a mark tree (trimmed or full-shape/padded)."""
        out = set()

        def rec(node, mark, idx):
            if node.left is None and node.right is None:
                if mark is not None and mark.label == '1':
                    out.add(idx)
                return idx + 1
            for child, mchild in ((node.left, mark.left if mark else None),
                                  (node.right, mark.right if mark else None)):
                if child is not None:
                    idx = rec(child, mchild, idx)
            return idx

        rec(self.shape, tree, 0)
        return out

    def to_networkx(self):
        import networkx as nx
        graph = nx.Graph()
        graph.add_nodes_from(self.vertices)
        graph.add_edges_from(tuple(edge) for edge in self.edges)
        return graph

    # ---------------- standard families ----------------

    @staticmethod
    def caterpillar(n: int) -> Tree:
        """The linear (caterpillar) decomposition: leaves 0..n-1 hang left
        to right off a left-deep spine."""
        if n < 1:
            raise ValueError("need at least one vertex")
        node = Tree('s')
        for _ in range(n - 1):
            node = Tree('s', node, Tree('s'))
        return node

    @classmethod
    def clique(cls, n: int) -> "RankWidthGraph":
        """K_n: every crossing block is all ones -- rank-width 1 on any
        decomposition."""
        return cls(cls.caterpillar(n),
                   [(i, j) for i in range(n) for j in range(i + 1, n)])

    @classmethod
    def path(cls, n: int) -> "RankWidthGraph":
        """P_n in path order: one edge crosses each caterpillar cut --
        rank-width 1."""
        return cls(cls.caterpillar(n), [(i, i + 1) for i in range(n - 1)])

    @classmethod
    def cycle(cls, n: int) -> "RankWidthGraph":
        """C_n: two edges cross the middle caterpillar cuts -- width 2 on
        this decomposition (and rank-width 2 for n >= 5)."""
        if n < 3:
            raise ValueError("a cycle needs at least three vertices")
        return cls(cls.caterpillar(n),
                   [(i, i + 1) for i in range(n - 1)] + [(n - 1, 0)])

    @classmethod
    def complete_bipartite(cls, left: int, right: int) -> "RankWidthGraph":
        """K_{left,right}, one part then the other: identical rows on every
        cut -- rank-width 1."""
        return cls(cls.caterpillar(left + right),
                   [(i, left + j) for i in range(left) for j in range(right)])


class RankWidthClass(SymbolicClassWrapper):
    """The uniformly tree-automatic class of graphs of rank-width <= r,
    presented over set-valued elements (MSO0 style: Sing, Subset, E).

    The advice is a rank decomposition annotated with the GF(2)
    factorization data of its cuts, exactly as the bounded-rank-width group
    classes annotate theirs: each node carries a basis-change matrix per
    child (w <- T w) and each binary node the bilinear form Q of its sibling
    block, so that two vertices x in the left and y in the right subtree are
    adjacent iff w_y^T Q w_x over F_2, where w is the vertex's interface
    vector (its column in the saturated basis of the cut, composed through
    the T maps). The letters are

        leaf:   'a' + w (r bits)          -- the vertex's interface vector
        unary:  'b' + T (r*r bits)
        binary: 'd' + TL + TR + Q (3 r*r bits)

    and every well-shaped advice presents some graph of rank-width <= r.
    Vertex sets are encoded as union-of-root-path marks (see
    `RankWidthGraph.encode_set`); the E automaton carries each marked
    vertex's interface vector -- O(2^{2r}) states however large the graph.

    `advice(graph)` compiles a `RankWidthGraph` whose decomposition has
    width <= r into the annotated advice (`chain_ring.saturate` /
    `solve_left` / `factor_two_sided` at p = 2, d = 1); `check_implicit` and
    `evaluate_implicit` run over the functional atoms without building any
    automaton. The flat letter alphabet caps r at 2: it grows as 2^{3r^2}
    binary letters, unlike the factored letters used by the group classes.
    """

    #: elements are vertex sets and E is the edge relation, not equality
    GRAPH = None
    #: extensional equality of sets
    EQUALITY = 'Subset(x,y) and Subset(y,x)'

    def __init__(self, r: int, max_states: Optional[int] = None):
        if r < 1:
            raise ValueError("rank-width bound must be >= 1")
        n_letters = 2 ** r + 2 ** (r * r) + 2 ** (3 * r * r)
        if n_letters > 20000:
            raise ValueError(
                f"the advice alphabet would have {n_letters} letters; "
                f"only r <= 2 is supported")
        self.r = r
        self.max_states = max_states
        vecs = list(it.product((0, 1), repeat=r))
        mats = list(it.product((0, 1), repeat=r * r))

        def mat(bits):
            return tuple(tuple(bits[i * r + j] for j in range(r))
                         for i in range(r))

        bstr = lambda bits: ''.join(str(b) for b in bits)
        self.leaf_letters = {'a' + bstr(w): w for w in vecs}
        self.unary_letters = {'b' + bstr(m): mat(m) for m in mats}
        self.binary_letters = {
            'd' + bstr(tl) + bstr(tr) + bstr(q): (mat(tl), mat(tr), mat(q))
            for tl in mats for tr in mats for q in mats}
        self.advice_letters = (set(self.leaf_letters)
                               | set(self.unary_letters)
                               | set(self.binary_letters))
        self.sigma = {PAD, '0', '1'} | self.advice_letters
        self.marks = {PAD, '0', '1'}
        self._cls = None

    @property
    def cls(self) -> UniformlyTreeAutomaticClass:
        """The presentation, built lazily (the r = 2 E automaton enumerates
        a few million transitions)."""
        if self._cls is None:
            self._cls = UniformlyTreeAutomaticClass({
                'U': self._universe_automaton(),
                'Sing': self._sing_automaton(),
                'Subset': self._subset_automaton(),
                'E': self._edge_automaton(),
            }, padding_symbol=PAD, max_states=self.max_states)
            self._cls.element_alphabet = [PAD, '0', '1']
        return self._cls

    # ---------------- letters ----------------

    def _kind(self, a):
        """('a', w) | ('b', T) | ('d', TL, TR, Q), or None."""
        if a in self.leaf_letters:
            return ('a', self.leaf_letters[a])
        if a in self.unary_letters:
            return ('b', self.unary_letters[a])
        if a in self.binary_letters:
            return ('d',) + self.binary_letters[a]
        return None

    @staticmethod
    def _shape_ok(kind, left, right) -> bool:
        if kind[0] == 'a':
            return left is None and right is None
        if kind[0] == 'd':
            return left is not None and right is not None
        return (left is None) != (right is None)

    def _dot(self, T, w):
        return tuple(sum(T[i][j] * w[j] for j in range(self.r)) % 2
                     for i in range(self.r))

    def _form(self, Q, wr, wl):
        return sum(wr[i] * Q[i][j] * wl[j]
                   for i in range(self.r) for j in range(self.r)) % 2

    # ---------------- the transition functions (shared with the atoms) ----

    def _u_delta(self, lq, rq, a, x):
        """U(advice, x): well-shaped advice, x a set encoding over the
        leaves. Phases A (no set domain below), S (a marked domain), Z (the
        empty set, only at the root)."""
        kind = self._kind(a)
        if kind is None or not self._shape_ok(kind, lq, rq):
            return 'dead'
        phases = [q for q in (lq, rq) if q is not None]
        if 'Z' in phases:
            return 'dead'
        if x == PAD:
            return 'A' if 'S' not in phases else 'dead'
        if x == '1':
            return 'S' if kind[0] == 'a' else 'dead'
        if x == '0':
            return 'S' if 'S' in phases else 'Z'
        return 'dead'

    def _sing_delta(self, lq, rq, a, x):
        if self._kind(a) is None:
            return 'dead'
        found = sum(1 for q in (lq, rq) if q == 'M')
        if x == PAD:
            return 'A' if found == 0 else 'dead'
        if x == '1':
            return 'M' if found == 0 and a in self.leaf_letters else 'dead'
        if x == '0':
            return 'M' if found == 1 else 'dead'
        return 'dead'

    _SUBSET_OK = {(PAD, PAD), (PAD, '0'), (PAD, '1'),
                  ('0', '0'), ('0', '1'), ('1', '1')}

    def _subset_delta(self, lq, rq, a, x, y):
        if self._kind(a) is None or (x, y) not in self._SUBSET_OK:
            return 'dead'
        return 'ok'

    def _e_delta(self, lq, rq, a, x, y):
        """E(advice, X, Y): X = {u}, Y = {v}, u != v adjacent. The state
        carries the interface vector of each marked vertex below; the
        binary node where they meet evaluates the bilinear form."""
        kind = self._kind(a)
        if kind is None or not self._shape_ok(kind, lq, rq):
            return 'dead'

        if kind[0] == 'a':
            if (x, y) == (PAD, PAD):
                return 'A'
            if (x, y) == ('1', PAD):
                return ('X', kind[1])
            if (x, y) == (PAD, '1'):
                return ('Y', kind[1])
            return 'dead'                    # '0' on a leaf, or u = v

        def parts(q):
            """(x below, w_x, y below, w_y, decided bit or None)"""
            if q in (None, 'A'):
                return False, None, False, None, None
            if q in ('D', 'N'):
                return True, None, True, None, 1 if q == 'D' else 0
            if q[0] == 'X':
                return True, q[1], False, None, None
            return False, None, True, q[1], None

        if kind[0] == 'b':
            q = lq if rq is None else rq
            hx, wx, hy, wy, bit = parts(q)
            T = kind[1]
            wx = self._dot(T, wx) if wx is not None else None
            wy = self._dot(T, wy) if wy is not None else None
        else:
            TL, TR, Q = kind[1], kind[2], kind[3]
            lhx, lwx, lhy, lwy, lbit = parts(lq)
            rhx, rwx, rhy, rwy, rbit = parts(rq)
            if (lhx and rhx) or (lhy and rhy):
                return 'dead'                # a set marked twice
            hx, hy = lhx or rhx, lhy or rhy
            bit = lbit if lbit is not None else rbit
            wx = (self._dot(TL, lwx) if lwx is not None else
                  self._dot(TR, rwx) if rwx is not None else None)
            wy = (self._dot(TL, lwy) if lwy is not None else
                  self._dot(TR, rwy) if rwy is not None else None)
            if bit is None and hx and hy:
                # the marks meet here: right-side vector against Q against
                # the left-side vector, in the children's own bases
                wr = rwx if rwx is not None else rwy
                wl = lwx if lwx is not None else lwy
                first = self._form(Q, wr, wl)
                bit = first
                wx = wy = None

        if x != ('0' if hx else PAD) or y != ('0' if hy else PAD):
            return 'dead'
        if bit is not None:
            return 'D' if bit else 'N'
        if hx and hy:
            return 'dead'                    # unreachable guard
        if hx:
            return ('X', wx)
        if hy:
            return ('Y', wy)
        return 'A'

    # ---------------- the automata ----------------

    def _universe_automaton(self) -> SparseTreeAutomaton:
        def delta(lq, rq, sym):
            return self._u_delta(lq, rq, sym[0], sym[1])
        return sta_from_delta(self.sigma, ['A', 'S', 'Z', 'dead'], 2, delta,
                              {'S', 'Z'},
                              tapes=[self.advice_letters, self.marks])

    def _sing_automaton(self) -> SparseTreeAutomaton:
        def delta(lq, rq, sym):
            return self._sing_delta(lq, rq, sym[0], sym[1])
        return sta_from_delta(self.sigma, ['A', 'M', 'dead'], 2, delta, {'M'},
                              tapes=[self.advice_letters, self.marks])

    def _subset_automaton(self) -> SparseTreeAutomaton:
        def delta(lq, rq, sym):
            return self._subset_delta(lq, rq, sym[0], sym[1], sym[2])
        return sta_from_delta(self.sigma, ['ok', 'dead'], 3, delta, {'ok'},
                              tapes=[self.advice_letters, self.marks,
                                     self.marks])

    def _edge_automaton(self) -> SparseTreeAutomaton:
        vecs = list(it.product((0, 1), repeat=self.r))
        states = (['A', 'D', 'N', 'dead'] +
                  [('X', w) for w in vecs] + [('Y', w) for w in vecs])

        def delta(lq, rq, sym):
            return self._e_delta(lq, rq, sym[0], sym[1], sym[2])
        return sta_from_delta(self.sigma, states, 3, delta, {'D'},
                              tapes=[self.advice_letters, self.marks,
                                     self.marks])

    # ---------------- the advice compiler ----------------

    def advice(self, graph: Union[RankWidthGraph, Tree]) -> Tree:
        """Compile a rank decomposition into the annotated advice; raises if
        some cut exceeds rank r. A pre-compiled advice tree passes through."""
        if not isinstance(graph, RankWidthGraph):
            return graph
        r = self.r
        bstr = lambda M: ''.join(str(int(v) % 2)
                                 for v in np.asarray(M).flatten())
        V: Dict[int, np.ndarray] = {}
        built: Dict[int, Tree] = {}

        def rec(node):
            lo, hi = graph.span[id(node)]
            Vt = np.zeros((r, hi - lo), dtype=np.int64)
            if hi - lo < graph.n:
                basis, _ = cr.saturate(graph.cut_matrix(node), 2, 1)
                if basis.shape[0] > r:
                    raise ValueError(
                        f"a cut has rank {basis.shape[0]} > r = {r}; this "
                        f"decomposition has width {graph.width}")
                Vt[:basis.shape[0]] = basis
            L, R = node.left, node.right
            if L is None and R is None:
                letter = 'a' + bstr(Vt[:, 0])
            elif L is None or R is None:
                child = L if L is not None else R
                rec(child)
                clo, chi = graph.span[id(child)]
                T = cr.solve_left(V[id(child)],
                                  Vt[:, clo - lo:chi - lo], 2, 1)
                letter = 'b' + bstr(T)
            else:
                rec(L)
                rec(R)
                llo, lhi = graph.span[id(L)]
                rlo, rhi = graph.span[id(R)]
                VL, VR = V[id(L)], V[id(R)]
                TL = cr.solve_left(VL, Vt[:, llo - lo:lhi - lo], 2, 1)
                TR = cr.solve_left(VR, Vt[:, rlo - lo:rhi - lo], 2, 1)
                # sibling block, rows the right leaves, columns the left --
                # the two-sided factorisation X = VR^T Q VL over F_2
                X = graph.adjacency[np.ix_(list(range(rlo, rhi)),
                                           list(range(llo, lhi)))]
                Q = cr.factor_two_sided(X, VL, VR, 2, 1)
                letter = 'd' + bstr(TL) + bstr(TR) + bstr(Q)
            V[id(node)] = Vt
            built[id(node)] = Tree(
                letter,
                built.get(id(L)) if L is not None else None,
                built.get(id(R)) if R is not None else None)

        rec(graph.shape)
        return built[id(graph.shape)]

    # ---------------- class-level operations ----------------

    def evaluate(self, phi):
        """Evaluate an MSO query over the class; variables range over vertex
        sets. See UniformlyTreeAutomaticClass.evaluate."""
        return self.cls.evaluate(phi)

    def check(self, phi, graph: Union[RankWidthGraph, Tree], **sets) -> bool:
        """Model check an MSO query against a single graph."""
        advice = self.advice(graph)
        if not sets:
            return self.cls.check(phi, advice)
        if not isinstance(graph, RankWidthGraph):
            raise ValueError("set assignments require a graph object")
        trees = {name: graph.encode_set(subset)
                 for name, subset in sets.items()}
        return self.cls.check(phi, advice, **trees)

    def _implicit_atoms(self) -> Dict:
        """Functional bottom-up atoms (Dom, Adv, Sing, Subset, E) from the
        shared transition functions -- nothing is built."""
        from autstr.implicit import ImplicitTA
        marks = self.marks

        def wrap2(delta, accepting):
            def build(args):
                adv, xv = args[0], args[1]
                return ImplicitTA(
                    args,
                    lambda sym, left, right: 'dead'
                    if 'dead' in (left, right) or sym[xv] not in marks
                    else delta(left, right, sym[adv], sym[xv]),
                    accepting)
            return build

        def wrap3(delta, accepting):
            def build(args):
                adv, xv, yv = args
                return ImplicitTA(
                    args,
                    lambda sym, left, right: 'dead'
                    if 'dead' in (left, right) or sym[xv] not in marks
                    or sym[yv] not in marks
                    else delta(left, right, sym[adv], sym[xv], sym[yv]),
                    accepting)
            return build

        def Adv(args):
            adv = args[0]
            return ImplicitTA(
                args,
                lambda sym, left, right: 'dead'
                if 'dead' in (left, right)
                else self._u_delta(left, right, sym[adv], PAD),
                lambda st: st == 'A')

        return {
            'Dom': wrap2(self._u_delta, lambda st: st in ('S', 'Z')),
            'Adv': Adv,
            'Sing': wrap2(self._sing_delta, lambda st: st == 'M'),
            'Subset': wrap3(self._subset_delta, lambda st: st == 'ok'),
            'E': wrap3(self._e_delta, lambda st: st == 'D'),
        }

    @property
    def implicit_cls(self):
        """The fully implicit presentation (functional atoms only)."""
        from autstr.implicit import ImplicitTreeClass
        return ImplicitTreeClass(self._implicit_atoms(), [PAD, '0', '1'])

    def check_implicit(self, phi, graph: Union[RankWidthGraph, Tree],
                       **sets) -> bool:
        """Like `check`, evaluated implicitly (no automaton is built). Set
        assignments are padded to the advice shape (the implicit evaluator
        runs all tapes synchronously)."""
        advice = self.advice(graph)
        trees = {}
        if sets:
            if not isinstance(graph, RankWidthGraph):
                raise ValueError("set assignments require a graph object")
            trees = {name: graph.encode_set_padded(subset)
                     for name, subset in sets.items()}
        return self.implicit_cls.check(phi, advice, **trees)

    def evaluate_implicit(self, phi, graph: Union[RankWidthGraph, Tree],
                          **sets):
        """The satisfying set of phi on the graph, computed implicitly:
        unassigned free variables stay open. Yields assignments
        {var: vertex set} when a graph object is given (raw mark trees for a
        bare advice); `len` is the exact solution count."""
        advice = self.advice(graph)
        trees = {}
        if sets:
            if not isinstance(graph, RankWidthGraph):
                raise ValueError("set assignments require a graph object")
            trees = {name: graph.encode_set_padded(subset)
                     for name, subset in sets.items()}
        sols = self.implicit_cls.evaluate(phi, advice, **trees)
        if isinstance(graph, RankWidthGraph):
            from autstr.implicit import MappedSolutions
            return MappedSolutions(sols, graph.decode_set)
        return sols

    def get_structure(self, graph) -> TreeAutomaticPresentation:
        """The MSO0-style tree-automatic presentation of a single graph."""
        return self.cls.get_structure(self.advice(graph))
