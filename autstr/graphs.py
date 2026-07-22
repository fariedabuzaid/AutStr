"""Graphs of bounded tree-depth and bounded pathwidth as uniformly automatic
classes.

Both classes present graphs over *sets* of vertices (as in MSO0), so
first-order logic over the presentation is monadic second-order logic over
the graph. The shared signature is

    Sing(x)      x is a singleton
    Subset(x,y)  x is a subset of y
    E(x,y)       x = {u}, y = {v} and u,v are adjacent

**Tree-depth <= d** (`TreeDepthClass`): the advice spells out a DFS traversal
of an elimination forest of height <= d, one letter per vertex encoding
(depth, adjacency profile to its ancestors).

**Pathwidth <= w** (`PathWidthClass`): the advice spells out a linear layout,
one letter per vertex encoding (register in {0..w}, adjacency profile to the
registers of its earlier neighbors). Introducing a vertex at register r
replaces the previous occupant of r; edges may only reach current occupants,
which is exactly the interval structure of a path decomposition of width w.

`TreeDepthGraph` / `PathWidthGraph` encapsulate the string representations of
single graphs and convert from/to networkx.
"""
import itertools as it
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import graphviz

from autstr.presentations import AutomaticPresentation
from autstr.sparse_automata import SparseDFA
from autstr.uniform import SymbolicClassWrapper, UniformlyAutomaticClass, dfa_from_delta

PAD = '*'


# ====================================================================
# String-encoded graphs
# ====================================================================

class StringGraph:
    """Base class for graphs encoded as strings of per-vertex letters."""

    def __init__(self, letters: Sequence[Tuple], nodes: Optional[Sequence] = None):
        self.letters = list(letters)
        self.nodes = list(nodes) if nodes is not None else list(range(len(self.letters)))
        if len(self.nodes) != len(self.letters):
            raise ValueError("nodes and letters must have the same length")

    @property
    def num_nodes(self) -> int:
        return len(self.letters)

    def edges(self) -> List[Tuple]:
        """Edge list (node names) decoded from the letters."""
        raise NotImplementedError

    def encode_set(self, subset) -> Tuple[str, ...]:
        """Encode a set of nodes as a {0,1}-word over the vertex positions."""
        subset = set(subset)
        unknown = subset - set(self.nodes)
        if unknown:
            raise ValueError(f"not nodes of this graph: {unknown}")
        return tuple('1' if v in subset else '0' for v in self.nodes)

    def to_networkx(self):
        """Convert back to a networkx graph (node names preserved)."""
        try:
            import networkx as nx
        except ImportError as e:
            raise ImportError("to_networkx requires networkx (pip install autstr[graphs])") from e
        graph = nx.Graph()
        graph.add_nodes_from(self.nodes)
        graph.add_edges_from(self.edges())
        return graph

    def to_graphviz(self, sets: Optional[Dict[str, Set]] = None,
                    filename: Optional[str] = None, format: str = "png",
                    view: bool = False) -> graphviz.Graph:
        """Visualize the graph; `sets` maps labels to node sets that are
        highlighted (colored and annotated with the label)."""
        palette = ['lightblue', 'lightsalmon', 'palegreen', 'plum', 'khaki', 'lightpink']
        membership = {}
        for idx, (label, subset) in enumerate((sets or {}).items()):
            for v in subset:
                membership.setdefault(v, []).append((label, palette[idx % len(palette)]))

        dot = graphviz.Graph(engine='dot')
        for v in self.nodes:
            tags = membership.get(v)
            if tags:
                label = f"{v}\\n({', '.join(t for t, _ in tags)})"
                dot.node(str(v), label=label, style='filled', fillcolor=tags[0][1])
            else:
                dot.node(str(v), label=str(v))
        for u, v in self.edges():
            dot.edge(str(u), str(v))
        if filename is not None:
            dot.render(filename=filename, format=format, view=view)
        return dot

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.num_nodes} nodes)"


class TreeDepthGraph(StringGraph):
    """A graph of bounded tree-depth in its string representation: the DFS
    traversal of an elimination forest, one (depth, profile) letter per
    vertex. profile[t-1] == 1 means the vertex is adjacent to its unique
    ancestor at depth t."""

    def __init__(self, letters: Sequence[Tuple[int, Tuple[int, ...]]],
                 nodes: Optional[Sequence] = None):
        letters = [(int(k), tuple(int(b) for b in profile)) for k, profile in letters]
        prev_depth = 0
        for i, (k, profile) in enumerate(letters):
            if not 1 <= k <= prev_depth + 1:
                raise ValueError(f"letter {i}: depth {k} after depth {prev_depth} is not a DFS traversal")
            if len(profile) != k - 1:
                raise ValueError(f"letter {i}: profile length {len(profile)} != depth-1 = {k - 1}")
            prev_depth = k
        super().__init__(letters, nodes)

    @property
    def height(self) -> int:
        """Height of the elimination forest (>= tree-depth of the graph)."""
        return max((k for k, _ in self.letters), default=0)

    def edges(self) -> List[Tuple]:
        stack = []  # stack[t-1] = position of the current ancestor at depth t
        edges = []
        for pos, (k, profile) in enumerate(self.letters):
            del stack[k - 1:]
            for t, bit in enumerate(profile, start=1):
                if bit:
                    edges.append((self.nodes[stack[t - 1]], self.nodes[pos]))
            stack.append(pos)
        return edges

    @classmethod
    def from_networkx(cls, graph, forest: Optional[Dict] = None,
                      exact_below: int = 13) -> 'TreeDepthGraph':
        """Build the string representation from a networkx graph.

        :param graph: undirected networkx graph
        :param forest: optional elimination forest as a dict node -> parent
            (roots map to None or are absent). Every edge of the graph must
            connect a vertex to one of its forest ancestors.
        :param exact_below: for graphs with fewer vertices, an optimal
            elimination forest is computed by exhaustive search; larger graphs
            fall back to a DFS forest (always valid, possibly deeper than the
            tree-depth).
        """
        if forest is None:
            if graph.number_of_nodes() < exact_below:
                forest = _optimal_elimination_forest(graph)
            else:
                forest = _dfs_elimination_forest(graph)

        children: Dict = {v: [] for v in graph.nodes}
        roots = []
        for v in sorted(graph.nodes, key=str):
            parent = forest.get(v)
            if parent is None:
                roots.append(v)
            else:
                children[parent].append(v)

        # DFS traversal emitting (depth, profile) letters
        letters = []
        order = []
        emitted_edges = 0
        adj = {v: set(graph.neighbors(v)) for v in graph.nodes}
        stack = [(root, []) for root in reversed(roots)]
        while stack:
            v, ancestors = stack.pop()
            depth = len(ancestors) + 1
            profile = tuple(int(a in adj[v]) for a in ancestors)
            emitted_edges += sum(profile)
            letters.append((depth, profile))
            order.append(v)
            chain = ancestors + [v]
            for c in reversed(children[v]):
                stack.append((c, chain))

        if len(order) != graph.number_of_nodes():
            raise ValueError("forest does not span all vertices")
        if emitted_edges != graph.number_of_edges():
            raise ValueError("not an elimination forest: some edge connects vertices "
                             "that are not in ancestor-descendant relation")
        return cls(letters, nodes=order)

    def __repr__(self) -> str:
        return f"TreeDepthGraph({self.num_nodes} nodes, height {self.height})"


class PathWidthGraph(StringGraph):
    """A graph of bounded pathwidth in its string representation: a linear
    layout, one (register, profile) letter per vertex. Introducing a vertex
    at register r replaces the previous occupant of r; profile lists the
    registers of the vertex's earlier neighbors (their current occupants)."""

    def __init__(self, letters: Sequence[Tuple[int, Tuple[int, ...]]],
                 nodes: Optional[Sequence] = None):
        letters = [(int(r), tuple(sorted(int(s) for s in profile))) for r, profile in letters]
        occupied: Set[int] = set()
        for i, (r, profile) in enumerate(letters):
            if r < 0 or any(s < 0 for s in profile):
                raise ValueError(f"letter {i}: negative register")
            if r in profile:
                raise ValueError(f"letter {i}: profile contains the vertex's own register {r}")
            if len(set(profile)) != len(profile):
                raise ValueError(f"letter {i}: duplicate registers in profile")
            if not set(profile) <= occupied:
                raise ValueError(f"letter {i}: profile references unoccupied registers "
                                 f"{set(profile) - occupied}")
            occupied.add(r)
        super().__init__(letters, nodes)

    @property
    def width(self) -> int:
        """Maximal register index (>= pathwidth of the graph)."""
        return max((r for r, _ in self.letters), default=0)

    def edges(self) -> List[Tuple]:
        occupant: Dict[int, int] = {}
        edges = []
        for pos, (r, profile) in enumerate(self.letters):
            for s in profile:
                edges.append((self.nodes[occupant[s]], self.nodes[pos]))
            occupant[r] = pos
        return edges

    @classmethod
    def from_networkx(cls, graph, order: Optional[Sequence] = None,
                      exact_below: int = 13) -> 'PathWidthGraph':
        """Build the string representation from a networkx graph.

        :param graph: undirected networkx graph
        :param order: optional vertex ordering (linear layout). If omitted, a
            minimum vertex-separation ordering is computed exhaustively for
            graphs with fewer than `exact_below` vertices; larger graphs fall
            back to a BFS ordering (valid, possibly wider than the pathwidth).
        """
        if order is None:
            if graph.number_of_nodes() < exact_below:
                order = _optimal_vertex_separation_order(graph)
            else:
                order = _bfs_order(graph)
        order = list(order)
        if set(order) != set(graph.nodes) or len(order) != graph.number_of_nodes():
            raise ValueError("order must enumerate every vertex exactly once")

        position = {v: i for i, v in enumerate(order)}
        last_needed = {
            v: max((position[u] for u in graph.neighbors(v)), default=position[v])
            for v in graph.nodes
        }

        letters = []
        register_of: Dict = {}
        for i, v in enumerate(order):
            in_use = {register_of[u] for u in order[:i] if last_needed[u] >= i}
            register = next(r for r in it.count() if r not in in_use)
            profile = tuple(sorted(
                register_of[u] for u in graph.neighbors(v) if position[u] < i
            ))
            register_of[v] = register
            letters.append((register, profile))
        return cls(letters, nodes=order)

    def __repr__(self) -> str:
        return f"PathWidthGraph({self.num_nodes} nodes, width {self.width})"


# ====================================================================
# Layout / decomposition search
# ====================================================================

def _dfs_elimination_forest(graph) -> Dict:
    """DFS forest of the graph: valid elimination forest because every
    non-tree edge of an undirected DFS is a back edge (ancestor pair)."""
    forest: Dict = {}
    visited = set()
    for root in sorted(graph.nodes, key=str):
        if root in visited:
            continue
        forest[root] = None
        visited.add(root)
        stack = [root]
        while stack:
            v = stack[-1]
            for w in sorted(graph.neighbors(v), key=str):
                if w not in visited:
                    visited.add(w)
                    forest[w] = v
                    stack.append(w)
                    break
            else:
                stack.pop()
    return forest


def _optimal_elimination_forest(graph) -> Dict:
    """Minimum-height elimination forest by exhaustive search (exponential;
    only for small graphs). Returns a dict node -> parent (None for roots)."""
    adj = {v: set(graph.neighbors(v)) for v in graph.nodes}
    memo: Dict[frozenset, Tuple[int, Dict, List]] = {}

    def components(vertices: frozenset) -> List[frozenset]:
        remaining = set(vertices)
        comps = []
        while remaining:
            comp = {remaining.pop()}
            frontier = [next(iter(comp))]
            while frontier:
                v = frontier.pop()
                for w in adj[v] & remaining:
                    remaining.discard(w)
                    comp.add(w)
                    frontier.append(w)
            comps.append(frozenset(comp))
        return comps

    def solve(vertices: frozenset) -> Tuple[int, Dict, List]:
        """Returns (height, parent map, roots) for G[vertices]."""
        if not vertices:
            return 0, {}, []
        if vertices in memo:
            return memo[vertices]

        comps = components(vertices)
        if len(comps) > 1:
            height, parents, roots = 0, {}, []
            for comp in comps:
                h, p, r = solve(comp)
                height = max(height, h)
                parents.update(p)
                roots.extend(r)
            memo[vertices] = (height, parents, roots)
            return memo[vertices]

        best = None
        for v in sorted(vertices, key=str):
            h, p, r = solve(vertices - {v})
            if best is None or h + 1 < best[0]:
                parents = dict(p)
                for root in r:
                    parents[root] = v
                best = (h + 1, parents, [v])
        memo[vertices] = best
        return best

    _, parents, roots = solve(frozenset(graph.nodes))
    forest = dict(parents)
    for root in roots:
        forest[root] = None
    return forest


def _optimal_vertex_separation_order(graph) -> List:
    """Minimum vertex-separation ordering by subset DP (exponential; only
    for small graphs). Vertex separation number equals pathwidth."""
    nodes = sorted(graph.nodes, key=str)
    n = len(nodes)
    index = {v: i for i, v in enumerate(nodes)}
    neighbor_mask = [0] * n
    for v in nodes:
        for u in graph.neighbors(v):
            neighbor_mask[index[v]] |= 1 << index[u]

    full = (1 << n) - 1

    def boundary(mask: int) -> int:
        return sum(1 for i in range(n)
                   if mask >> i & 1 and neighbor_mask[i] & ~mask)

    cost = {0: 0}
    best_last = {}
    for mask in range(1, full + 1):
        b = boundary(mask)
        best = None
        m = mask
        while m:
            bit = m & -m
            prev = cost[mask ^ bit]
            value = max(prev, b)
            if best is None or value < best[0]:
                best = (value, bit)
            m ^= bit
        cost[mask] = best[0]
        best_last[mask] = best[1]

    order = []
    mask = full
    while mask:
        bit = best_last[mask]
        order.append(nodes[bit.bit_length() - 1])
        mask ^= bit
    order.reverse()
    return order


def _bfs_order(graph) -> List:
    """BFS ordering (per component, from a minimum-degree vertex)."""
    order = []
    visited = set()
    for start in sorted(graph.nodes, key=lambda v: (graph.degree(v), str(v))):
        if start in visited:
            continue
        queue = [start]
        visited.add(start)
        while queue:
            v = queue.pop(0)
            order.append(v)
            for u in sorted(graph.neighbors(v), key=str):
                if u not in visited:
                    visited.add(u)
                    queue.append(u)
    return order


# ====================================================================
# Shared presentation automata (set signature)
# ====================================================================

def _sing_automaton(sigma, letter_symbols) -> SparseDFA:
    """Sing(p, X): X contains exactly one position."""
    states = ['zero', 'one', 'one_end', 'pad', 'dead']

    def delta(q, sym):
        a, x = sym
        if q == 'dead':
            return 'dead'
        if q == 'pad':
            return 'pad' if (a, x) == (PAD, PAD) else 'dead'
        is_letter = a in letter_symbols
        if q == 'zero':
            if is_letter and x == '0':
                return 'zero'
            if is_letter and x == '1':
                return 'one'
            return 'dead'
        if q == 'one':
            if is_letter and x == '0':
                return 'one'
            if is_letter and x == PAD:
                return 'one_end'
            if (a, x) == (PAD, PAD):
                return 'pad'
            return 'dead'
        if q == 'one_end':
            if is_letter and x == PAD:
                return 'one_end'
            if (a, x) == (PAD, PAD):
                return 'pad'
            return 'dead'

    return dfa_from_delta(sigma, states, 2, delta, 'zero', {'one', 'one_end', 'pad'})


def _subset_automaton(sigma, letter_symbols) -> SparseDFA:
    """Subset(p, X, Y): X is a subset of Y (positionwise X <= Y)."""
    states = ['ok', 'pad', 'dead']
    allowed = {('0', '0'), ('0', '1'), ('1', '1'),
               ('0', PAD), (PAD, '0'), (PAD, '1'), (PAD, PAD)}

    def delta(q, sym):
        a, x, y = sym
        if q == 'dead':
            return 'dead'
        if q == 'pad':
            return 'pad' if sym == (PAD, PAD, PAD) else 'dead'
        if a in letter_symbols and (x, y) in allowed:
            return 'ok'
        if sym == (PAD, PAD, PAD):
            return 'pad'
        return 'dead'

    return dfa_from_delta(sigma, states, 3, delta, 'ok', {'ok', 'pad'})


class _SetGraphClass(SymbolicClassWrapper):
    """Shared class-level operations of the set-signature graph classes."""

    #: elements are vertex *sets* and E is the edge relation, not equality
    GRAPH = None
    #: extensional equality of sets
    EQUALITY = 'Subset(x,y) and Subset(y,x)'

    def __init__(self, automata: Dict[str, SparseDFA],
                 eager_equality: bool = False):
        self.cls = UniformlyAutomaticClass(automata, padding_symbol=PAD)
        self._declare_equality(eager_equality)

    def advice(self, graph) -> List[str]:
        raise NotImplementedError

    def evaluate(self, phi) -> Tuple[SparseDFA, List[str]]:
        """Evaluate an MSO query over the class; see
        UniformlyAutomaticClass.evaluate. Variables range over vertex sets."""
        return self.cls.evaluate(phi)

    def symbolic(self, signature=None):
        """A symbolic interface to this class; see
        UniformlyAutomaticClass.symbolic."""
        return self.cls.symbolic(signature)

    def check(self, phi, graph, **sets) -> bool:
        """Model check a query against a single graph.

        :param phi: formula over Sing/Subset/E; free variables can be
            assigned via `sets` (name = set of nodes), unassigned ones are
            quantified existentially. Note that nltk only treats names
            matching [a-df-z][0-9]* (single lowercase letter except 'e',
            optional digits) as variables.
        :param graph: a StringGraph of this class (or a raw advice string).
        :param sets: assignments for free variables, as sets of nodes.
        """
        advice = self.advice(graph)
        if not sets:
            return self.cls.check(phi, advice)
        if not isinstance(graph, StringGraph):
            raise ValueError("set assignments require a graph object")
        words = {name: list(graph.encode_set(subset)) for name, subset in sets.items()}
        return self.cls.check(phi, advice, **words)

    def get_structure(self, graph) -> AutomaticPresentation:
        """The MSO0-style automatic presentation of a single graph."""
        return self.cls.get_structure(self.advice(graph))


# ====================================================================
# Tree-depth
# ====================================================================

def _letter_symbol(depth: int, profile: Tuple[int, ...]) -> str:
    """Advice alphabet symbol for a tree-depth vertex letter."""
    return f"a{depth}_" + "".join(str(b) for b in profile)


class TreeDepthClass(_SetGraphClass):
    """The uniformly automatic class of graphs of tree-depth <= d, presented
    over set-valued elements (MSO0 style)."""

    def __init__(self, d: int):
        if d < 1:
            raise ValueError("depth bound must be >= 1")
        self.d = d
        self.letters = [
            (k, profile)
            for k in range(1, d + 1)
            for profile in it.product((0, 1), repeat=k - 1)
        ]
        self.symbol_of = {letter: _letter_symbol(*letter) for letter in self.letters}
        self.letter_of = {sym: letter for letter, sym in self.symbol_of.items()}
        self.sigma = {PAD, '0', '1'} | set(self.symbol_of.values())

        super().__init__({
            'U': self._universe_automaton(),
            'Sing': _sing_automaton(self.sigma, self.letter_of),
            'Subset': _subset_automaton(self.sigma, self.letter_of),
            'E': self._edge_automaton(),
        })

    def _depth(self, symbol: str) -> Optional[int]:
        letter = self.letter_of.get(symbol)
        return letter[0] if letter else None

    def _profile(self, symbol: str) -> Optional[Tuple[int, ...]]:
        letter = self.letter_of.get(symbol)
        return letter[1] if letter else None

    def _universe_automaton(self) -> SparseDFA:
        """U(p, x): p is a DFS traversal of a forest of height <= d and x is a
        bitvector over its positions (monotone trailing padding)."""
        states = [('s', prev, ended) for prev in range(self.d + 1) for ended in (False, True)]
        states += ['pad', 'dead']

        def delta(q, sym):
            a, x = sym
            if q == 'dead':
                return 'dead'
            if q == 'pad':
                return 'pad' if (a, x) == (PAD, PAD) else 'dead'
            _, prev, ended = q
            if a == PAD:
                return 'pad' if x == PAD else 'dead'
            depth = self._depth(a)
            if depth is None or depth > prev + 1:
                return 'dead'
            if x == PAD:
                return ('s', depth, True)
            if x in ('0', '1') and not ended:
                return ('s', depth, False)
            return 'dead'

        finals = {q for q in states if q != 'dead'}
        return dfa_from_delta(self.sigma, states, 2, delta, ('s', 0, False), finals)

    def _edge_automaton(self) -> SparseDFA:
        """E(p, X, Y): X = {u}, Y = {v}, u != v adjacent. u must be a forest
        ancestor of v (or vice versa): after the first marker at depth k, all
        letters up to the second marker must have depth > k, and the second
        vertex's profile bit for depth k must be set."""
        states = ['init', 'done', 'pad', 'dead']
        states += [('X', k) for k in range(1, self.d + 1)]
        states += [('Y', k) for k in range(1, self.d + 1)]

        def delta(q, sym):
            a, x, y = sym
            if q == 'dead':
                return 'dead'
            if q == 'pad':
                return 'pad' if sym == (PAD, PAD, PAD) else 'dead'
            depth = self._depth(a)
            if q == 'init':
                if depth is None:
                    return 'dead'
                if (x, y) == ('0', '0'):
                    return 'init'
                if (x, y) == ('1', '0'):
                    return ('X', depth)
                if (x, y) == ('0', '1'):
                    return ('Y', depth)
                return 'dead'
            if q == 'done':
                if depth is not None and x in ('0', PAD) and y in ('0', PAD):
                    return 'done'
                if sym == (PAD, PAD, PAD):
                    return 'pad'
                return 'dead'
            phase, k = q
            first, second = (x, y) if phase == 'X' else (y, x)
            if depth is None or depth <= k:
                return 'dead'
            if second == '1' and first in ('0', PAD):
                # the second marker: adjacent iff its profile links depth k
                return 'done' if self._profile(a)[k - 1] == 1 else 'dead'
            if second == '0' and first in ('0', PAD):
                return (phase, k)
            return 'dead'

        return dfa_from_delta(self.sigma, states, 3, delta, 'init', {'done', 'pad'})

    def advice(self, graph: Union[TreeDepthGraph, Sequence[str]]) -> List[str]:
        """The advice string of a graph (its letters as alphabet symbols)."""
        if not isinstance(graph, TreeDepthGraph):
            return list(graph)
        if graph.height > self.d:
            raise ValueError(f"graph has elimination-forest height {graph.height} > d = {self.d}")
        return [self.symbol_of[letter] for letter in graph.letters]


# ====================================================================
# Pathwidth
# ====================================================================

def _register_symbol(register: int, profile: Tuple[int, ...]) -> str:
    """Advice alphabet symbol for a pathwidth vertex letter."""
    return f"r{register}s" + "".join(str(s) for s in profile)


class PathWidthClass(_SetGraphClass):
    """The uniformly automatic class of graphs of pathwidth <= w, presented
    over set-valued elements (MSO0 style)."""

    def __init__(self, w: int):
        if w < 0:
            raise ValueError("width bound must be >= 0")
        self.w = w
        registers = range(w + 1)
        self.letters = [
            (r, profile)
            for r in registers
            for size in range(w + 1)
            for profile in it.combinations([s for s in registers if s != r], size)
        ]
        self.symbol_of = {letter: _register_symbol(*letter) for letter in self.letters}
        self.letter_of = {sym: letter for letter, sym in self.symbol_of.items()}
        self.sigma = {PAD, '0', '1'} | set(self.symbol_of.values())

        super().__init__({
            'U': self._universe_automaton(),
            'Sing': _sing_automaton(self.sigma, self.letter_of),
            'Subset': _subset_automaton(self.sigma, self.letter_of),
            'E': self._edge_automaton(),
        })

    def _universe_automaton(self) -> SparseDFA:
        """U(p, x): p is a valid layout (profiles only reference registers
        already in use) and x is a bitvector over its positions."""
        registers = list(range(self.w + 1))
        subsets = [frozenset(c) for size in range(self.w + 2)
                   for c in it.combinations(registers, size)]
        states = [('s', occupied, ended) for occupied in subsets for ended in (False, True)]
        states += ['pad', 'dead']

        def delta(q, sym):
            a, x = sym
            if q == 'dead':
                return 'dead'
            if q == 'pad':
                return 'pad' if (a, x) == (PAD, PAD) else 'dead'
            _, occupied, ended = q
            if a == PAD:
                return 'pad' if x == PAD else 'dead'
            letter = self.letter_of.get(a)
            if letter is None or not set(letter[1]) <= occupied:
                return 'dead'
            occupied = frozenset(occupied | {letter[0]})
            if x == PAD:
                return ('s', occupied, True)
            if x in ('0', '1') and not ended:
                return ('s', occupied, False)
            return 'dead'

        finals = {q for q in states if q != 'dead'}
        return dfa_from_delta(self.sigma, states, 2, delta, ('s', frozenset(), False), finals)

    def _edge_automaton(self) -> SparseDFA:
        """E(p, X, Y): X = {u}, Y = {v}, u != v adjacent. After the first
        marker at register r, no letter up to the second marker may reuse r
        (that would replace u), and the second vertex's profile must list r."""
        states = ['init', 'done', 'pad', 'dead']
        states += [('X', r) for r in range(self.w + 1)]
        states += [('Y', r) for r in range(self.w + 1)]

        def delta(q, sym):
            a, x, y = sym
            if q == 'dead':
                return 'dead'
            if q == 'pad':
                return 'pad' if sym == (PAD, PAD, PAD) else 'dead'
            letter = self.letter_of.get(a)
            if q == 'init':
                if letter is None:
                    return 'dead'
                if (x, y) == ('0', '0'):
                    return 'init'
                if (x, y) == ('1', '0'):
                    return ('X', letter[0])
                if (x, y) == ('0', '1'):
                    return ('Y', letter[0])
                return 'dead'
            if q == 'done':
                if letter is not None and x in ('0', PAD) and y in ('0', PAD):
                    return 'done'
                if sym == (PAD, PAD, PAD):
                    return 'pad'
                return 'dead'
            phase, r = q
            first, second = (x, y) if phase == 'X' else (y, x)
            if letter is None:
                return 'dead'
            if second == '1' and first in ('0', PAD):
                # the second marker: adjacent iff its profile lists register r
                return 'done' if r in letter[1] else 'dead'
            if second == '0' and first in ('0', PAD):
                # the first vertex is replaced if its register is reused
                return 'dead' if letter[0] == r else (phase, r)
            return 'dead'

        return dfa_from_delta(self.sigma, states, 3, delta, 'init', {'done', 'pad'})

    def advice(self, graph: Union[PathWidthGraph, Sequence[str]]) -> List[str]:
        """The advice string of a graph (its letters as alphabet symbols)."""
        if not isinstance(graph, PathWidthGraph):
            return list(graph)
        if graph.width > self.w:
            raise ValueError(f"graph has layout width {graph.width} > w = {self.w}")
        return [self.symbol_of[letter] for letter in graph.letters]
