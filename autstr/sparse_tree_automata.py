"""Sparse bottom-up tree automata.

The tree analog of `autstr.sparse_automata`: deterministic bottom-up automata
over binary trees (general trees embed via the first-child/next-sibling
encoding), stored sparsely and processed with batched numpy throughout.

**States.** Real states are 0..num_states-1; the virtual *absent* state
``BOT = num_states`` represents a missing child, so a single transition table
covers leaves (both children absent), unary and binary nodes:

    state(node) = delta(state(left) or BOT, state(right) or BOT, label(node)).

**Sparsity.** Transitions are stored as a sorted table from the child pair
``(left, right)`` to a *shared multi-terminal BDD* over the binary digits of
the symbol (see `autstr.mtbdd`); pairs absent from the table map every symbol
to the global ``default_state``. Nothing in the pipeline ever enumerates the
convolution alphabet: a symbol is a variable assignment, so a transition that
ignores a tape simply does not test that tape's variables. Boolean
combinations are pairwise `apply` on the diagrams, complementation relabels
acceptance and touches no diagram at all, and hash-consing makes two states
with the same transition function share one node.
"""
from collections import defaultdict, deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from autstr.mtbdd import NONE, STORE, num_bits, var_tables
from autstr.utils.misc import encode_symbol


# ====================================================================
# Trees
# ====================================================================

class Tree:
    """An immutable labelled binary tree (convenience representation)."""

    __slots__ = ("label", "left", "right")

    def __init__(self, label, left: Optional["Tree"] = None,
                 right: Optional["Tree"] = None):
        self.label = label
        self.left = left
        self.right = right

    def __eq__(self, other):
        if not isinstance(other, Tree):
            return NotImplemented
        # iterative structural equality
        stack = [(self, other)]
        while stack:
            a, b = stack.pop()
            if a is None or b is None:
                if a is not b:
                    return False
                continue
            if a.label != b.label:
                return False
            stack.append((a.left, b.left))
            stack.append((a.right, b.right))
        return True

    def __repr__(self):
        if self.left is None and self.right is None:
            return f"Tree({self.label!r})"
        return f"Tree({self.label!r}, {self.left!r}, {self.right!r})"

    def size(self) -> int:
        n, stack = 0, [self]
        while stack:
            t = stack.pop()
            if t is not None:
                n += 1
                stack.append(t.left)
                stack.append(t.right)
        return n


def tree_to_arrays(tree: Tree, base_alphabet: Set, arity: int = 1):
    """Convert a Tree with tuple/symbol labels to the post-order array format
    (labels encoded as integers over base_alphabet^arity)."""
    base = frozenset(base_alphabet)
    labels, lefts, rights = [], [], []
    # iterative post-order: (node, child indices resolved?) via two-phase stack
    stack: List[Tuple[Tree, bool]] = [(tree, False)]
    index: dict = {}
    while stack:
        node, expanded = stack.pop()
        if node is None:
            continue
        if not expanded:
            stack.append((node, True))
            stack.append((node.right, False))
            stack.append((node.left, False))
        else:
            label = node.label if isinstance(node.label, tuple) else (node.label,)
            labels.append(encode_symbol(label, base))
            lefts.append(index[id(node.left)] if node.left is not None else -1)
            rights.append(index[id(node.right)] if node.right is not None else -1)
            index[id(node)] = len(labels) - 1
    return (np.array(labels, dtype=np.int64),
            np.array(lefts, dtype=np.int64),
            np.array(rights, dtype=np.int64))


def convolve_trees(trees: Sequence[Tree], base_alphabet: Set,
                   padding_symbol) -> Tree:
    """Overlay k trees into one tree over the tuple alphabet: the domain is
    the union of the domains, absent positions are padded."""
    def merge(nodes):
        if all(n is None for n in nodes):
            return None
        label = tuple(n.label if n is not None else padding_symbol
                      for n in nodes)
        left = merge([n.left if n is not None else None for n in nodes])
        right = merge([n.right if n is not None else None for n in nodes])
        return Tree(label, left, right)

    # bounded recursion is fine for convenience use; large inputs should be
    # generated directly in array form
    import sys
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old, 10000))
    try:
        return merge(list(trees))
    finally:
        sys.setrecursionlimit(old)


# ====================================================================
# The automaton
# ====================================================================

class SparseTreeAutomaton:
    """Deterministic bottom-up tree automaton with MTBDD transitions.

    The constructor takes the transition function in the flat form that is
    convenient to write down by hand — a global default, optional per-pair
    defaults, and ``(left, right, symbol) -> target`` exceptions — and
    compiles it into one decision diagram per child pair.

    :param num_states: number of real states (0..num_states-1); the virtual
        absent-child state is ``BOT = num_states``.
    :param default_state: target of every transition not listed below.
    :param exc_left, exc_right, exc_symbol, exc_target: parallel arrays of
        exception transitions delta(exc_left, exc_right, exc_symbol) =
        exc_target. Children may be BOT; targets are real states.
    :param is_accepting: boolean array over the real states (acceptance is
        checked at the root).
    :param pd_left, pd_right, pd_target: parallel arrays of pair defaults
        ``delta(pd_left, pd_right, *) = pd_target`` for symbols without an
        exception. Pairs not listed fall back to the global default.
    :param pair_keys, pair_nodes: the compiled form (sorted packed pair keys
        and their diagram roots); passed by the pipeline instead of the flat
        arrays.
    """

    def __init__(self, num_states: int, default_state: int,
                 exc_left=(), exc_right=(), exc_symbol=(), exc_target=(),
                 is_accepting=(), symbol_arity: int = 1,
                 base_alphabet: Optional[Set] = None,
                 pd_left=(), pd_right=(), pd_target=(),
                 pair_keys=None, pair_nodes=None):
        self.num_states = int(num_states)
        self.default_state = int(default_state)
        self.is_accepting = np.asarray(is_accepting, dtype=bool)
        self.symbol_arity = int(symbol_arity)
        self.base_alphabet = base_alphabet or {0}
        self.base_alphabet_frozen = frozenset(self.base_alphabet)

        self.store = STORE
        self.m = len(self.base_alphabet_frozen)
        self.bits = num_bits(self.m)
        self.nvars = self.symbol_arity * self.bits
        self.default_node = self.store.const(self.default_state,
                                             self.symbol_arity, self.m,
                                             self.bits)
        if pair_keys is not None:
            order = np.argsort(np.asarray(pair_keys, dtype=np.int64),
                               kind="stable")
            self.pair_keys = np.asarray(pair_keys, dtype=np.int64)[order]
            self.pair_nodes = np.asarray(pair_nodes, dtype=np.int64)[order]
        else:
            self._compile(exc_left, exc_right, exc_symbol, exc_target,
                          pd_left, pd_right, pd_target)
        self._run_cache: Dict[int, int] = {}

    # ---------------- compilation of the flat form ----------------

    def _compile(self, exc_left, exc_right, exc_symbol, exc_target,
                 pd_left, pd_right, pd_target) -> None:
        base = self.num_states + 1
        left = np.asarray(exc_left, dtype=np.int64).reshape(-1)
        right = np.asarray(exc_right, dtype=np.int64).reshape(-1)
        symbol = np.asarray(exc_symbol, dtype=np.int64).reshape(-1)
        target = np.asarray(exc_target, dtype=np.int64).reshape(-1)
        pdl = np.asarray(pd_left, dtype=np.int64).reshape(-1)
        pdr = np.asarray(pd_right, dtype=np.int64).reshape(-1)
        pdt = np.asarray(pd_target, dtype=np.int64).reshape(-1)

        # a pair's base value is its pair default, the global default otherwise
        pd_keys = pdl * base + pdr
        order = np.argsort(pd_keys, kind="stable")
        pd_keys, pdt = pd_keys[order], pdt[order]

        exc_keys = left * base + right
        # the first row of a duplicated (pair, symbol) wins, as with the
        # leftmost binary search the flat representation used
        order = np.lexsort((symbol, exc_keys))
        exc_keys, symbol, target = exc_keys[order], symbol[order], target[order]
        if len(exc_keys):
            packed = exc_keys * self.num_symbols + symbol
            _, first = np.unique(packed, return_index=True)
            exc_keys, symbol, target = exc_keys[first], symbol[first], target[first]

        pairs = np.union1d(exc_keys, pd_keys).astype(np.int64)
        starts = np.searchsorted(exc_keys, pairs, 'left')
        ends = np.searchsorted(exc_keys, pairs, 'right')
        if len(pd_keys):
            pd_pos = np.minimum(np.searchsorted(pd_keys, pairs),
                                len(pd_keys) - 1)
            pd_hit = pd_keys[pd_pos] == pairs
        else:
            pd_pos = np.zeros(len(pairs), dtype=np.int64)
            pd_hit = np.zeros(len(pairs), dtype=bool)

        keys, nodes = [], []
        for i, key in enumerate(pairs.tolist()):
            value = int(pdt[pd_pos[i]]) if pd_hit[i] else self.default_state
            node = self.store.build_rows(
                symbol[starts[i]:ends[i]], target[starts[i]:ends[i]],
                value, self.symbol_arity, self.m, self.bits)
            if node != self.default_node:
                keys.append(key)
                nodes.append(node)
        self.pair_keys = np.array(keys, dtype=np.int64)
        self.pair_nodes = np.array(nodes, dtype=np.int64)

    # ---------------- basics ----------------

    @property
    def BOT(self) -> int:
        return self.num_states

    @property
    def num_symbols(self) -> int:
        return self.m ** self.symbol_arity

    @property
    def num_nodes(self) -> int:
        """Distinct diagram nodes carrying this automaton's transitions."""
        return self.store.size(self.pair_nodes.tolist() + [self.default_node])

    def pair_node(self, left, right) -> np.ndarray:
        """Batched lookup of the diagram of each child pair."""
        keys = np.asarray(left, dtype=np.int64) * (self.num_states + 1) + \
            np.asarray(right, dtype=np.int64)
        if len(self.pair_keys) == 0:
            return np.full(keys.shape, self.default_node, dtype=np.int64)
        pos = np.minimum(np.searchsorted(self.pair_keys, keys),
                         len(self.pair_keys) - 1)
        hit = self.pair_keys[pos] == keys
        return np.where(hit, self.pair_nodes[pos], self.default_node)

    def transitions(self, left, right, symbol) -> np.ndarray:
        """Batched transition lookup: find each pair's diagram, then descend
        it along the symbol's digits."""
        symbol = np.asarray(symbol, dtype=np.int64)
        nodes = self.pair_node(left, right)
        return self.store.eval_batch(nodes, symbol, self.symbol_arity,
                                     self.m, self.bits)

    def dense_delta(self, max_entries: int = 10 ** 7) -> np.ndarray:
        """The full transition table ``(BOT+1, BOT+1, num_symbols)``. For
        inspection and for reference oracles on small automata."""
        n, S = self.num_states, self.num_symbols
        if (n + 1) ** 2 * S > max_entries:
            raise ValueError("transition table too large to materialize")
        left, right, symbol = np.meshgrid(np.arange(n + 1), np.arange(n + 1),
                                          np.arange(S), indexing='ij')
        return self.transitions(left.ravel(), right.ravel(), symbol.ravel()
                                ).reshape(n + 1, n + 1, S)

    def exceptions(self, max_entries: int = 10 ** 7):
        """The transitions that differ from the global default, as flat
        ``(left, right, symbol, target)`` arrays (inspection only)."""
        table = self.dense_delta(max_entries)
        left, right, symbol = np.nonzero(table != self.default_state)
        return left, right, symbol, table[left, right, symbol]

    # ---------------- running trees ----------------

    def run(self, labels, lefts, rights) -> int:
        """State at the root of a post-order array tree.

        Adaptive evaluation: children resolve before parents in post-order, so
        each vectorized round computes every node whose children are already
        known — one round per tree level, ideal for bushy trees. Long unary
        chains (e.g. string-like spines) are inherently sequential, so when a
        round stops being productive the remaining nodes are finished by a
        scalar post-order sweep instead of degenerating to O(n^2)."""
        labels = np.asarray(labels, dtype=np.int64)
        lefts = np.asarray(lefts, dtype=np.int64)
        rights = np.asarray(rights, dtype=np.int64)

        n = len(labels)
        states = np.full(n + 1, -1, dtype=np.int64)   # slot n aliases BOT
        pending = np.ones(n, dtype=bool)
        child_l = np.where(lefts < 0, n, lefts)       # -1 -> resolved BOT slot
        child_r = np.where(rights < 0, n, rights)
        states[n] = self.BOT

        min_batch = max(1024, n // 64)
        while pending.any():
            ready = pending & (states[child_l] >= 0) & (states[child_r] >= 0)
            count = int(ready.sum())
            if count == 0:
                raise ValueError("tree arrays are not in a valid child-first order")
            if count < min_batch:
                break                                  # chain regime: go scalar
            idx = np.flatnonzero(ready)
            states[idx] = self.transitions(states[child_l[idx]],
                                           states[child_r[idx]],
                                           labels[idx])
            pending[idx] = False

        if pending.any():
            self._run_scalar(labels, child_l, child_r, states, pending)
        return int(states[n - 1])

    def _run_scalar(self, labels, child_l, child_r, states, pending):
        """Finish the remaining pending nodes with a plain post-order sweep.

        A diagram descent costs one step per variable, which is more than the
        single table probe of the flat representation, so results are memoized
        per (child pair, symbol): the chains this path exists for reuse the
        same few transitions over and over."""
        from bisect import bisect_left
        from autstr.mtbdd import TOP
        store = self.store
        var, lo, hi, term = store.var, store.lo, store.hi, store.term
        div, shift = var_tables(self.symbol_arity, self.m, self.bits)
        div, shift = div.tolist(), shift.tolist()
        pair_keys = self.pair_keys.tolist()
        pair_nodes = self.pair_nodes.tolist()
        num_pairs = len(pair_keys)
        default_node = self.default_node
        cache = self._run_cache
        base = self.num_states + 1
        S = self.num_symbols
        m = self.m

        st = states.tolist()
        lab = labels.tolist()
        cl = child_l.tolist()
        cr = child_r.tolist()
        for i in np.flatnonzero(pending).tolist():
            pair = st[cl[i]] * base + st[cr[i]]
            symbol = lab[i]
            key = pair * S + symbol
            target = cache.get(key)
            if target is None:
                pos = bisect_left(pair_keys, pair)
                node = pair_nodes[pos] if pos < num_pairs and \
                    pair_keys[pos] == pair else default_node
                while var[node] != TOP:
                    v = var[node]
                    node = hi[node] if (symbol // div[v]) % m >> shift[v] & 1 \
                        else lo[node]
                target = cache[key] = term[node]
            st[i] = target
        states[:] = st

    def accepts(self, *trees) -> bool:
        """Does the automaton accept the convolution of the given trees?
        Accepts `Tree` objects (one per tape) or a single pre-encoded array
        tree given as the tuple (labels, lefts, rights)."""
        if len(trees) == 1 and isinstance(trees[0], tuple) and \
                len(trees[0]) == 3 and not isinstance(trees[0][0], Tree):
            labels, lefts, rights = trees[0]
        else:
            tree = trees[0] if len(trees) == 1 else convolve_trees(
                trees, self.base_alphabet_frozen, sorted(self.base_alphabet)[0])
            labels, lefts, rights = tree_to_arrays(
                tree, self.base_alphabet_frozen, self.symbol_arity)
        root = self.run(labels, lefts, rights)
        return bool(self.is_accepting[root])

    # ---------------- boolean operations ----------------

    def complement(self) -> "SparseTreeAutomaton":
        """Flip acceptance — the transition diagrams are untouched."""
        return SparseTreeAutomaton(
            self.num_states, self.default_state,
            is_accepting=~self.is_accepting, symbol_arity=self.symbol_arity,
            base_alphabet=self.base_alphabet,
            pair_keys=self.pair_keys, pair_nodes=self.pair_nodes)

    def intersection(self, other) -> "SparseTreeAutomaton":
        return self._product(other, np.logical_and)

    def union(self, other) -> "SparseTreeAutomaton":
        return self._product(other, np.logical_or)

    def _product(self, other: "SparseTreeAutomaton", combine
                 ) -> "SparseTreeAutomaton":
        if self.symbol_arity != other.symbol_arity:
            raise ValueError("product requires the same symbol arity")
        if self.base_alphabet_frozen != other.base_alphabet_frozen:
            raise ValueError("product requires the same base alphabet")

        # Product states are pairs of states; the transition diagram of a
        # product pair is the pairwise `apply` of the factors' diagrams, whose
        # terminal operation allocates product state ids on demand. Discovery
        # is a bottom-up reachability fixpoint over child options (discovered
        # pairs plus the BOT pair): the targets of a combo are exactly the
        # terminals of its diagram.
        store = self.store
        nb = other.num_states
        state_ids: Dict[int, int] = {}
        pairs_a: List[int] = []
        pairs_b: List[int] = []
        pending: List[int] = []

        def get_id(a: int, b: int) -> int:
            key = a * (nb + 1) + b
            idx = state_ids.get(key)
            if idx is None:
                idx = state_ids[key] = len(pairs_a)
                pairs_a.append(a)
                pairs_b.append(b)
                pending.append(idx)
            return idx

        def op(ta: int, tb: int) -> int:
            if ta == NONE or tb == NONE:
                return NONE
            return get_id(ta, tb)

        default_id = get_id(self.default_state, other.default_state)
        pending.clear()                            # the default is not a child
        default_node = store.const(default_id, self.symbol_arity, self.m,
                                   self.bits)
        cache: Dict[int, int] = {}

        keys: List[Tuple[int, int]] = []
        nodes: List[int] = []
        new_options = [-1, default_id]             # -1 encodes the BOT pair
        all_options: List[int] = []

        def components(options: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            absent = options < 0
            safe = np.maximum(options, 0)
            a = np.where(absent, self.BOT,
                         np.array(pairs_a, dtype=np.int64)[safe])
            b = np.where(absent, other.BOT,
                         np.array(pairs_b, dtype=np.int64)[safe])
            return a, b

        while new_options:
            # each combo is enumerated exactly once, in the round where its
            # later member was discovered (no quadratic dedup set)
            round_new = new_options
            combos = [(x, y) for x in round_new for y in all_options]
            combos += [(y, x) for x in round_new for y in all_options]
            combos += [(x, y) for x in round_new for y in round_new]
            all_options.extend(round_new)
            pending.clear()
            if not combos:
                break

            cl = np.array([c[0] for c in combos], dtype=np.int64)
            cr = np.array([c[1] for c in combos], dtype=np.int64)
            la, lb = components(cl)
            ra, rb = components(cr)
            nodes_a = self.pair_node(la, ra).tolist()
            nodes_b = other.pair_node(lb, rb).tolist()

            for i, (fa, fb) in enumerate(zip(nodes_a, nodes_b)):
                node = store.apply2(fa, fb, op, cache)
                if node != default_node:
                    keys.append(combos[i])
                    nodes.append(node)
            new_options = list(pending)

        num_states = len(pairs_a)
        packed = [(num_states if l < 0 else l) * (num_states + 1) +
                  (num_states if r < 0 else r) for l, r in keys]
        acc = combine(self.is_accepting[np.array(pairs_a, dtype=np.int64)],
                      other.is_accepting[np.array(pairs_b, dtype=np.int64)])
        return SparseTreeAutomaton(
            num_states, default_id, is_accepting=acc,
            symbol_arity=self.symbol_arity, base_alphabet=self.base_alphabet,
            pair_keys=np.array(packed, dtype=np.int64),
            pair_nodes=np.array(nodes, dtype=np.int64))

    # ---------------- emptiness ----------------

    def reachable_states(self) -> np.ndarray:
        """Boolean mask of states reachable by some tree (bottom-up fixpoint).
        The targets of an available child pair are the terminals of its
        diagram; the global default joins as soon as some available pair is
        absent from the table."""
        base = self.num_states + 1
        available = np.zeros(base, dtype=bool)
        available[self.BOT] = True
        left = self.pair_keys // base
        right = self.pair_keys % base
        default_seen = False
        while True:
            usable = available[left] & available[right]
            targets = [np.asarray(self.store.terminals(int(node)),
                                  dtype=np.int64)
                       for node in self.pair_nodes[usable]]
            targets = [t for t in targets if len(t)]
            new = np.zeros(base, dtype=bool)
            if targets:
                new[np.concatenate(targets)] = True
            if not default_seen:
                count = int(available.sum())
                if count * count > int(usable.sum()):
                    default_seen = True             # some pair is unlisted
                    new[self.default_state] = True
            new &= ~available
            if not new.any():
                break
            available |= new
        return available[:self.num_states]

    def is_empty(self) -> bool:
        reach = self.reachable_states()
        return not bool((reach & self.is_accepting).any())

    def _transitions(self, available: np.ndarray):
        """Yield ``(left, right, targets)`` for every child pair both of whose
        children are available, ``BOT`` included. Unlisted pairs fall to the
        global default, so they are enumerated too -- which is why this is
        quadratic in the number of available states and reserved for the
        analyses below rather than the hot pipeline."""
        base = self.num_states + 1
        listed = {int(k): int(n) for k, n in
                  zip(self.pair_keys, self.pair_nodes)}
        usable = [int(s) for s in np.flatnonzero(available)] + [self.BOT]
        for left in usable:
            for right in usable:
                node = listed.get(left * base + right)
                if node is None:
                    yield left, right, (self.default_state,)
                else:
                    yield left, right, self.store.terminals(node)

    def co_reachable_states(self, available: Optional[np.ndarray] = None
                            ) -> np.ndarray:
        """Boolean mask of states that can occur in an accepting run: a state
        is co-reachable if it is accepting (as the root) or it is a child in
        some transition whose target is co-reachable and whose sibling subtree
        exists. The top-down companion to `reachable_states`."""
        if available is None:
            available = self.reachable_states()
        co = self.is_accepting.copy()
        transitions = list(self._transitions(available))
        while True:
            new = False
            for left, right, targets in transitions:
                if not any(t < self.num_states and co[t] for t in targets):
                    continue
                for child in (left, right):
                    if child < self.num_states and not co[child]:
                        co[child] = True
                        new = True
            if not new:
                return co

    def is_finite(self) -> bool:
        """Whether the automaton accepts finitely many trees.

        A state that can occur strictly below itself pumps: the context
        between the two occurrences can be repeated without bound. So the
        language is infinite exactly when the "child of" graph, restricted to
        states that are both reachable and co-reachable, has a cycle.
        """
        available = self.reachable_states()
        usable = available & self.co_reachable_states(available)
        if not usable.any():
            return True                          # empty language

        successors = defaultdict(set)
        indegree = defaultdict(int)
        nodes = set(int(s) for s in np.flatnonzero(usable))
        for left, right, targets in self._transitions(available):
            for target in targets:
                if target >= self.num_states or not usable[target]:
                    continue
                for child in (left, right):
                    if child >= self.num_states or not usable[child]:
                        continue
                    if target not in successors[child]:
                        successors[child].add(target)
                        indegree[target] += 1

        # Kahn's algorithm: anything left over sits on a cycle.
        queue = deque(n for n in nodes if indegree[n] == 0)
        removed = 0
        while queue:
            node = queue.popleft()
            removed += 1
            for target in successors[node]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)
        return removed == len(nodes)

    def __repr__(self):
        return (f"SparseTreeAutomaton({self.num_states} states, "
                f"{len(self.pair_keys)} pairs, {self.num_nodes} nodes, "
                f"default={self.default_state})")
