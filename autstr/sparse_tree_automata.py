"""Sparse bottom-up tree automata.

The tree analog of `autstr.sparse_automata`: deterministic bottom-up automata
over binary trees (general trees embed via the first-child/next-sibling
encoding), stored sparsely and processed with batched numpy throughout.

**States.** Real states are 0..num_states-1; the virtual *absent* state
``BOT = num_states`` represents a missing child, so a single transition table
covers leaves (both children absent), unary and binary nodes:

    state(node) = delta(state(left) or BOT, state(right) or BOT, label(node)).

**Sparsity (two levels).** Transition lookup resolves in three tiers:

    delta(l, r, s) = exception(l, r, s)  ??  pair_default(l, r)  ??  default

Exceptions are parallel arrays ``(left, right, symbol) -> target`` kept
sorted by the packed integer key ``(left*(num_states+1) + right)*S + symbol``
(S = symbol-space size), so a batch of transitions is one
``np.searchsorted``; *pair defaults* are a sparse sorted table
``(left, right) -> target`` consulted on exception misses, and the global
``default_state`` (typically a rejecting sink) catches the rest.

Pair defaults are what keeps boolean combinations sparse: complementation
turns the rejecting sink into an accepting loop, and states that loop to
themselves on almost every symbol would otherwise need one exception per
symbol — fatal over large convolution alphabets. The representation is
closed under product on both levels: default x default = default, and the
pair default of a product pair is the pair of the factors' pair defaults, so
a product transition deviates from its pair default exactly where one of the
factors has a symbol exception — candidate enumeration stays exception-
driven. (This is the tree analog of the string engine's per-state
``default_states``.)

**Trees.** The convenience `Tree` class holds labelled binary trees; the
computational format is the *array tree*: post-order numpy arrays
``(labels, lefts, rights)`` with -1 for an absent child and encoded integer
labels. Conversions are iterative (no recursion limits).
"""
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

from autstr.utils.misc import decode_symbol, encode_symbol


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
    """Deterministic bottom-up tree automaton with sparse transitions.

    :param num_states: number of real states (0..num_states-1); the virtual
        absent-child state is ``BOT = num_states``.
    :param default_state: target of every transition not listed below.
    :param exc_left, exc_right, exc_symbol, exc_target: parallel arrays of
        exception transitions delta(exc_left, exc_right, exc_symbol) =
        exc_target. Children may be BOT; targets are real states.
    :param is_accepting: boolean array over the real states (acceptance is
        checked at the root).
    :param pd_left, pd_right, pd_target: parallel arrays of pair defaults
        delta(pd_left, pd_right, *) = pd_target for symbols without an
        exception. Children may be BOT; targets are real states. Pairs not
        listed fall back to the global default.
    """

    def __init__(self, num_states: int, default_state: int,
                 exc_left, exc_right, exc_symbol, exc_target,
                 is_accepting, symbol_arity: int = 1,
                 base_alphabet: Optional[Set] = None,
                 pd_left=(), pd_right=(), pd_target=()):
        self.num_states = int(num_states)
        self.default_state = int(default_state)
        self.is_accepting = np.asarray(is_accepting, dtype=bool)
        self.symbol_arity = symbol_arity
        self.base_alphabet = base_alphabet or {0}
        self.base_alphabet_frozen = frozenset(self.base_alphabet)

        left = np.asarray(exc_left, dtype=np.int64)
        right = np.asarray(exc_right, dtype=np.int64)
        symbol = np.asarray(exc_symbol, dtype=np.int64)
        target = np.asarray(exc_target, dtype=np.int64)

        order = np.argsort(self._keys(left, right, symbol), kind="stable")
        self.exc_left = left[order]
        self.exc_right = right[order]
        self.exc_symbol = symbol[order]
        self.exc_target = target[order]
        self._sorted_keys = self._keys(self.exc_left, self.exc_right,
                                       self.exc_symbol)

        pdl = np.asarray(pd_left, dtype=np.int64)
        pdr = np.asarray(pd_right, dtype=np.int64)
        pdt = np.asarray(pd_target, dtype=np.int64)
        order = np.argsort(pdl * (self.num_states + 1) + pdr, kind="stable")
        self.pd_left = pdl[order]
        self.pd_right = pdr[order]
        self.pd_target = pdt[order]
        self._pd_keys = self.pd_left * (self.num_states + 1) + self.pd_right

    # ---------------- basics ----------------

    @property
    def BOT(self) -> int:
        return self.num_states

    @property
    def num_symbols(self) -> int:
        return len(self.base_alphabet_frozen) ** self.symbol_arity

    def _keys(self, left, right, symbol):
        base = self.num_states + 1
        return (left * base + right) * self.num_symbols + symbol

    def pair_defaults(self, left, right) -> np.ndarray:
        """Batched pair-default lookup (global default on misses)."""
        left = np.asarray(left, dtype=np.int64)
        right = np.asarray(right, dtype=np.int64)
        keys = left * (self.num_states + 1) + right
        if len(self._pd_keys) == 0:
            return np.full(keys.shape, self.default_state, dtype=np.int64)
        pos = np.minimum(np.searchsorted(self._pd_keys, keys),
                         len(self._pd_keys) - 1)
        hit = self._pd_keys[pos] == keys
        return np.where(hit, self.pd_target[pos], self.default_state)

    def transitions(self, left, right, symbol) -> np.ndarray:
        """Batched transition lookup (binary search over exception keys,
        pair defaults on misses)."""
        left = np.asarray(left, dtype=np.int64)
        right = np.asarray(right, dtype=np.int64)
        symbol = np.asarray(symbol, dtype=np.int64)
        fallback = self.pair_defaults(left, right)
        if len(self._sorted_keys) == 0:
            return fallback
        keys = self._keys(left, right, symbol)
        pos = np.minimum(np.searchsorted(self._sorted_keys, keys),
                         len(self._sorted_keys) - 1)
        hit = self._sorted_keys[pos] == keys
        return np.where(hit, self.exc_target[pos], fallback)

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
        """Finish the remaining pending nodes with a plain post-order sweep
        (valid because children precede parents in the arrays)."""
        from bisect import bisect_left
        keys = self._sorted_keys.tolist()
        targets = self.exc_target.tolist()
        pd_keys = self._pd_keys.tolist()
        pd_targets = self.pd_target.tolist()
        n_keys = len(keys)
        n_pd = len(pd_keys)
        default = self.default_state
        base = self.num_states + 1
        S = self.num_symbols
        st = states.tolist()
        lab = labels.tolist()
        cl = child_l.tolist()
        cr = child_r.tolist()
        for i in np.flatnonzero(pending).tolist():
            pair = st[cl[i]] * base + st[cr[i]]
            key = pair * S + lab[i]
            pos = bisect_left(keys, key)
            if pos < n_keys and keys[pos] == key:
                st[i] = targets[pos]
            else:
                pos = bisect_left(pd_keys, pair)
                st[i] = pd_targets[pos] if pos < n_pd and pd_keys[pos] == pair \
                    else default
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
        return SparseTreeAutomaton(
            self.num_states, self.default_state,
            self.exc_left, self.exc_right, self.exc_symbol, self.exc_target,
            ~self.is_accepting, self.symbol_arity, self.base_alphabet,
            self.pd_left, self.pd_right, self.pd_target)

    def intersection(self, other) -> "SparseTreeAutomaton":
        return self._product(other, np.logical_and)

    def union(self, other) -> "SparseTreeAutomaton":
        return self._product(other, np.logical_or)

    def _exception_groups(self):
        """Group the (sorted) exception table by (left, right): returns the
        group keys ``left*(n+1)+right`` (sorted, unique) and the start/end
        offsets of each group in the exception arrays."""
        base = self.num_states + 1
        pair_keys = self.exc_left * base + self.exc_right
        starts = np.flatnonzero(np.r_[True, pair_keys[1:] != pair_keys[:-1]]) \
            if len(pair_keys) else np.array([], dtype=np.int64)
        ends = np.r_[starts[1:], len(pair_keys)] if len(starts) else starts
        return pair_keys[starts] if len(starts) else pair_keys, starts, ends

    def _product(self, other: "SparseTreeAutomaton", combine) -> "SparseTreeAutomaton":
        if self.symbol_arity != other.symbol_arity:
            raise ValueError("product requires the same symbol arity")
        if self.base_alphabet_frozen != other.base_alphabet_frozen:
            raise ValueError("product requires the same base alphabet")

        # Product states are pairs (a, b) of real states; the product BOT is
        # (BOT_a, BOT_b), the product default is (default_a, default_b) and
        # the pair default of a product pair is the pair of the factors' pair
        # defaults: unlisted transitions resolve tier by tier in both factors,
        # so the representation is closed on all levels. Discovery is a
        # bottom-up reachability fixpoint: child slots range over discovered
        # pairs plus the BOT pair, and a product transition deviates from its
        # combo's pair default only where some factor has a symbol exception.
        nb = other.num_states
        BOT_pair_key = self.BOT * (nb + 1) + other.BOT

        def pair_key(a, b):
            return a * (nb + 1) + b

        keys_a, starts_a, ends_a = self._exception_groups()
        keys_b, starts_b, ends_b = other._exception_groups()

        state_ids = {}
        pairs_a: List[int] = []
        pairs_b: List[int] = []

        def get_id(a, b):
            key = pair_key(int(a), int(b))
            idx = state_ids.get(key)
            if idx is None:
                idx = state_ids[key] = len(pairs_a)
                pairs_a.append(int(a))
                pairs_b.append(int(b))
            return idx

        default_id = get_id(self.default_state, other.default_state)

        # exception/pair-default output, accumulated as numpy chunks (python
        # lists of ints would dominate memory on million-row tables)
        exc_chunks: List[np.ndarray] = []
        pd_l: List[int] = []
        pd_r: List[int] = []
        pd_t: List[int] = []

        # child options: product ids, or -1 encoding the BOT pair
        new_options = [-1, default_id]
        all_options = []

        def component_states(option_ids):
            ids = np.asarray(option_ids, dtype=np.int64)
            a = np.where(ids < 0, self.BOT, np.array(pairs_a, dtype=np.int64)[
                np.maximum(ids, 0)])
            b = np.where(ids < 0, other.BOT, np.array(pairs_b, dtype=np.int64)[
                np.maximum(ids, 0)])
            return a, b

        while new_options:
            # each combo enumerated exactly once, in the round where its later
            # member was discovered (no quadratic dedup set)
            round_new = new_options
            new_options = []
            combos = [(x, y) for x in round_new for y in all_options]
            combos += [(y, x) for x in round_new for y in all_options]
            combos += [(x, y) for x in round_new for y in round_new]
            all_options.extend(round_new)
            if not combos:
                break

            cl = np.array([c[0] for c in combos], dtype=np.int64)
            cr = np.array([c[1] for c in combos], dtype=np.int64)
            la, lb = component_states(cl)
            ra, rb = component_states(cr)

            pda = self.pair_defaults(la, ra)
            pdb = other.pair_defaults(lb, rb)
            # symbols with an exception in either factor, per combo
            ga = np.searchsorted(keys_a, la * (self.num_states + 1) + ra)
            gb = np.searchsorted(keys_b, lb * (other.num_states + 1) + rb)
            for i in range(len(combos)):
                before = len(pairs_a)
                pd_id = get_id(pda[i], pdb[i])
                if len(pairs_a) > before:
                    new_options.append(pd_id)
                if pd_id != default_id:
                    pd_l.append(int(cl[i]))     # -1 encodes the BOT pair
                    pd_r.append(int(cr[i]))
                    pd_t.append(pd_id)

                syms = []
                if ga[i] < len(keys_a) and keys_a[ga[i]] == la[i] * (self.num_states + 1) + ra[i]:
                    syms.append(self.exc_symbol[starts_a[ga[i]]:ends_a[ga[i]]])
                if gb[i] < len(keys_b) and keys_b[gb[i]] == lb[i] * (other.num_states + 1) + rb[i]:
                    syms.append(other.exc_symbol[starts_b[gb[i]]:ends_b[gb[i]]])
                if not syms:
                    continue
                symbols = np.unique(np.concatenate(syms))
                ta = self.transitions(np.full(len(symbols), la[i]),
                                      np.full(len(symbols), ra[i]), symbols)
                tb = other.transitions(np.full(len(symbols), lb[i]),
                                       np.full(len(symbols), rb[i]), symbols)
                deviates = (ta != pda[i]) | (tb != pdb[i])
                if not deviates.any():
                    continue
                sel_syms = symbols[deviates]
                keys = ta[deviates] * (nb + 1) + tb[deviates]
                # bulk target-id assignment: python only per distinct new pair
                uniq, inverse = np.unique(keys, return_inverse=True)
                uids = np.empty(len(uniq), dtype=np.int64)
                for j, key in enumerate(uniq.tolist()):
                    idx = state_ids.get(key)
                    if idx is None:
                        idx = state_ids[key] = len(pairs_a)
                        pairs_a.append(key // (nb + 1))
                        pairs_b.append(key % (nb + 1))
                        new_options.append(idx)
                    uids[j] = idx
                chunk = np.empty((4, len(sel_syms)), dtype=np.int64)
                chunk[0] = cl[i]                # -1 encodes the BOT pair
                chunk[1] = cr[i]
                chunk[2] = sel_syms
                chunk[3] = uids[inverse]
                exc_chunks.append(chunk)

        num_states = len(pairs_a)
        exc = np.concatenate(exc_chunks, axis=1) if exc_chunks else \
            np.empty((4, 0), dtype=np.int64)
        # rewrite the -1 BOT sentinels now that the state count is known
        exc_l_arr = np.where(exc[0] < 0, num_states, exc[0])
        exc_r_arr = np.where(exc[1] < 0, num_states, exc[1])
        pd_l_arr = np.where(np.array(pd_l, dtype=np.int64) < 0,
                            num_states, np.array(pd_l, dtype=np.int64))
        pd_r_arr = np.where(np.array(pd_r, dtype=np.int64) < 0,
                            num_states, np.array(pd_r, dtype=np.int64))

        acc = combine(self.is_accepting[np.array(pairs_a)],
                      other.is_accepting[np.array(pairs_b)])
        return SparseTreeAutomaton(
            num_states, default_id, exc_l_arr, exc_r_arr,
            exc[2], exc[3],
            acc, self.symbol_arity, self.base_alphabet,
            pd_l_arr, pd_r_arr, np.array(pd_t, dtype=np.int64))

    # ---------------- emptiness ----------------

    def reachable_states(self) -> np.ndarray:
        """Boolean mask of states reachable by some tree (bottom-up fixpoint).
        Sparse-aware: a pair default (or the global default) is reachable as
        soon as some available child pair leaves a symbol un-excepted. Pair
        defaults are examined once per pair, in the round where the pair's
        later member became available."""
        available = np.zeros(self.num_states + 1, dtype=bool)
        available[self.BOT] = True
        keys, starts, ends = self._exception_groups()
        counts = ends - starts if len(starts) else np.array([], dtype=np.int64)
        base = self.num_states + 1

        frontier = np.array([self.BOT], dtype=np.int64)
        while len(frontier):
            av = np.flatnonzero(available)
            new = frontier
            pl = np.concatenate([np.repeat(new, len(av)),
                                 np.tile(av, len(new))])
            pr = np.concatenate([np.tile(av, len(new)),
                                 np.repeat(new, len(av))])
            pk = pl * base + pr
            covered = np.zeros(len(pk), dtype=np.int64)
            if len(keys):
                pos = np.searchsorted(keys, pk)
                valid = pos < len(keys)
                match = valid.copy()
                match[valid] = keys[pos[valid]] == pk[valid]
                covered[match] = counts[pos[match]]
            cand = self.pair_defaults(pl, pr)[covered < self.num_symbols]
            newly = np.unique(cand[~available[cand]]) if len(cand) else cand
            # exception targets with available children
            mask = available[self.exc_left] & available[self.exc_right] & \
                ~available[self.exc_target]
            if mask.any():
                newly = np.unique(np.concatenate(
                    [newly, self.exc_target[mask]]))
            available[newly] = True
            frontier = newly
        return available[:self.num_states]

    def is_empty(self) -> bool:
        reach = self.reachable_states()
        return not bool((reach & self.is_accepting).any())

    def __repr__(self):
        return (f"SparseTreeAutomaton({self.num_states} states, "
                f"{len(self.exc_target)} exceptions, "
                f"{len(self.pd_target)} pair defaults, "
                f"default={self.default_state})")
