"""The first-order pipeline for sparse tree automata: cylindrification
(expand), existential projection, padding closure, and minimization.

Design notes.

*Symbols are variable assignments.* Every transition is a multi-terminal BDD
over the binary digits of the convolution symbol, tape-major (see
`autstr.mtbdd`), which is what makes the pipeline affordable:

- `expand` renames variable blocks. The tapes it adds are simply never tested,
  so a k-tape transition widened to k+j tapes costs *nothing* — no row is
  duplicated once per letter of each new tape. Sending two source tapes to the
  same target block substitutes one variable block for the other, which is how
  a relation R(x, x) is formed.
- `project` quantifies one tape's variable block: the m cofactors of a
  transition are combined by set union, giving the nondeterministic transition
  as a diagram over *sets* of states, and the subset construction then folds
  those diagrams over the members of each child subset. No symbol is ever
  enumerated, and no "does this pair except all m preimages?" counting is
  needed — invalid binary codes carry the reserved NONE terminal.
- `minimize` refines over diagram identity: hash-consing means two states have
  the same behavior on a child pair exactly when the class-relabelled diagrams
  of that pair are the same integer.

*Padding has two directions.*

- `project` handles the *absent* direction: projecting away tape i turns the
  automaton nondeterministic **and** changes the domain semantics: the
  ∃-witness tree may extend below the remaining tapes' domains, leaving
  regions labelled all-padding that are trimmed from the projected
  convolution. In a bottom-up run of the trimmed tree an absent child may
  therefore correspond to any state reachable by some pure-padding tree (the
  padding closure P0), so the subset determinization runs with the
  absent-child subset S⊥ = {BOT} ∪ P0.
- `attach_padding` handles the *present* direction: it accepts every tree of
  the language with arbitrary all-padding regions attached below, by making
  such regions behave exactly like absent children (a single fresh PAD state,
  no subset construction — the source is deterministic). It must be applied
  before `expand` widens a relation to more tapes, because the wider
  convolution's domain is the union of all tapes' domains.

*Sparsity.* A child pair absent from the transition table sends every symbol
to the global default. Products, projections and minimization all drop a pair
again as soon as its diagram is the default constant, so the pair tables stay
driven by genuine deviations.
"""
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from autstr.mtbdd import NONE, ComputedTable, bits_of, var_tables
from autstr.sparse_tree_automata import SparseTreeAutomaton
from autstr.utils.misc import encode_symbol


def _symbol_assignment(symbol: int, arity: int, m: int, bits: int) -> List[int]:
    """The binary variable assignment of a convolution symbol."""
    div, shift = var_tables(arity, m, bits)
    return [int((symbol // div[v]) % m) >> int(shift[v]) & 1
            for v in range(arity * bits)]


# ====================================================================
# Cylindrification
# ====================================================================

def expand(sta: SparseTreeAutomaton, new_arity: int, pos: List[int]
           ) -> SparseTreeAutomaton:
    """Expand a k-tape automaton to new_arity tapes, placing original tape t
    at position pos[t]; the remaining positions range over all letters.

    This is a variable renaming on the transition diagrams: the new tapes'
    variables do not occur, so the automaton ignores them. Repeated entries in
    `pos` identify tapes (the diagonal of the relation).

    Note: like the string `expand`, this widens only the *alphabet*; apply
    `attach_padding` first so regions contributed solely by the new tapes
    (all-padding on the original tapes) are accepted.
    """
    store = sta.store
    bits, m = sta.bits, sta.m
    varmap = [pos[v // bits] * bits + v % bits for v in range(sta.nvars)]
    valid = store.const(0, new_arity, m, bits)

    def keep_valid(target: int, ok: int) -> int:
        # the new tapes are unconstrained by the source, so restrict their
        # invalid binary codes explicitly
        return NONE if (target == NONE or ok == NONE) else target

    rename_cache: Dict[int, int] = {}
    mask_cache: Dict[int, int] = {}
    default_node = store.const(sta.default_state, new_arity, m, bits)

    keys, nodes = [], []
    for key, node in zip(sta.pair_keys.tolist(), sta.pair_nodes.tolist()):
        renamed = store.rename(node, varmap, rename_cache)
        renamed = store.apply2(renamed, valid, keep_valid, mask_cache)
        if renamed != default_node:
            keys.append(key)
            nodes.append(renamed)

    return SparseTreeAutomaton(
        sta.num_states, sta.default_state, is_accepting=sta.is_accepting,
        symbol_arity=new_arity, base_alphabet=sta.base_alphabet,
        pair_keys=np.array(keys, dtype=np.int64),
        pair_nodes=np.array(nodes, dtype=np.int64))


def permute_tapes(sta: SparseTreeAutomaton, perm: List[int]
                  ) -> SparseTreeAutomaton:
    """Reorder the tapes of a convolution automaton: new tape i carries what
    was tape perm[i]."""
    k = sta.symbol_arity
    if sorted(perm) != list(range(k)):
        raise ValueError(f"perm must be a permutation of range({k})")
    inverse = [0] * k
    for new_tape, old_tape in enumerate(perm):
        inverse[old_tape] = new_tape
    bits = sta.bits
    varmap = [inverse[v // bits] * bits + v % bits for v in range(sta.nvars)]
    cache: Dict[int, int] = {}
    nodes = [sta.store.rename(node, varmap, cache)
             for node in sta.pair_nodes.tolist()]
    return SparseTreeAutomaton(
        sta.num_states, sta.default_state, is_accepting=sta.is_accepting,
        symbol_arity=k, base_alphabet=sta.base_alphabet,
        pair_keys=sta.pair_keys, pair_nodes=np.array(nodes, dtype=np.int64))


# ====================================================================
# Existential projection
# ====================================================================

def project(sta: SparseTreeAutomaton, tape: int, padding_symbol,
            max_states: Optional[int] = None) -> SparseTreeAutomaton:
    """Existentially quantify one tape: accept the convolution of the
    remaining tapes iff some witness tree exists on the projected tape
    (including witnesses whose domain extends below the remaining tapes,
    which is what the padding closure of the absent-child set captures).

    Subset construction is worst-case exponential; `max_states` aborts with a
    clear error instead of exhausting memory."""
    k = sta.symbol_arity
    if k < 2:
        raise ValueError("cannot project the only tape")
    if not 0 <= tape < k:
        raise ValueError(f"tape must be in [0, {k})")

    store = sta.store
    m, bits = sta.m, sta.bits
    n, BOT = sta.num_states, sta.BOT
    new_arity = k - 1

    # ---- subsets of source states, as diagram terminals (integer bitsets:
    # every union result is interned, intermediates included) ----
    subsets: List[int] = []
    subset_ids: Dict[int, int] = {}

    def subset_id(mask: int) -> int:
        idx = subset_ids.get(mask)
        if idx is None:
            idx = subset_ids[mask] = len(subsets)
            subsets.append(mask)
        return idx

    def singleton(target: int) -> int:
        return subset_id(1 << target)

    def union(a: int, b: int) -> int:
        if a == NONE or b == NONE:
            return NONE
        return subset_id(subsets[a] | subsets[b])

    # ---- the nondeterministic transition of each source pair ----
    # relabel targets to singletons, union the quantified tape's m cofactors,
    # then drop the tape's variable block
    varmap = [(v // bits - (1 if v // bits > tape else 0)) * bits + v % bits
              for v in range(sta.nvars)]
    singleton_cache: Dict[int, int] = {}
    union_cache = ComputedTable(23)
    rename_cache: Dict[int, int] = {}

    def nondeterministic(node: int) -> int:
        node = store.apply1(node, singleton, singleton_cache)
        node = store.quantify_letter(node, tape, m, bits, union, union_cache)
        return store.rename(node, varmap, rename_cache)

    default_set_node = nondeterministic(sta.default_node)
    set_nodes = np.full((n + 1, n + 1), default_set_node, dtype=np.int64)
    for key, node in zip(sta.pair_keys.tolist(), sta.pair_nodes.tolist()):
        set_nodes[key // (n + 1), key % (n + 1)] = nondeterministic(node)

    # ---- the absent-child subset: BOT plus the padding closure ----
    pad_new = encode_symbol((padding_symbol,) * new_arity,
                            sta.base_alphabet_frozen)
    closure = 0                                    # bitset of the padding closure
    while True:
        available = np.array(bits_of(closure | (1 << BOT)), dtype=np.int64)
        left = np.repeat(available, len(available))
        right = np.tile(available, len(available))
        reached = store.eval_batch(set_nodes[left, right],
                                   np.full(len(left), pad_new, dtype=np.int64),
                                   new_arity, m, bits)
        grown = closure
        for sid in reached.tolist():
            grown |= subsets[sid]
        if grown == closure:
            break
        closure = grown
    absent = closure | (1 << BOT)

    # ---- subset construction over the new states ----
    state_subsets: List[int] = []
    state_ids: Dict[int, int] = {}
    fresh: List[int] = []

    def state_of(sid: int) -> int:
        mask = subsets[sid]
        idx = state_ids.get(mask)
        if idx is None:
            if max_states is not None and len(state_subsets) >= max_states:
                raise RuntimeError(
                    f"subset determinization exceeded max_states={max_states}")
            idx = state_ids[mask] = len(state_subsets)
            state_subsets.append(mask)
            fresh.append(idx)
        return idx

    default_id = state_of(singleton(sta.default_state))
    fresh.clear()                                  # the default is not a child
    default_node = store.const(default_id, new_arity, m, bits)
    state_cache: Dict[int, int] = {}

    member_cache: Dict[int, np.ndarray] = {}

    def members(option: int) -> np.ndarray:
        arr = member_cache.get(option)
        if arr is None:
            mask = absent if option < 0 else state_subsets[option]
            arr = member_cache[option] = np.array(bits_of(mask), dtype=np.int64)
        return arr

    keys: List[Tuple[int, int]] = []
    nodes: List[int] = []
    new_options = [-1, default_id]                 # -1 encodes the absent set
    all_options: List[int] = []

    while new_options:
        # each combo is enumerated exactly once, in the round where its later
        # member was discovered — no dedup set (whose quadratic growth would
        # dominate memory long before the state cap)
        round_new = new_options
        combos = [(x, y) for x in round_new for y in all_options]
        combos += [(y, x) for x in round_new for y in all_options]
        combos += [(x, y) for x in round_new for y in round_new]
        all_options.extend(round_new)
        fresh.clear()

        for left_option, right_option in combos:
            grid = set_nodes[np.ix_(members(left_option),
                                    members(right_option))]
            # hash-consing collapses most members onto the same diagram, and
            # `apply2` memoizes the folds, so only the distinct ones cost
            distinct = np.unique(grid).tolist()
            node = distinct[0]
            for other in distinct[1:]:
                node = store.apply2(node, other, union, union_cache)
            node = store.apply1(node, state_of, state_cache)
            if node != default_node:
                keys.append((left_option, right_option))
                nodes.append(node)
        new_options = list(fresh)

    num_states = len(state_subsets)
    accepting_mask = 0
    for q in np.flatnonzero(sta.is_accepting).tolist():
        accepting_mask |= 1 << q
    packed = [(num_states if l < 0 else l) * (num_states + 1) +
              (num_states if r < 0 else r) for l, r in keys]
    return SparseTreeAutomaton(
        num_states, default_id,
        is_accepting=[bool(mask & accepting_mask) for mask in state_subsets],
        symbol_arity=new_arity, base_alphabet=sta.base_alphabet,
        pair_keys=np.array(packed, dtype=np.int64),
        pair_nodes=np.array(nodes, dtype=np.int64))


# ====================================================================
# Padding
# ====================================================================

def attach_padding(sta: SparseTreeAutomaton, padding_symbol,
                   max_states: Optional[int] = None) -> SparseTreeAutomaton:
    """Accept exactly the trees whose maximal all-padding subtrees, once
    trimmed away, the source accepts — the tree analog of the string
    pipeline's `pad`, required before `expand` widens the convolution (the
    wider convolution's domain is the union of all tapes' domains, so the
    original tapes see attached regions as padding).

    The source is deterministic, so no subset construction is needed: one
    fresh PAD state absorbs pure-padding regions, and every child pair with an
    absent child gains a copy with PAD in that position, making a padding
    region behave exactly like an absent child. Any native transitions the
    source had on all-padding leaves are overridden — canonical convolutions
    contain no all-padding node, so those transitions carry no meaning."""
    store = sta.store
    k, m, bits = sta.symbol_arity, sta.m, sta.bits
    n, old_bot = sta.num_states, sta.BOT
    PAD, BOT = n, n + 1
    old_base, base = n + 1, n + 2

    pad_assignment = _symbol_assignment(
        encode_symbol((padding_symbol,) * k, sta.base_alphabet_frozen),
        k, m, bits)

    pairs: Dict[int, int] = {}
    for key, node in zip(sta.pair_keys.tolist(), sta.pair_nodes.tolist()):
        left, right = key // old_base, key % old_base
        lefts = (BOT, PAD) if left == old_bot else (left,)
        rights = (BOT, PAD) if right == old_bot else (right,)
        for a in lefts:
            for b in rights:
                pairs[a * base + b] = node

    for a in (BOT, PAD):                           # pure padding starts here
        for b in (BOT, PAD):
            key = a * base + b
            pairs[key] = store.set_path(pairs.get(key, sta.default_node),
                                        pad_assignment, PAD)

    listed = [(key, node) for key, node in pairs.items()
              if node != sta.default_node]
    return SparseTreeAutomaton(
        n + 1, sta.default_state,
        is_accepting=np.r_[sta.is_accepting, False], symbol_arity=k,
        base_alphabet=sta.base_alphabet,
        pair_keys=np.array([p for p, _ in listed], dtype=np.int64),
        pair_nodes=np.array([node for _, node in listed], dtype=np.int64))


# ====================================================================
# Single-tree and string-language automata
# ====================================================================

def canonical(sta: SparseTreeAutomaton, padding_symbol) -> SparseTreeAutomaton:
    """Keep only the canonical convolution of each tuple: trees in which no
    node is padding on *every* tape.

    `attach_padding` deliberately accepts each tuple with arbitrary
    all-padding regions hanging below it, so the *tree* language of a
    saturated relation automaton is infinite as soon as the relation is
    non-empty. Restricting to canonical trees first is what makes finiteness
    and counting questions be about tuples rather than about trees -- the tree
    analog of `automata_tools.canonical`.
    """
    store = sta.store
    k, m, bits = sta.symbol_arity, sta.m, sta.bits
    CLEAN, DEAD, BOT = 0, 1, 2
    base = 3

    pad_assignment = _symbol_assignment(
        encode_symbol((padding_symbol,) * k, sta.base_alphabet_frozen),
        k, m, bits)

    # Stay clean on every symbol but the all-padding one; once dead, dead.
    clean = store.set_path(store.const(CLEAN, k, m, bits),
                           pad_assignment, DEAD)
    dead = store.const(DEAD, k, m, bits)

    keys, nodes = [], []
    for left in (CLEAN, DEAD, BOT):
        for right in (CLEAN, DEAD, BOT):
            keys.append(left * base + right)
            nodes.append(dead if DEAD in (left, right) else clean)

    no_padding = SparseTreeAutomaton(
        num_states=2, default_state=DEAD, is_accepting=[True, False],
        symbol_arity=k, base_alphabet=sta.base_alphabet,
        pair_keys=keys, pair_nodes=nodes)
    return minimize(sta.intersection(no_padding))


def tree_automaton(tree, base_alphabet, symbol_arity: int = 1
                   ) -> SparseTreeAutomaton:
    """Automaton accepting exactly the given tree. Subtrees are hash-consed
    (one state per distinct subtree), so equal keys always share a target and
    the exception table stays deterministic; state 0 is the dead default."""
    from autstr.sparse_tree_automata import tree_to_arrays
    base = frozenset(base_alphabet)
    labels, lefts, rights = tree_to_arrays(tree, base, symbol_arity)

    ids: Dict[tuple, int] = {}
    state_of: List[int] = []
    for i in range(len(labels)):
        l = state_of[lefts[i]] if lefts[i] >= 0 else -1
        r = state_of[rights[i]] if rights[i] >= 0 else -1
        key = (l, r, int(labels[i]))
        s = ids.get(key)
        if s is None:
            s = ids[key] = len(ids) + 1
        state_of.append(s)

    n = len(ids) + 1
    exc = sorted((n if l < 0 else l, n if r < 0 else r, sym, t)
                 for (l, r, sym), t in ids.items())
    acc = np.zeros(n, dtype=bool)
    acc[state_of[-1]] = True                    # root is last in post-order
    return SparseTreeAutomaton(
        n, 0,
        [e[0] for e in exc], [e[1] for e in exc],
        [e[2] for e in exc], [e[3] for e in exc],
        acc, symbol_arity, set(base_alphabet))


def string_chain(word):
    """Embed a word as a unary left-spine tree: the first letter labels the
    root, each next letter its left child. Chain convolution then aligns
    positions from the root and pads at the bottom — exactly the string
    convolution convention."""
    from autstr.sparse_tree_automata import Tree
    node = None
    for letter in reversed(list(word)):
        node = Tree(letter, node, None)
    if node is None:
        raise ValueError("cannot embed the empty word as a tree")
    return node


def from_string_dfa(dfa) -> SparseTreeAutomaton:
    """Embed a string DFA's language as chain trees (see `string_chain`).

    A bottom-up run reads the chain from the last letter to the first, so
    this is the reversal-determinization of the DFA: the tree state after a
    suffix v is the set {p : reading v from p reaches acceptance}, computed
    with pre-images over a dense next-state table (validation-scale sizes).
    The root accepts iff the DFA's start state lies in the set."""
    n = dfa.num_states
    S = dfa.num_symbols
    table = dfa.dense_next()                     # validation-scale sizes

    accepting = np.asarray(dfa.is_accepting, dtype=bool)

    def pre_sets(subset: FrozenSet[int]):
        member = np.zeros(n, dtype=bool)
        member[list(subset)] = True
        in_t = member[table]                     # (n, S)
        return [frozenset(np.flatnonzero(in_t[:, a]).tolist())
                for a in range(S)]

    ids: Dict[FrozenSet[int], int] = {}
    order: List[FrozenSet[int]] = []

    def get_id(fs):
        if fs not in ids:
            ids[fs] = len(order)
            order.append(fs)
        return ids[fs]

    dead = get_id(frozenset())                   # the default sink
    f_set = frozenset(np.flatnonzero(accepting).tolist())

    triples = []
    frontier = [None]                            # None encodes the leaf case
    processed = set()
    while frontier:
        current = frontier.pop()
        if current in processed:
            continue
        processed.add(current)
        base_set = f_set if current is None else order[current]
        for a, pre in enumerate(pre_sets(base_set)):
            if not pre:
                continue                         # empty set == dead default
            before = len(order)
            tid = get_id(pre)
            if len(order) > before:
                frontier.append(tid)
            child = -1 if current is None else current
            triples.append((child, a, tid))

    num_states = len(order)
    BOT = num_states
    exc = sorted((BOT if c < 0 else c, BOT, a, t) for c, a, t in triples)
    is_acc = [dfa.start_state in fs for fs in order]
    return SparseTreeAutomaton(
        num_states, dead,
        [e[0] for e in exc], [e[1] for e in exc],
        [e[2] for e in exc], [e[3] for e in exc],
        is_acc, dfa.symbol_arity, dfa.base_alphabet)


# ====================================================================
# Minimization
# ====================================================================

_M1 = np.uint64(0x9E3779B97F4A7C15)
_M2 = np.uint64(0xC2B2AE3D27D4EB4F)
_M3 = np.uint64(0x165667B19E3779F9)


def _digest(values: np.ndarray, seed: int) -> np.ndarray:
    h = values.astype(np.uint64) * _M1 + np.uint64(seed)
    h ^= h >> np.uint64(33)
    h *= _M2
    h ^= h >> np.uint64(29)
    h *= _M3
    h ^= h >> np.uint64(32)
    return h


def _entry_hashes(side: np.ndarray, partner: np.ndarray,
                  node: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Two independent 64-bit hashes of the refinement entries
    ``(side, partner, class-relabelled diagram)``."""
    packed = (node.astype(np.uint64) * _M1 +
              partner.astype(np.uint64) * _M2 +
              side.astype(np.uint64) * _M3)
    return _digest(packed, 0x5851F42D4C957F2D), _digest(packed, 0x14057B7EF767814F)


def minimize(sta: SparseTreeAutomaton) -> SparseTreeAutomaton:
    """Minimize by Moore refinement over class-relabelled transition diagrams.

    A state q is characterized by, for each side and each partner class c, the
    function symbol -> target class it induces together with any partner in c.
    Relabelling a pair's diagram by the current classes turns that function
    into a hash-consed node, so a state's signature is the *set* of triples
    ``(side, partner class, node)`` — one `apply1` per listed pair per round.
    Child pairs absent from the table contribute the default's class on every
    symbol, and are folded in by counting how much of each class a state has
    listed. Unreachable states are pruned first."""
    store = sta.store
    k, m, bits = sta.symbol_arity, sta.m, sta.bits

    reach = sta.reachable_states()
    keep = np.flatnonzero(reach)
    if len(keep) == 0:
        return SparseTreeAutomaton(1, 0, is_accepting=[False], symbol_arity=k,
                                   base_alphabet=sta.base_alphabet)

    old_base = sta.num_states + 1
    new_of_old = np.full(sta.num_states + 1, -1, dtype=np.int64)
    new_of_old[keep] = np.arange(len(keep))
    n = len(keep)
    accepting = sta.is_accepting[keep]
    default = int(new_of_old[sta.default_state])
    if default < 0:
        # the default was unreachable (every available pair is listed, so no
        # kept diagram mentions it): give it a fresh dead state
        default = n
        n += 1
        accepting = np.r_[accepting, False]
        new_of_old[sta.default_state] = default
    BOT = n
    new_of_old[sta.BOT] = BOT

    relabel_cache: Dict[int, int] = {}
    pair_left, pair_right, pair_nodes = [], [], []
    for key, node in zip(sta.pair_keys.tolist(), sta.pair_nodes.tolist()):
        left, right = new_of_old[key // old_base], new_of_old[key % old_base]
        if left < 0 or right < 0:
            continue
        pair_left.append(int(left))
        pair_right.append(int(right))
        pair_nodes.append(store.apply1(node, lambda t: int(new_of_old[t]),
                                       relabel_cache))
    pair_left = np.array(pair_left, dtype=np.int64)
    pair_right = np.array(pair_right, dtype=np.int64)
    pair_nodes = np.array(pair_nodes, dtype=np.int64)

    # entries: each listed pair is seen from both sides
    owner = np.concatenate([pair_left, pair_right])
    partner = np.concatenate([pair_right, pair_left])
    side = np.concatenate([np.zeros(len(pair_left), dtype=np.int64),
                           np.ones(len(pair_right), dtype=np.int64)])
    node_of_entry = np.concatenate([pair_nodes, pair_nodes])
    real = owner < BOT
    owner, partner, side, node_of_entry = (owner[real], partner[real],
                                           side[real], node_of_entry[real])

    classes = accepting.astype(np.int64)
    num_classes = len(np.unique(classes))
    while True:
        round_cache: Dict[int, int] = {}
        relabelled = np.array(
            [store.apply1(int(node), lambda t: int(classes[t]), round_cache)
             for node in pair_nodes], dtype=np.int64) if len(pair_nodes) \
            else np.empty(0, dtype=np.int64)
        entry_node = np.concatenate([relabelled, relabelled])[real]
        default_node = store.const(int(classes[default]), k, m, bits)

        # An entry names the *concrete* partner, not its class. Keying by the
        # partner's class and comparing the resulting sets is strictly weaker:
        # a state sending partner p to X and p' to Y (both of class c) would
        # get the same entry set as one that swaps them, and the refinement
        # can then stabilize on a partition that is not a congruence.
        # Pairs a state does not list, and listed pairs whose relabelled
        # diagram is the default constant, behave identically for every state,
        # so only the deviating entries carry information.
        sig1 = np.zeros(n, dtype=np.uint64)
        sig2 = np.zeros(n, dtype=np.uint64)
        with np.errstate(over='ignore'):
            deviates = entry_node != default_node
            if deviates.any():
                rows = np.unique(np.stack([owner[deviates], side[deviates],
                                           partner[deviates],
                                           entry_node[deviates]], axis=1),
                                 axis=0)
                h1, h2 = _entry_hashes(rows[:, 1], rows[:, 2], rows[:, 3])
                np.add.at(sig1, rows[:, 0], h1)
                np.bitwise_xor.at(sig2, rows[:, 0], h2)

        signature = np.stack([classes.astype(np.uint64), sig1, sig2], axis=1)
        _, refined = np.unique(signature, axis=0, return_inverse=True)
        refined = refined.astype(np.int64)
        stable = len(np.unique(refined)) == num_classes
        classes = refined
        num_classes = len(np.unique(classes))
        if stable:
            break

    # ---- rebuild on the classes ----
    P = num_classes
    classes_full = np.r_[classes, P]
    default_class = int(classes[default])
    default_node = store.const(default_class, k, m, bits)
    final_cache: Dict[int, int] = {}
    keys, nodes = [], []
    seen: Set[int] = set()
    for left, right, node in zip(pair_left.tolist(), pair_right.tolist(),
                                 pair_nodes.tolist()):
        key = int(classes_full[left]) * (P + 1) + int(classes_full[right])
        if key in seen:
            continue
        seen.add(key)
        node = store.apply1(node, lambda t: int(classes[t]), final_cache)
        if node != default_node:
            keys.append(key)
            nodes.append(node)

    new_accepting = np.zeros(P, dtype=bool)
    new_accepting[classes[np.flatnonzero(accepting)]] = True
    return SparseTreeAutomaton(
        P, default_class, is_accepting=new_accepting, symbol_arity=k,
        base_alphabet=sta.base_alphabet,
        pair_keys=np.array(keys, dtype=np.int64),
        pair_nodes=np.array(nodes, dtype=np.int64))


# ====================================================================
# Equivalence (exact, via boolean closure + emptiness)
# ====================================================================

def equivalent(a: SparseTreeAutomaton, b: SparseTreeAutomaton) -> bool:
    return a.intersection(b.complement()).is_empty() and \
        b.intersection(a.complement()).is_empty()
