"""The first-order pipeline for sparse tree automata: cylindrification
(expand), existential projection, padding closure, and minimization.

Design notes.

*Padding has two directions.*

- `project` handles the *absent* direction: projecting away tape i turns the
  automaton nondeterministic (several source symbols map to the same
  projected symbol) **and** changes the domain semantics: the ∃-witness tree
  may extend below the remaining tapes' domains, leaving regions labelled
  all-padding that are trimmed from the projected convolution. In a bottom-up
  run of the trimmed tree, an absent child may therefore correspond to any
  state reachable by some pure-padding tree (the padding closure P0), so the
  subset determinization runs with the absent-child subset S⊥ = {BOT} ∪ P0.
- `attach_padding` handles the *present* direction: it accepts every tree of
  the language with arbitrary all-padding regions attached below, by making
  such regions behave exactly like absent children (a single fresh PAD state,
  no subset construction — the source is deterministic). It must be applied
  before `expand` widens a relation to more tapes, because the wider
  convolution's domain is the union of all tapes' domains.

*Sparsity is preserved on both levels.* The subset automaton's global
default is the subset {d} of the source default; each child combo's pair
default is the set of its members' pair defaults; and a subset transition
deviates from that only where some source exception applies, so candidates
are generated from the exception tables alone.

*Minimization* is Moore refinement over exception signatures: transitions not
listed as exceptions go to the global default irrespective of the state, so
two states can only be distinguished by exceptions that mention them. A
state's signature is its set of (side, partner class, symbol, target class)
entries, filtered to targets outside the default's class; refinement is
O(#exceptions) per round.
"""
from itertools import chain
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from autstr.sparse_tree_automata import SparseTreeAutomaton
from autstr.utils.misc import encode_symbol


# ====================================================================
# Cylindrification
# ====================================================================

def expand(sta: SparseTreeAutomaton, new_arity: int, pos: List[int]
           ) -> SparseTreeAutomaton:
    """Expand a k-tape automaton to new_arity tapes, placing original tape t
    at position pos[t]; the remaining positions range over all letters. Every
    expanded exception is a closed-form block of an original exception.

    Note: like the string `expand`, this widens only the *alphabet*; apply
    `attach_padding` first so regions contributed solely by the new tapes
    (all-padding on the original tapes) are accepted.
    """
    k = sta.symbol_arity
    m = len(sta.base_alphabet_frozen)

    symbols = sta.exc_symbol
    powers_old = m ** np.arange(k - 1, -1, -1, dtype=np.int64)
    digits = (symbols[:, None] // powers_old) % m          # (E, k)
    powers_new = m ** np.arange(new_arity - 1, -1, -1, dtype=np.int64)

    # consistency for duplicated positions + fixed part of the new encoding
    valid = np.ones(len(symbols), dtype=bool)
    fixed = np.zeros(len(symbols), dtype=np.int64)
    first_at: Dict[int, int] = {}
    for old_idx, new_idx in enumerate(pos):
        if new_idx in first_at:
            valid &= digits[:, old_idx] == digits[:, first_at[new_idx]]
        else:
            first_at[new_idx] = old_idx
            fixed += digits[:, old_idx] * powers_new[new_idx]

    free_pos = [p for p in range(new_arity) if p not in first_at]
    if free_pos:
        grid = np.indices((m,) * len(free_pos)).reshape(len(free_pos), -1).T
        offsets = grid @ powers_new[free_pos]
    else:
        offsets = np.zeros(1, dtype=np.int64)
    K = len(offsets)

    keep = np.flatnonzero(valid)
    new_symbols = (fixed[keep, None] + offsets).ravel()
    repeat = np.repeat(keep, K)
    return SparseTreeAutomaton(
        sta.num_states, sta.default_state,
        sta.exc_left[repeat], sta.exc_right[repeat],
        new_symbols, sta.exc_target[repeat],
        sta.is_accepting, new_arity, sta.base_alphabet,
        sta.pd_left, sta.pd_right, sta.pd_target)


# ====================================================================
# Subset determinization (shared by projection and padding closure)
# ====================================================================

def _padding_closure(sta: SparseTreeAutomaton, mapped: np.ndarray,
                     pad_symbol: int, preimage_count: int) -> Set[int]:
    """States reachable by some tree whose mapped labels are all pad_symbol.
    Sparse-aware: an available pair's pair default (or the global default)
    joins the closure as soon as the pair leaves a pad-preimage un-excepted."""
    BOT = sta.BOT
    base = sta.num_states + 1
    available: Set[int] = {BOT}
    is_pad = mapped == pad_symbol
    pad_pair_keys = np.sort((sta.exc_left * base + sta.exc_right)[is_pad])
    changed = True
    while changed:
        changed = False
        av = np.array(sorted(available), dtype=np.int64)
        lok = np.isin(sta.exc_left, av)
        rok = np.isin(sta.exc_right, av)
        new = set(sta.exc_target[lok & rok & is_pad].tolist()) - available
        # per available pair: pad-preimages not fully excepted -> defaults
        pl = np.repeat(av, len(av))
        pr = np.tile(av, len(av))
        pk = pl * base + pr
        covered = np.searchsorted(pad_pair_keys, pk, 'right') - \
            np.searchsorted(pad_pair_keys, pk, 'left')
        defaults = sta.pair_defaults(pl, pr)
        new |= set(defaults[covered < preimage_count].tolist()) - available
        if new:
            available |= new
            changed = True
    available.discard(BOT)
    return available


def _subset_determinize(sta: SparseTreeAutomaton, mapped: np.ndarray,
                        num_new_symbols: int, new_arity: int,
                        preimage_count: int,
                        pad_symbol: Optional[int],
                        max_states: Optional[int] = None) -> SparseTreeAutomaton:
    """Determinize the NTA obtained by relabelling every exception symbol via
    `mapped` (each new symbol having `preimage_count` source symbols), with
    absent children ranging over {BOT} plus the padding closure.

    Subset construction is worst-case exponential; `max_states` aborts with a
    clear error instead of exhausting memory."""
    BOT = sta.BOT
    d = sta.default_state

    if pad_symbol is not None:
        p0 = _padding_closure(sta, mapped, pad_symbol, preimage_count)
    else:
        p0 = set()
    s_bot: FrozenSet[int] = frozenset(p0 | {BOT})

    # exceptions grouped by (left, right); mapped symbols aligned with the
    # automaton's sorted exception order
    group_keys, starts, ends = sta._exception_groups()
    base = sta.num_states + 1

    subset_ids: Dict[FrozenSet[int], int] = {}
    subsets: List[FrozenSet[int]] = []

    def get_id(fs: FrozenSet[int]) -> int:
        idx = subset_ids.get(fs)
        if idx is None:
            if max_states is not None and len(subsets) >= max_states:
                raise RuntimeError(
                    f"subset determinization exceeded max_states={max_states} "
                    f"({len(sta.exc_target)} source exceptions)")
            idx = subset_ids[fs] = len(subsets)
            subsets.append(fs)
        return idx

    default_id = get_id(frozenset({d}))

    exc_l: List[int] = []
    exc_r: List[int] = []
    exc_s: List[int] = []
    exc_t: List[int] = []
    pd_l: List[int] = []
    pd_r: List[int] = []
    pd_t: List[int] = []

    new_options = [-1, default_id]          # -1 encodes the absent-child set
    all_options: List[int] = []

    member_cache: Dict[int, np.ndarray] = {}

    def members(option: int) -> np.ndarray:
        arr = member_cache.get(option)
        if arr is None:
            fs = s_bot if option < 0 else subsets[option]
            arr = member_cache[option] = np.array(sorted(fs), dtype=np.int64)
        return arr

    # dense pair probes (n is the SOURCE state count — small next to the
    # subset space): which child pairs have exceptions at all, and every
    # pair's default, so unproductive combos cost one fancy-index each
    has_exc = np.zeros((base, base), dtype=bool)
    has_exc.ravel()[group_keys] = True
    all_pairs = np.arange(base * base, dtype=np.int64)
    pd_dense = sta.pair_defaults(all_pairs // base,
                                 all_pairs % base).reshape(base, base)
    no_pd = len(sta._pd_keys) == 0

    while new_options:
        # each combo is enumerated exactly once, in the round where its later
        # member was discovered — no dedup set (whose quadratic growth would
        # dominate memory long before the state cap)
        round_new = new_options
        new_options = []
        combo_iter = chain(
            ((x, y) for x in round_new for y in all_options),
            ((y, x) for x in round_new for y in all_options),
            ((x, y) for x in round_new for y in round_new))
        all_options.extend(round_new)

        for cl, cr in combo_iter:
            A, B = members(cl), members(cr)
            productive = bool(has_exc[np.ix_(A, B)].any())
            if no_pd and not productive:
                continue                     # transitions all default
            pl = np.repeat(A, len(B))
            pr = np.tile(B, len(A))
            pair_defs = pd_dense[pl, pr]

            # the combo's pair default: the set of the members' pair defaults
            D0 = frozenset(pair_defs.tolist())
            if D0 != subsets[default_id]:
                before = len(subsets)
                pid = get_id(D0)
                if len(subsets) > before:
                    new_options.append(pid)
                pd_l.append(cl)
                pd_r.append(cr)
                pd_t.append(pid)

            if not productive:
                continue
            pair_keys = pl * base + pr
            gpos = np.searchsorted(group_keys, pair_keys)
            valid = gpos < len(group_keys)
            valid[valid] &= group_keys[gpos[valid]] == pair_keys[valid]
            # pairs without exceptions contribute their default everywhere
            invariant = set(pair_defs[~valid].tolist())
            vdefs = pair_defs[valid]
            spans = [(starts[g], ends[g]) for g in gpos[valid]]
            span_syms = [np.sort(mapped[s:e]) for s, e in spans]
            rows = np.concatenate([np.arange(s, e) for s, e in spans])
            row_syms = mapped[rows]
            row_targets = sta.exc_target[rows]
            order = np.argsort(row_syms, kind="stable")
            sorted_syms = row_syms[order]
            sorted_targets = row_targets[order]

            candidates = np.unique(row_syms)
            lo = np.searchsorted(sorted_syms, candidates, 'left')
            hi = np.searchsorted(sorted_syms, candidates, 'right')
            # which valid pairs except all preimages of each candidate
            cover = np.zeros((len(spans), len(candidates)), dtype=bool)
            for i, ss in enumerate(span_syms):
                cover[i] = (np.searchsorted(ss, candidates, 'right') -
                            np.searchsorted(ss, candidates, 'left')
                            ) >= preimage_count
            for j, sym in enumerate(candidates.tolist()):
                targets = set(sorted_targets[lo[j]:hi[j]].tolist())
                targets |= invariant
                targets |= set(vdefs[~cover[:, j]].tolist())
                fs = frozenset(targets)
                if fs == D0:
                    continue
                before = len(subsets)
                tid = get_id(fs)
                if len(subsets) > before:
                    new_options.append(tid)
                exc_l.append(cl)
                exc_r.append(cr)
                exc_s.append(sym)
                exc_t.append(tid)

    num_states = len(subsets)
    exc_l_arr = np.array(exc_l, dtype=np.int64)
    exc_r_arr = np.array(exc_r, dtype=np.int64)
    exc_l_arr = np.where(exc_l_arr < 0, num_states, exc_l_arr)
    exc_r_arr = np.where(exc_r_arr < 0, num_states, exc_r_arr)
    pd_l_arr = np.array(pd_l, dtype=np.int64)
    pd_r_arr = np.array(pd_r, dtype=np.int64)
    pd_l_arr = np.where(pd_l_arr < 0, num_states, pd_l_arr)
    pd_r_arr = np.where(pd_r_arr < 0, num_states, pd_r_arr)
    accepting = np.array([bool(fs & set(np.flatnonzero(sta.is_accepting)))
                          for fs in subsets])
    return SparseTreeAutomaton(
        num_states, default_id, exc_l_arr, exc_r_arr,
        np.array(exc_s, dtype=np.int64), np.array(exc_t, dtype=np.int64),
        accepting, new_arity, sta.base_alphabet,
        pd_l_arr, pd_r_arr, np.array(pd_t, dtype=np.int64))


def project(sta: SparseTreeAutomaton, tape: int, padding_symbol,
            max_states: Optional[int] = None) -> SparseTreeAutomaton:
    """Existentially quantify one tape: accept the convolution of the
    remaining tapes iff some witness tree exists on the projected tape
    (including witnesses whose domain extends below the remaining tapes,
    which is what the padding closure of the absent-child set captures)."""
    k = sta.symbol_arity
    if k < 2:
        raise ValueError("cannot project the only tape")
    if not 0 <= tape < k:
        raise ValueError(f"tape must be in [0, {k})")
    m = len(sta.base_alphabet_frozen)

    p = m ** (k - 1 - tape)
    mapped = (sta.exc_symbol // (p * m)) * p + sta.exc_symbol % p
    pad_new = encode_symbol((padding_symbol,) * (k - 1),
                            sta.base_alphabet_frozen) if k > 1 else None
    return _subset_determinize(sta, mapped, m ** (k - 1), k - 1,
                               preimage_count=m, pad_symbol=pad_new,
                               max_states=max_states)


def attach_padding(sta: SparseTreeAutomaton, padding_symbol,
                   max_states: Optional[int] = None) -> SparseTreeAutomaton:
    """Accept exactly the trees whose maximal all-padding subtrees, once
    trimmed away, the source accepts — the tree analog of the string
    pipeline's `pad`, required before `expand` widens the convolution (the
    wider convolution's domain is the union of all tapes' domains, so the
    original tapes see attached regions as padding).

    The source is deterministic, so no subset construction is needed: one
    fresh PAD state absorbs pure-padding regions, and every exception with an
    absent child gains a copy with PAD in that position, making a padding
    region behave exactly like an absent child. Any native transitions the
    source had on all-padding leaves are overridden: canonical convolutions
    contain no all-padding node, so those transitions carry no meaning."""
    pad = encode_symbol((padding_symbol,) * sta.symbol_arity,
                        sta.base_alphabet_frozen)
    n = sta.num_states
    PAD, BOT = n, n + 1
    src_bot = sta.BOT

    def with_pad_variants(l, r, cols):
        """Duplicate every row with an absent child so PAD acts like BOT."""
        lb = l == src_bot
        rb = r == src_bot
        both = lb & rb
        l0 = np.where(lb, BOT, l)
        r0 = np.where(rb, BOT, r)
        L = np.concatenate([l0, np.full(int(lb.sum()), PAD, dtype=np.int64),
                            l0[rb], np.full(int(both.sum()), PAD,
                                            dtype=np.int64)])
        R = np.concatenate([r0, r0[lb],
                            np.full(int(rb.sum()), PAD, dtype=np.int64),
                            np.full(int(both.sum()), PAD, dtype=np.int64)])
        C = [np.concatenate([c, c[lb], c[rb], c[both]]) for c in cols]
        return L, R, C

    eL, eR, (eS, eT) = with_pad_variants(
        sta.exc_left, sta.exc_right, [sta.exc_symbol, sta.exc_target])
    # pure-padding leaves are overridden: they start a PAD region
    absent = np.isin(eL, (BOT, PAD)) & np.isin(eR, (BOT, PAD))
    keep = ~(absent & (eS == pad))
    eL, eR, eS, eT = eL[keep], eR[keep], eS[keep], eT[keep]
    over = np.array([(BOT, BOT), (BOT, PAD), (PAD, BOT), (PAD, PAD)],
                    dtype=np.int64)
    eL = np.r_[eL, over[:, 0]]
    eR = np.r_[eR, over[:, 1]]
    eS = np.r_[eS, np.full(4, pad, dtype=np.int64)]
    eT = np.r_[eT, np.full(4, PAD, dtype=np.int64)]

    pL, pR, (pT,) = with_pad_variants(
        sta.pd_left, sta.pd_right, [sta.pd_target])

    return SparseTreeAutomaton(
        n + 1, sta.default_state, eL, eR, eS, eT,
        np.r_[sta.is_accepting, False], sta.symbol_arity, sta.base_alphabet,
        pL, pR, pT)


def permute_tapes(sta: SparseTreeAutomaton, perm: List[int]
                  ) -> SparseTreeAutomaton:
    """Reorder the tapes of a convolution automaton: new tape i carries what
    was tape perm[i]. Closed-form relabelling of the exception symbols."""
    k = sta.symbol_arity
    if sorted(perm) != list(range(k)):
        raise ValueError(f"perm must be a permutation of range({k})")
    m = len(sta.base_alphabet_frozen)
    powers = m ** np.arange(k - 1, -1, -1, dtype=np.int64)
    digits = (sta.exc_symbol[:, None] // powers) % m          # (E, k)
    new_symbols = digits[:, perm] @ powers
    return SparseTreeAutomaton(
        sta.num_states, sta.default_state,
        sta.exc_left, sta.exc_right, new_symbols, sta.exc_target,
        sta.is_accepting, k, sta.base_alphabet,
        sta.pd_left, sta.pd_right, sta.pd_target)


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


# ====================================================================
# Minimization
# ====================================================================

def minimize(sta: SparseTreeAutomaton) -> SparseTreeAutomaton:
    """Minimize by Moore refinement over exception and pair-default
    signatures.

    A state pair's baseline behavior is its pair default (falling back to
    the global default), so two states are distinguishable only through pair
    defaults naming them whose target class differs from the global
    default's class, or through exceptions naming them whose target class
    differs from their pair's baseline class. Unreachable states are pruned
    first."""
    reach = sta.reachable_states()
    keep = np.flatnonzero(reach)
    if len(keep) == 0:
        alphabet = sta.base_alphabet
        return SparseTreeAutomaton(1, 0, [], [], [], [], [False],
                                   sta.symbol_arity, alphabet)
    new_of_old = np.full(sta.num_states + 1, -1, dtype=np.int64)
    new_of_old[keep] = np.arange(len(keep))
    new_of_old[sta.BOT] = len(keep)                      # BOT stays BOT

    lm = new_of_old[sta.exc_left]
    rm = new_of_old[sta.exc_right]
    tmask = (lm >= 0) & (rm >= 0)
    lm, rm = lm[tmask], rm[tmask]
    sym = sta.exc_symbol[tmask]
    tgt = new_of_old[sta.exc_target[tmask]]
    tmask2 = tgt >= 0
    lm, rm, sym, tgt = lm[tmask2], rm[tmask2], sym[tmask2], tgt[tmask2]

    pdl = new_of_old[sta.pd_left]
    pdr = new_of_old[sta.pd_right]
    pmask = (pdl >= 0) & (pdr >= 0)
    pdl, pdr = pdl[pmask], pdr[pmask]
    pdt = new_of_old[sta.pd_target[pmask]]
    pmask2 = pdt >= 0
    pdl, pdr, pdt = pdl[pmask2], pdr[pmask2], pdt[pmask2]

    n = len(keep)
    BOT = n
    acc = sta.is_accepting[keep]
    default = int(new_of_old[sta.default_state])
    if default < 0:
        # the default was unreachable: introduce a fresh dead state for it
        default = n
        n += 1
        BOT = n
        acc = np.r_[acc, False]
        lm = np.where(lm == n - 1, n, lm)                 # keep BOT index last
        rm = np.where(rm == n - 1, n, rm)
        pdl = np.where(pdl == n - 1, n, pdl)
        pdr = np.where(pdr == n - 1, n, pdr)

    # baseline state per exception row: its pair's default
    base2 = BOT + 1
    pd_keys = pdl * base2 + pdr
    order = np.argsort(pd_keys, kind="stable")
    pd_keys_sorted = pd_keys[order]
    pdt_sorted = pdt[order]
    exc_pair = lm * base2 + rm
    if len(pd_keys_sorted):
        pos = np.minimum(np.searchsorted(pd_keys_sorted, exc_pair),
                         len(pd_keys_sorted) - 1)
        hit = pd_keys_sorted[pos] == exc_pair
        exc_base = np.where(hit, pdt_sorted[pos], default)
    else:
        exc_base = np.full(len(exc_pair), default, dtype=np.int64)

    # ---- refinement (vectorized: hashed set-signatures per state) ----
    # entry rows: (owner state, side, partner, sym, target-state); classes of
    # partner/target are resolved per round, entries hashed and combined per
    # owner with two independent 64-bit digests (sum and xor of the deduped
    # entry hashes) — the tree analog of the string engine's hashed Moore.
    PD_SYM = -1                                           # marks pd entries
    # exception entries, both sides
    own_parts = [lm, rm, pdl, pdr]
    side_parts = [np.zeros(len(lm), np.int64), np.ones(len(rm), np.int64),
                  np.zeros(len(pdl), np.int64), np.ones(len(pdr), np.int64)]
    part_parts = [rm, lm, pdr, pdl]
    sym_parts = [sym, sym,
                 np.full(len(pdl), PD_SYM), np.full(len(pdr), PD_SYM)]
    tgt_parts = [tgt, tgt, pdt, pdt]
    is_pd_entry = np.concatenate([np.zeros(len(lm) + len(rm), dtype=bool),
                                  np.ones(2 * len(pdl), dtype=bool)])
    e_own = np.concatenate(own_parts)
    e_side = np.concatenate(side_parts)
    e_part = np.concatenate(part_parts)
    e_sym = np.concatenate(sym_parts)
    e_tgt = np.concatenate(tgt_parts)
    e_base = np.concatenate([exc_base, exc_base, pdt, pdt])  # pd rows unused
    real_own = e_own < BOT

    M1 = np.uint64(0x9E3779B97F4A7C15)
    M2 = np.uint64(0xC2B2AE3D27D4EB4F)
    M3 = np.uint64(0x165667B19E3779F9)

    def digest(values, seed):
        h = values.astype(np.uint64) * M1 + np.uint64(seed)
        h ^= h >> np.uint64(33)
        h *= M2
        h ^= h >> np.uint64(29)
        h *= M3
        h ^= h >> np.uint64(32)
        return h

    classes = acc.astype(np.int64)                        # 0/1 by acceptance
    num_classes = len(np.unique(classes))
    with np.errstate(over='ignore'):
        while True:
            cls_of = np.r_[classes, np.int64(-1)]         # BOT -> -1
            dcls = classes[default]
            live = np.where(is_pd_entry,
                            classes[e_tgt] != dcls,
                            cls_of[e_tgt] != classes[e_base]) & real_own
            idx = np.flatnonzero(live)
            packed = ((e_side[idx] * (n + 2) + cls_of[e_part[idx]] + 1)
                      * (int(np.max(e_sym) if len(e_sym) else 0) + 3)
                      + e_sym[idx] + 2)
            packed = packed * np.int64(n + 1) + cls_of[e_tgt[idx]] + 1
            h1 = digest(packed, 0x5851F42D4C957F2D)
            h2 = digest(packed, 0x14057B7EF767814F)
            # dedup identical entries per owner (set semantics)
            key = np.stack([e_own[idx], h1.view(np.int64)], axis=1)
            order = np.lexsort((h2.view(np.int64), key[:, 1], key[:, 0]))
            so, sh1, sh2 = e_own[idx][order], h1[order], h2[order]
            if len(so):
                keepm = np.r_[True, (so[1:] != so[:-1]) |
                              (sh1[1:] != sh1[:-1]) | (sh2[1:] != sh2[:-1])]
                so, sh1, sh2 = so[keepm], sh1[keepm], sh2[keepm]
            sig1 = np.zeros(n, dtype=np.uint64)
            sig2 = np.zeros(n, dtype=np.uint64)
            if len(so):
                starts_o = np.flatnonzero(np.r_[True, so[1:] != so[:-1]])
                sums = np.add.reduceat(sh1, starts_o)
                xors = np.bitwise_xor.reduceat(sh2, starts_o)
                sig1[so[starts_o]] = sums
                sig2[so[starts_o]] = xors
            rows = np.stack([classes.astype(np.uint64), sig1, sig2], axis=1)
            _, new_classes = np.unique(rows, axis=0, return_inverse=True)
            new_classes = new_classes.astype(np.int64)
            stable = len(np.unique(new_classes)) == num_classes
            classes = new_classes        # always adopt: labels are 0..P-1
            num_classes = len(np.unique(classes))
            if stable:
                break

    # ---- rebuild on classes (vectorized) ----
    P = num_classes
    cls_of = np.r_[classes, P]                            # BOT -> class P
    dcls = int(classes[default])
    keep_pd = classes[pdt] != dcls
    pd_keys2 = cls_of[pdl][keep_pd] * (P + 1) + cls_of[pdr][keep_pd]
    pd_vals2 = classes[pdt][keep_pd]
    if len(pd_keys2):
        uk, first = np.unique(pd_keys2, return_index=True)
        pd_keys2, pd_vals2 = uk, pd_vals2[first]
    # exception baseline per row on class level
    exc_keys2 = cls_of[lm] * (P + 1) + cls_of[rm]
    if len(pd_keys2):
        pos = np.minimum(np.searchsorted(pd_keys2, exc_keys2),
                         len(pd_keys2) - 1)
        hit = pd_keys2[pos] == exc_keys2
        baseline = np.where(hit, pd_vals2[pos], dcls)
    else:
        baseline = np.full(len(exc_keys2), dcls, dtype=np.int64)
    new_t = classes[tgt]
    live = new_t != baseline
    S = sta.num_symbols
    packed = (exc_keys2[live] * S + sym[live])
    uk, first = np.unique(packed, return_index=True)
    out_l = (uk // S) // (P + 1)
    out_r = (uk // S) % (P + 1)
    out_s = uk % S
    out_t = new_t[live][first]
    new_acc = np.zeros(P, dtype=bool)
    new_acc[classes[np.flatnonzero(acc)]] = True
    return SparseTreeAutomaton(
        P, dcls, out_l, out_r, out_s, out_t,
        new_acc, sta.symbol_arity, sta.base_alphabet,
        pd_keys2 // (P + 1), pd_keys2 % (P + 1), pd_vals2)


# ====================================================================
# Equivalence (exact, via boolean closure + emptiness)
# ====================================================================

def equivalent(a: SparseTreeAutomaton, b: SparseTreeAutomaton) -> bool:
    return a.intersection(b.complement()).is_empty() and \
        b.intersection(a.complement()).is_empty()


# ====================================================================
# Embedding string languages (unary chains)
# ====================================================================

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
    S = len(dfa.base_alphabet_frozen) ** dfa.symbol_arity

    # dense next-state table (n, S)
    table = np.tile(np.asarray(dfa.default_states, dtype=np.int64)[:, None],
                    (1, S))
    for q in range(n):
        for s, t in zip(dfa.exception_symbols[q], dfa.exception_states[q]):
            if s >= 0:
                table[q, int(s)] = int(t)

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
