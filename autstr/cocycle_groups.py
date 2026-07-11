"""Distributed-center class-2 groups: cocycle tensors on site trees.

Validation layer for the tensor cut-rank generalisation of the bounded
rank-width group classes (see paper/theorem3-notes.md). A *site tree* is a
binary tree whose nodes are generators: 'x' sites and central 'z' sites.
The commutator data is a tensor T[j, i, v] over Z_p (i < j x-positions in
post-order, v a z-position), presenting the central extension with the
bilinear cocycle

    (b, a)(b', a') = (b + b' + C(a, a'), a + a'),
    C(a, a')_v     = sum_{i<j} T[j, i, v] a_j a'_i .

This module provides the reference group law and the *six crossing
flattenings* whose ranks measure, per subtree cut, the traffic a bottom-up
automaton must carry: upward digit functionals (F_y, F_x), upward pair-sums
(F_m), inward claims (F_g), and the mixed exports whose products flow back
into inside checks (F_py, F_px). The flattenings are genuinely different
matrices -- reshaping changes rank -- so the width is their maximum, not a
single bipartition rank. The uniformly tree-automatic presentation
(claim-and-verify automaton with microcode advice) builds on these measures
and is developed next; the classes `CutRankTreeGroups` (all z-sites on a
chain above the root) and `TreeExtraspecialGroups` (z-sites at the leaves,
laminar targets, width 1) are the two known corners of the class.
"""
from typing import Dict, List, Sequence, Tuple

import numpy as np

from autstr.groups import _rref_mod
from autstr.sparse_tree_automata import Tree

XSITE = 'x'
ZSITE = 'z'

FLATTENINGS = ('F_y', 'F_x', 'F_m', 'F_g', 'F_py', 'F_px')


def _layout(shape: Tree):
    """Post-order sequence, 1-based positions and subtree sizes."""
    seq = []
    stack = [(shape, False)]
    while stack:
        node, done = stack.pop()
        if node is None:
            continue
        if done:
            seq.append(node)
            continue
        stack.append((node, True))
        stack.append((node.right, False))
        stack.append((node.left, False))
    pos = {id(node): i + 1 for i, node in enumerate(seq)}
    size = {}
    for node in seq:
        size[id(node)] = 1 + sum(size[id(c)] for c in (node.left, node.right)
                                 if c is not None)
    return seq, pos, size


class CocycleSites:
    """A site tree over Z_p with its cocycle tensors: reference law and
    cut-width measures. Elements are (b, a) with b indexed by the z-sites
    and a by the x-sites, both in ascending post-order."""

    def __init__(self, p: int, shape: Tree):
        if p < 2 or any(p % d == 0 for d in range(2, int(p ** 0.5) + 1)):
            raise ValueError(f"p must be prime, got {p}")
        self.p = p
        self.shape = shape
        self.seq, self.pos, self.size = _layout(shape)
        self.n_sites = len(self.seq)
        self.site = {}
        for node in self.seq:
            if node.label not in (XSITE, ZSITE):
                raise ValueError(f"site labels must be '{XSITE}' or '{ZSITE}'")
            self.site[self.pos[id(node)]] = node.label
        self.X = [t for t in range(1, self.n_sites + 1) if self.site[t] == XSITE]
        self.Z = [t for t in range(1, self.n_sites + 1) if self.site[t] == ZSITE]
        self._xi = {t: idx for idx, t in enumerate(self.X)}
        self._zi = {v: idx for idx, v in enumerate(self.Z)}

    # ---------------- tensors and the reference law ----------------

    def check_tensor(self, T: Dict[Tuple[int, int, int], int]):
        for (j, i, v) in T:
            if i not in self._xi or j not in self._xi or not i < j:
                raise ValueError(f"({j}, {i}) must be x-positions with i < j")
            if v not in self._zi:
                raise ValueError(f"{v} is not a z-position")

    def multiply(self, T: Dict, g, h):
        """The reference group law of G(T)."""
        (b1, a1), (b2, a2) = g, h
        b = [(u + w) % self.p for u, w in zip(b1, b2)]
        for (j, i, v), coeff in T.items():
            b[self._zi[v]] = (b[self._zi[v]]
                              + coeff * a1[self._xi[j]] * a2[self._xi[i]]) % self.p
        return tuple(b), tuple((u + w) % self.p for u, w in zip(a1, a2))

    def identity(self):
        return (0,) * len(self.Z), (0,) * len(self.X)

    # ---------------- the six crossing flattenings ----------------

    def _flattening_entries(self, T: Dict, lo: int, hi: int):
        """Sort the tensor entries of a cut into the six flattenings,
        as {name: {(row_key, col_key): coeff}}."""
        inside = lambda t: lo <= t <= hi
        out = {name: {} for name in FLATTENINGS}
        for (j, i, v), coeff in T.items():
            ji, ii, vi = inside(j), inside(i), inside(v)
            if ii and ji and vi:
                continue                              # fully internal
            if not ii and not ji and not vi:
                continue                              # fully external
            if ii and not ji and not vi:
                out['F_y'][(i, (j, v))] = coeff
            elif ji and not ii and not vi:
                out['F_x'][(j, (i, v))] = coeff
            elif ii and ji and not vi:
                out['F_m'][((j, i), v)] = coeff
            elif ii and not ji and vi:
                out['F_py'][(i, (j, v))] = coeff
            elif ji and not ii and vi:
                out['F_px'][(j, (i, v))] = coeff
            if vi and not (ii and ji):
                out['F_g'][(v, (j, i))] = coeff       # claims, all shapes
        return out

    @staticmethod
    def _rank(entries: Dict, p: int) -> int:
        if not entries:
            return 0
        rows = sorted({r for r, _ in entries})
        cols = sorted({c for _, c in entries})
        M = np.zeros((len(rows), len(cols)), dtype=np.int64)
        rindex = {r: a for a, r in enumerate(rows)}
        cindex = {c: a for a, c in enumerate(cols)}
        for (r, c), coeff in entries.items():
            M[rindex[r], cindex[c]] = coeff % p
        return len(_rref_mod(M, p)[1])

    def cut_profile(self, T: Dict) -> Dict[int, Dict[str, int]]:
        """For every proper subtree cut (keyed by its root position), the
        ranks of the six crossing flattenings."""
        self.check_tensor(T)
        profile = {}
        for node in self.seq:
            t, sz = self.pos[id(node)], self.size[id(node)]
            if sz == self.n_sites:
                continue
            entries = self._flattening_entries(T, t - sz + 1, t)
            profile[t] = {name: self._rank(entries[name], self.p)
                          for name in FLATTENINGS}
        return profile

    def cut_width(self, T: Dict) -> int:
        """The width of this layout for the tensor: the maximum flattening
        rank over all subtree cuts."""
        profile = self.cut_profile(T)
        return max((max(ranks.values()) for ranks in profile.values()),
                   default=0)


# ---------------- embeddings of the two known corner classes ----------------

def fixed_k_sites(p: int, layout_shape: Tree, form: Dict[Tuple[int, int], Sequence[int]],
                  k: int) -> Tuple[CocycleSites, Dict]:
    """Embed a `CutRankTreeGroups` instance: an all-x copy of the layout
    with a chain of k z-sites above the root. Positions of the x-layout are
    preserved; the z-chain occupies positions n+1 (innermost = center
    coordinate 0) through n+k."""
    def convert(node):
        if node is None:
            return None
        return Tree(XSITE, convert(node.left), convert(node.right))
    site_root = convert(layout_shape)
    n = len(_layout(layout_shape)[0])
    for _ in range(k):
        site_root = Tree(ZSITE, site_root, None)
    sites = CocycleSites(p, site_root)
    T = {}
    for (j, i), label in form.items():
        for l in range(k):
            if label[l] % p:
                T[(j, i, n + 1 + l)] = label[l] % p
    return sites, T


def laminar_sites(p: int, shape: Tree) -> Tuple[CocycleSites, Dict, Dict]:
    """Embed a `TreeExtraspecialGroups` instance: every inner node w of the
    shape becomes a chain x-site (for x_w) over x-site (for y_w), every leaf
    becomes a z-site, and [x_w, y_w] hits every leaf below w. Returns the
    sites, the tensor, and the address map {shape address: site positions}."""
    def convert(node, addr):
        if node is None:
            return None, {}
        left, lmap = convert(node.left, addr + '0')
        right, rmap = convert(node.right, addr + '1')
        amap = {**lmap, **rmap}
        if node.left is None and node.right is None:
            return Tree(ZSITE), {**amap, addr: ('z',)}
        lower = Tree(XSITE, left, right)
        upper = Tree(XSITE, lower, None)
        return upper, {**amap, addr: ('i', lower, upper)}
    site_root, amap = convert(shape, '')
    sites = CocycleSites(p, site_root)
    addr_pos = {}
    for addr, entry in amap.items():
        if entry[0] == 'z':
            pass
        else:
            addr_pos[addr] = (sites.pos[id(entry[2])], sites.pos[id(entry[1])])
    # leaves: locate by re-walking shape and site tree in parallel
    def zwalk(node, snode, addr, acc):
        if node is None:
            return
        if node.left is None and node.right is None:
            acc[addr] = sites.pos[id(snode)]
            return
        lower = snode.left
        zwalk(node.left, lower.left, addr + '0', acc)
        zwalk(node.right, lower.right, addr + '1', acc)
    leaf_pos = {}
    zwalk(shape, site_root, '', leaf_pos)
    T = {}
    for addr, (xw, yw) in addr_pos.items():
        for leaf_addr, v in leaf_pos.items():
            if leaf_addr.startswith(addr):
                T[(xw, yw, v)] = 1
    return sites, T, {'inner': addr_pos, 'leaf': leaf_pos}


def point_target_sites(p: int, shape: Tree) -> Tuple[CocycleSites, Dict]:
    """A width-1 family covered by neither corner class: as `laminar_sites`,
    but every commutator [x_w, y_w] hits only the *leftmost* leaf below w
    (point targets: the center grows with the tree, the law is not laminar)."""
    sites, T_laminar, maps = laminar_sites(p, shape)
    T = {}
    for (xw, yw, v) in T_laminar:
        best = min(u for (a, b, u) in T_laminar if (a, b) == (xw, yw))
        T[(xw, yw, best)] = 1
    return sites, T


def scattered_sites(p: int, m: int) -> Tuple[CocycleSites, Dict]:
    """The lower-bound family: m private z-sites on a chain at the bottom,
    m commuting x-pairs above, pair t targeting exactly z_t. The cut at the
    z-chain has an identity claim flattening: width m."""
    node = None
    for _ in range(m):
        node = Tree(ZSITE, node, None)
    for _ in range(2 * m):
        node = Tree(XSITE, node, None)
    sites = CocycleSites(p, node)
    T = {}
    for t in range(1, m + 1):
        i = m + 2 * t - 1
        j = m + 2 * t
        T[(j, i, t)] = 1
    return sites, T
