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
import itertools as it
from typing import Dict, List, Sequence, Tuple

import numpy as np

from autstr.groups import _rref_mod
from autstr.sparse_tree_automata import Tree
from autstr.tree_presentations import TreeAutomaticPresentation
from autstr.tree_uniform import UniformlyTreeAutomaticClass, sta_from_delta

PAD = '*'

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


# ====================================================================
# The claim-and-verify automaton (width 1) with microcode advice
# ====================================================================

def _vratio(target: Dict, base: Dict, p: int, ctx: str) -> int:
    """The scalar c with target == c * base (as sparse vectors); raises
    AssertionError if the vectors are not proportional -- every call is a
    machine-checked instance of a restriction lemma."""
    base = {k: v % p for k, v in base.items() if v % p}
    target = {k: v % p for k, v in target.items() if v % p}
    if not base:
        if target:
            raise AssertionError(f"lemma failure ({ctx}): target outside span")
        return 0
    key = next(iter(base))
    c = target.get(key, 0) * pow(base[key], p - 2, p) % p
    if {k: c * v % p for k, v in base.items() if c * v % p} != target:
        raise AssertionError(f"lemma failure ({ctx}): not proportional")
    return c


def _vratio2(entries: Dict, u: Dict, w: Dict, p: int, ctx: str) -> int:
    """The scalar c with entries[(a, b)] == c * u[a] * w[b]."""
    outer = {(a, b): u[a] * w[b] % p for a in u for b in w if u[a] * w[b] % p}
    return _vratio(entries, outer, p, ctx)


def _vratio3(entries: Dict, u: Dict, w: Dict, x: Dict, p: int, ctx: str) -> int:
    """The scalar c with entries[(a, b, e)] == c * u[a] * w[b] * x[e]."""
    outer = {(a, b, e): u[a] * w[b] * x[e] % p
             for a in u for b in w for e in x if u[a] * w[b] * x[e] % p}
    return _vratio(entries, outer, p, ctx)


def _vsolve2(target: Dict, b1: Dict, b2: Dict, p: int, ctx: str):
    """(c1, c2) with target == c1*b1 + c2*b2; AssertionError if unsolvable."""
    target = {k: v % p for k, v in target.items() if v % p}
    if not target:
        return 0, 0
    keys = sorted(set(target) | set(b1) | set(b2))
    A = np.array([[b1.get(k, 0) % p, b2.get(k, 0) % p] for k in keys],
                 dtype=np.int64)
    B = np.array([[target.get(k, 0) % p] for k in keys], dtype=np.int64)
    from autstr.groups import _solve_xa_eq_b
    try:
        X = _solve_xa_eq_b(A.T, B.T, p)      # (1 x 2) with X . A^T = B^T
    except ValueError:
        raise AssertionError(f"lemma failure ({ctx}): 2-term system unsolvable")
    return int(X[0, 0]) % p, int(X[0, 1]) % p


class CocycleRankWidthGroups:
    """The uniformly tree-automatic class of distributed-center class-2
    groups of tensor cut-rank 1 (see paper/theorem3-notes.md): the common
    generalisation, at width r = 1, of `CutRankTreeGroups` (fixed center on
    a chain) and `TreeExtraspecialGroups` (laminar leaf targets).

    The advice is *microcode*: each site of a `CocycleSites` layout expands
    into its site letter followed by a chain of micro-op letters, each
    carrying at most two Z_p constants, over an alphabet of O(p^5) letters
    independent of the tensor. Element digits repeat along a site's stretch
    (the universe automaton enforces constancy), so the multiplication
    automaton's state is exactly the seven-register machine of the
    protocol:

        wy, wx : upward digit functionals      (F_y, F_x)
        phy,phx: mixed exports for inside targets (F_py, F_px)
        m      : pair-sum exports in the F_m basis
        g      : claim coordinates (W g = residuals), unique by
                 full-column-rank bases; inconsistency = rejection
        bg     : the sibling's claim, buffered between a merge and its
                 consistency check (the generalised "siblings must agree")

    `advice(sites, T)` compiles a tensor of cut-width 1 into microcode;
    every scalar it derives is guarded by an AssertionError that is a
    machine-checked instance of a restriction lemma, so a compilation that
    succeeds *is* a certificate that the factorisations of the protocol
    exist for this instance. Width r >= 2 needs entry-op chains for the
    matrix constants and is deliberately not implemented yet."""

    OPS1 = {'sy', 'sx', 'sq', 'sh', 'sm', 'sg', 'iy', 'ix', 'iq', 'ih',
            'qy', 'hx', 'rm', 'rg', 'gm', 'z0', 'bg', 'bc'}

    def __init__(self, p: int, merge_letters=None):
        """With `merge_letters=None` the full ISA is available: the
        simulator and the compiler work, but the presentation automata are
        too large for the enumeration-based builder. Passing an explicit
        set of merge letters (e.g. collected from compiled advices via
        `used_merge_letters`) instantiates the automata over that
        sub-alphabet."""
        if p < 2 or p > 9 or any(p % d == 0 for d in range(2, int(p ** 0.5) + 1)):
            raise ValueError(f"p must be a prime in 2..9, got {p}")
        self.p = p
        self._merge_letters = None if merge_letters is None \
            else frozenset(merge_letters)
        self.digits = [str(d) for d in range(p)]
        self._digit_val = {d: int(d) for d in self.digits}
        self._parsed = {}
        self.element_letters = set(self.digits)
        self._cls = None

    @property
    def advice_letters(self):
        """The full microcode ISA. The merge letters carry 13 constants
        (folds to the joint bases, the three sibling products, the two
        cross m->claim translations), so the flat alphabet has 2 p^13 + O(p^2)
        letters: fine for the transition function and the simulator, too
        large for the enumeration-based automaton builder -- building the
        presentation automata is gated until factored transition letters
        land in the engine (see theorem3-notes.md)."""
        p = self.p
        letters = ['L', 'K', 'U', 'V', 'n', 'g0', 'bl']
        for op in ('sy', 'sx', 'sq', 'sh', 'sm', 'sg'):
            letters += [f'{op}{c}' for c in range(p)]
        for op in ('iy', 'ix', 'iq', 'ih', 'qy', 'hx', 'rm', 'rg', 'gm',
                   'b1'):
            letters += [f'{op}{c}' for c in range(1, p)]
        for op in ('za', 'zc', 'bk'):
            letters += [f'{op}{c}{d}' for c in range(p) for d in range(p)]
        letters += [f'z0{c}' for c in range(p)]
        if self._merge_letters is not None:
            return set(letters) | set(self._merge_letters)
        for kind in ('M', 'W'):
            letters += [kind + ''.join(map(str, cs))
                        for cs in it.product(range(p), repeat=13)]
        return set(letters)

    @property
    def sigma(self):
        return {PAD} | set(self.digits) | self.advice_letters

    @property
    def cls(self) -> UniformlyTreeAutomaticClass:
        """The uniformly tree-automatic presentation (built lazily: the
        seven-register multiplication automaton enumerates a large
        state-pair x letter product)."""
        if self._cls is None:
            self._cls = UniformlyTreeAutomaticClass({
                'U': self._universe_automaton(),
                'M': self._multiplication_automaton(),
                'Eq': self._eq_automaton(),
            }, padding_symbol=PAD)
        return self._cls

    # ---------------- the automata ----------------

    def _shape_step(self, a, lq, rq):
        """Shape legality shared by U and Eq: which child pattern each
        advice letter admits. Returns False if illegal."""
        if a in ('L', 'K'):
            return lq is None and rq is None
        if a[0] in ('M', 'W'):
            return lq is not None and rq is not None
        # unary site letters and all micro-ops
        return (lq is None) != (rq is None)

    def _universe_automaton(self) -> SparseTreeAutomaton:
        """Shape plus per-stretch digit constancy: a site letter records the
        element digit, the micro-ops above it must repeat it."""
        def delta(lq, rq, sym):
            a, x = sym
            if 'dead' in (lq, rq) or x not in self.element_letters:
                return 'dead'
            if not self._shape_step(a, lq, rq):
                return 'dead'
            if a in ('L', 'K', 'U', 'V') or a[0] in ('M', 'W'):
                return ('d', x)
            child = lq if rq is None else rq
            return child if child == ('d', x) else 'dead'

        states = ['dead'] + [('d', d) for d in self.digits]
        return sta_from_delta(self.sigma, states, 2, delta,
                              {('d', d) for d in self.digits},
                              tapes=[self.advice_letters, self.element_letters])

    def _eq_automaton(self) -> SparseTreeAutomaton:
        def delta(lq, rq, sym):
            a, x, y = sym
            if 'dead' in (lq, rq) or x != y or x not in self.element_letters:
                return 'dead'
            return 'ok' if self._shape_step(a, lq, rq) else 'dead'

        return sta_from_delta(self.sigma, ['ok', 'dead'], 3, delta, {'ok'},
                              tapes=[self.advice_letters] +
                                    [self.element_letters] * 2)

    _OP_CODES = {'sy': 9, 'sx': 10, 'sq': 11, 'sh': 12, 'sm': 13,
                 'sg': 14, 'iy': 15, 'ix': 16, 'iq': 17, 'ih': 18,
                 'qy': 19, 'hx': 20, 'rm': 21, 'rg': 22, 'gm': 23,
                 'b1': 24, 'za': 25, 'zc': 26, 'z0': 27, 'bk': 28}
    _SITE_CODES = {'L': 0, 'K': 1, 'U': 2, 'V': 3, 'n': 6, 'g0': 7, 'bl': 8}

    def _parse_letter(self, a):
        """Letter -> (code, consts), memoized; None if malformed."""
        hit = self._parsed.get(a)
        if hit is not None:
            return hit
        if a in self._SITE_CODES:
            out = (self._SITE_CODES[a], ())
        elif a[0] in ('M', 'W') and len(a) == 14:
            out = (4 if a[0] == 'M' else 5, tuple(int(c) for c in a[1:]))
        elif a[:2] in self._OP_CODES:
            out = (self._OP_CODES[a[:2]], tuple(int(c) for c in a[2:]))
        else:
            return None
        self._parsed[a] = out
        return out

    def _m_delta(self, lq, rq, sym):
        """The seven-register claim-and-verify transition (shared by the
        automaton construction and the direct simulator)."""
        p = self.p
        a, x, y, z = sym
        if lq == 'dead' or rq == 'dead':
            return 'dead'
        info = self._parse_letter(a)
        dx = self._digit_val.get(x)
        dy = self._digit_val.get(y)
        dz = self._digit_val.get(z)
        if info is None or dx is None or dy is None or dz is None:
            return 'dead'
        code, cs = info
        d0 = (dz - dx - dy) % p
        if lq is not None and rq is not None:
            if code == 4 and d0 == 0 or code == 5:
                fy, fq, fyq, fx, fh, fxh, fml, fmr, cpm, a5, a6, c2, c3 = cs
                wyl, wxl, qyl, hxl, ml, gl, _ = lq
                wyr, wxr, qyr, hxr, mr, gr, _ = rq
                return ((fy * wyl + wyr) % p,
                        (wxl + fx * wxr) % p,
                        (fq * qyl + fyq * wyl + qyr) % p,
                        (hxl + fh * hxr + fxh * wxr) % p,
                        (fml * ml + fmr * mr + cpm * wxr * wyl) % p,
                        (gl + a5 * wxr * qyl + c3 * mr) % p,
                        (gr + a6 * wyl * hxr + c2 * ml) % p)
            return 'dead'
        if lq is None and rq is None:
            if code == 0 and d0 == 0 or code == 1:
                return (0,) * 7
            return 'dead'
        q = lq if rq is None else rq
        if code == 2:
            return q if d0 == 0 else 'dead'
        if code == 3:
            return q
        if code in (0, 1, 4, 5):
            return 'dead'
        wy, wx, qy, hx, m, g, bg = q
        if code == 6:
            return q
        if code == 7:
            return q if g == 0 else 'dead'
        if code == 8:
            return (wy, wx, qy, hx, m, g, 0)
        if code == 9:
            return ((cs[0] * wy) % p, wx, qy, hx, m, g, bg)
        if code == 10:
            return (wy, (cs[0] * wx) % p, qy, hx, m, g, bg)
        if code == 11:
            return (wy, wx, (cs[0] * qy) % p, hx, m, g, bg)
        if code == 12:
            return (wy, wx, qy, (cs[0] * hx) % p, m, g, bg)
        if code == 13:
            return (wy, wx, qy, hx, (cs[0] * m) % p, g, bg)
        if code == 14:
            return (wy, wx, qy, hx, m, (cs[0] * g) % p, bg)
        if code == 15:
            return ((wy + cs[0] * dy) % p, wx, qy, hx, m, g, bg)
        if code == 16:
            return (wy, (wx + cs[0] * dx) % p, qy, hx, m, g, bg)
        if code == 17:
            return (wy, wx, (qy + cs[0] * dy) % p, hx, m, g, bg)
        if code == 18:
            return (wy, wx, qy, (hx + cs[0] * dx) % p, m, g, bg)
        if code == 19:
            return (wy, wx, (qy + cs[0] * wy) % p, hx, m, g, bg)
        if code == 20:
            return (wy, wx, qy, (hx + cs[0] * wx) % p, m, g, bg)
        if code == 21:
            return (wy, wx, qy, hx, (m + cs[0] * dx * wy) % p, g, bg)
        if code == 22:
            return (wy, wx, qy, hx, m, (g + cs[0] * dx * qy) % p, bg)
        if code == 23:
            return (wy, wx, qy, hx, m, (g + cs[0] * m) % p, bg)
        if code == 24:
            return (wy, wx, qy, hx, m, (g + cs[0] * bg) % p, bg)
        if code == 25:
            return (wy, wx, qy, hx, m, (cs[0] * (d0 - cs[1] * m)) % p, bg)
        if code == 26:
            return q if (cs[1] * g - d0 + cs[0] * m) % p == 0 else 'dead'
        if code == 27:
            return q if (d0 - cs[0] * m) % p == 0 else 'dead'
        if code == 28:
            return q if (cs[0] * g + cs[1] * bg) % p == 0 else 'dead'
        return 'dead'

    def simulate(self, advice: Tree, tx: Tree, ty: Tree, tz: Tree) -> bool:
        """Run the multiplication transition directly over the convolved
        trees (fast oracle for compiler development; the automaton built by
        `_multiplication_automaton` realises exactly this run)."""
        def run(a, x, y, z):
            if a is None:
                return None
            lq = run(a.left, x.left if x else None, y.left if y else None,
                     z.left if z else None)
            rq = run(a.right, x.right if x else None, y.right if y else None,
                     z.right if z else None)
            if lq == 'dead' or rq == 'dead':
                return 'dead'
            sym = (a.label,
                   x.label if x else PAD, y.label if y else PAD,
                   z.label if z else PAD)
            return self._m_delta(lq, rq, sym)
        return run(advice, tx, ty, tz) == (0,) * 7

    def _multiplication_automaton(self) -> SparseTreeAutomaton:
        states = ['dead'] + list(it.product(range(self.p), repeat=7))
        return sta_from_delta(self.sigma, states, 4, self._m_delta,
                              {(0,) * 7},
                              tapes=[self.advice_letters] +
                                    [self.element_letters] * 3)

    # ---------------- the advice compiler ----------------

    def _r1_bases(self, sites: CocycleSites, T: Dict, lo: int, hi: int):
        """Width-1 basis vectors of the six flattenings of the interval, as
        sparse dicts. Vy/Vx/Vpy/Vpx live over x-positions, Bm and w over
        z-positions."""
        p = self.p
        ent = sites._flattening_entries(T, lo, hi)

        def basis(e, key_of, ctx):
            groups = {}
            for pair_key, coeff in e.items():
                own, other = key_of(pair_key)
                groups.setdefault(other, {})[own] = coeff % p
            vec = {}
            for cand in groups.values():
                if any(v % p for v in cand.values()):
                    if not vec:
                        vec = {k: v % p for k, v in cand.items() if v % p}
                    else:
                        _vratio(cand, vec, p, ctx)
            if vec:                                   # canonical: leading 1
                lead = pow(vec[min(vec)], p - 2, p)
                vec = {k: v * lead % p for k, v in vec.items()}
            return vec

        return {
            'Vy': basis(ent['F_y'], lambda k: (k[0], k[1]), 'Vy rank'),
            'Vx': basis(ent['F_x'], lambda k: (k[0], k[1]), 'Vx rank'),
            'Vpy': basis(ent['F_py'], lambda k: (k[0], k[1]), 'Vpy rank'),
            'Vpx': basis(ent['F_px'], lambda k: (k[0], k[1]), 'Vpx rank'),
            'Bm': basis(ent['F_m'], lambda k: (k[1], k[0]), 'Bm rank'),
            'w': basis(ent['F_g'], lambda k: (k[0], k[1]), 'claim rank'),
        }

    @staticmethod
    def _restrict(vec: Dict, lo: int, hi: int) -> Dict:
        return {k: v for k, v in vec.items() if lo <= k <= hi}

    def _pair_block(self, sites, T, t, lo, hi, targets):
        """Entries of the read-off pairs {t, i}, i in [lo, hi], keyed by
        (i, v) with v restricted by the predicate `targets`."""
        out = {}
        for (j, i, v), coeff in T.items():
            if j == t and lo <= i <= hi and targets(v):
                out[(i, v)] = coeff % self.p
        return out

    def _bank_ops(self, old, new, lo, hi, own, digit_pos, p, banks=None):
        """Rebase + injection ops from child bases `old` to node bases
        `new` over the child interval [lo, hi]; `own` is the node's own
        position (or None) for the digit injections."""
        ops = []
        # mixed banks first: their qy/hx ops consume the *old* wy/wx
        table = (('Vpy', 'sq', 'iq', 'qy', 'Vy'),
                 ('Vpx', 'sh', 'ih', 'hx', 'Vx'),
                 ('Vy', 'sy', 'iy', None, None),
                 ('Vx', 'sx', 'ix', None, None))
        for bank, scale, inject, mix, mixsrc in table:
            if banks is not None and bank not in banks:
                continue
            target = self._restrict(new[bank], lo, hi)
            if mix is None:
                theta = _vratio(target, self._restrict(old[bank], lo, hi),
                                p, f'{bank} restriction')
                cmix = 0
            else:
                theta, cmix = _vsolve2(target,
                                       self._restrict(old[bank], lo, hi),
                                       self._restrict(old[mixsrc], lo, hi),
                                       p, f'{bank} mixed restriction')
            if theta != 1:
                ops.append(f'{scale}{theta}')
            if cmix:
                ops.append(f'{mix}{cmix}')
            if own is not None:
                cinj = new[bank].get(own, 0)
                if cinj:
                    ops.append(f'{inject}{cinj}')
        return ops

    def _claim_rebase_ops(self, w_old, w_new, p, ctx):
        """Ops taking claim coordinates from basis w_old to w_new over the
        same z-set (w_new = phi * w_old)."""
        if not w_old:
            if w_new:
                raise AssertionError(f'lemma failure ({ctx}): claim born late')
            return []
        if not w_new:
            return ['g0']
        phi = _vratio(w_new, w_old, p, ctx)
        if phi == 0:
            raise AssertionError(f'lemma failure ({ctx}): rebase kills basis')
        inv = pow(phi, p - 2, p)
        return [f'sg{inv}'] if inv != 1 else []

    def advice(self, sites: CocycleSites, T: Dict) -> Tree:
        """Compile a width-1 tensor over the site layout into microcode
        advice. Raises ValueError if some cut exceeds width 1; every scalar
        derivation asserts its restriction lemma."""
        p = self.p
        if sites.p != p:
            raise ValueError("sites and class must share p")
        sites.check_tensor(T)
        width = sites.cut_width(T)
        if width > 1:
            raise ValueError(f"tensor has cut-width {width} > 1")

        bases: Dict[int, Dict] = {}
        adv: Dict[int, Tree] = {}

        def chain(tree, ops):
            for op in ops:
                tree = Tree(op, tree, None)
            return tree

        for node in sites.seq:
            t = sites.pos[id(node)]
            sz = sites.size[id(node)]
            lo = t - sz + 1
            kind = sites.site[t]
            B_S = self._r1_bases(sites, T, lo, t)
            L, R = node.left, node.right

            if L is None and R is None:
                letter = 'L' if kind == XSITE else 'K'
                ops = []
                if kind == XSITE:
                    for bank, inject in (('Vy', 'iy'), ('Vx', 'ix'),
                                         ('Vpy', 'iq'), ('Vpx', 'ih')):
                        c = B_S[bank].get(t, 0)
                        if c:
                            ops.append(f'{inject}{c}')
                else:
                    w = B_S['w']
                    if w:
                        cc = pow(w[t], p - 2, p)
                        ops.append(f'za{cc}0')
                    else:
                        ops.append('z00')
                adv[t] = chain(Tree(letter), ops)
                bases[t] = B_S
                continue

            if L is None or R is None:
                child = L if L is not None else R
                cpos = sites.pos[id(child)]
                B_C = bases[cpos]
                below = chain(adv[cpos], [])
                letter = 'U' if kind == XSITE else 'V'
                ops = self._site_ops(sites, T, t, lo, t - 1, B_C, B_S, kind)
                adv[t] = chain(Tree(letter, below, None), ops)
                bases[t] = B_S
                continue

            # binary: L keeps wy/qy raw for the merge products, R keeps
            # wx/hx raw; the merge letter folds the raw sides into the
            # joint bases of J = [lo, t-1] and consumes the products.
            lp, rp = sites.pos[id(L)], sites.pos[id(R)]
            B_L, B_R = bases[lp], bases[rp]
            B_J = self._r1_bases(sites, T, lo, t - 1)
            pre_L = self._bank_ops(B_L, B_J, lo, lp, None, None, p,
                                   banks=('Vpx', 'Vx'))
            pre_R = self._bank_ops(B_R, B_J, lp + 1, t - 1, None, None, p,
                                   banks=('Vpy', 'Vy'))

            # folds of the raw sides (solved against the child bases)
            fy = _vratio(self._restrict(B_J['Vy'], lo, lp), B_L['Vy'], p,
                         'Vy fold L')
            fq, fyq = _vsolve2(self._restrict(B_J['Vpy'], lo, lp),
                               B_L['Vpy'], B_L['Vy'], p, 'Vpy fold L')
            fx = _vratio(self._restrict(B_J['Vx'], lp + 1, t - 1), B_R['Vx'],
                         p, 'Vx fold R')
            fh, fxh = _vsolve2(self._restrict(B_J['Vpx'], lp + 1, t - 1),
                               B_R['Vpx'], B_R['Vx'], p, 'Vpx fold R')
            fml = _vratio({k: v for k, v in B_L['Bm'].items()
                           if not lp + 1 <= k <= t - 1}, B_J['Bm'], p,
                          'Bm fold L')
            fmr = _vratio({k: v for k, v in B_R['Bm'].items()
                           if not lo <= k <= lp}, B_J['Bm'], p, 'Bm fold R')

            # sibling products and cross m->claim translations
            wL, wR, wJ = B_L['w'], B_R['w'], B_J['w']
            new_pairs = {(j, i, v): c % p for (j, i, v), c in T.items()
                         if lo <= i <= lp and lp + 1 <= j <= t - 1}
            blockL = {k: c for k, c in new_pairs.items() if lo <= k[2] <= lp}
            blockR = {k: c for k, c in new_pairs.items()
                      if lp + 1 <= k[2] <= t - 1}
            blockO = {k: c for k, c in new_pairs.items()
                      if not lo <= k[2] <= t - 1}
            c_gl = _vratio3(blockL, B_R['Vx'], B_L['Vpy'], wL, p,
                            'sibling block -> L claims')
            c_gr = _vratio3({(k[1], k[0], k[2]): c for k, c in blockR.items()},
                            B_L['Vy'], B_R['Vpx'], wR, p,
                            'sibling block -> R claims')
            cpm = _vratio3(blockO, B_R['Vx'], B_L['Vy'], B_J['Bm'], p,
                           'sibling block -> exports')
            psiL = _vratio(self._restrict(B_L['Bm'], lp + 1, t - 1), wR, p,
                           'm_L -> R claims')
            psiR = _vratio(self._restrict(B_R['Bm'], lo, lp), wL, p,
                           'm_R -> L claims')
            consts = (fy, fq, fyq, fx, fh, fxh, fml, fmr, cpm,
                      (-c_gl) % p, (-c_gr) % p, (-psiL) % p, (-psiR) % p)
            letter = ('M' if kind == XSITE else 'W') + ''.join(map(str, consts))
            if self._merge_letters is not None \
                    and letter not in self._merge_letters:
                raise ValueError(
                    f"merge letter {letter!r} is not in the instantiated "
                    f"sub-alphabet; collect it with used_merge_letters first")

            # claim join on the stretch above the merge
            ops = []
            phiL = _vratio(self._restrict(wJ, lo, lp), wL, p, 'claim join L')
            phiR = _vratio(self._restrict(wJ, lp + 1, t - 1), wR, p,
                           'claim join R')
            if wL and phiL:
                inv = pow(phiL, p - 2, p)
                if inv != 1:
                    ops.append(f'sg{inv}')
                if wR:
                    ops.append(f'bk{phiR}{p - 1}')
            elif wL and not phiL:
                ops.append('g0')
                if wR and phiR:
                    ops.append(f'b1{pow(phiR, p - 2, p)}')
                elif wR:
                    ops.append('bk01')
            else:                                     # no L claims
                if wR and phiR:
                    ops.append(f'b1{pow(phiR, p - 2, p)}')
                elif wR:
                    ops.append('bk01')
            ops.append('bl')
            ops += self._site_ops(sites, T, t, lo, t - 1, B_J, B_S, kind)
            adv[t] = chain(Tree(letter, chain(adv[lp], pre_L),
                           chain(adv[rp], pre_R)), ops)
            bases[t] = B_S

        return adv[sites.pos[id(sites.seq[-1])]]

    @staticmethod
    def used_merge_letters(advice: Tree):
        """The merge letters occurring in a compiled advice tree (for
        instantiating a sub-alphabet class that can build its automata)."""
        out = set()
        stack = [advice]
        while stack:
            node = stack.pop()
            if node is None:
                continue
            if node.label and node.label[0] in ('M', 'W'):
                out.add(node.label)
            stack.append(node.left)
            stack.append(node.right)
        return out

    def _z_in(self, sites, v, lo, hi):
        return lo <= v <= hi

    @staticmethod
    def _restrict_keys(vec: Dict, other: Dict) -> Dict:
        return {k: v for k, v in vec.items() if k in other}

    def _m_rebase_ops(self, bm_old, bm_new, drop_lo, drop_hi, p, ctx):
        """Rebase m from bm_old to bm_new, where the columns in
        [drop_lo, drop_hi] left the outside (their mass moves via merge
        constants, not here)."""
        kept = {k: v for k, v in bm_old.items() if not drop_lo <= k <= drop_hi}
        theta = _vratio(kept, bm_new, p, ctx) if bm_new else 0
        if bm_new:
            theta = _vratio(kept, bm_new, p, ctx)
            # kept = theta' * bm_new: we need m_new with m_new*bm_new = m*kept
            return [f'sm{theta}'] if theta != 1 else []
        if kept:
            raise AssertionError(f'lemma failure ({ctx}): exports survive '
                                 f'without a basis')
        return ['sm0'] if bm_old else []

    def _site_ops(self, sites, T, t, lo, hi, B_C, B_S, kind):
        """The stretch above a site node t whose (possibly joint) child
        covers [lo, hi]: m rebase, read-offs, claim work, bank work."""
        p = self.p
        ops = []
        if kind == XSITE:
            # m rebase first (read-offs emit in the S basis)
            ops += self._m_rebase_ops(B_C['Bm'], B_S['Bm'], t, t - 1, p,
                                      'Bm x-site')
            # read-off of pairs {t, i}: targets outside S
            out_block = self._pair_block(
                sites, T, t, lo, hi, lambda v: not lo <= v <= t)
            crm = _vratio2(out_block, B_C['Vy'], B_S['Bm'], p,
                           'read-off -> exports')
            if crm:
                ops.append(f'rm{crm}')
            # read-off targets inside (discharge the claim)
            in_block = self._pair_block(sites, T, t, lo, hi,
                                        lambda v: lo <= v <= hi)
            crg = _vratio2(in_block, B_C['Vpy'], B_C['w'], p,
                           'read-off -> claims')
            if crg:
                ops.append(f'rg{(-crg) % p}')
            ops += self._claim_rebase_ops(B_C['w'], B_S['w'], p,
                                          'claim rebase x-site')
            ops += self._bank_ops(B_C, B_S, lo, hi, t, t, p)
        else:
            # z-site: the residual enters the claim; cm couples m
            cm = B_C['Bm'].get(t, 0)
            wC, wS = B_C['w'], B_S['w']
            if wC:
                phi = _vratio(self._restrict(wS, lo, hi), wC, p,
                              'claim extend restriction')
                omega = wS.get(t, 0)
                if phi:
                    inv = pow(phi, p - 2, p)
                    c2 = omega * inv % p
                    ops.append(f'zc{cm}{c2}')
                    if inv != 1:
                        ops.append(f'sg{inv}')
                else:
                    ops.append('g0')
                    if omega:
                        ops.append(f'za{pow(omega, p - 2, p)}{cm}')
                    else:
                        ops.append(f'z0{cm}')
            else:
                omega = wS.get(t, 0)
                if omega:
                    ops.append(f'za{pow(omega, p - 2, p)}{cm}')
                else:
                    ops.append(f'z0{cm}')
            ops += self._m_rebase_ops(B_C['Bm'], B_S['Bm'], t, t, p,
                                      'Bm z-site')
            ops += self._bank_ops(B_C, B_S, lo, hi, None, None, p)
        return ops

    # ---------------- encodings and class operations ----------------

    def encode(self, element, sites: CocycleSites, advice: Tree) -> Tree:
        """Element tree of the advice's shape: each site's digit repeats
        along its stretch (micro-ops inherit the digit of the site below)."""
        b, a = element
        if len(b) != len(sites.Z) or len(a) != len(sites.X):
            raise ValueError("element does not fit the site layout")
        if not all(0 <= d < self.p for d in tuple(b) + tuple(a)):
            raise ValueError(f"components must lie in Z_{self.p}")
        zi = {v: idx for idx, v in enumerate(sites.Z)}
        xi = {w: idx for idx, w in enumerate(sites.X)}
        counter = [0]

        def build(node):
            if node is None:
                return None, None
            left, ldig = build(node.left)
            right, rdig = build(node.right)
            letter = node.label
            if letter in ('L', 'K', 'U', 'V') or letter[0] in ('M', 'W'):
                counter[0] += 1
                t = counter[0]
                digit = (str(a[xi[t]]) if sites.site[t] == XSITE
                         else str(b[zi[t]]))
            else:
                digit = ldig if ldig is not None else rdig
            return Tree(digit, left, right), digit

        tree, _ = build(advice)
        if counter[0] != sites.n_sites:
            raise ValueError("advice does not match the site layout")
        return tree

    def multiply(self, sites: CocycleSites, T: Dict, g, h):
        return sites.multiply(T, g, h)

    def evaluate(self, phi):
        return self.cls.evaluate(phi)

    def check(self, phi, sites: CocycleSites, advice: Tree, **elements) -> bool:
        trees = {name: self.encode(el, sites, advice)
                 for name, el in elements.items()}
        return self.cls.check(phi, advice, **trees)

    def get_structure(self, advice: Tree) -> TreeAutomaticPresentation:
        return self.cls.get_structure(advice)
