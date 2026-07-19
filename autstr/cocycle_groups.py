"""Distributed-center class-2 groups: cocycle tensors on site trees.

Validation layer for the tensor cut-rank generalisation of the bounded
rank-width group classes (the companion paper's master theorem). A *site
tree* is a
binary tree whose nodes are generators: 'x' sites and central 'z' sites.
The commutator data is a tensor T[j, i, v] over the chain ring R = Z/p^d
(i < j x-positions in post-order, v a z-position), presenting the central
extension with the bilinear cocycle

    (b, a)(b', a') = (b + b' + C(a, a'), a + a'),
    C(a, a')_v     = sum_{i<j} T[j, i, v] a_j a'_i .

Both coordinate blocks range over R = Z/p^d (an exponent-p^d center forces
an exponent-p^d quotient); the default d = 1 is the field case R = F_p. The
width is the *module cut-rank*: the minimal number of generators of each
flattening's module, which ``chain_ring`` computes via Smith normal form.

This module provides the reference group law, the *six crossing
flattenings* whose module ranks measure, per subtree cut, the traffic a
bottom-up automaton must carry -- upward digit functionals (F_y, F_x),
upward pair-sums (F_m), inward claims (F_g), and the mixed exports whose
products flow back into inside checks (F_py, F_px); reshaping changes rank,
so the width is their maximum -- and `CocycleRankWidthGroups`, the
uniformly tree-automatic presentation realising the master theorem's
six-register protocol at any width r and depth d. The classes
`CutRankTreeGroups` (all z-sites on a chain above the root) and
`TreeExtraspecialGroups` (z-sites at the leaves, laminar targets, width 1)
are corner cases.
"""
import itertools as it
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

from autstr import chain_ring as cr
from autstr.sparse_tree_automata import SparseTreeAutomaton, Tree
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
    """A site tree over the chain ring R = Z/p^d with its cocycle tensors:
    reference law and cut-width measures.

    Elements are (b, a) with b the center coordinates (one per z-site) and a
    the quotient coordinates (one per x-site), both over R = Z/p^d and in
    ascending post-order. With ``d = 1`` (the default) R is the field F_p and
    this is exactly the original field construction; ``d > 1`` is the
    exponent-p^d case (center of exponent p^d), where the
    commutator cocycle is R-bilinear and the tensor coefficients live in R. The
    quotient shares the exponent p^d: a class-2 group whose commutator subgroup
    has exponent p^d cannot have an exponent-p quotient.

    The width measure ``cut_width`` is the *module cut-rank*: the free rank of
    the saturated interface (``chain_ring.module_cut_rank``), which coincides
    with the ordinary F_p flattening rank when d = 1 but, over the ring,
    correctly counts valuation-carrying (p-divisible) generators that vanish
    under a naive mod-p reduction.
    """

    def __init__(self, p: int, shape: Tree, d: int = 1):
        if p < 2 or any(p % f == 0 for f in range(2, int(p ** 0.5) + 1)):
            raise ValueError(f"p must be prime, got {p}")
        if d < 1:
            raise ValueError(f"center ring depth d must be >= 1, got {d}")
        self.p = p
        self.d = d
        self.q = p ** d            # size of the center ring R = Z/p^d
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
        """The reference group law of G(T) over R = Z/p^d.

        Both the center coordinates b and the quotient coordinates a range over
        R; the cocycle C is R-bilinear, which is what makes the law associative
        over the ring. (Keeping a over F_p while C is R-valued would break
        associativity for d > 1: a carry a_j + a'_j >= p drops a term p*T*a''
        that is nonzero mod p^d. In a class-2 group with commutator subgroup of
        exponent p^d the quotient necessarily has exponent p^d as well, since
        [x,y]^p = [x^p,y] = 1 would otherwise force G' to have exponent p.)"""
        (b1, a1), (b2, a2) = g, h
        b = [(u + w) % self.q for u, w in zip(b1, b2)]
        for (j, i, v), coeff in T.items():
            b[self._zi[v]] = (b[self._zi[v]]
                              + coeff * a1[self._xi[j]] * a2[self._xi[i]]) % self.q
        return tuple(b), tuple((u + w) % self.q for u, w in zip(a1, a2))

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
    def _rank(entries: Dict, p: int, d: int = 1) -> int:
        """The module cut-rank of one flattening over R = Z/p^d: the free rank
        of the saturation of its row module (``chain_ring.module_cut_rank``).
        At d = 1 this is the ordinary F_p rank; at d > 1 it correctly counts
        p-divisible generators (a valuation-1 coefficient still contributes a
        rank), which a naive mod-p reduction would drop."""
        if not entries:
            return 0
        rows = sorted({r for r, _ in entries})
        cols = sorted({c for _, c in entries})
        M = np.zeros((len(rows), len(cols)), dtype=np.int64)
        rindex = {r: a for a, r in enumerate(rows)}
        cindex = {c: a for a, c in enumerate(cols)}
        q = p ** d
        for (r, c), coeff in entries.items():
            M[rindex[r], cindex[c]] = coeff % q
        return cr.module_cut_rank(M, p, d)

    def cut_profile(self, T: Dict) -> Dict[int, Dict[str, int]]:
        """For every proper subtree cut (keyed by its root position), the
        module cut-ranks of the six crossing flattenings over R = Z/p^d."""
        self.check_tensor(T)
        profile = {}
        for node in self.seq:
            t, sz = self.pos[id(node)], self.size[id(node)]
            if sz == self.n_sites:
                continue
            entries = self._flattening_entries(T, t - sz + 1, t)
            profile[t] = {name: self._rank(entries[name], self.p, self.d)
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
                  k: int, d: int = 1) -> Tuple[CocycleSites, Dict]:
    """Embed a `CutRankTreeGroups` instance: an all-x copy of the layout
    with a chain of k z-sites above the root. Positions of the x-layout are
    preserved; the z-chain occupies positions n+1 (innermost = center
    coordinate 0) through n+k. With ``d > 1`` the sites and the form labels
    live over the chain ring R = Z/p^d."""
    q = p ** d
    def convert(node):
        if node is None:
            return None
        return Tree(XSITE, convert(node.left), convert(node.right))
    site_root = convert(layout_shape)
    n = len(_layout(layout_shape)[0])
    for _ in range(k):
        site_root = Tree(ZSITE, site_root, None)
    sites = CocycleSites(p, site_root, d=d)
    T = {}
    for (j, i), label in form.items():
        for l in range(k):
            if label[l] % q:
                T[(j, i, n + 1 + l)] = label[l] % q
    return sites, T


def laminar_sites(p: int, shape: Tree, d: int = 1) -> Tuple[CocycleSites, Dict, Dict]:
    """Embed a `TreeExtraspecialGroups` instance: every inner node w of the
    shape becomes a chain x-site (for x_w) over x-site (for y_w), every leaf
    becomes a z-site, and [x_w, y_w] hits every leaf below w. Returns the
    sites, the tensor, and the address map {shape address: site positions}.
    With ``d > 1`` the sites live over the chain ring R = Z/p^d (the tensor
    keeps unit coefficients; the center gains exponent p^d)."""
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
    sites = CocycleSites(p, site_root, d=d)
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


def point_target_sites(p: int, shape: Tree, d: int = 1) -> Tuple[CocycleSites, Dict]:
    """A width-1 family covered by neither corner class: as `laminar_sites`,
    but every commutator [x_w, y_w] hits only the *leftmost* leaf below w
    (point targets: the center grows with the tree, the law is not laminar).
    With ``d > 1`` the sites live over the chain ring R = Z/p^d."""
    sites, T_laminar, maps = laminar_sites(p, shape, d=d)
    T = {}
    for (xw, yw, v) in T_laminar:
        best = min(u for (a, b, u) in T_laminar if (a, b) == (xw, yw))
        T[(xw, yw, best)] = 1
    return sites, T


def scattered_sites(p: int, m: int, d: int = 1) -> Tuple[CocycleSites, Dict]:
    """The lower-bound family: m private z-sites on a chain at the bottom,
    m commuting x-pairs above, pair t targeting exactly z_t. The cut at the
    z-chain has an identity claim flattening: width m (the module cut-rank
    over R = Z/p^d equally, for ``d > 1``)."""
    node = None
    for _ in range(m):
        node = Tree(ZSITE, node, None)
    for _ in range(2 * m):
        node = Tree(XSITE, node, None)
    sites = CocycleSites(p, node, d=d)
    T = {}
    for t in range(1, m + 1):
        i = m + 2 * t - 1
        j = m + 2 * t
        T[(j, i, t)] = 1
    return sites, T


# ====================================================================
# The claim-and-verify automaton: the six-register protocol
# ====================================================================

class CocycleRankWidthGroups:
    """The uniformly tree-automatic class of distributed-center class-2
    groups of tensor cut-rank <= r over R = Z/p^d -- the implementation of
    the master theorem (paper: thm:master). The bottom-up automaton's
    state is six R^r registers (plus three scratch slots):

        wy, wx : upward digit functionals   (column modules of F_y, F_x)
        qy, hx : mixed exports for inside targets       (F_py, F_px)
        m      : representative of the exports element of E_S = rowsp(F_m)
        g      : representative of the residuals element of the claim
                 module Gamma_S = colsp(F_g); a residual leaving the
                 module is a rejection

    The advice is a *table-driven instruction stream*: each site expands
    into its marker letter followed by a chain of operations, each either
    linear -- a streamed matrix or injection column, the folds and
    read-off coefficients of the restriction calculus (paper
    lem:restrict) -- or a streamed *table* keyed on register values and
    the residual digit: the pairing tables of lem:tables and the claim
    extensions and joins of lem:claims. Over the ring these functions are
    well-defined and bilinear on the interface images but need not be
    matrices. Merges are one-step: the binary marker consumes both
    children's raw registers and the stretch above it folds directly to
    the parent cut; no joint-interval interfaces exist.

    `advice(sites, T)` compiles a tensor of module cut-width <= r; every
    linear coefficient and every table entry is derived through
    `chain_ring` solves whose solvability instantiates a lemma of the
    paper, guarded by an assertion -- a compilation that succeeds is a
    machine-checked certificate for that instance. Interfaces are minimal
    generating sets of the flattening modules (Smith normal form with the
    p-power factors kept); at d = 1 they are ordinary bases and every
    table is semantically a matrix, though the letter format is uniform
    in d.

    The explicit presentation automata are beyond the enumeration builder
    by construction (the instruction phase is part of the state);
    `simulate` runs the exact transition function over the convolved
    trees, and `check_implicit` / `evaluate_implicit` decide first-order
    properties through the functional atoms.
    """

    #: marker letters: (site kind, arity)
    MARKERS = {'L': (XSITE, 0), 'K': (ZSITE, 0), 'U': (XSITE, 1),
               'V': (ZSITE, 1), 'M': (XSITE, 2), 'W': (ZSITE, 2)}
    _XMARK = ('L', 'U', 'M')
    _ZMARK = ('K', 'V', 'W')

    #: register ids: out slots 00-08 (wy wx qy hx m g t0 t1 t2), left
    #: child 10-15, right child 20-25; value sources 30 (x digit), 31
    #: (y digit), 33 (the residual digit d0 = z - x - y)
    _OUT = {'wy': '00', 'wx': '01', 'qy': '02', 'hx': '03',
            'm': '04', 'g': '05', 't0': '06', 't1': '07', 't2': '08'}
    _DIGSRC = {'30': 'x', '31': 'y', '33': 'd0'}

    def __init__(self, p: int, r: int = 1, d: int = 1):
        if p < 2 or p > 9 or any(p % f == 0 for f in range(2, int(p ** 0.5) + 1)):
            raise ValueError(f"p must be a prime in 2..9, got {p}")
        if r < 1:
            raise ValueError(f"width r must be >= 1, got {r}")
        if d < 1:
            raise ValueError(f"center ring depth d must be >= 1, got {d}")
        self.p, self.r, self.d = p, r, d
        self.q = p ** d
        self.digits = [str(v) for v in range(self.q)]
        self._digit_val = {s: int(s) for s in self.digits}
        self.element_letters = set(self.digits)
        self._vecs = list(it.product(range(self.q), repeat=r))
        self._vec_index = {v: i for i, v in enumerate(self._vecs)}
        self._zero = (0,) * r
        self._parsed = {}

    # ---------------- letters ----------------

    def _enc(self, c: int) -> str:
        """A ring scalar as an entry letter: 'e' + d base-p digits."""
        return 'e' + ''.join(str(dd)
                             for dd in cr.to_digits(int(c) % self.q,
                                                    self.p, self.d))

    def _src_size(self, src: str) -> int:
        return self.q if src in self._DIGSRC else self.q ** self.r

    def _parse_letter(self, a):
        """Letter -> spec, memoized; None if malformed. Specs:
        ('mark', kind, arity) | ('P', tgt, src, total) |
        ('D', tgt, digsrc, total) | ('T', tgt, srcs, total) |
        ('C', srcs, total) | ('e', value)."""
        hit = self._parsed.get(a)
        if hit is not None:
            return hit
        out = None
        r = self.r
        if a in self.MARKERS:
            kind, arity = self.MARKERS[a]
            out = ('mark', kind, arity)
        elif a and a[0] == 'e' and len(a) == 1 + self.d \
                and all(c.isdigit() and int(c) < self.p for c in a[1:]):
            out = ('e', cr.from_digits([int(c) for c in a[1:]], self.p))
        elif a and a[0] == 'P' and len(a) == 5:
            out = ('P', a[1:3], a[3:5], r * r)
        elif a and a[0] == 'D' and len(a) == 5 and a[3:5] in self._DIGSRC:
            out = ('D', a[1:3], a[3:5], r)
        elif a and a[0] == 'T' and len(a) >= 4 and a[3].isdigit():
            n = int(a[3])
            if len(a) == 4 + 2 * n:
                srcs = tuple(a[4 + 2 * i: 6 + 2 * i] for i in range(n))
                total = 1
                for s in srcs:
                    total *= self._src_size(s)
                out = ('T', a[1:3], srcs, total * (r + 1))
        elif a and a[0] == 'C' and len(a) >= 2 and a[1].isdigit():
            n = int(a[1])
            if len(a) == 2 + 2 * n:
                srcs = tuple(a[2 + 2 * i: 4 + 2 * i] for i in range(n))
                total = 1
                for s in srcs:
                    total *= self._src_size(s)
                out = ('C', srcs, total)
        if out is not None:
            self._parsed[a] = out
        return out

    # ---------------- the transition function ----------------

    def _read(self, frame, digs, src):
        """A source's current value: register tuple or digit scalar."""
        if src in self._DIGSRC:
            return digs[self._DIGSRC[src]]
        bank, idx = src[0], int(src[1])
        if bank == '0':
            return frame[1][idx]
        if bank == '1':
            return frame[2][idx]
        return frame[3][idx]

    def _combo_index(self, frame, digs, srcs):
        """The mixed-radix index of the sources' current values, in the
        enumeration order the compiler streams tables in."""
        idx = 0
        for src in srcs:
            v = self._read(frame, digs, src)
            idx = idx * self._src_size(src) + \
                (v if src in self._DIGSRC else self._vec_index[v])
        return idx

    def _digs(self, x, y, z):
        dx = self._digit_val.get(x)
        dy = self._digit_val.get(y)
        dz = self._digit_val.get(z)
        if dx is None or dy is None or dz is None:
            return None
        return {'x': dx, 'y': dy, 'd0': (dz - dx - dy) % self.q}

    def _m_step(self, lq, rq, sym):
        """One bottom-up step: sym = (advice letter, x, y, z labels).
        States are frames ('f', out9, inL6, inR6, op, phase); an op in
        progress carries the combo index it froze at its header."""
        q, r = self.q, self.r
        if lq == 'dead' or rq == 'dead':
            return 'dead'
        a, x, y, z = sym
        digs = self._digs(x, y, z)
        spec = self._parse_letter(a)
        if digs is None or spec is None:
            return 'dead'
        if spec[0] == 'mark':
            _, kind, arity = spec
            kids = [s for s in (lq, rq) if s is not None]
            if len(kids) != arity:
                return 'dead'
            regs = []
            for k in kids:
                if k[0] != 'f' or k[4] is not None:
                    return 'dead'
                regs.append(tuple(k[1][:6]))
            zero6 = (self._zero,) * 6
            inL = regs[0] if arity >= 1 else zero6
            inR = regs[1] if arity == 2 else zero6
            return ('f', (self._zero,) * 9, inL, inR, None, 0)
        # operation letters are unary
        child = lq if rq is None else (rq if lq is None else None)
        if child is None or child[0] != 'f':
            return 'dead'
        _, out, inL, inR, op, ph = child
        if spec[0] in ('P', 'D'):
            if op is not None:
                return 'dead'
            return ('f', out, inL, inR, spec, 0)
        if spec[0] == 'T':
            if op is not None:
                return 'dead'
            myidx = self._combo_index(child, digs, spec[2])
            return ('f', out, inL, inR, spec + (myidx,), 0)
        if spec[0] == 'C':
            if op is not None:
                return 'dead'
            myidx = self._combo_index(child, digs, spec[1])
            return ('f', out, inL, inR, spec + (myidx,), 0)
        if spec[0] != 'e' or op is None:
            return 'dead'
        val = spec[1]
        kind = op[0]

        def put(o, ti, i, v):
            row = list(o[ti])
            row[i] = v % q
            return o[:ti] + (tuple(row),) + o[ti + 1:]

        if kind == 'P':
            _, tgt, src, total = op
            i, j = divmod(ph, r)
            w = self._read(child, digs, src)
            ti = int(tgt[1])
            out = put(out, ti, i, out[ti][i] + val * w[j])
        elif kind == 'D':
            _, tgt, dsrc, total = op
            ti = int(tgt[1])
            out = put(out, ti, ph,
                      out[ti][ph] + val * digs[self._DIGSRC[dsrc]])
        elif kind == 'T':
            _, tgt, srcs, total, myidx = op
            combo, slot = divmod(ph, r + 1)
            if combo == myidx:
                if slot == 0:
                    if val == 0:
                        return 'dead'
                else:
                    out = put(out, int(tgt[1]), slot - 1, val)
        else:                                            # 'C'
            _, srcs, total, myidx = op
            if ph == myidx and val == 0:
                return 'dead'
        ph += 1
        if ph == op[-2] if kind in ('T', 'C') else ph == op[3]:
            return ('f', out, inL, inR, None, 0)
        return ('f', out, inL, inR, op, ph)

    def _m_accepting(self, state) -> bool:
        return state != 'dead' and state is not None and state[0] == 'f' \
            and state[4] is None

    # ---------------- interfaces and elements ----------------

    def _generators(self, vectors, width_ctx):
        """A minimal generating set (<= r; Smith normal form with the
        p-power factors kept) of the module spanned by the vectors."""
        vecs = [v for v in vectors if any(c % self.q for c in v.values())]
        if not vecs:
            return []
        keys = sorted({k for v in vecs for k in v})
        M = np.array([[v.get(k, 0) % self.q for k in keys] for v in vecs],
                     dtype=np.int64)
        basis, exps = cr.saturate(M, self.p, self.d)
        if basis.shape[0] > self.r:
            raise ValueError(
                f"{width_ctx}: module rank {basis.shape[0]} > r = {self.r}")
        out = []
        for row, e in zip(basis, exps):
            scaled = {k: int(row[i] * self.p ** e) % self.q
                      for i, k in enumerate(keys)
                      if row[i] * self.p ** e % self.q}
            out.append(scaled)
        return out

    def _elem(self, gens, rep):
        """The module element represented by `rep` over the generators."""
        out = {}
        for c, g in zip(rep, gens):
            if c % self.q == 0:
                continue
            for k, v in g.items():
                out[k] = (out.get(k, 0) + c * v) % self.q
        return {k: v for k, v in out.items() if v}

    def _canon(self, gens, elem, width=None):
        """The canonical representative of an element, or None if it lies
        outside the module (the membership check of the protocol)."""
        width = self.r if width is None else width
        elem = {k: v % self.q for k, v in elem.items() if v % self.q}
        if not elem:
            return (0,) * width
        if not gens:
            return None
        keys = sorted({k for g in gens for k in g} | set(elem))
        V = np.array([[g.get(k, 0) for k in keys] for g in gens],
                     dtype=np.int64)
        B = np.array([[elem.get(k, 0) for k in keys]], dtype=np.int64)
        try:
            X = cr.solve_left(V, B, self.p, self.d)
        except ValueError:
            return None
        rep = [0] * width
        for i in range(len(gens)):
            rep[i] = int(X[0, i]) % self.q
        return tuple(rep)

    def _coeffs(self, gens, vec, ctx, width=None):
        """`vec` as an R-combination of the generators (a restriction-lemma
        certificate); AssertionError if it is not one."""
        rep = self._canon(gens, vec, width=width)
        if rep is None:
            raise AssertionError(f"lemma failure ({ctx}): vector outside "
                                 f"the generated module")
        return rep

    def _span_preimages(self, gens):
        """The image of a |-> (gens . a) with one preimage per value (at
        most q^r values, closure of the column span)."""
        keys = sorted({k for g in gens for k in g})
        span = {self._zero: {}}
        frontier = [self._zero]
        while frontier:
            nxt = []
            for w in frontier:
                a = span[w]
                for k in keys:
                    col = tuple(g.get(k, 0) for g in gens) + \
                        (0,) * (self.r - len(gens))
                    w2 = tuple((wv + col[i]) % self.q
                               for i, wv in enumerate(w))
                    if w2 not in span:
                        a2 = dict(a)
                        a2[k] = (a2.get(k, 0) + 1) % self.q
                        span[w2] = a2
                        nxt.append(w2)
            frontier = nxt
        return span

    def _value(self, lam, w):
        return sum(l * wv for l, wv in zip(lam, w)) % self.q

    def _add_elems(self, gens_a, rep_a, gens_b, rep_b, sign=1):
        ea = self._elem(gens_a, rep_a)
        for k, v in self._elem(gens_b, rep_b).items():
            ea[k] = (ea.get(k, 0) + sign * v) % self.q
        return ea

    # ---------------- op emission ----------------

    def _emit_P(self, out, tgt, src, mat):
        if not any(any(c % self.q for c in row) for row in mat):
            return
        out.append('P' + tgt + src)
        for row in mat:
            for c in row:
                out.append(self._enc(c))

    def _emit_D(self, out, tgt, dsrc, col):
        if not any(c % self.q for c in col):
            return
        out.append('D' + tgt + dsrc)
        for c in col:
            out.append(self._enc(c))

    def _spaces(self, srcs):
        return [range(self.q) if s in self._DIGSRC else self._vecs
                for s in srcs]

    def _emit_T(self, out, tgt, srcs, fn):
        out.append('T' + tgt + str(len(srcs)) + ''.join(srcs))
        for combo in it.product(*self._spaces(srcs)):
            rep = fn(*combo)
            if rep is None:
                out.append(self._enc(0))
                out.extend(self._enc(0) for _ in range(self.r))
            else:
                out.append(self._enc(1))
                out.extend(self._enc(c) for c in rep)

    def _emit_C(self, out, srcs, fn):
        out.append('C' + str(len(srcs)) + ''.join(srcs))
        for combo in it.product(*self._spaces(srcs)):
            out.append(self._enc(1 if fn(*combo) else 0))

    def _emit_add(self, out, tgt, other, gens, sign=1):
        """tgt := tgt (+/-) other, both representatives over `gens`."""
        self._emit_T(out, tgt, (tgt, other),
                     lambda a_, b_: self._canon(
                         gens, self._add_elems(gens, a_, gens, b_,
                                               sign=sign)))

    # ---------------- the compiler ----------------

    def _bank_data(self, sites, T, lo, hi):
        """The six interface generator sets of the cut [lo, hi]."""
        ent = sites._flattening_entries(T, lo, hi)

        def group(e, own_of, other_of):
            cols = {}
            for k, c in e.items():
                cols.setdefault(other_of(k), {})[own_of(k)] = c % self.q
            return list(cols.values())

        w = f"cut [{lo}, {hi}]"
        first = lambda k: k[0]
        second = lambda k: k[1]
        return {
            'Vy': self._generators(group(ent['F_y'], first, second), w),
            'Vx': self._generators(group(ent['F_x'], first, second), w),
            'Vpy': self._generators(group(ent['F_py'], first, second), w),
            'Vpx': self._generators(group(ent['F_px'], first, second), w),
            'Em': self._generators(group(ent['F_m'], second, first), w),
            'Cg': self._generators(group(ent['F_g'], first, second), w),
        }

    def _fold_ops(self, out, banks_S, banks_C, in_prefix, lo_c, hi_c, t,
                  inject_kind):
        """Fold the four functional registers from a child cut into the
        parent registers (paper lem:restrict(1)); at an x-site, inject the
        parent's own digit column once (`inject_kind` gates it so a binary
        node injects during one child pass only)."""
        table = (('Vy', 'wy', None, '31'), ('Vx', 'wx', None, '30'),
                 ('Vpy', 'qy', 'Vy', '31'), ('Vpx', 'hx', 'Vx', '30'))
        mixreg = {'Vy': 'wy', 'Vx': 'wx'}
        inside = lambda k: lo_c <= k <= hi_c
        for bank, reg, mix, dsrc in table:
            gens_S = banks_S[bank]
            stacked = banks_C[bank] + (banks_C[mix] if mix else [])
            n1 = len(banks_C[bank])
            mat_main = [[0] * self.r for _ in range(self.r)]
            mat_mix = [[0] * self.r for _ in range(self.r)]
            for i, g in enumerate(gens_S):
                restr = {k: v for k, v in g.items() if inside(k)}
                if not any(v % self.q for v in restr.values()):
                    continue
                lam = self._coeffs(stacked, restr,
                                   f"{bank} fold at {t}",
                                   width=max(len(stacked), 1))
                for jdx, c in enumerate(lam[:len(stacked)]):
                    if jdx < n1:
                        mat_main[i][jdx] = c
                    else:
                        mat_mix[i][jdx - n1] = c
            self._emit_P(out, self._OUT[reg],
                         in_prefix + self._OUT[reg][1], mat_main)
            if mix:
                self._emit_P(out, self._OUT[reg],
                             in_prefix + self._OUT[mixreg[mix]][1],
                             mat_mix)
            if inject_kind == XSITE:
                col = [g.get(t, 0) for g in gens_S]
                col += [0] * (self.r - len(col))
                self._emit_D(out, self._OUT[reg], dsrc, col)

    def _restrict_fn(self, gens_from, gens_to, keep, ctx):
        def fn(rep):
            elem = {k: v for k, v in self._elem(gens_from, rep).items()
                    if keep(k)}
            out = self._canon(gens_to, elem)
            if out is None:
                raise AssertionError(f"lemma failure ({ctx})")
            return out
        return fn

    def _pairing_certify(self, block, gens_R, gens_L, ctx):
        """Membership certificates of a target-keyed sibling block
        (paper lem:tables): i-vectors in the L-module, j-vectors in the
        R-module."""
        rows, cols = {}, {}
        for v, b in block.items():
            for (j, i), c in b.items():
                rows.setdefault((j, v), {})[i] = c
                cols.setdefault((i, v), {})[j] = c
        for vec in rows.values():
            self._coeffs(gens_L, vec, ctx)
        for vec in cols.values():
            self._coeffs(gens_R, vec, ctx)

    def _pairing_fn(self, block, gens_R, gens_L, gens_out, ctx):
        """The pairing table of a target-keyed sibling block: the two
        register values determine the represented target element."""
        self._pairing_certify(block, gens_R, gens_L, ctx)
        spanR = self._span_preimages(gens_R)
        spanL = self._span_preimages(gens_L)

        def fn(wr, wl):
            aR, aL = spanR.get(wr), spanL.get(wl)
            if aR is None or aL is None:
                return self._zero
            elem = {}
            for v, b in block.items():
                s = 0
                for (j, i), c in b.items():
                    s += c * aR.get(j, 0) * aL.get(i, 0)
                if s % self.q:
                    elem[v] = s % self.q
            rep = self._canon(gens_out, elem)
            if rep is None:
                raise AssertionError(f"lemma failure ({ctx})")
            return rep
        return fn

    def _readoff_split(self, T, t, lo_c, hi_c, lo, hi):
        """Coefficient vectors of the pairs {t, i}, i in the child
        interval, split by target: (outside S, inside the child,
        inside the sibling)."""
        outs, own, sib = {}, {}, {}
        for (j, i, v), c in T.items():
            if j != t or not lo_c <= i <= hi_c:
                continue
            if not lo <= v <= hi:
                outs.setdefault(v, {})[i] = c % self.q
            elif lo_c <= v <= hi_c:
                own.setdefault(v, {})[i] = c % self.q
            else:
                sib.setdefault(v, {})[i] = c % self.q
        return outs, own, sib

    def _readoff_map(self, blocks, gens_in, gens_out, ctx):
        """The read-off table: the x digit and one child register value
        determine the represented target element. Register values outside
        the interface image cannot occur in a live run and get don't-care
        entries; for values in the image the membership is a lemma
        instance."""
        lams = {v: self._coeffs(gens_in, b, ctx) for v, b in blocks.items()}
        image = set(self._span_preimages(gens_in))

        def fn(xd, w):
            if w not in image:
                return self._zero
            elem = {v: (xd * self._value(lam, w)) % self.q
                    for v, lam in lams.items()}
            rep = self._canon(gens_out, elem)
            if rep is None:
                raise AssertionError(f"lemma failure ({ctx})")
            return rep
        return fn

    def advice(self, sites: CocycleSites, T: Dict) -> Tree:
        """Compile a tensor of module cut-width <= r into the instruction
        stream; ValueError beyond the width, AssertionError on any failed
        lemma instance."""
        if sites.p != self.p or sites.d != self.d:
            raise ValueError("sites and class must share p and d")
        sites.check_tensor(T)
        width = sites.cut_width(T)
        if width > self.r:
            raise ValueError(f"tensor has cut-width {width} > r = {self.r}")
        OUT = self._OUT
        banks: Dict[int, Dict] = {}
        built: Dict[int, Tree] = {}

        for node in sites.seq:
            t = sites.pos[id(node)]
            sz = sites.size[id(node)]
            lo, hi = t - sz + 1, t
            kind = sites.site[t]
            B_S = self._bank_data(sites, T, lo, hi)
            Lc, Rc = node.left, node.right
            ops: list = []

            if kind == XSITE:
                self._emit_C(ops, ('33',), lambda d0: d0 == 0)

            if Lc is None and Rc is None:                       # leaf
                if kind == XSITE:
                    for bank, reg, dsrc in (('Vy', 'wy', '31'),
                                            ('Vx', 'wx', '30'),
                                            ('Vpy', 'qy', '31'),
                                            ('Vpx', 'hx', '30')):
                        col = [g.get(t, 0) for g in B_S[bank]]
                        col += [0] * (self.r - len(col))
                        self._emit_D(ops, OUT[reg], dsrc, col)
                else:
                    self._emit_T(ops, OUT['g'], ('33',),
                                 lambda d0: self._canon(B_S['Cg'],
                                                        {t: d0}))
                marker = 'L' if kind == XSITE else 'K'
                sub = Tree(marker)

            elif Lc is None or Rc is None:                      # unary
                child = Lc if Lc is not None else Rc
                cp = sites.pos[id(child)]
                csz = sites.size[id(child)]
                B_C = banks[cp]
                lo_c, hi_c = cp - csz + 1, cp
                self._fold_ops(ops, B_S, B_C, '1', lo_c, hi_c, t, kind)
                outside = lambda v: not lo <= v <= hi
                if kind == XSITE:
                    self._emit_T(ops, OUT['m'], ('14',),
                                 self._restrict_fn(
                                     B_C['Em'], B_S['Em'], outside,
                                     f"exports restrict at {t}"))
                    self._emit_T(ops, OUT['t1'], ('15',),
                                 lambda grep: tuple(grep))
                    outs, own, sib = self._readoff_split(T, t, lo_c, hi_c,
                                                         lo, hi)
                    if outs:
                        self._emit_T(ops, OUT['t0'], ('30', '10'),
                                     self._readoff_map(
                                         outs, B_C['Vy'], B_S['Em'],
                                         f"read-off exports at {t}"))
                        self._emit_add(ops, OUT['m'], OUT['t0'],
                                       B_S['Em'])
                    if own:
                        self._emit_T(ops, OUT['t0'], ('30', '12'),
                                     self._readoff_map(
                                         own, B_C['Vpy'], B_C['Cg'],
                                         f"read-off claims at {t}"))
                        self._emit_add(ops, OUT['t1'], OUT['t0'],
                                       B_C['Cg'], sign=-1)
                    self._emit_T(ops, OUT['g'], (OUT['t1'],),
                                 lambda grep: self._canon(
                                     B_S['Cg'],
                                     self._elem(B_C['Cg'], grep)))
                else:
                    Em_C, Cg_C = B_C['Em'], B_C['Cg']

                    def fn_ext(grep, mrep, d0):
                        rho = (d0 - self._elem(Em_C, mrep).get(t, 0)) \
                            % self.q
                        elem = self._elem(Cg_C, grep)
                        elem[t] = rho
                        return self._canon(B_S['Cg'], elem)

                    self._emit_T(ops, OUT['g'], ('15', '14', '33'), fn_ext)
                    self._emit_T(ops, OUT['m'], ('14',),
                                 self._restrict_fn(
                                     B_C['Em'], B_S['Em'], outside,
                                     f"exports restrict at {t}"))
                marker = 'U' if kind == XSITE else 'V'
                sub = Tree(marker, built[id(child)], None)

            else:                                               # binary
                lp, rp = sites.pos[id(Lc)], sites.pos[id(Rc)]
                lsz, rsz = sites.size[id(Lc)], sites.size[id(Rc)]
                B_L, B_R = banks[lp], banks[rp]
                lo_l, hi_l = lp - lsz + 1, lp
                lo_r, hi_r = rp - rsz + 1, rp
                outside = lambda v: not lo <= v <= hi
                in_l = lambda v: lo_l <= v <= hi_l
                in_r = lambda v: lo_r <= v <= hi_r
                self._fold_ops(ops, B_S, B_L, '1', lo_l, hi_l, t, kind)
                self._fold_ops(ops, B_S, B_R, '2', lo_r, hi_r, t, ZSITE)
                # exports: restrict and add both children
                self._emit_T(ops, OUT['m'], ('14',),
                             self._restrict_fn(
                                 B_L['Em'], B_S['Em'], outside,
                                 f"exports restrict L at {t}"))
                self._emit_T(ops, OUT['t0'], ('24',),
                             self._restrict_fn(
                                 B_R['Em'], B_S['Em'], outside,
                                 f"exports restrict R at {t}"))
                self._emit_add(ops, OUT['m'], OUT['t0'], B_S['Em'])
                # split-pair blocks by target
                blocks = {}
                for (j, i, v), c in T.items():
                    if lo_l <= i <= hi_l and lo_r <= j <= hi_r:
                        blocks.setdefault(v, {})[(j, i)] = c % self.q
                b_out = {v: b for v, b in blocks.items() if outside(v)}
                b_L = {v: b for v, b in blocks.items() if in_l(v)}
                b_R = {v: b for v, b in blocks.items() if in_r(v)}
                b_t = {t: blocks[t]} if t in blocks else {}
                if b_out:
                    self._emit_T(ops, OUT['t0'], ('21', '10'),
                                 self._pairing_fn(
                                     b_out, B_R['Vx'], B_L['Vy'],
                                     B_S['Em'],
                                     f"sibling exports at {t}"))
                    self._emit_add(ops, OUT['m'], OUT['t0'], B_S['Em'])
                # claims: initialise both, then discharge
                self._emit_T(ops, OUT['t1'], ('15',),
                             lambda grep: tuple(grep))
                self._emit_T(ops, OUT['t2'], ('25',),
                             lambda grep: tuple(grep))
                if b_L:
                    self._emit_T(ops, OUT['t0'], ('21', '12'),
                                 self._pairing_fn(
                                     b_L, B_R['Vx'], B_L['Vpy'],
                                     B_L['Cg'],
                                     f"sibling claims L at {t}"))
                    self._emit_add(ops, OUT['t1'], OUT['t0'], B_L['Cg'],
                                   sign=-1)
                if b_R:
                    self._emit_T(ops, OUT['t0'], ('23', '10'),
                                 self._pairing_fn(
                                     b_R, B_R['Vpx'], B_L['Vy'],
                                     B_R['Cg'],
                                     f"sibling claims R at {t}"))
                    self._emit_add(ops, OUT['t2'], OUT['t0'], B_R['Cg'],
                                   sign=-1)
                # cross exports: each child's m at the sibling's targets
                self._emit_T(ops, OUT['t0'], ('24',),
                             self._restrict_fn(
                                 B_R['Em'], B_L['Cg'], in_l,
                                 f"cross exports R->L at {t}"))
                self._emit_add(ops, OUT['t1'], OUT['t0'], B_L['Cg'],
                               sign=-1)
                self._emit_T(ops, OUT['t0'], ('14',),
                             self._restrict_fn(
                                 B_L['Em'], B_R['Cg'], in_r,
                                 f"cross exports L->R at {t}"))
                self._emit_add(ops, OUT['t2'], OUT['t0'], B_R['Cg'],
                               sign=-1)
                if kind == XSITE:
                    # read-offs of {t, i} over each child, targets sorted
                    for pre, blo, bhi, B_C, own_tmp, sib_tmp, B_sib in (
                            ('1', lo_l, hi_l, B_L, OUT['t1'], OUT['t2'],
                             B_R),
                            ('2', lo_r, hi_r, B_R, OUT['t2'], OUT['t1'],
                             B_L)):
                        outs, own, sib = self._readoff_split(
                            T, t, blo, bhi, lo, hi)
                        if outs:
                            self._emit_T(ops, OUT['t0'], ('30', pre + '0'),
                                         self._readoff_map(
                                             outs, B_C['Vy'], B_S['Em'],
                                             f"read-off exports at {t}"))
                            self._emit_add(ops, OUT['m'], OUT['t0'],
                                           B_S['Em'])
                        if own:
                            self._emit_T(ops, OUT['t0'], ('30', pre + '2'),
                                         self._readoff_map(
                                             own, B_C['Vpy'], B_C['Cg'],
                                             f"read-off claims at {t}"))
                            self._emit_add(ops, own_tmp, OUT['t0'],
                                           B_C['Cg'], sign=-1)
                        if sib:
                            self._emit_T(ops, OUT['t0'], ('30', pre + '0'),
                                         self._readoff_map(
                                             sib, B_C['Vy'], B_sib['Cg'],
                                             f"read-off sibling claims "
                                             f"at {t}"))
                            self._emit_add(ops, sib_tmp, OUT['t0'],
                                           B_sib['Cg'], sign=-1)
                    self._emit_T(ops, OUT['g'], (OUT['t1'], OUT['t2']),
                                 lambda gl, gr: self._canon(
                                     B_S['Cg'],
                                     {**self._elem(B_L['Cg'], gl),
                                      **self._elem(B_R['Cg'], gr)}))
                else:
                    # z-site: accumulate the t-coordinate of the exports
                    # and the sibling pairs targeting t, then join with
                    # the residual
                    Em_L, Em_R = B_L['Em'], B_R['Em']
                    self._emit_T(ops, OUT['t0'], ('14', '24'),
                                 lambda mL, mR: (
                                     (self._elem(Em_L, mL).get(t, 0)
                                      + self._elem(Em_R, mR).get(t, 0))
                                     % self.q,) + (0,) * (self.r - 1))
                    if b_t:
                        self._pairing_certify(b_t, B_R['Vx'], B_L['Vy'],
                                              f"sibling residual at {t}")
                        spanR = self._span_preimages(B_R['Vx'])
                        spanL = self._span_preimages(B_L['Vy'])
                        bt = b_t[t]

                        def fn_pt(acc, wr, wl):
                            s = acc[0]
                            aR, aL = spanR.get(wr), spanL.get(wl)
                            if aR is not None and aL is not None:
                                for (j, i), c in bt.items():
                                    s += c * aR.get(j, 0) * aL.get(i, 0)
                            return (s % self.q,) + (0,) * (self.r - 1)

                        self._emit_T(ops, OUT['t0'],
                                     (OUT['t0'], '21', '10'), fn_pt)

                    def fn_join(gl, gr, acc, d0):
                        elem = {**self._elem(B_L['Cg'], gl),
                                **self._elem(B_R['Cg'], gr)}
                        elem[t] = (d0 - acc[0]) % self.q
                        return self._canon(B_S['Cg'], elem)

                    self._emit_T(ops, OUT['g'],
                                 (OUT['t1'], OUT['t2'], OUT['t0'], '33'),
                                 fn_join)
                marker = 'M' if kind == XSITE else 'W'
                sub = Tree(marker, built[id(Lc)], built[id(Rc)])

            for op in ops:
                sub = Tree(op, sub, None)
            built[id(node)] = sub
            banks[t] = B_S

        return built[id(sites.seq[-1])]

    # ---------------- encodings and class operations ----------------

    def encode(self, element, sites: CocycleSites, advice: Tree) -> Tree:
        """Element tree of the advice's shape: each site's digit repeats
        along its instruction stretch."""
        b, a = element
        if len(b) != len(sites.Z) or len(a) != len(sites.X):
            raise ValueError("element does not fit the site layout")
        if not all(0 <= v < self.q for v in tuple(b) + tuple(a)):
            raise ValueError(f"components must lie in Z/{self.q}")
        zi = {v: idx for idx, v in enumerate(sites.Z)}
        xi = {w: idx for idx, w in enumerate(sites.X)}
        counter = [0]

        def build(node):
            if node is None:
                return None, None
            left, ldig = build(node.left)
            right, rdig = build(node.right)
            if node.label in self.MARKERS:
                counter[0] += 1
                t = counter[0]
                digit = (str(a[xi[t]]) if sites.site[t] == XSITE
                         else str(b[zi[t]]))
            else:
                digit = ldig if ldig is not None else rdig
            return Tree(digit, left, right), digit

        tree, _ = self._with_depth(advice, lambda: build(advice))
        if counter[0] != sites.n_sites:
            raise ValueError("advice does not match the site layout")
        return tree

    def decode(self, tree: Tree, advice: Tree):
        """Inverse of `encode`: the digit at each marker; z-sites to b and
        x-sites to a, in ascending post-order."""
        bs, xs = [], []

        def rec(an, en):
            if an is None:
                return
            rec(an.left, en.left)
            rec(an.right, en.right)
            if an.label in self._XMARK:
                xs.append(int(en.label))
            elif an.label in self._ZMARK:
                bs.append(int(en.label))

        self._with_depth(advice, lambda: rec(advice, tree))
        return tuple(bs), tuple(xs)

    def multiply(self, sites: CocycleSites, T: Dict, g, h):
        return sites.multiply(T, g, h)

    def simulate(self, advice: Tree, tx: Tree, ty: Tree, tz: Tree) -> bool:
        """Run the transition function over the convolved trees, with an
        explicit stack (instruction stretches can be long)."""
        results = {}
        stack = [(advice, tx, ty, tz, False)]
        while stack:
            an, xn, yn, zn, done = stack.pop()
            if an is None:
                continue
            if not done:
                stack.append((an, xn, yn, zn, True))
                stack.append((an.left, xn.left if xn else None,
                              yn.left if yn else None,
                              zn.left if zn else None, False))
                stack.append((an.right, xn.right if xn else None,
                              yn.right if yn else None,
                              zn.right if zn else None, False))
                continue
            lq = results.get(id(an.left)) if an.left is not None else None
            rq = results.get(id(an.right)) if an.right is not None else None
            sym = (an.label, xn.label if xn else PAD,
                   yn.label if yn else PAD, zn.label if zn else PAD)
            results[id(an)] = self._m_step(lq, rq, sym)
        return self._m_accepting(results[id(advice)])

    # ---------------- the (gated) explicit presentation ----------------

    @property
    def cls(self):
        raise ValueError(
            "the protocol automaton's explicit presentation is beyond the "
            "enumeration builder (the instruction phase is part of the "
            "state); use check_implicit / evaluate_implicit / simulate")

    def evaluate(self, phi):
        return self.cls

    def check(self, phi, sites: CocycleSites, advice: Tree,
              **elements) -> bool:
        return self.cls

    def get_structure(self, advice: Tree):
        return self.cls

    # ---------------- implicit atoms ----------------

    def _shape_step(self, a, lq, rq):
        spec = self._parse_letter(a)
        if spec is None:
            return None
        if spec[0] == 'mark':
            have = (lq is not None) + (rq is not None)
            return spec if have == spec[2] else None
        return spec if (lq is None) != (rq is None) else None

    def _implicit_atoms(self) -> Dict:
        """Functional bottom-up atoms (Dom, Adv, M, Eq); nothing built."""
        from autstr.implicit import ImplicitTA
        element_letters = self.element_letters

        def shape(args, q_of):
            adv = args[0]

            def step(sym, left, right):
                if left == 'dead' or right == 'dead' or not q_of(sym):
                    return 'dead'
                spec = self._shape_step(sym[adv], left, right)
                if spec is None:
                    return 'dead'
                if spec[0] == 'mark':
                    return ('d', sym[args[1]]) if len(args) > 1 else 'ok'
                if len(args) == 1:
                    return 'ok'
                child = left if right is None else right
                return child if child == ('d', sym[args[1]]) else 'dead'
            return ImplicitTA(args, step,
                              lambda st: st != 'dead' and st is not None)

        def Dom(args):
            return shape(args, lambda sym: sym[args[1]] in element_letters)

        def Adv(args):
            return shape(args, lambda sym: True)

        def Eq(args):
            def step(sym, left, right):
                if left == 'dead' or right == 'dead' \
                        or sym[args[1]] not in element_letters \
                        or sym[args[1]] != sym[args[2]]:
                    return 'dead'
                return 'ok' if self._shape_step(sym[args[0]], left,
                                                right) is not None \
                    else 'dead'
            return ImplicitTA(args, step, lambda st: st == 'ok')

        def M(args):
            adv, xv, yv, zv = args
            return ImplicitTA(
                args,
                lambda sym, left, right: self._m_step(
                    left, right, (sym[adv], sym[xv], sym[yv], sym[zv])),
                self._m_accepting)

        return {'Dom': Dom, 'Adv': Adv, 'M': M, 'Eq': Eq}

    @property
    def implicit_cls(self):
        from autstr.implicit import ImplicitTreeClass
        return ImplicitTreeClass(self._implicit_atoms(),
                                 list(self.element_letters))

    def _with_depth(self, advice, fn):
        """Instruction stretches are long unary chains; raise the
        recursion limit for tree walks proportionally."""
        n = 0
        stack = [advice]
        while stack:
            node = stack.pop()
            if node is None:
                continue
            n += 1
            stack.append(node.left)
            stack.append(node.right)
        old = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old, 4 * n + 10000))
        try:
            return fn()
        finally:
            sys.setrecursionlimit(old)

    def check_implicit(self, phi, sites: CocycleSites, advice: Tree,
                       **elements) -> bool:
        """First-order model checking over the functional atoms."""
        trees = {name: self.encode(el, sites, advice)
                 for name, el in elements.items()}
        return self._with_depth(
            advice, lambda: self.implicit_cls.check(phi, advice, **trees))

    def evaluate_implicit(self, phi, sites: CocycleSites, advice: Tree,
                          **elements):
        """The satisfying set of phi, computed implicitly; yields
        assignments {var: (b, a)}."""
        from autstr.implicit import MappedSolutions
        trees = {name: self.encode(el, sites, advice)
                 for name, el in elements.items()}
        sols = self._with_depth(
            advice,
            lambda: self.implicit_cls.evaluate(phi, advice, **trees))
        return MappedSolutions(sols, lambda tr: self.decode(tr, advice))
