"""Distributed-center class-2 groups: cocycle tensors on site trees.

Validation layer for the tensor cut-rank generalisation of the bounded
rank-width group classes (see paper/theorem3-notes.md). A *site tree* is a
binary tree whose nodes are generators: 'x' sites and central 'z' sites.
The commutator data is a tensor T[j, i, v] over the chain ring R = Z/p^d
(i < j x-positions in post-order, v a z-position), presenting the central
extension with the bilinear cocycle

    (b, a)(b', a') = (b + b' + C(a, a'), a + a'),
    C(a, a')_v     = sum_{i<j} T[j, i, v] a_j a'_i .

The center coordinates b range over R = Z/p^d (exponent-p^d center, "Idea 2"
of paper/scratch-chainring.tex); the quotient coordinates a range over F_p.
The default d = 1 is the field case R = F_p. Over the ring the width is the
*module cut-rank* -- the free rank of the saturated interface -- which the
``chain_ring`` module supplies (Lemma "lem:sat"); the two-sided factorisation
that the tree merge needs also lives there (``chain_ring.factor_two_sided``,
Corollary "cor:merge").

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
    exponent-p^d case of "Idea 2" (paper/scratch-chainring.tex), where the
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
# The claim-and-verify automaton (width 1) with microcode advice
# ====================================================================

def _vratio(target: Dict, base: Dict, p: int, ctx: str, d: int = 1) -> int:
    """The scalar c over R = Z/p^d with target == c * base (as sparse
    vectors); raises AssertionError if target is outside the R-span of base --
    every call is a machine-checked instance of a restriction lemma. At d = 1
    (the field) c is the unique proportionality factor; over the ring c may
    carry valuation and is the minimal `chain_ring.solve_left` solution."""
    q = p ** d
    base = {k: v % q for k, v in base.items() if v % q}
    target = {k: v % q for k, v in target.items() if v % q}
    if not base:
        if target:
            raise AssertionError(f"lemma failure ({ctx}): target outside span")
        return 0
    keys = sorted(set(base) | set(target))
    V = np.array([[base.get(k, 0) for k in keys]], dtype=np.int64)
    B = np.array([[target.get(k, 0) for k in keys]], dtype=np.int64)
    try:
        return int(cr.solve_left(V, B, p, d)[0, 0]) % q
    except ValueError:
        raise AssertionError(f"lemma failure ({ctx}): not proportional")


def _vratio2(entries: Dict, u: Dict, w: Dict, p: int, ctx: str,
             d: int = 1) -> int:
    """The scalar c with entries[(a, b)] == c * u[a] * w[b] over R = Z/p^d."""
    q = p ** d
    outer = {(a, b): u[a] * w[b] % q for a in u for b in w if u[a] * w[b] % q}
    return _vratio(entries, outer, p, ctx, d)


def _vratio3(entries: Dict, u: Dict, w: Dict, x: Dict, p: int, ctx: str,
             d: int = 1) -> int:
    """The scalar c with entries[(a, b, e)] == c * u[a] * w[b] * x[e]
    over R = Z/p^d."""
    q = p ** d
    outer = {(a, b, e): u[a] * w[b] * x[e] % q
             for a in u for b in w for e in x if u[a] * w[b] * x[e] % q}
    return _vratio(entries, outer, p, ctx, d)


def _vsolve2(target: Dict, b1: Dict, b2: Dict, p: int, ctx: str, d: int = 1):
    """(c1, c2) over R = Z/p^d with target == c1*b1 + c2*b2; AssertionError
    if unsolvable."""
    q = p ** d
    target = {k: v % q for k, v in target.items() if v % q}
    if not target:
        return 0, 0
    keys = sorted(set(target) | set(b1) | set(b2))
    V = np.array([[b1.get(k, 0) % q for k in keys],
                  [b2.get(k, 0) % q for k in keys]], dtype=np.int64)
    B = np.array([[target.get(k, 0) for k in keys]], dtype=np.int64)
    try:
        X = cr.solve_left(V, B, p, d)
    except ValueError:
        raise AssertionError(f"lemma failure ({ctx}): 2-term system unsolvable")
    return int(X[0, 0]) % q, int(X[0, 1]) % q


def _unit_val(c: int, p: int, d: int):
    """Decompose c in R = Z/p^d as (u, s) with c == u * p^s and u a unit
    (u = 1, s = d for c = 0)."""
    q = p ** d
    c = int(c) % q
    if c == 0:
        return 1, d
    s = cr.valuation(c, p, d)
    return (c // p ** s) % q, s


class CocycleRankWidthGroups:
    """The uniformly tree-automatic class of distributed-center class-2
    groups of tensor cut-rank 1 (see paper/theorem3-notes.md): the common
    generalisation, at width r = 1, of `CutRankTreeGroups` (fixed center on
    a chain) and `TreeExtraspecialGroups` (laminar leaf targets). With
    ``d > 1`` everything runs over the chain ring R = Z/p^d ("Idea 2"): the
    bases are the saturated free interfaces (`chain_ring.saturate`), the
    registers and constants range over R, and the claim register carries the
    *determined truncation* p^s * P of the claim coordinate, re-anchored by
    the ring-only `zr` op when a later z-position determines more of it.
    The default d = 1 is byte-identical to the original field construction.

    The advice is *microcode*: each site of a `CocycleSites` layout expands
    into its site letter followed by a chain of micro-op letters, each
    carrying at most three R-constants (d base-p digits each), over an
    alphabet of O(q^5) letters independent of the tensor. Element digits
    repeat along a site's stretch (the universe automaton enforces
    constancy), so the multiplication automaton's state is exactly the
    seven-register machine of the protocol:

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
            'qy', 'hx', 'rm', 'rg', 'gm', 'z0', 'zr', 'bg', 'bc'}

    def __init__(self, p: int, merge_letters=None, d: int = 1):
        """With `merge_letters=None` the full ISA is available: the
        simulator and the compiler work, but the presentation automata are
        too large for the enumeration-based builder. Passing an explicit
        set of merge letters (e.g. collected from compiled advices via
        `used_merge_letters`) instantiates the automata over that
        sub-alphabet. With `d > 1` the center ring is R = Z/p^d ("Idea 2"):
        registers and micro-op constants range over R, each constant spelled
        as d base-p digits in the letter names (so d = 1 is byte-identical
        to the original field encoding)."""
        if p < 2 or p > 9 or any(p % f == 0 for f in range(2, int(p ** 0.5) + 1)):
            raise ValueError(f"p must be a prime in 2..9, got {p}")
        if d < 1:
            raise ValueError(f"center ring depth d must be >= 1, got {d}")
        self.p = p
        self.d = d
        self.q = p ** d                    # the ring R = Z/p^d
        self._merge_letters = None if merge_letters is None \
            else frozenset(merge_letters)
        self.digits = [str(x) for x in range(self.q)]
        self._digit_val = {x: int(x) for x in self.digits}
        self._parsed = {}
        self.element_letters = set(self.digits)
        self._cls = None

    def _enc(self, c: int) -> str:
        """A ring constant as d base-p digits (least significant first);
        the d = 1 encoding is the original single-digit form."""
        return ''.join(str(dig)
                       for dig in cr.to_digits(int(c) % self.q, self.p, self.d))

    @property
    def advice_letters(self):
        """The full microcode ISA. The merge letters carry 13 constants
        (folds to the joint bases, the three sibling products, the two
        cross m->claim translations), so the flat alphabet has 2 q^13 + O(q^3)
        letters: fine for the transition function and the simulator, too
        large for the enumeration-based automaton builder -- building the
        presentation automata is gated until factored transition letters
        land in the engine (see theorem3-notes.md)."""
        q, enc = self.q, self._enc
        letters = ['L', 'K', 'U', 'V', 'n', 'g0', 'bl']
        for op in ('sy', 'sx', 'sq', 'sh', 'sm', 'sg'):
            letters += [f'{op}{enc(c)}' for c in range(q)]
        for op in ('iy', 'ix', 'iq', 'ih', 'qy', 'hx', 'rm', 'rg', 'gm',
                   'b1'):
            letters += [f'{op}{enc(c)}' for c in range(1, q)]
        for op in ('za', 'zc', 'bk'):
            letters += [f'{op}{enc(c)}{enc(e)}' for c in range(q)
                        for e in range(q)]
        letters += [f'z0{enc(c)}' for c in range(q)]
        if self.d > 1:                     # the ring re-anchor op
            letters += [f'zr{enc(c)}{enc(a)}{enc(b)}' for c in range(q)
                        for a in range(q) for b in range(q)]
        if self._merge_letters is not None:
            return set(letters) | set(self._merge_letters)
        if 2 * q ** 13 > 20_000_000:
            raise ValueError(
                f"the full merge ISA has 2 * {q}^13 letters -- too many to "
                f"enumerate; instantiate with merge_letters="
                f"used_merge_letters(advice), or use check_implicit / "
                f"simulate, which never enumerate the alphabet")
        for kind in ('M', 'W'):
            letters += [kind + ''.join(enc(c) for c in cs)
                        for cs in it.product(range(q), repeat=13)]
        return set(letters)

    def _isa_count(self) -> int:
        """|advice_letters| computed arithmetically (the full merge ISA may
        be too large to enumerate)."""
        q = self.q
        n = 7 + 6 * q + 10 * (q - 1) + 3 * q * q + q
        if self.d > 1:
            n += q ** 3
        n += (len(self._merge_letters) if self._merge_letters is not None
              else 2 * q ** 13)
        return n

    @property
    def sigma(self):
        return {PAD} | set(self.digits) | self.advice_letters

    #: cap on the transition enumerations of the lazy `cls` build; beyond it
    #: the build would run for minutes to hours, so `cls` raises instead
    _CLS_ENUMERATION_CAP = 50_000_000

    def _cls_cost(self) -> int:
        """Transition enumerations the lazy `cls` build performs (dominated by
        the seven-register multiplication automaton: state pairs x letters x
        digits^3)."""
        n_states = 1 + self.q ** 7
        return ((n_states + 1) ** 2 * self._isa_count()
                * len(self.element_letters) ** 3)

    @property
    def cls(self) -> UniformlyTreeAutomaticClass:
        """The uniformly tree-automatic presentation (built lazily: the
        seven-register multiplication automaton enumerates a large
        state-pair x letter product)."""
        if self._cls is None:
            cost = self._cls_cost()
            if cost > self._CLS_ENUMERATION_CAP:
                hint = ("instantiate with merge_letters="
                        "used_merge_letters(advice) for a sub-alphabet, or "
                        if self._merge_letters is None else "")
                raise ValueError(
                    f"advice alphabet too large to build the explicit "
                    f"presentation automata: {self._isa_count()} "
                    f"letters, ~{cost:.0e} transition enumerations "
                    f"(cap {self._CLS_ENUMERATION_CAP:.0e}); {hint}"
                    f"use check_implicit or simulate instead")
            self._cls = UniformlyTreeAutomaticClass({
                'U': self._universe_automaton(),
                'M': self._multiplication_automaton(),
                'Eq': self._eq_automaton(),
            }, padding_symbol=PAD)
            self._cls.element_alphabet = list(self.element_letters)
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
                 'b1': 24, 'za': 25, 'zc': 26, 'z0': 27, 'bk': 28,
                 'zr': 29}
    _OP_NCONSTS = {'sy': 1, 'sx': 1, 'sq': 1, 'sh': 1, 'sm': 1, 'sg': 1,
                   'iy': 1, 'ix': 1, 'iq': 1, 'ih': 1, 'qy': 1, 'hx': 1,
                   'rm': 1, 'rg': 1, 'gm': 1, 'b1': 1, 'za': 2, 'zc': 2,
                   'z0': 1, 'bk': 2, 'zr': 3}
    _SITE_CODES = {'L': 0, 'K': 1, 'U': 2, 'V': 3, 'n': 6, 'g0': 7, 'bl': 8}

    def _parse_letter(self, a):
        """Letter -> (code, consts), memoized; None if malformed. Constants
        are d base-p digits each (least significant first)."""
        hit = self._parsed.get(a)
        if hit is not None:
            return hit
        d, p = self.d, self.p

        def consts(s, n):
            if len(s) != n * d or not all(c.isdigit() and int(c) < p
                                          for c in s):
                return None
            return tuple(cr.from_digits([int(c) for c in s[i * d:(i + 1) * d]],
                                        p)
                         for i in range(n))

        if a in self._SITE_CODES:
            out = (self._SITE_CODES[a], ())
        elif a[0] in ('M', 'W') and len(a) == 1 + 13 * d:
            cs = consts(a[1:], 13)
            if cs is None:
                return None
            out = (4 if a[0] == 'M' else 5, cs)
        elif a[:2] in self._OP_CODES:
            cs = consts(a[2:], self._OP_NCONSTS[a[:2]])
            if cs is None:
                return None
            out = (self._OP_CODES[a[:2]], cs)
        else:
            return None
        self._parsed[a] = out
        return out

    def _m_delta(self, lq, rq, sym):
        """The seven-register claim-and-verify transition over R = Z/p^d
        (shared by the automaton construction and the direct simulator)."""
        p = self.q
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
        if code == 29:
            # ring re-anchor: the entering z-position determines more of the
            # truncated claim than the positions seen so far -- check the old
            # truncation against the new residual, then re-seed the register
            val = (d0 - cs[0] * m) % p
            if (g - cs[1] * val) % p:
                return 'dead'
            return (wy, wx, qy, hx, m, (cs[2] * val) % p, bg)
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
        states = ['dead'] + list(it.product(range(self.q), repeat=7))
        return sta_from_delta(self.sigma, states, 4, self._m_delta,
                              {(0,) * 7},
                              tapes=[self.advice_letters] +
                                    [self.element_letters] * 3)

    # ---------------- the advice compiler ----------------

    def _r1_bases(self, sites: CocycleSites, T: Dict, lo: int, hi: int):
        """Width-1 SATURATED basis vectors of the six flattenings of the
        interval over R = Z/p^d, as sparse dicts. Vy/Vx/Vpy/Vpx live over
        x-positions, Bm and w over z-positions. The saturated free basis
        (`chain_ring.saturate`) is what makes the restriction and
        factorisation lemmas hold over the ring; at d = 1 it is the ordinary
        normalized field basis."""
        p, d, q = self.p, self.d, self.q
        ent = sites._flattening_entries(T, lo, hi)

        def basis(e, key_of, ctx):
            groups = {}
            for pair_key, coeff in e.items():
                own, other = key_of(pair_key)
                groups.setdefault(other, {})[own] = coeff % q
            cands = [c for c in groups.values() if any(v % q
                                                       for v in c.values())]
            if not cands:
                return {}
            keys = sorted({k for c in cands for k in c})
            M = np.array([[c.get(k, 0) for k in keys] for c in cands],
                         dtype=np.int64)
            sat, _ = cr.saturate(M, p, d)
            if sat.shape[0] > 1:
                raise AssertionError(f"lemma failure ({ctx}): module rank > 1")
            vec = {k: int(sat[0, i]) % q for i, k in enumerate(keys)
                   if sat[0, i] % q}
            # canonical: the unit part of the leading entry scaled to 1
            u, _ = _unit_val(vec[min(vec)], p, d)
            inv = cr.unit_inverse(u, p, d)
            vec = {k: v * inv % q for k, v in vec.items()}
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
                out[(i, v)] = coeff % self.q
        return out

    def _bank_ops(self, old, new, lo, hi, own, digit_pos, p, banks=None):
        """Rebase + injection ops from child bases `old` to node bases
        `new` over the child interval [lo, hi]; `own` is the node's own
        position (or None) for the digit injections. The functional
        registers flow forward, so ring constants (possibly with valuation)
        need no inversion."""
        d, enc = self.d, self._enc
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
                                p, f'{bank} restriction', d)
                cmix = 0
            else:
                theta, cmix = _vsolve2(target,
                                       self._restrict(old[bank], lo, hi),
                                       self._restrict(old[mixsrc], lo, hi),
                                       p, f'{bank} mixed restriction', d)
            if theta != 1:
                ops.append(f'{scale}{enc(theta)}')
            if cmix:
                ops.append(f'{mix}{enc(cmix)}')
            if own is not None:
                cinj = new[bank].get(own, 0)
                if cinj:
                    ops.append(f'{inject}{enc(cinj)}')
        return ops

    def _claim_rebase_ops(self, w_old, w_new, s_old, p, ctx):
        """Ops taking the claim register from basis w_old (truncation
        valuation s_old) to basis w_new over the same z-set; returns
        (ops, s_new). Two saturated bases of the same rank-1 module differ
        by a unit, so the rebase is always invertible."""
        d, q, enc = self.d, self.q, self._enc
        if not w_old:
            if w_new:
                raise AssertionError(f'lemma failure ({ctx}): claim born late')
            return [], 0
        if not w_new:
            return ['g0'], 0
        phi = _vratio(w_new, w_old, p, ctx, d)
        if phi % p == 0:
            raise AssertionError(f'lemma failure ({ctx}): rebase kills basis')
        inv = cr.unit_inverse(phi, p, d)
        return ([f'sg{enc(inv)}'] if inv != 1 else []), s_old

    def advice(self, sites: CocycleSites, T: Dict) -> Tree:
        """Compile a width-1 tensor over the site layout into microcode
        advice over R = Z/p^d. Raises ValueError if some cut exceeds width 1
        (module cut-rank over the ring); every scalar derivation asserts its
        restriction lemma.

        Over the ring the claim register can only carry the *determined
        truncation* p^s * P of the claim coordinate P, where s is the least
        valuation of the (saturated) claim basis over the z-positions read
        so far; the compiler tracks s statically per node, normalizes unit
        parts through `sg`, cross-multiplies consistency checks by the
        matching p-powers, and re-anchors with the ring-only `zr` op when a
        new z-position determines more than the ones before it. At d = 1
        every valuation is 0 and the compiled stream is the original field
        microcode, byte for byte."""
        p, d, q, enc = self.p, self.d, self.q, self._enc
        if sites.p != p or sites.d != d:
            raise ValueError("sites and class must share p and d")
        sites.check_tensor(T)
        width = sites.cut_width(T)
        if width > 1:
            raise ValueError(f"tensor has cut-width {width} > 1")

        bases: Dict[int, Dict] = {}
        claim_s: Dict[int, int] = {}
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
                s_t = 0
                if kind == XSITE:
                    for bank, inject in (('Vy', 'iy'), ('Vx', 'ix'),
                                         ('Vpy', 'iq'), ('Vpx', 'ih')):
                        c = B_S[bank].get(t, 0)
                        if c:
                            ops.append(f'{inject}{enc(c)}')
                else:
                    w = B_S['w']
                    if w:
                        # singleton saturated basis: w[t] is a unit
                        u, s_t = _unit_val(w[t], p, d)
                        cc = cr.unit_inverse(u, p, d)
                        ops.append(f'za{enc(cc)}{enc(0)}')
                    else:
                        ops.append(f'z0{enc(0)}')
                adv[t] = chain(Tree(letter), ops)
                bases[t] = B_S
                claim_s[t] = s_t
                continue

            if L is None or R is None:
                child = L if L is not None else R
                cpos = sites.pos[id(child)]
                B_C = bases[cpos]
                below = chain(adv[cpos], [])
                letter = 'U' if kind == XSITE else 'V'
                ops, s_t = self._site_ops(sites, T, t, lo, t - 1, B_C, B_S,
                                          kind, claim_s[cpos])
                adv[t] = chain(Tree(letter, below, None), ops)
                bases[t] = B_S
                claim_s[t] = s_t
                continue

            # binary: L keeps wy/qy raw for the merge products, R keeps
            # wx/hx raw; the merge letter folds the raw sides into the
            # joint bases of J = [lo, t-1] and consumes the products.
            lp, rp = sites.pos[id(L)], sites.pos[id(R)]
            B_L, B_R = bases[lp], bases[rp]
            s_L, s_R = claim_s[lp], claim_s[rp]
            B_J = self._r1_bases(sites, T, lo, t - 1)
            pre_L = self._bank_ops(B_L, B_J, lo, lp, None, None, p,
                                   banks=('Vpx', 'Vx'))
            pre_R = self._bank_ops(B_R, B_J, lp + 1, t - 1, None, None, p,
                                   banks=('Vpy', 'Vy'))

            # folds of the raw sides (solved against the child bases)
            fy = _vratio(self._restrict(B_J['Vy'], lo, lp), B_L['Vy'], p,
                         'Vy fold L', d)
            fq, fyq = _vsolve2(self._restrict(B_J['Vpy'], lo, lp),
                               B_L['Vpy'], B_L['Vy'], p, 'Vpy fold L', d)
            fx = _vratio(self._restrict(B_J['Vx'], lp + 1, t - 1), B_R['Vx'],
                         p, 'Vx fold R', d)
            fh, fxh = _vsolve2(self._restrict(B_J['Vpx'], lp + 1, t - 1),
                               B_R['Vpx'], B_R['Vx'], p, 'Vpx fold R', d)
            fml = _vratio({k: v for k, v in B_L['Bm'].items()
                           if not lp + 1 <= k <= t - 1}, B_J['Bm'], p,
                          'Bm fold L', d)
            fmr = _vratio({k: v for k, v in B_R['Bm'].items()
                           if not lo <= k <= lp}, B_J['Bm'], p, 'Bm fold R', d)

            # sibling products and cross m->claim translations; the claim
            # discharges land in the children's truncated coordinates, so
            # their constants carry the matching p-powers
            wL, wR, wJ = B_L['w'], B_R['w'], B_J['w']
            new_pairs = {(j, i, v): c % q for (j, i, v), c in T.items()
                         if lo <= i <= lp and lp + 1 <= j <= t - 1}
            blockL = {k: c for k, c in new_pairs.items() if lo <= k[2] <= lp}
            blockR = {k: c for k, c in new_pairs.items()
                      if lp + 1 <= k[2] <= t - 1}
            blockO = {k: c for k, c in new_pairs.items()
                      if not lo <= k[2] <= t - 1}
            c_gl = _vratio3(blockL, B_R['Vx'], B_L['Vpy'], wL, p,
                            'sibling block -> L claims', d)
            c_gr = _vratio3({(k[1], k[0], k[2]): c for k, c in blockR.items()},
                            B_L['Vy'], B_R['Vpx'], wR, p,
                            'sibling block -> R claims', d)
            cpm = _vratio3(blockO, B_R['Vx'], B_L['Vy'], B_J['Bm'], p,
                           'sibling block -> exports', d)
            psiL = _vratio(self._restrict(B_L['Bm'], lp + 1, t - 1), wR, p,
                           'm_L -> R claims', d)
            psiR = _vratio(self._restrict(B_R['Bm'], lo, lp), wL, p,
                           'm_R -> L claims', d)
            consts = (fy, fq, fyq, fx, fh, fxh, fml, fmr, cpm,
                      (-c_gl * p ** s_L) % q, (-c_gr * p ** s_R) % q,
                      (-psiL * p ** s_R) % q, (-psiR * p ** s_L) % q)
            letter = (('M' if kind == XSITE else 'W')
                      + ''.join(enc(c) for c in consts))
            if self._merge_letters is not None \
                    and letter not in self._merge_letters:
                raise ValueError(
                    f"merge letter {letter!r} is not in the instantiated "
                    f"sub-alphabet; collect it with used_merge_letters first")

            # claim join on the stretch above the merge: pick the child
            # whose truncation determines more of the joint coordinate
            ops = []
            phiL = _vratio(self._restrict(wJ, lo, lp), wL, p, 'claim join L',
                           d)
            phiR = _vratio(self._restrict(wJ, lp + 1, t - 1), wR, p,
                           'claim join R', d)
            uL, eL = _unit_val(phiL, p, d)
            uR, eR = _unit_val(phiR, p, d)
            sLp = min(s_L + eL, d) if wL else d
            sRp = min(s_R + eR, d) if wR else d
            s_J = 0
            if wL and phiL:
                if not wR or sLp <= sRp:
                    inv = cr.unit_inverse(uL, p, d)
                    if inv != 1:
                        ops.append(f'sg{enc(inv)}')
                    if wR:
                        c = (uR * p ** (sRp - sLp)) % q
                        ops.append(f'bk{enc(c)}{enc(q - 1)}')
                    s_J = sLp
                else:                     # re-anchor to the right claim
                    cc = (uL * p ** (sLp - sRp)
                          * cr.unit_inverse(uR, p, d)) % q
                    ops.append(f'bk{enc(1)}{enc(-cc)}')
                    ops.append(f'sg{enc(0)}')
                    ops.append(f'b1{enc(cr.unit_inverse(uR, p, d))}')
                    s_J = sRp
            elif wL and not phiL:
                ops.append('g0')
                if wR and phiR:
                    ops.append(f'b1{enc(cr.unit_inverse(uR, p, d))}')
                    s_J = sRp
                elif wR:
                    ops.append(f'bk{enc(0)}{enc(1)}')
            else:                                     # no L claims
                if wR and phiR:
                    ops.append(f'b1{enc(cr.unit_inverse(uR, p, d))}')
                    s_J = sRp
                elif wR:
                    ops.append(f'bk{enc(0)}{enc(1)}')
            ops.append('bl')
            site_ops, s_t = self._site_ops(sites, T, t, lo, t - 1, B_J, B_S,
                                           kind, s_J)
            ops += site_ops
            adv[t] = chain(Tree(letter, chain(adv[lp], pre_L),
                           chain(adv[rp], pre_R)), ops)
            bases[t] = B_S
            claim_s[t] = s_t

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
        d, enc = self.d, self._enc
        kept = {k: v for k, v in bm_old.items() if not drop_lo <= k <= drop_hi}
        if bm_new:
            theta = _vratio(kept, bm_new, p, ctx, d)
            # kept = theta' * bm_new: we need m_new with m_new*bm_new = m*kept
            return [f'sm{enc(theta)}'] if theta != 1 else []
        if kept:
            raise AssertionError(f'lemma failure ({ctx}): exports survive '
                                 f'without a basis')
        return [f'sm{enc(0)}'] if bm_old else []

    def _site_ops(self, sites, T, t, lo, hi, B_C, B_S, kind, s_C):
        """The stretch above a site node t whose (possibly joint) child
        covers [lo, hi]: m rebase, read-offs, claim work, bank work. `s_C`
        is the claim truncation valuation entering the stretch; returns
        (ops, s_S) with the valuation leaving it."""
        p, d, q, enc = self.p, self.d, self.q, self._enc
        ops = []
        if kind == XSITE:
            # m rebase first (read-offs emit in the S basis)
            ops += self._m_rebase_ops(B_C['Bm'], B_S['Bm'], t, t - 1, p,
                                      'Bm x-site')
            # read-off of pairs {t, i}: targets outside S
            out_block = self._pair_block(
                sites, T, t, lo, hi, lambda v: not lo <= v <= t)
            crm = _vratio2(out_block, B_C['Vy'], B_S['Bm'], p,
                           'read-off -> exports', d)
            if crm:
                ops.append(f'rm{enc(crm)}')
            # read-off targets inside (discharge the truncated claim)
            in_block = self._pair_block(sites, T, t, lo, hi,
                                        lambda v: lo <= v <= hi)
            crg = _vratio2(in_block, B_C['Vpy'], B_C['w'], p,
                           'read-off -> claims', d)
            c_dis = (-crg * p ** s_C) % q
            if c_dis:
                ops.append(f'rg{enc(c_dis)}')
            claim_ops, s_S = self._claim_rebase_ops(B_C['w'], B_S['w'], s_C,
                                                    p, 'claim rebase x-site')
            ops += claim_ops
            ops += self._bank_ops(B_C, B_S, lo, hi, t, t, p)
        else:
            # z-site: the residual enters the claim; cm couples m
            cm = B_C['Bm'].get(t, 0) % q
            wC, wS = B_C['w'], B_S['w']
            omega = wS.get(t, 0) % q
            u_o, s_o = _unit_val(omega, p, d)
            if wC:
                phi = _vratio(self._restrict(wS, lo, hi), wC, p,
                              'claim extend restriction', d)
                u_phi, e_phi = _unit_val(phi, p, d)
                if phi:
                    s_kept = min(s_C + e_phi, d)
                    inv = cr.unit_inverse(u_phi, p, d)
                    if omega and s_o < s_kept:
                        # ring re-anchor: position t determines more of the
                        # claim than everything read so far
                        u_o_inv = cr.unit_inverse(u_o, p, d)
                        ca = (p ** (s_kept - s_o) * u_o_inv * u_phi) % q
                        ops.append(f'zr{enc(cm)}{enc(ca)}{enc(u_o_inv)}')
                        s_S = s_o
                    else:
                        c2 = 0 if not omega else \
                            (u_o * p ** (s_o - s_kept) * inv) % q
                        ops.append(f'zc{enc(cm)}{enc(c2)}')
                        if inv != 1:
                            ops.append(f'sg{enc(inv)}')
                        s_S = s_kept
                else:
                    ops.append('g0')
                    if omega:
                        ops.append(f'za{enc(cr.unit_inverse(u_o, p, d))}'
                                   f'{enc(cm)}')
                        s_S = s_o
                    else:
                        ops.append(f'z0{enc(cm)}')
                        s_S = 0
            else:
                if omega:
                    ops.append(f'za{enc(cr.unit_inverse(u_o, p, d))}'
                               f'{enc(cm)}')
                    s_S = s_o
                else:
                    ops.append(f'z0{enc(cm)}')
                    s_S = 0
            ops += self._m_rebase_ops(B_C['Bm'], B_S['Bm'], t, t, p,
                                      'Bm z-site')
            ops += self._bank_ops(B_C, B_S, lo, hi, None, None, p)
        return ops, s_S

    # ---------------- encodings and class operations ----------------

    def encode(self, element, sites: CocycleSites, advice: Tree) -> Tree:
        """Element tree of the advice's shape: each site's digit repeats
        along its stretch (micro-ops inherit the digit of the site below)."""
        b, a = element
        if len(b) != len(sites.Z) or len(a) != len(sites.X):
            raise ValueError("element does not fit the site layout")
        if not all(0 <= x < self.q for x in tuple(b) + tuple(a)):
            raise ValueError(f"components must lie in Z/{self.q}")
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

    def _implicit_atoms(self) -> Dict:
        """Functional bottom-up base automata (Dom, Adv, M, Eq) built straight
        from the microcode deltas -- no explicit product automaton, so this
        works for the full-ISA and ring members whose `cls` cannot be built."""
        from autstr.implicit import ImplicitTA
        element_letters = self.element_letters

        def legal(a):
            return (self._parse_letter(a) is not None
                    if isinstance(a, str) and a else False)

        def shape(args, q_of):
            adv = args[0]

            def step(sym, left, right):
                if left == 'dead' or right == 'dead' or not q_of(sym):
                    return 'dead'
                a = sym[adv]
                if not legal(a) or not self._shape_step(
                        a, None if left is None else 'q',
                        None if right is None else 'q'):
                    return 'dead'
                if a in ('L', 'K', 'U', 'V') or a[0] in ('M', 'W'):
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
            xv, yv = args[1], args[2]

            def step(sym, left, right):
                if left == 'dead' or right == 'dead' \
                        or sym[xv] not in element_letters or sym[xv] != sym[yv]:
                    return 'dead'
                a = sym[args[0]]
                if not legal(a) or not self._shape_step(
                        a, None if left is None else 'q',
                        None if right is None else 'q'):
                    return 'dead'
                return 'ok'
            return ImplicitTA(args, step, lambda st: st == 'ok')

        def M(args):
            adv, xv, yv, zv = args
            return ImplicitTA(
                args,
                lambda sym, left, right: self._m_delta(
                    left, right, (sym[adv], sym[xv], sym[yv], sym[zv])),
                lambda st: st == (0,) * 7)

        return {'Dom': Dom, 'Adv': Adv, 'M': M, 'Eq': Eq}

    def decode(self, tree: Tree, advice: Tree):
        """Inverse of `encode` over the member's advice: the digit at each
        site letter, z-sites to b and x-sites to a (ascending post-order)."""
        bs, xs = [], []

        def rec(an, en):
            if an is None:
                return
            rec(an.left, en.left)
            rec(an.right, en.right)
            letter = an.label
            if letter in ('L', 'U') or letter[0] == 'M':
                xs.append(int(en.label))
            elif letter in ('K', 'V') or letter[0] == 'W':
                bs.append(int(en.label))

        rec(advice, tree)
        return tuple(bs), tuple(xs)

    @property
    def implicit_cls(self):
        """The fully implicit presentation of this class (functional atoms
        only, nothing compiled): an `autstr.implicit.ImplicitTreeClass` over
        raw element trees."""
        from autstr.implicit import ImplicitTreeClass
        return ImplicitTreeClass(self._implicit_atoms(),
                                 list(self.element_letters))

    def check_implicit(self, phi, sites: CocycleSites, advice: Tree, **elements) -> bool:
        """Like `check`, evaluated implicitly (no query or base tree
        automaton) -- the only viable model checker for the full-ISA and ring
        members whose `cls` cannot be built. See `autstr.implicit`."""
        trees = {name: self.encode(el, sites, advice)
                 for name, el in elements.items()}
        return self.implicit_cls.check(phi, advice, **trees)

    def evaluate_implicit(self, phi, sites: CocycleSites, advice: Tree,
                          **elements):
        """The satisfying set of phi on the member presented by the advice,
        computed implicitly: unassigned free variables stay open and are
        solved for. Yields assignments {var: (b, a)}; `len` is the exact
        solution count without enumeration."""
        from autstr.implicit import MappedSolutions
        trees = {name: self.encode(el, sites, advice)
                 for name, el in elements.items()}
        sols = self.implicit_cls.evaluate(phi, advice, **trees)
        return MappedSolutions(sols, lambda t: self.decode(t, advice))

    def get_structure(self, advice: Tree) -> TreeAutomaticPresentation:
        return self.cls.get_structure(advice)
