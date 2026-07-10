"""Tree-indexed generalizations of the extraspecial p-groups as a uniformly
tree-automatic class.

The advice is an arbitrary binary *shape tree*. It presents a class-2 group
G_t with one generator pair x_w, y_w per **inner** node w, one central
generator z_v per **leaf** v, exponent-p relations, and commutators

    [x_w, y_w] = prod over the leaves v below w of z_v,

all other generator pairs commuting. The commutator supports form the
laminar family of the tree: a *spine* advice (every inner node with a single
child, one leaf at the bottom) makes every commutator hit the same z, which
is exactly the extraspecial group p^(1+2n) of exponent p (p odd); general
shapes interpolate between that and the direct sum of Heisenberg groups.

Concretely G_t = { (a, b, c) : a, b assign F_p to inner nodes, c assigns F_p
to leaves } with the central-extension law

    (a1,b1,c1)(a2,b2,c2) = (a1+a2, b1+b2, c),
    c(v) = c1(v) + c2(v) + sum over inner w < v of a1(w)*b2(w)   (mod p).

Elements are encoded as trees of the advice's exact shape: inner node w is
labelled 'i{a(w)}{b(w)}', leaf v is labelled 'l{c(v)}'. The multiplication
automaton is the running-sum trick evaluated along all paths of the tree: a
bottom-up state in F_p tracks the deficit c_z - c_x - c_y still owed by the
ancestors, each inner node subtracts its commutator contribution
a_x(w)*b_y(w), and siblings must agree on the owed amount when their
branches merge — p + 1 states in total, independent of the shape.

**Bounded rank-width** (`CutRankTreeGroups(p, k, r)`): the tree analog of
`autstr.groups.CutRankGroups`. A member is a class-2 central extension of
Z_p^n by Z_p^k whose commutation form admits a *tree* layout in which every
subtree's crossing block has rank <= r over F_p; the advice spells out the
factorizations node by node, and a spine layout is exactly the word class.
"""
import itertools as it
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from autstr.groups import _rref_mod, _solve_xa_eq_b
from autstr.sparse_tree_automata import SparseTreeAutomaton, Tree
from autstr.tree_presentations import TreeAutomaticPresentation
from autstr.tree_uniform import UniformlyTreeAutomaticClass, sta_from_delta

PAD = '*'
SHAPE = 's'
CMARK = 'c'


class TreeExtraspecialGroups:
    """The uniformly tree-automatic class of tree-indexed extraspecial
    p-groups (advice = shape tree, elements = coordinate labellings)."""

    def __init__(self, p: int, max_states: Optional[int] = None):
        if p < 2 or p > 9:
            raise ValueError("p must be a prime in 2..9 (single digits)")
        self.p = p
        self.inner_letters = {f'i{a}{b}': (a, b)
                              for a in range(p) for b in range(p)}
        self.leaf_letters = {f'l{c}': c for c in range(p)}
        self.sigma = {PAD, SHAPE} | set(self.inner_letters) \
            | set(self.leaf_letters)
        self.advice_letters = {SHAPE}
        self.element_letters = set(self.inner_letters) | set(self.leaf_letters)

        self.cls = UniformlyTreeAutomaticClass({
            'U': self._universe_automaton(),
            'M': self._multiplication_automaton(),
            'E': self._equality_automaton(),
        }, padding_symbol=PAD, max_states=max_states)

    # ---------------- the automata ----------------

    def _universe_automaton(self) -> SparseTreeAutomaton:
        """U(advice, x): x has the advice's exact shape, inner letters on
        inner nodes, leaf letters on leaves."""
        def delta(lq, rq, sym):
            adv, x = sym
            if adv != SHAPE:
                return 'dead'
            if lq is None and rq is None:
                return 'ok' if x in self.leaf_letters else 'dead'
            return 'ok' if x in self.inner_letters else 'dead'

        return sta_from_delta(self.sigma, ['ok', 'dead'], 2, delta, {'ok'},
                              tapes=[self.advice_letters,
                                     self.element_letters])

    def _equality_automaton(self) -> SparseTreeAutomaton:
        def delta(lq, rq, sym):
            adv, x, y = sym
            if adv != SHAPE or x != y:
                return 'dead'
            if lq is None and rq is None:
                return 'ok' if x in self.leaf_letters else 'dead'
            return 'ok' if x in self.inner_letters else 'dead'

        return sta_from_delta(self.sigma, ['ok', 'dead'], 3, delta, {'ok'},
                              tapes=[self.advice_letters] +
                                    [self.element_letters] * 2)

    def _multiplication_automaton(self) -> SparseTreeAutomaton:
        """M(advice, x, y, z): z = x·y. The state is the deficit still owed
        to the current subtree's leaves by the ancestors' commutator
        contributions; sibling branches must owe the same amount."""
        p = self.p

        def delta(lq, rq, sym):
            adv, x, y, z = sym
            if adv != SHAPE:
                return 'dead'
            if lq is None and rq is None:
                if not (x in self.leaf_letters and y in self.leaf_letters
                        and z in self.leaf_letters):
                    return 'dead'
                return (self.leaf_letters[z] - self.leaf_letters[x]
                        - self.leaf_letters[y]) % p
            if not (x in self.inner_letters and y in self.inner_letters
                    and z in self.inner_letters):
                return 'dead'
            ax, bx = self.inner_letters[x]
            ay, by = self.inner_letters[y]
            az, bz = self.inner_letters[z]
            if az != (ax + ay) % p or bz != (bx + by) % p:
                return 'dead'
            if lq is not None and rq is not None and lq != rq:
                return 'dead'                 # branches disagree on the debt
            owed = lq if lq is not None else rq
            return (owed - ax * by) % p

        states = list(range(p)) + ['dead']
        return sta_from_delta(self.sigma, states, 4, delta, {0},
                              tapes=[self.advice_letters] +
                                    [self.element_letters] * 3)

    # ---------------- encodings ----------------

    @staticmethod
    def spine(n: int) -> Tree:
        """Shape with n inner nodes over a single leaf: presents the
        extraspecial group p^(1+2n)."""
        node = Tree(SHAPE)
        for _ in range(n):
            node = Tree(SHAPE, node, None)
        return node

    def advice(self, shape: Tree) -> Tree:
        """Normalize a shape tree to advice (labels forced to 's')."""
        def convert(node):
            if node is None:
                return None
            return Tree(SHAPE, convert(node.left), convert(node.right))
        return convert(shape)

    def encode(self, shape: Tree, a: Dict[str, int] = (),
               b: Dict[str, int] = (), c: Dict[str, int] = ()) -> Tree:
        """The element (a, b, c) of G_shape as a tree. Coordinates are given
        per node address ('' = root, then '0'/'1' for left/right children);
        omitted coordinates are 0."""
        a, b, c = dict(a), dict(b), dict(c)

        def convert(node, addr):
            if node is None:
                return None
            if node.left is None and node.right is None:
                label = f'l{c.get(addr, 0) % self.p}'
            else:
                label = (f'i{a.get(addr, 0) % self.p}'
                         f'{b.get(addr, 0) % self.p}')
            return Tree(label, convert(node.left, addr + '0'),
                        convert(node.right, addr + '1'))
        return convert(shape, '')

    # ---------------- class-level operations ----------------

    def evaluate(self, phi):
        """Evaluate a first-order query over the class; see
        UniformlyTreeAutomaticClass.evaluate."""
        return self.cls.evaluate(phi)

    def check(self, phi, shape: Tree, **elements) -> bool:
        """Model check a formula against the group G_shape. Free variables
        can be assigned element trees (see `encode`); unassigned ones are
        quantified existentially."""
        return self.cls.check(phi, self.advice(shape), **elements)

    def get_structure(self, shape: Tree) -> TreeAutomaticPresentation:
        """The tree-automatic presentation of the single group G_shape."""
        return self.cls.get_structure(self.advice(shape))


# ====================================================================
# Class-2 groups of bounded rank-width (tree layouts of bounded cut-rank)
# ====================================================================

class CutRankTreeGroups:
    """For a fixed prime p, center dimension k and width r, the uniformly
    tree-automatic class of class-2 central extensions of Z_p^n by Z_p^k
    whose commutation form admits a *tree* layout of cut-rank <= r over F_p
    (bounded rank-width) — the tree analog of `autstr.groups.CutRankGroups`,
    which is recovered exactly on spine layouts.

    Generators are the post-order positions 1..n of a binary layout tree;
    the form is a dict {(j, i): label in Z_p^k, i < j} presenting
    x_j x_i = x_i x_j y^B[j,i], and the group law is the same bilinear
    cocycle as in the word class. Elements are (b, a) encoded as digit
    labellings of the advice's shape; the k center digits live on a chain
    of 'c' nodes above the layout root.

    Bottom-up, the state at a subtree S is (s, wx, wy): the correction
    accumulated by pairs inside S, and r linear functionals w = V·(digits
    of S) of each factor's digits, where V is a row basis of S's crossing
    block. Two vectors are needed because a crossing pair can be consumed
    in two ways: its larger endpoint is an ancestor of S (the read-off
    pairs y-functionals of S with that ancestor's x-digit), or the pair is
    split between siblings (the merge pairs the right child's x-functionals
    against the left child's y-functionals — in post-order the left subtree
    lies entirely below the right one). The advice letter at a node holds
    the factorization data over Z_p::

        leaf:   'a' + v (r)
        unary:  'b' + T (r x r) + v (r) + R (k x r)
        binary: 'd' + T_L, T_R (r x r) + v (r) + R_L, R_R (k x r)
                    + Q (k x (r x r))

    with w <- T_L w_L + T_R w_R + v*digit (both factors), correction
    s <- s_L + s_R + x_t*(R_L wy_L + R_R wy_R) + wx_R^T Q_l wy_L, and the
    'c' chain consuming s coordinate by coordinate: z_c - x_c - y_c = s.
    That is p^(k+2r) layout states however large the tree. `advice`
    compiles (shape, form) into letters — rref row bases of the crossing
    blocks plus solving the consistency systems, including the two-sided
    factorization V_R^T Q V_L of the sibling block — and fails precisely
    when some subtree's cut-rank exceeds r; `tree_cut_rank` measures the
    width a layout needs. Every well-shaped advice presents some group in
    the class: the streamed cocycle is bilinear by construction.

    Signature: M(x,y,z), Eq(x,y); the center is first-order definable.
    """

    def __init__(self, p: int, k: int = 1, r: int = 1):
        if p < 2 or p > 9 or any(p % d == 0 for d in range(2, int(p ** 0.5) + 1)):
            raise ValueError(f"p must be a prime in 2..9 (single digits), got {p}")
        if k < 1 or r < 1:
            raise ValueError("need k >= 1 central and r >= 1 state dimensions")
        n_letters = (p ** r + p ** (r * r + r + k * r)
                     + p ** (2 * r * r + r + 2 * k * r + k * r * r))
        if n_letters > 20000:
            raise ValueError(
                f"advice alphabet would have {n_letters} letters; choose "
                f"smaller p, k or r (factored letters are future work)")
        self.p, self.k, self.r = p, k, r
        self.digits = [str(d) for d in range(p)]
        self._digitset = set(self.digits)

        def mat(digs, off, rows, cols):
            return tuple(tuple(digs[off + i * cols + j] for j in range(cols))
                         for i in range(rows))

        self.leaf_letters = {
            'a' + ''.join(map(str, digs)): tuple(digs)
            for digs in it.product(range(p), repeat=r)}
        self.unary_letters = {}
        for digs in it.product(range(p), repeat=r * r + r + k * r):
            T = mat(digs, 0, r, r)
            v = digs[r * r: r * r + r]
            R = mat(digs, r * r + r, k, r)
            self.unary_letters['b' + ''.join(map(str, digs))] = (T, v, R)
        self.binary_letters = {}
        for digs in it.product(range(p), repeat=2 * r * r + r + 2 * k * r + k * r * r):
            off = 0
            TL = mat(digs, off, r, r); off += r * r
            TR = mat(digs, off, r, r); off += r * r
            v = digs[off: off + r]; off += r
            RL = mat(digs, off, k, r); off += k * r
            RR = mat(digs, off, k, r); off += k * r
            Q = tuple(mat(digs, off + l * r * r, r, r) for l in range(k))
            self.binary_letters['d' + ''.join(map(str, digs))] = (TL, TR, v, RL, RR, Q)

        self.sigma = ({PAD, CMARK} | self._digitset | set(self.leaf_letters)
                      | set(self.unary_letters) | set(self.binary_letters))
        self.advice_letters = ({CMARK} | set(self.leaf_letters)
                               | set(self.unary_letters) | set(self.binary_letters))
        self.element_letters = self._digitset

        self.cls = UniformlyTreeAutomaticClass({
            'U': self._universe_automaton(),
            'M': self._multiplication_automaton(),
            'Eq': self._eq_automaton(),
        }, padding_symbol=PAD)

    # ---------------- the automata ----------------

    def _shape_delta(self, q_of):
        """Shared shape logic for U and Eq: q_of(sym) is the digit condition."""
        k = self.k

        def delta(lq, rq, sym):
            a = sym[0]
            if 'dead' in (lq, rq) or not q_of(sym):
                return 'dead'
            if a in self.leaf_letters:
                return 'lay' if lq is None and rq is None else 'dead'
            if a in self.unary_letters:
                child = lq if rq is None else (rq if lq is None else None)
                return 'lay' if child == 'lay' else 'dead'
            if a in self.binary_letters:
                return 'lay' if lq == 'lay' and rq == 'lay' else 'dead'
            if a == CMARK:
                child = lq if rq is None else (rq if lq is None else None)
                if child == 'lay':
                    return ('c', 1)
                if isinstance(child, tuple) and child[0] == 'c' and child[1] < k:
                    return ('c', child[1] + 1)
            return 'dead'

        states = ['lay', 'dead'] + [('c', j) for j in range(1, k + 1)]
        return delta, states, {('c', k)}

    def _universe_automaton(self) -> SparseTreeAutomaton:
        delta, states, finals = self._shape_delta(
            lambda sym: sym[1] in self._digitset)
        return sta_from_delta(self.sigma, states, 2, delta, finals,
                              tapes=[self.advice_letters, self.element_letters])

    def _eq_automaton(self) -> SparseTreeAutomaton:
        delta, states, finals = self._shape_delta(
            lambda sym: sym[1] in self._digitset and sym[1] == sym[2])
        return sta_from_delta(self.sigma, states, 3, delta, finals,
                              tapes=[self.advice_letters] +
                                    [self.element_letters] * 2)

    def _multiplication_automaton(self) -> SparseTreeAutomaton:
        """M(advice, x, y, z): z = x·y via the factored streaming cocycle."""
        p, k, r = self.p, self.k, self.r

        def dot(M, w):
            return tuple(sum(row[j] * w[j] for j in range(r)) % p for row in M)

        def delta(lq, rq, sym):
            a, x, y, z = sym
            if 'dead' in (lq, rq):
                return 'dead'
            if not all(s in self._digitset for s in (x, y, z)):
                return 'dead'
            xi, yi, zi = int(x), int(y), int(z)
            if a == CMARK:
                child = lq if rq is None else (rq if lq is None else None)
                if child is None or not isinstance(child, tuple):
                    return 'dead'
                if child[0] == 't':
                    j, s = 0, child[1]
                elif child[0] == 'c':
                    j, s = child[1], child[2]
                else:
                    return 'dead'
                if j >= k or (zi - xi - yi) % p != s[j]:
                    return 'dead'
                return ('c', j + 1, s[:j] + (0,) + s[j + 1:])
            if (xi + yi - zi) % p:
                return 'dead'
            if a in self.leaf_letters:
                if lq is not None or rq is not None:
                    return 'dead'
                v = self.leaf_letters[a]
                return ('t', (0,) * k,
                        tuple(v[i] * xi % p for i in range(r)),
                        tuple(v[i] * yi % p for i in range(r)))
            if a in self.unary_letters:
                child = lq if rq is None else (rq if lq is None else None)
                if child is None or child[0] != 't':
                    return 'dead'
                T, v, R = self.unary_letters[a]
                s, wx, wy = child[1], child[2], child[3]
                read = dot(R, wy)
                s = tuple((s[l] + xi * read[l]) % p for l in range(k))
                Twx, Twy = dot(T, wx), dot(T, wy)
                return ('t', s,
                        tuple((Twx[i] + v[i] * xi) % p for i in range(r)),
                        tuple((Twy[i] + v[i] * yi) % p for i in range(r)))
            if a in self.binary_letters:
                if lq is None or rq is None or lq[0] != 't' or rq[0] != 't':
                    return 'dead'
                TL, TR, v, RL, RR, Q = self.binary_letters[a]
                sL, wxL, wyL = lq[1], lq[2], lq[3]
                sR, wxR, wyR = rq[1], rq[2], rq[3]
                readL, readR = dot(RL, wyL), dot(RR, wyR)
                s = tuple((sL[l] + sR[l] + xi * (readL[l] + readR[l])
                           + sum(Q[l][u][t] * wxR[u] * wyL[t]
                                 for u in range(r) for t in range(r))) % p
                          for l in range(k))
                TwxL, TwxR = dot(TL, wxL), dot(TR, wxR)
                TwyL, TwyR = dot(TL, wyL), dot(TR, wyR)
                return ('t', s,
                        tuple((TwxL[i] + TwxR[i] + v[i] * xi) % p for i in range(r)),
                        tuple((TwyL[i] + TwyR[i] + v[i] * yi) % p for i in range(r)))
            return 'dead'

        states = ['dead']
        vecs_r = list(it.product(range(p), repeat=r))
        for s in it.product(range(p), repeat=k):
            for wx in vecs_r:
                for wy in vecs_r:
                    states.append(('t', s, wx, wy))
        for j in range(1, k + 1):
            for rest in it.product(range(p), repeat=k - j):
                states.append(('c', j, (0,) * j + rest))
        finals = {('c', k, (0,) * k)}
        return sta_from_delta(self.sigma, states, 4, delta, finals,
                              tapes=[self.advice_letters] +
                                    [self.element_letters] * 3)

    # ---------------- layouts, forms and the advice compiler ----------------

    @staticmethod
    def spine(n: int) -> Tree:
        """The word layout: a left chain, post-order = bottom-up."""
        if n < 1:
            raise ValueError("need at least one generator")
        node = Tree(SHAPE)
        for _ in range(n - 1):
            node = Tree(SHAPE, node, None)
        return node

    @staticmethod
    def balanced(n: int) -> Tree:
        """A balanced binary layout with n nodes."""
        if n < 1:
            raise ValueError("need at least one generator")

        def build(m):
            if m == 0:
                return None
            left = (m - 1 + 1) // 2
            return Tree(SHAPE, build(left), build(m - 1 - left))
        return build(n)

    @staticmethod
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

    def _check_form(self, n: int, form: Dict[Tuple[int, int], Sequence[int]]):
        for (j, i), label in form.items():
            if not 1 <= i < j <= n:
                raise ValueError(f"label position {(j, i)} needs 1 <= i < j <= {n}")
            if len(label) != self.k:
                raise ValueError(f"label {label} at {(j, i)} must have length k={self.k}")

    def _crossing(self, n: int, form: Dict, lo: int, hi: int) -> np.ndarray:
        """Crossing block of the post-order interval [lo, hi]: one row per
        (outside position, center coordinate), unoriented pair labels."""
        outside = {j: idx for idx, j in enumerate(
            [j for j in range(1, n + 1) if j < lo or j > hi])}
        M = np.zeros((len(outside) * self.k, hi - lo + 1), dtype=np.int64)
        for (j, i), label in form.items():
            i_in, j_in = lo <= i <= hi, lo <= j <= hi
            if i_in == j_in:
                continue
            inner, outer = (i, j) if i_in else (j, i)
            for l in range(self.k):
                M[outside[outer] * self.k + l, inner - lo] = label[l] % self.p
        return M

    def _pair_rows(self, form: Dict, t: int, lo: int, hi: int) -> np.ndarray:
        """The read-off rows at node t: labels of the pairs {t, i} for i in
        [lo, hi] (all below t in post-order), as a k x (hi-lo+1) matrix."""
        B = np.zeros((self.k, hi - lo + 1), dtype=np.int64)
        for i in range(lo, hi + 1):
            label = form.get((t, i))
            if label:
                B[:, i - lo] = [c % self.p for c in label]
        return B

    def tree_cut_rank(self, shape: Tree, form: Dict) -> int:
        """The width the given tree layout needs: the maximal rank over Z_p
        of the crossing blocks of its subtrees."""
        seq, pos, size = self._layout(shape)
        n = len(seq)
        self._check_form(n, form)
        best = 0
        for node in seq:
            t, sz = pos[id(node)], size[id(node)]
            if sz < n:
                rank = len(_rref_mod(self._crossing(n, form, t - sz + 1, t),
                                     self.p)[1])
                best = max(best, rank)
        return best

    def advice(self, shape: Tree, form: Dict[Tuple[int, int], Sequence[int]]) -> Tree:
        """Compile a layout and a form into the advice tree; raises if some
        subtree's crossing block exceeds rank r."""
        p, k, r = self.p, self.k, self.r
        seq, pos, size = self._layout(shape)
        n = len(seq)
        self._check_form(n, form)
        digs = lambda A: ''.join(str(int(d) % p) for d in np.asarray(A).flatten())
        V: Dict[int, np.ndarray] = {}
        built: Dict[int, Tree] = {}
        for node in seq:
            t, sz = pos[id(node)], size[id(node)]
            lo = t - sz + 1
            Vt = np.zeros((r, sz), dtype=np.int64)
            if sz < n:
                basis, pivots = _rref_mod(self._crossing(n, form, lo, t), p)
                if len(pivots) > r:
                    raise ValueError(
                        f"subtree at position {t} has crossing rank "
                        f"{len(pivots)} > r = {r}; this layout needs width "
                        f"{self.tree_cut_rank(shape, form)}")
                Vt[:basis.shape[0]] = basis
            L, R = node.left, node.right
            if L is None and R is None:
                letter = 'a' + digs(Vt[:, 0])
            elif L is None or R is None:
                child = L if L is not None else R
                cp, csz = pos[id(child)], size[id(child)]
                T = _solve_xa_eq_b(V[cp], Vt[:, :csz], p)
                Rm = _solve_xa_eq_b(V[cp], self._pair_rows(form, t, lo, t - 1), p)
                letter = 'b' + digs(T) + digs(Vt[:, -1]) + digs(Rm)
            else:
                lp, rp = pos[id(L)], pos[id(R)]
                lsz, rsz = size[id(L)], size[id(R)]
                VL, VR = V[lp], V[rp]
                TL = _solve_xa_eq_b(VL, Vt[:, :lsz], p)
                TR = _solve_xa_eq_b(VR, Vt[:, lsz:sz - 1], p)
                RL = _solve_xa_eq_b(VL, self._pair_rows(form, t, lo, lp), p)
                RR = _solve_xa_eq_b(VR, self._pair_rows(form, t, lp + 1, t - 1), p)
                letter = ('d' + digs(TL) + digs(TR) + digs(Vt[:, -1])
                          + digs(RL) + digs(RR))
                for l in range(k):
                    # sibling block X[j, i] = B[j, i][l], i in L, j in R;
                    # factor as V_R^T Q_l V_L
                    X = np.zeros((rsz, lsz), dtype=np.int64)
                    for jj in range(lp + 1, t):
                        for ii in range(lo, lp + 1):
                            label = form.get((jj, ii))
                            if label:
                                X[jj - lp - 1, ii - lo] = label[l] % p
                    Y = _solve_xa_eq_b(VR, X.T, p).T   # Q_l V_L, (r x lsz)
                    Ql = _solve_xa_eq_b(VL, Y, p)
                    letter += digs(Ql)
            V[t] = Vt
            built[id(node)] = Tree(letter,
                                   built.get(id(L)) if L is not None else None,
                                   built.get(id(R)) if R is not None else None)
        root = built[id(seq[-1])]
        for _ in range(k):
            root = Tree(CMARK, root, None)
        return root

    def clique_form(self, n: int, label: Sequence[int] = None) -> Dict:
        """Nothing commutes; every crossing block is all-ones — cut-rank 1
        on every layout."""
        label = tuple(label) if label is not None else (1,) + (0,) * (self.k - 1)
        return {(j, i): label for j in range(2, n + 1) for i in range(1, j)}

    def matching_form(self, n: int) -> Dict:
        """Disjoint commutator pairs of post-order neighbours: the
        extraspecial layout."""
        e1 = (1,) + (0,) * (self.k - 1)
        return {(2 * t, 2 * t - 1): e1 for t in range(1, n // 2 + 1)}

    # ---------------- encodings and class operations ----------------

    def multiply(self, n: int, form: Dict, g, h):
        """Reference implementation of the group law (identical to the word
        class: the group depends on the form, not the layout)."""
        (b1, a1), (b2, a2) = g, h
        b = [(u + v) % self.p for u, v in zip(b1, b2)]
        for (j, i), label in form.items():
            c = a1[j - 1] * a2[i - 1]
            for l in range(self.k):
                b[l] = (b[l] + label[l] * c) % self.p
        return tuple(b), tuple((u + v) % self.p for u, v in zip(a1, a2))

    def identity(self, n: int):
        return (0,) * self.k, (0,) * n

    def _strip_center(self, tree: Tree) -> Tree:
        node = tree
        for _ in range(self.k):
            if node is None or node.label != CMARK or node.right is not None:
                raise ValueError("expected a chain of k center nodes on top")
            node = node.left
        return node

    def encode(self, element, shape: Tree) -> Tree:
        """Encode (b, a) over a layout shape (an advice tree is accepted
        too — its center chain is stripped). a is indexed by post-order."""
        if shape.label == CMARK:
            shape = self._strip_center(shape)
        b, a = element
        seq, pos, size = self._layout(shape)
        if len(b) != self.k or len(a) != len(seq):
            raise ValueError(f"element must be (Z_p^{self.k}, Z_p^{len(seq)})")
        if not all(0 <= d < self.p for d in tuple(b) + tuple(a)):
            raise ValueError(f"components must lie in Z_{self.p}")
        built = {}
        for node in seq:
            built[id(node)] = Tree(
                str(a[pos[id(node)] - 1]),
                built.get(id(node.left)) if node.left is not None else None,
                built.get(id(node.right)) if node.right is not None else None)
        root = built[id(seq[-1])]
        for j in range(self.k):
            root = Tree(str(b[j]), root, None)
        return root

    def evaluate(self, phi):
        return self.cls.evaluate(phi)

    def check(self, phi, advice: Tree, **elements) -> bool:
        """Model check against the member presented by the advice; free
        variables can be assigned elements as (b, a) tuples."""
        trees = {name: self.encode(el, advice) for name, el in elements.items()}
        return self.cls.check(phi, advice, **trees)

    def get_structure(self, advice: Tree) -> TreeAutomaticPresentation:
        return self.cls.get_structure(advice)
