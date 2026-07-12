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

from autstr import chain_ring as cr
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
        self.cls.element_alphabet = list(self.element_letters)

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

    def check_implicit(self, phi, shape: Tree, **elements) -> bool:
        """Like `check`, evaluated implicitly (no query tree automaton)."""
        return self.cls.check_implicit(phi, self.advice(shape), **elements)

    def get_structure(self, shape: Tree) -> TreeAutomaticPresentation:
        """The tree-automatic presentation of the single group G_shape."""
        return self.cls.get_structure(self.advice(shape))


# ====================================================================
# Class-2 groups of bounded rank-width (tree layouts of bounded cut-rank)
# ====================================================================

class CutRankTreeGroups:
    """For a fixed prime p, center dimension k, width r and ring depth d, the
    uniformly tree-automatic class of class-2 groups over R = Z/p^d whose
    commutation form admits a *tree* layout of module cut-rank <= r (bounded
    rank-width over R) — the tree analog of `autstr.groups.CutRankGroups`, which
    is recovered exactly on spine layouts. With d = 1 (the default) R is the
    field F_p and this is the original construction; d > 1 is the exponent-p^d
    ("Idea 2") case. The tree merge is the one step the field proof genuinely
    uses a field: the sibling block factorises as V_R^T Q V_L, which is false
    over R with a naive interface but holds once the carried bases are the
    SATURATED free interfaces (chain_ring.factor_two_sided, Cor. cor:merge).

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
    the factorization data over R = Z/p^d (each entry as d base-p digits)::

        leaf:   'a' + v (r)
        unary:  'b' + T (r x r) + v (r) + R (k x r)
        binary: 'd' + T_L, T_R (r x r) + v (r) + R_L, R_R (k x r)
                    + Q (k x (r x r))

    with w <- T_L w_L + T_R w_R + v*digit (both factors), correction
    s <- s_L + s_R + x_t*(R_L wy_L + R_R wy_R) + wx_R^T Q_l wy_L, and the
    'c' chain consuming s coordinate by coordinate: z_c - x_c - y_c = s.
    That is q^(k+2r) layout states however large the tree. `advice` compiles
    (shape, form) into letters — saturated free bases of the crossing blocks
    plus solving the consistency systems, including the two-sided factorization
    V_R^T Q V_L of the sibling block — and fails precisely when some subtree's
    module cut-rank exceeds r; `tree_cut_rank` measures the width a layout needs.
    Every well-shaped advice presents some group in the class: the streamed
    cocycle is bilinear by construction.

    The multiplication automaton is a product over the whole letter alphabet, so
    building it is only feasible for a small ring alphabet; `cls` is therefore
    lazy and `simulate` runs the transition directly over the tapes for larger q.

    Signature: M(x,y,z), Eq(x,y); the center is first-order definable.
    """

    def __init__(self, p: int, k: int = 1, r: int = 1, d: int = 1):
        if p < 2 or p > 9 or any(p % f == 0 for f in range(2, int(p ** 0.5) + 1)):
            raise ValueError(f"p must be a prime in 2..9 (single digits), got {p}")
        if k < 1 or r < 1:
            raise ValueError("need k >= 1 central and r >= 1 state dimensions")
        if d < 1:
            raise ValueError(f"center ring depth d must be >= 1, got {d}")
        q = p ** d
        n_letters = (q ** r + q ** (r * r + r + k * r)
                     + q ** (2 * r * r + r + 2 * k * r + k * r * r))
        if n_letters > 20000:
            raise ValueError(
                f"advice alphabet would have {n_letters} letters; choose "
                f"smaller p, k, r or d (factored letters are future work)")
        self.p, self.k, self.r, self.d = p, k, r, d
        self.q = q                              # center/quotient ring R = Z/p^d
        self.digits = [str(x) for x in range(q)]
        self._digitset = set(self.digits)

        def mat(entries, off, rows, cols):
            return tuple(tuple(entries[off + i * cols + j] for j in range(cols))
                         for i in range(rows))

        self.leaf_letters = {
            self._letter_name('a', entries): tuple(entries)
            for entries in it.product(range(q), repeat=r)}
        self.unary_letters = {}
        for entries in it.product(range(q), repeat=r * r + r + k * r):
            T = mat(entries, 0, r, r)
            v = entries[r * r: r * r + r]
            R = mat(entries, r * r + r, k, r)
            self.unary_letters[self._letter_name('b', entries)] = (T, v, R)
        self.binary_letters = {}
        for entries in it.product(range(q), repeat=2 * r * r + r + 2 * k * r + k * r * r):
            off = 0
            TL = mat(entries, off, r, r); off += r * r
            TR = mat(entries, off, r, r); off += r * r
            v = entries[off: off + r]; off += r
            RL = mat(entries, off, k, r); off += k * r
            RR = mat(entries, off, k, r); off += k * r
            Q = tuple(mat(entries, off + l * r * r, r, r) for l in range(k))
            self.binary_letters[self._letter_name('d', entries)] = (TL, TR, v, RL, RR, Q)

        self.sigma = ({PAD, CMARK} | self._digitset | set(self.leaf_letters)
                      | set(self.unary_letters) | set(self.binary_letters))
        self.advice_letters = ({CMARK} | set(self.leaf_letters)
                               | set(self.unary_letters) | set(self.binary_letters))
        self.element_letters = self._digitset
        self._cls = None

    def _letter_name(self, prefix: str, entries: Sequence[int]) -> str:
        """Advice-letter key: the prefix ('a' leaf, 'b' unary, 'd' binary) plus
        every ring entry spelled as its d base-p digits (fixed width d, so the
        d = 1 field encoding is the original single-digit form)."""
        return prefix + ''.join(
            str(dig) for e in entries
            for dig in cr.to_digits(int(e) % self.q, self.p, self.d))

    @property
    def cls(self) -> UniformlyTreeAutomaticClass:
        """The uniformly tree-automatic presentation, built lazily: the tree
        multiplication automaton is a product over the whole letter alphabet, so
        its construction is only feasible for a small ring alphabet. The
        reference law, `advice` compiler, width measure and `simulate` never
        need it and stay cheap for any q."""
        if self._cls is None:
            self._cls = UniformlyTreeAutomaticClass({
                'U': self._universe_automaton(),
                'M': self._multiplication_automaton(),
                'Eq': self._eq_automaton(),
            }, padding_symbol=PAD)
            self._cls.element_alphabet = list(self.digits)
        return self._cls

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
        p, k, r, q = self.p, self.k, self.r, self.q

        def dot(M, w):
            return tuple(sum(row[j] * w[j] for j in range(r)) % q for row in M)

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
                if j >= k or (zi - xi - yi) % q != s[j]:
                    return 'dead'
                return ('c', j + 1, s[:j] + (0,) + s[j + 1:])
            if (xi + yi - zi) % q:
                return 'dead'
            if a in self.leaf_letters:
                if lq is not None or rq is not None:
                    return 'dead'
                v = self.leaf_letters[a]
                return ('t', (0,) * k,
                        tuple(v[i] * xi % q for i in range(r)),
                        tuple(v[i] * yi % q for i in range(r)))
            if a in self.unary_letters:
                child = lq if rq is None else (rq if lq is None else None)
                if child is None or child[0] != 't':
                    return 'dead'
                T, v, R = self.unary_letters[a]
                s, wx, wy = child[1], child[2], child[3]
                read = dot(R, wy)
                s = tuple((s[l] + xi * read[l]) % q for l in range(k))
                Twx, Twy = dot(T, wx), dot(T, wy)
                return ('t', s,
                        tuple((Twx[i] + v[i] * xi) % q for i in range(r)),
                        tuple((Twy[i] + v[i] * yi) % q for i in range(r)))
            if a in self.binary_letters:
                if lq is None or rq is None or lq[0] != 't' or rq[0] != 't':
                    return 'dead'
                TL, TR, v, RL, RR, Q = self.binary_letters[a]
                sL, wxL, wyL = lq[1], lq[2], lq[3]
                sR, wxR, wyR = rq[1], rq[2], rq[3]
                readL, readR = dot(RL, wyL), dot(RR, wyR)
                s = tuple((sL[l] + sR[l] + xi * (readL[l] + readR[l])
                           + sum(Q[l][u][t] * wxR[u] * wyL[t]
                                 for u in range(r) for t in range(r))) % q
                          for l in range(k))
                TwxL, TwxR = dot(TL, wxL), dot(TR, wxR)
                TwyL, TwyR = dot(TL, wyL), dot(TR, wyR)
                return ('t', s,
                        tuple((TwxL[i] + TwxR[i] + v[i] * xi) % q for i in range(r)),
                        tuple((TwyL[i] + TwyR[i] + v[i] * yi) % q for i in range(r)))
            return 'dead'

        states = ['dead']
        vecs_r = list(it.product(range(q), repeat=r))
        for s in it.product(range(q), repeat=k):
            for wx in vecs_r:
                for wy in vecs_r:
                    states.append(('t', s, wx, wy))
        for j in range(1, k + 1):
            for rest in it.product(range(q), repeat=k - j):
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
                M[outside[outer] * self.k + l, inner - lo] = label[l] % self.q
        return M

    def _pair_rows(self, form: Dict, t: int, lo: int, hi: int) -> np.ndarray:
        """The read-off rows at node t: labels of the pairs {t, i} for i in
        [lo, hi] (all below t in post-order), as a k x (hi-lo+1) matrix."""
        B = np.zeros((self.k, hi - lo + 1), dtype=np.int64)
        for i in range(lo, hi + 1):
            label = form.get((t, i))
            if label:
                B[:, i - lo] = [c % self.q for c in label]
        return B

    def tree_cut_rank(self, shape: Tree, form: Dict) -> int:
        """The width the given tree layout needs: the maximal module cut-rank
        over R = Z/p^d of the crossing blocks of its subtrees (the free rank of
        the saturated interface; the ordinary F_p rank when d = 1)."""
        seq, pos, size = self._layout(shape)
        n = len(seq)
        self._check_form(n, form)
        best = 0
        for node in seq:
            t, sz = pos[id(node)], size[id(node)]
            if sz < n:
                rank = cr.module_cut_rank(self._crossing(n, form, t - sz + 1, t),
                                          self.p, self.d)
                best = max(best, rank)
        return best

    def advice(self, shape: Tree, form: Dict[Tuple[int, int], Sequence[int]]) -> Tree:
        """Compile a layout and a form into the advice tree; raises if some
        subtree's crossing block exceeds rank r."""
        p, k, r, d, q = self.p, self.k, self.r, self.d, self.q
        seq, pos, size = self._layout(shape)
        n = len(seq)
        self._check_form(n, form)
        digs = lambda A: ''.join(
            str(dig) for e in np.asarray(A).flatten()
            for dig in cr.to_digits(int(e) % q, p, d))
        V: Dict[int, np.ndarray] = {}
        built: Dict[int, Tree] = {}
        for node in seq:
            t, sz = pos[id(node)], size[id(node)]
            lo = t - sz + 1
            Vt = np.zeros((r, sz), dtype=np.int64)
            if sz < n:
                # carry the SATURATED free basis of the subtree's crossing block
                basis, _ = cr.saturate(self._crossing(n, form, lo, t), p, d)
                if basis.shape[0] > r:
                    raise ValueError(
                        f"subtree at position {t} has module cut-rank "
                        f"{basis.shape[0]} > r = {r}; this layout needs width "
                        f"{self.tree_cut_rank(shape, form)}")
                Vt[:basis.shape[0]] = basis
            L, R = node.left, node.right
            if L is None and R is None:
                letter = 'a' + digs(Vt[:, 0])
            elif L is None or R is None:
                child = L if L is not None else R
                cp, csz = pos[id(child)], size[id(child)]
                T = cr.solve_left(V[cp], Vt[:, :csz], p, d)
                Rm = cr.solve_left(V[cp], self._pair_rows(form, t, lo, t - 1), p, d)
                letter = 'b' + digs(T) + digs(Vt[:, -1]) + digs(Rm)
            else:
                lp, rp = pos[id(L)], pos[id(R)]
                lsz, rsz = size[id(L)], size[id(R)]
                VL, VR = V[lp], V[rp]
                TL = cr.solve_left(VL, Vt[:, :lsz], p, d)
                TR = cr.solve_left(VR, Vt[:, lsz:sz - 1], p, d)
                RL = cr.solve_left(VL, self._pair_rows(form, t, lo, lp), p, d)
                RR = cr.solve_left(VR, self._pair_rows(form, t, lp + 1, t - 1), p, d)
                letter = ('d' + digs(TL) + digs(TR) + digs(Vt[:, -1])
                          + digs(RL) + digs(RR))
                for l in range(k):
                    # sibling block X[j, i] = B[j, i][l], i in L, j in R,
                    # factored as V_R^T Q_l V_L over R = Z/p^d. This is the one
                    # step that is FALSE over the ring with a naive interface;
                    # the saturated bases VL, VR make it hold (Cor. cor:merge).
                    X = np.zeros((rsz, lsz), dtype=np.int64)
                    for jj in range(lp + 1, t):
                        for ii in range(lo, lp + 1):
                            label = form.get((jj, ii))
                            if label:
                                X[jj - lp - 1, ii - lo] = label[l] % q
                    Ql = cr.factor_two_sided(X, VL, VR, p, d)
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
        """Reference implementation of the group law over R = Z/p^d (identical
        to the word class: the group depends on the form, not the layout)."""
        (b1, a1), (b2, a2) = g, h
        b = [(u + v) % self.q for u, v in zip(b1, b2)]
        for (j, i), label in form.items():
            c = a1[j - 1] * a2[i - 1]
            for l in range(self.k):
                b[l] = (b[l] + label[l] * c) % self.q
        return tuple(b), tuple((u + v) % self.q for u, v in zip(a1, a2))

    def identity(self, n: int):
        return (0,) * self.k, (0,) * n

    def simulate(self, advice: Tree, gx, gy, gz) -> bool:
        """Run the multiplication automaton directly over the convolved trees:
        True iff the advice accepts gx * gy = gz. A bottom-up pass of the shared
        `_m_step` transition, without building the product tree automaton, so the
        saturated tree merge can be checked against the reference law for any ring
        alphabet. gx, gy, gz are (b, a) elements over the advice shape."""
        tx, ty, tz = (self.encode(g, advice) for g in (gx, gy, gz))

        def rec(an, xn, yn, zn):
            left = (rec(an.left, xn.left, yn.left, zn.left)
                    if an.left is not None else None)
            right = (rec(an.right, xn.right, yn.right, zn.right)
                     if an.right is not None else None)
            return self._m_step(an.label, xn.label, yn.label, zn.label,
                                left, right)

        return self._m_accepting(rec(advice, tx, ty, tz))

    # ---------------- the multiplication transition (shared) ----------------

    def _m_accepting(self, state) -> bool:
        return state == ('c', self.k, (0,) * self.k)

    def _m_step(self, a, x, y, z, left, right):
        """One bottom-up step of the tree multiplication automaton over
        R = Z/p^d, shared by `simulate` and the implicit M atom. `a` is the
        advice label; `x, y, z` the element labels; `left`/`right` the child
        states (None for a missing child, 'dead' once any child died)."""
        k, r, q = self.k, self.r, self.q
        if left == 'dead' or right == 'dead':
            return 'dead'
        if not (x in self._digitset and y in self._digitset and z in self._digitset):
            return 'dead'
        xi, yi, zi = int(x), int(y), int(z)

        def dot(M, w):
            return tuple(sum(row[j] * w[j] for j in range(r)) % q for row in M)

        if a == CMARK:
            child = left if right is None else (right if left is None else None)
            if child is None or not isinstance(child, tuple):
                return 'dead'
            if child[0] == 't':
                j, s = 0, child[1]
            elif child[0] == 'c':
                j, s = child[1], child[2]
            else:
                return 'dead'
            if j >= k or (zi - xi - yi) % q != s[j]:
                return 'dead'
            return ('c', j + 1, s[:j] + (0,) + s[j + 1:])
        if (xi + yi - zi) % q:
            return 'dead'
        if a in self.leaf_letters:
            if left is not None or right is not None:
                return 'dead'
            v = self.leaf_letters[a]
            return ('t', (0,) * k,
                    tuple(v[i] * xi % q for i in range(r)),
                    tuple(v[i] * yi % q for i in range(r)))
        if a in self.unary_letters:
            child = left if right is None else (right if left is None else None)
            if child is None or child[0] != 't':
                return 'dead'
            T, v, R = self.unary_letters[a]
            s, wx, wy = child[1], child[2], child[3]
            s = tuple((s[l] + xi * dot(R, wy)[l]) % q for l in range(k))
            Twx, Twy = dot(T, wx), dot(T, wy)
            return ('t', s,
                    tuple((Twx[i] + v[i] * xi) % q for i in range(r)),
                    tuple((Twy[i] + v[i] * yi) % q for i in range(r)))
        if a in self.binary_letters:
            if left is None or right is None \
                    or left[0] != 't' or right[0] != 't':
                return 'dead'
            TL, TR, v, RL, RR, Q = self.binary_letters[a]
            sL, wxL, wyL = left[1], left[2], left[3]
            sR, wxR, wyR = right[1], right[2], right[3]
            readL, readR = dot(RL, wyL), dot(RR, wyR)
            s = tuple((sL[l] + sR[l] + xi * (readL[l] + readR[l])
                       + sum(Q[l][u][t] * wxR[u] * wyL[t]
                             for u in range(r) for t in range(r))) % q
                      for l in range(k))
            TwxL, TwxR = dot(TL, wxL), dot(TR, wxR)
            TwyL, TwyR = dot(TL, wyL), dot(TR, wyR)
            return ('t', s,
                    tuple((TwxL[i] + TwxR[i] + v[i] * xi) % q for i in range(r)),
                    tuple((TwyL[i] + TwyR[i] + v[i] * yi) % q for i in range(r)))
        return 'dead'

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
            raise ValueError(f"element must be (R^{self.k}, R^{len(seq)}), R = Z/{self.q}")
        if not all(0 <= x < self.q for x in tuple(b) + tuple(a)):
            raise ValueError(f"components must lie in Z/{self.q}")
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

    def _implicit_atoms(self) -> Dict:
        """Functional bottom-up base automata (Dom, Adv, M, Eq) from the tree
        deltas -- no explicit product, so this works for the large-q members
        whose `cls` cannot be built. Each entry is a builder ``args -> ImplicitTA``."""
        from autstr.implicit import ImplicitTA
        k = self.k
        digitset = self._digitset

        def shape(args, q_of):
            adv = args[0]

            def step(sym, left, right):
                if left == 'dead' or right == 'dead' or not q_of(sym):
                    return 'dead'
                a = sym[adv]
                if left is None and right is None:
                    return 'lay' if a in self.leaf_letters else 'dead'
                if (left is None) != (right is None):
                    child = left if right is None else right
                    if a in self.unary_letters:
                        return 'lay' if child == 'lay' else 'dead'
                    if a == CMARK:
                        if child == 'lay':
                            return ('c', 1)
                        if isinstance(child, tuple) and child[0] == 'c' \
                                and child[1] < k:
                            return ('c', child[1] + 1)
                    return 'dead'
                if a in self.binary_letters:
                    return 'lay' if left == 'lay' and right == 'lay' else 'dead'
                return 'dead'
            return ImplicitTA(args, step, lambda st: st == ('c', k))

        def Dom(args):
            return shape(args, lambda sym: sym[args[1]] in digitset)

        def Adv(args):
            return shape(args, lambda sym: True)

        def Eq(args):
            return shape(args, lambda sym: sym[args[1]] in digitset
                         and sym[args[1]] == sym[args[2]])

        def M(args):
            adv, xv, yv, zv = args
            return ImplicitTA(
                args,
                lambda sym, left, right: self._m_step(
                    sym[adv], sym[xv], sym[yv], sym[zv], left, right),
                self._m_accepting)

        return {'Dom': Dom, 'Adv': Adv, 'M': M, 'Eq': Eq}

    def check_implicit(self, phi, advice: Tree, **elements) -> bool:
        """Like `check`, but evaluated implicitly (no query or base tree
        automaton) -- the only viable model checker for the large-alphabet ring
        members whose `cls` cannot be built. See `autstr.implicit`."""
        from autstr import implicit
        from autstr.uniform import UniformlyAutomaticClass
        trees = {name: self.encode(el, advice) for name, el in elements.items()}
        return implicit.check_class_tree(
            phi, advice, trees, self._implicit_atoms(), list(self.digits),
            UniformlyAutomaticClass._relativize,
            UniformlyAutomaticClass._variable_names)

    def get_structure(self, advice: Tree) -> TreeAutomaticPresentation:
        return self.cls.get_structure(advice)
