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
"""
import itertools as it
from typing import Dict, Optional, Union

from autstr.sparse_tree_automata import SparseTreeAutomaton, Tree
from autstr.tree_presentations import TreeAutomaticPresentation
from autstr.tree_uniform import UniformlyTreeAutomaticClass, sta_from_delta

PAD = '*'
SHAPE = 's'


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
