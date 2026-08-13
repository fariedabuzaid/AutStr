"""The countable atomless boolean algebra, as a tree-automatic structure.

There is exactly one countable atomless boolean algebra up to isomorphism —
Cantor's back-and-forth argument again — and it is the infinite counterpart of
the `autstr.algebra.FiniteBooleanAlgebras` class, carrying the same signature
``Leq``, ``Meet``, ``Join``, ``Compl``, ``Atom``. The same formula runs against
both, and the one that separates them is the definition of the name: over a
finite algebra ``exists x. Atom(x)`` is true, and here it is false.

**Elements are clopen subsets of Cantor space.** A clopen subset of
:math:`2^\\omega` is a finite union of cylinders, so it is decided by finitely
many bits of a point — that is, by a finite binary decision tree with leaves
labelled in or out. Making the tree *reduced* (no split whose two sides are the
same constant) makes the encoding a bijection, which matters here: a
non-canonical encoding would need a quotient, and quotients are exactly what
the tree engine cannot yet supply (see `autstr.interpretations`).

**One authored automaton.** Reduction is what makes the order cheap. In a
reduced tree a subtree is constant exactly when it is a leaf, so at any
position the comparison is already decided unless *both* sides split there::

    x is 0 here            -> fine, nothing of x to contain
    y is 1 here            -> fine, y contains everything
    x is 1, y is not       -> y misses a point of x
    y is 0, x is not       -> x has a point y misses
    both split             -> recurse

Nothing below a leaf is ever consulted, so ``Leq`` is a two-state bottom-up
automaton. Everything else — ``Meet``, ``Join``, ``Compl``, ``Atom``, and
equality — is first-order over it and is defined rather than authored, built
the first time a query asks for it.

    >>> B = AtomlessBooleanAlgebra()
    >>> x, y = B.symbolic().vars("x y")
    >>> (x * y).eq({'00'}).check()          # some meet is the cylinder 00
    True
    >>> B.check('exists x.(Atom(x))')       # atomless, by construction
    False

A clopen set is written as the set of binary strings whose cylinders it
contains: ``{'0', '10'}`` is everything starting ``0`` or ``10``, ``set()`` is
empty and ``{''}`` is everything. Any covering set may be given — ``{'0','1'}``
and ``{''}`` denote the same element — and `decode` returns the canonical one.
"""
from __future__ import annotations

import itertools
from typing import FrozenSet, Iterable, Optional

from autstr.sparse_tree_automata import SparseTreeAutomaton, Tree
from autstr.symbolic import FunctionCodec, Signature
from autstr.tree_presentations import TreeAutomaticPresentation
from autstr.utils.misc import encode_symbol

#: a clopen set, as the binary strings whose cylinders it contains
Clopen = Iterable[str]


class AtomlessBooleanAlgebra(TreeAutomaticPresentation):
    """The unique countable atomless boolean algebra.

    :param max_states: optional cap on the subset determinizations inside
        projection.

    Elements are the clopen subsets of Cantor space, encoded as reduced binary
    decision trees: ``'0'`` and ``'1'`` label leaves that are out and in, and
    ``'n'`` labels a split on the next bit of a point.
    """

    #: split, out-leaf, in-leaf, padding
    SPLIT, OUT, IN, PAD = 'n', '0', '1', '*'
    LETTERS = frozenset({SPLIT, OUT, IN, PAD})

    #: `x` is the bottom / the top, spelled out so that the definitions below
    #: stay first-order over `Leq` alone
    _BOTTOM = 'all v.(Leq({0},v))'
    _TOP = 'all v.(Leq(v,{0}))'

    def __init__(self, max_states: Optional[int] = None) -> None:
        super().__init__({'U': self._universe(), 'Leq': self._order()},
                         padding_symbol=self.PAD, max_states=max_states)
        bottom, top = self._BOTTOM, self._TOP
        self._declare_deferred({
            # antisymmetry is equality, as in the finite algebras
            'Eq': 'Leq(x,y) & Leq(y,x)',
            # the greatest lower bound, and dually the least upper bound
            'Meet': ('Leq(z,x) & Leq(z,y) & '
                     'all w.((Leq(w,x) & Leq(w,y)) -> Leq(w,z))'),
            'Join': ('Leq(x,z) & Leq(y,z) & '
                     'all w.((Leq(x,w) & Leq(y,w)) -> Leq(z,w))'),
            # y is the complement of x: they meet in the bottom and join to
            # the top
            'Compl': (f'(all w.((Leq(w,x) & Leq(w,y)) -> {bottom.format("w")}))'
                      f' & (all w.((Leq(x,w) & Leq(y,w)) -> '
                      f'{top.format("w")}))'),
            # nothing lies strictly between an atom and the bottom -- here the
            # relation is empty, which is the whole point
            'Atom': (f'(not {bottom.format("x")}) & '
                     f'all y.(Leq(y,x) -> ({bottom.format("y")} | Leq(x,y)))'),
        })

    def default_signature(self):
        """Meet as ``*``, join as ``+``, complement as unary ``-``, the order
        as ``.leq`` and equality as ``.eq``.

        The lattice operations are terms, not connectives: ``&``, ``|`` and
        ``~`` already mean conjunction, disjunction and negation of *formulas*,
        so binding them here would make ``x & y`` ambiguous — the same reason
        Büchi arithmetic spells its divisibility relation
        ``divided_by_power``.
        """
        signature = Signature(codec=FunctionCodec(self.encode, self.decode))
        signature.function('meet', graph='Meet', out=2)
        signature.function('join', graph='Join', out=2)
        signature.function('compl', graph='Compl', out=1)
        signature.operator('*', 'meet')
        signature.operator('+', 'join')
        signature.operator('-', 'compl')
        signature.operator('leq', 'Leq')
        signature.operator('eq', 'Eq')
        # the operations are built on first use, so their arities cannot be
        # read off automata that do not exist yet
        signature.relations = {'Meet': 3, 'Join': 3, 'Compl': 2, 'Eq': 2,
                               'Atom': 1}
        return signature

    def is_atomless(self) -> bool:
        """Whether every non-empty element strictly contains a non-empty one —
        decidable, being first-order, and the property that pins the algebra
        down to isomorphism."""
        bottom = self._BOTTOM
        return self.check(
            f'all x.((not {bottom.format("x")}) -> exists y.('
            f'(not {bottom.format("y")}) & Leq(y,x) & (not Leq(x,y))))')

    def __repr__(self):
        return "<AtomlessBooleanAlgebra>"

    # ---------------- encoding elements ----------------
    def encode(self, clopen: Clopen) -> Tree:
        """The reduced decision tree of a clopen set, written as the binary
        strings whose cylinders it contains. Any covering set is accepted and
        canonicalized."""
        cylinders = frozenset(clopen)
        for cylinder in cylinders:
            if set(cylinder) - {self.OUT, self.IN}:
                raise ValueError(
                    f"{cylinder!r} is not a binary string, so it names no "
                    f"cylinder of Cantor space")
        return self._reduce(cylinders, '')

    def decode(self, tree: Tree) -> FrozenSet[str]:
        """The clopen set a tree encodes, as its canonical cylinders."""
        found = set()

        def walk(node, prefix):
            if node is None:
                raise ValueError("not a decision tree")
            if node.label == self.IN:
                found.add(prefix)
            elif node.label == self.SPLIT:
                walk(node.left, prefix + self.OUT)
                walk(node.right, prefix + self.IN)
            elif node.label != self.OUT:
                raise ValueError(f"{node.label!r} labels no decision node")

        walk(tree, '')
        return frozenset(found)

    def _reduce(self, cylinders: FrozenSet[str], prefix: str) -> Tree:
        """The reduced tree of `cylinders` below `prefix`: a leaf as soon as
        the answer is settled, and a split only where it is not."""
        if any(prefix.startswith(cylinder) for cylinder in cylinders):
            return Tree(self.IN)                  # inside one of them already
        if not any(cylinder.startswith(prefix) for cylinder in cylinders):
            return Tree(self.OUT)                 # none of them reaches here
        left = self._reduce(cylinders, prefix + self.OUT)
        right = self._reduce(cylinders, prefix + self.IN)
        if left.left is None and right.left is None and \
                left.label == right.label:
            return Tree(left.label)               # a split that decides nothing
        return Tree(self.SPLIT, left, right)

    # ---------------- the automata ----------------
    @classmethod
    def _enc(cls, letters) -> int:
        return encode_symbol(tuple(letters), cls.LETTERS)

    @classmethod
    def _universe(cls) -> SparseTreeAutomaton:
        """The reduced decision trees — one per clopen set. A split whose two
        sides are the same constant is the one thing forbidden; it would give
        that constant a second encoding."""
        out, in_, split, dead = 0, 1, 2, 3
        bot = 4
        exc = [(bot, bot, (cls.OUT,), out), (bot, bot, (cls.IN,), in_)]
        exc += [(left, right, (cls.SPLIT,), split)
                for left, right in itertools.product((out, in_, split),
                                                     repeat=2)
                if (left, right) not in ((out, out), (in_, in_))]
        return SparseTreeAutomaton(
            4, dead,
            [e[0] for e in exc], [e[1] for e in exc],
            [cls._enc(e[2]) for e in exc], [e[3] for e in exc],
            [True, True, True, False], 1, set(cls.LETTERS))

    @classmethod
    def _order(cls) -> SparseTreeAutomaton:
        """Containment. Because the trees are reduced, a subtree is constant
        exactly when it is a leaf, so a leaf on either side settles the whole
        cylinder and nothing below it is ever consulted."""
        ok, bad, bot = 0, 1, 2

        def verdict(one: str, other: str, below_left: int,
                    below_right: int) -> int:
            if one == cls.PAD or other == cls.PAD:
                return ok                  # settled by a leaf further up
            if one == cls.OUT or other == cls.IN:
                return ok                  # x is empty here, or y is full
            if one == cls.IN or other == cls.OUT:
                return bad                 # x has a point that y misses
            return ok if below_left == ok == below_right else bad

        exc = [(left, right, (one, other),
                verdict(one, other, ok if left == bot else left,
                        ok if right == bot else right))
               for left in (ok, bad, bot) for right in (ok, bad, bot)
               for one in sorted(cls.LETTERS) for other in sorted(cls.LETTERS)]
        exc = [e for e in exc if e[3] == ok]       # the default is a failure

        return SparseTreeAutomaton(
            2, bad,
            [e[0] for e in exc], [e[1] for e in exc],
            [cls._enc(e[2]) for e in exc], [e[3] for e in exc],
            [True, False], 2, set(cls.LETTERS))
