"""Ordinals as automatic structures, on both engines.

Delhommé's theorem draws two lines. The word-automatic ordinals are precisely
those below :math:`\\omega^\\omega`, and the tree-automatic ones precisely those
below :math:`\\omega^{\\omega^\\omega}`. Every ordinal below the first line
lives in some :math:`\\omega^n` and every ordinal below the second in some
:math:`\\omega^{\\omega^n}`, so two factories cover both classes exactly:
`Ordinal` on the string engine and `TreeOrdinal` on the tree engine. Neither
boundary ordinal is itself reachable — :math:`\\omega^\\omega` is not
word-automatic, :math:`\\omega^{\\omega^\\omega}` not tree-automatic — which is
why both take an exponent rather than being a single structure.

`Ordinal` is *derived, not authored*. By Cantor normal form an ordinal
:math:`\\alpha < \\omega^n` is uniquely :math:`\\sum_{i<n} \\omega^i c_i` with
natural coefficients, so :math:`\\omega^n` is the set of n-tuples of naturals
under reverse-lexicographic order — an n-dimensional first-order interpretation
of Büchi arithmetic::

    >>> W2 = Ordinal(2)                       # the ordinals below ω²
    >>> S = W2.symbolic()
    >>> a, b = S.vars("a b")
    >>> a.lt(b).evaluate().contains(a=(0, 7), b=(1, 0))   # 7 < ω
    True

Ordinals are written as Python tuples of Cantor coefficients in *descending*
exponent order, so ``(3, 2, 5)`` in ``Ordinal(3)`` is
:math:`\\omega^2 \\cdot 3 + \\omega \\cdot 2 + 5` and tuples compare
lexicographically exactly as the ordinals do. A plain integer is the finite
ordinal of that value.

The signature carries the strict order ``.lt``, equality ``.eq`` and the
successor ``.succ``. Everything else is first-order over those and can be
defined on the spot: a limit ordinal is one that is neither zero nor a
successor, and both structures have a definable least element but no greatest.

`TreeOrdinal` reaches past the word barrier by nesting the same idea: an
ordinal below :math:`\\omega^{\\omega^n}` is a finite sequence of blocks
indexed by the leading exponent coefficient, each block an ordinal below
:math:`\\omega^{\\omega^{n-1}}`, bottoming out at a natural number. That nesting
is a tree, and it is authored rather than interpreted — see `TreeOrdinal`.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from autstr.buildin.presentations import BuechiArithmetic
from autstr.interpretations import interpret
from autstr.sparse_tree_automata import SparseTreeAutomaton, Tree
from autstr.symbolic import FunctionCodec, order_signature
from autstr.tree_presentations import TreeAutomaticPresentation
from autstr.utils.misc import encode_symbol

#: base-2 encoding of a natural, least significant bit first — the convention
#: of the Büchi arithmetic presentation over ℕ
_PAD = '*'

#: the constant 1 of Büchi arithmetic: the least power of two
_ONE = 'Pt(o) & (all p.(Pt(p) -> (not Lt(p,o))))'

#: a Python ordinal: coefficients in descending exponent order, or an integer
OrdinalValue = Union[int, Sequence[int]]


def _encode_nat(value: int) -> List[str]:
    """The word encoding a natural number: binary, least significant bit
    first."""
    if value < 0:
        raise ValueError(f"a Cantor coefficient must be a natural, got {value}")
    return list(format(value, 'b')[::-1])


def _decode_nat(word) -> int:
    """The natural number encoded by a word, ignoring padding."""
    bits = ''.join(symbol for symbol in word if symbol != _PAD)
    if not bits:
        raise ValueError("empty encoding")
    return int(bits[::-1], base=2)


class Ordinal:
    """The ordinals below :math:`\\omega^n`, under their natural order.

    :param n: the exponent (n ≥ 1). ``Ordinal(1)`` is :math:`(\\omega, <)`,
        i.e. the naturals; ``Ordinal(2)`` is :math:`(\\omega^2, <)`.

    The presentation carries ``Lt`` (the strict order), ``Eq`` and ``Succ``
    (the graph of the successor function, ``Succ(x, y)`` iff ``y = x + 1``).
    """

    def __init__(self, n: int = 1) -> None:
        if n < 1:
            raise ValueError("the exponent must be >= 1")
        self.n = n
        coefficients = [f'x{i}' for i in range(n)]      # x_i: coefficient of ω^i
        others = [f'y{i}' for i in range(n)]
        pair = coefficients + others

        self.presentation = interpret(
            BuechiArithmetic(),
            domain=(' & '.join(f'Eq({c},{c})' for c in coefficients),
                    coefficients),
            relations={
                'Lt': (self._less(n), pair),
                'Eq': (' & '.join(f'Eq(x{i},y{i})' for i in range(n)), pair),
                'Succ': (self._successor(n), pair),
            },
            dimension=n)

    @staticmethod
    def _less(n: int) -> str:
        """Reverse-lexicographic order on the coefficients: compare the most
        significant one first. First-order for each fixed n — a disjunction
        over which exponent the two ordinals first differ at."""
        terms = []
        for i in reversed(range(n)):
            agree = [f'Eq(x{j},y{j})' for j in range(i + 1, n)]
            terms.append(' & '.join([f'Lt(x{i},y{i})'] + agree))
        return ' | '.join(f'({term})' for term in terms)

    @staticmethod
    def _successor(n: int) -> str:
        """``y = x + 1``: add one to the constant coefficient and hold the
        rest. Every ordinal has a successor, so this is a total function."""
        held = [f'Eq(x{j},y{j})' for j in range(1, n)]
        add_one = f'exists o.(({_ONE}) & A(x0,o,y0))'
        return ' & '.join([f'({add_one})'] + held)

    # -- element codec: a Cantor normal form <-> its folded encoding ----
    def encode(self, value: OrdinalValue) -> list:
        """The encoding of an ordinal written as its Cantor coefficients in
        descending exponent order (or as a plain integer, for a finite
        ordinal)."""
        coefficients = self._coefficients(value)
        words = [_encode_nat(c) for c in reversed(coefficients)]  # x0 first
        if self.n == 1:
            return words[0]
        length = max(len(word) for word in words)
        return [tuple(word[k] if k < len(word) else _PAD for word in words)
                for k in range(length)]

    def decode(self, word) -> Tuple[int, ...]:
        """The ordinal encoded by a word, as a tuple of Cantor coefficients in
        descending exponent order."""
        if self.n == 1:
            return (_decode_nat(word),)
        return tuple(reversed([_decode_nat([letter[i] for letter in word])
                               for i in range(self.n)]))

    def _coefficients(self, value: OrdinalValue) -> Tuple[int, ...]:
        """`value` as a full n-tuple of coefficients, descending exponent."""
        if isinstance(value, int):
            value = (value,)
        coefficients = tuple(value)
        if len(coefficients) > self.n:
            raise ValueError(
                f"{value!r} has {len(coefficients)} coefficients, more than "
                f"the {self.n} of an ordinal below omega^{self.n}")
        return (0,) * (self.n - len(coefficients)) + coefficients

    # -- interface -----------------------------------------------------
    def default_signature(self):
        """The signature `symbolic()` uses when none is given: ``.lt``,
        ``.eq`` and ``.succ``, with ordinals written as coefficient tuples."""
        return order_signature(self.presentation.get_relation_symbols(),
                               less='Lt', methods={'succ': 'Succ'},
                               codec=FunctionCodec(self.encode, self.decode))

    def symbolic(self, signature=None):
        """A symbolic interface to the order; write ``a.lt(b)`` and ordinals as
        Python coefficient tuples."""
        return self.presentation.symbolic(signature or self.default_signature())

    def check(self, phi) -> bool:
        """Truth of a formula over the ordinals (free variables existential)."""
        return self.presentation.check(phi)

    def evaluate(self, phi):
        """The relation of satisfying assignments of a formula."""
        return self.presentation.evaluate(phi)

    def get_relation_symbols(self):
        """All relation symbols of the structure ('U' is the domain)."""
        return self.presentation.get_relation_symbols()

    def is_total_order(self) -> bool:
        """Whether ``Lt`` is a strict total order — decidable, being a
        first-order question. Well-foundedness is *not* first-order, so it is
        not checkable here; it holds by construction."""
        return self.presentation.check(
            'all x.(not Lt(x,x)) & '
            'all x.(all y.(all z.((Lt(x,y) & Lt(y,z)) -> Lt(x,z)))) & '
            'all x.(all y.(Lt(x,y) | Lt(y,x) | Eq(x,y)))')

    def __repr__(self):
        return f"<Ordinal omega^{self.n}>"


#: a Python ordinal below omega^(omega^n): Cantor coefficients keyed by their
#: exponent, itself an ordinal below omega^n written as a coefficient tuple
CantorForm = Union[int, Mapping[Union[int, Sequence[int]], int]]


class TreeOrdinal(TreeAutomaticPresentation):
    """The ordinals below :math:`\\omega^{\\omega^n}`, under their natural
    order — the tree-automatic ordinals, which reach exactly this far.

    :param n: the exponent (n ≥ 1). ``TreeOrdinal(1)`` is
        :math:`(\\omega^\\omega, <)`, the first ordinal past the word barrier.
    :param max_states: optional cap on the subset determinizations inside
        projection.

    **The encoding.** By Cantor normal form an ordinal below
    :math:`\\omega^{\\omega^n}` is a finite sum :math:`\\sum \\omega^{\\beta_i}
    c_i` with exponents :math:`\\beta_i < \\omega^n` — and an exponent below
    :math:`\\omega^n` is an n-tuple of naturals, exactly what `Ordinal` orders.
    Splitting off the leading coefficient of the exponent turns that into a
    recursion: an ordinal below :math:`\\omega^{\\omega^n}` is a sequence of
    *blocks* indexed densely by that leading coefficient, each block an ordinal
    below :math:`\\omega^{\\omega^{n-1}}`, and at n = 0 a natural number. So the
    encoding is a left spine of blocks whose payloads hang right and are
    themselves spines, bottoming out in binary chains — the shape the tree
    engine reads natively.

    Indexing the blocks *densely* by position is what makes the order easy:
    two ordinals are aligned by construction, so the comparison is decided at
    the deepest position where they differ, and a single three-state automaton
    (``less``, ``equal``, ``greater``) computes it bottom-up at every level of
    the nesting at once. Deeper is more significant everywhere — a deeper spine
    node is a higher exponent, a deeper bit is a higher power of two — and a
    position one ordinal has and the other lacks is a term the other is
    missing, so present beats absent.

    **Authored, not interpreted.** For n = 1 the trees coincide with Skolem
    arithmetic's, since both encode a finitely supported map from
    :math:`\\mathbb{N}` to :math:`\\mathbb{N}`. The order is *not* an
    interpretation of it: permuting the primes is an automorphism of
    :math:`(\\mathbb{N}_{>0}, \\cdot)`, so no ordering of the exponent
    positions is definable there, while the encoding fixes one. The comparison
    automaton is small enough that authoring it is the honest route.

    Ordinals are written as maps from exponent to coefficient::

        >>> W = TreeOrdinal(1)                    # ordinals below ω^ω
        >>> a, b = W.symbolic().vars("a b")
        >>> a.lt(b).evaluate().contains(a={2: 5}, b={3: 1})   # ω²·5 < ω³
        True

    An exponent is written as `Ordinal` writes one — an integer, or a tuple of
    coefficients in descending order, left-padded to length n — and a plain
    integer ordinal is the finite ordinal of that value.
    """

    #: spine node, tree root, the two bits, and padding
    SPINE, ROOT, PAD = 's', 'o', '*'
    LETTERS = frozenset({SPINE, ROOT, '0', '1', PAD})

    #: the three verdicts the comparison automaton computes bottom-up
    _EQUAL, _LESS, _GREATER = 0, 1, 2

    #: y is the immediate successor of x -- first-order over the order, since
    #: every ordinal has one, so it is defined rather than authored
    _SUCCESSOR = 'Lt(x,y) & (not exists z.(Lt(x,z) & Lt(z,y)))'

    def __init__(self, n: int = 1, max_states: Optional[int] = None) -> None:
        if n < 1:
            raise ValueError("the exponent must be >= 1")
        self.n = n
        super().__init__(
            {'U': self._universe(n),
             'Lt': self._comparison({self._LESS}),
             'Eq': self._comparison({self._EQUAL})},
            padding_symbol=self.PAD, max_states=max_states)
        self._declare_deferred({'Succ': self._SUCCESSOR})

    def default_signature(self):
        """The strict order as ``.lt``, equality as ``.eq``, the successor as
        ``.succ``, and ordinals written as Cantor maps."""
        return order_signature(self.get_relation_symbols(), less='Lt',
                               methods={'succ': 'Succ'},
                               codec=FunctionCodec(self.encode, self.decode))

    def is_total_order(self) -> bool:
        """Whether ``Lt`` is a strict total order — decidable, being
        first-order. Well-foundedness is not first-order, so it is not
        checkable here; it holds by construction."""
        return self.check(
            'all x.(not Lt(x,x)) & '
            'all x.(all y.(all z.((Lt(x,y) & Lt(y,z)) -> Lt(x,z)))) & '
            'all x.(all y.(Lt(x,y) | Lt(y,x) | Eq(x,y)))')

    def __repr__(self):
        return f"<TreeOrdinal omega^(omega^{self.n})>"

    # ---------------- encoding elements ----------------
    def encode(self, value: CantorForm) -> Tree:
        """The tree encoding an ordinal, written as a map from exponent to
        coefficient (or as a plain integer, for a finite ordinal)."""
        return Tree(self.ROOT, self._nest(self._normalize(value), self.n), None)

    def decode(self, tree: Tree) -> Dict[Tuple[int, ...], int]:
        """The ordinal a tree encodes, as a map from exponent tuple to
        coefficient; the empty map is the ordinal zero."""
        if tree is None or tree.label != self.ROOT or tree.right is not None:
            raise ValueError("not an ordinal tree")
        return self._unnest(tree.left, self.n)

    def _normalize(self, value: CantorForm) -> Dict[Tuple[int, ...], int]:
        """`value` as a map from full-length exponent tuples to non-zero
        coefficients."""
        if isinstance(value, int):
            value = {(0,) * self.n: value}
        normalized = {}
        for exponent, coefficient in value.items():
            if coefficient < 0:
                raise ValueError(
                    f"a Cantor coefficient must be a natural, got {coefficient}")
            if not coefficient:
                continue                       # a zero term is no term
            normalized[self._exponent(exponent)] = coefficient
        return normalized

    def _exponent(self, exponent) -> Tuple[int, ...]:
        """An exponent as a full n-tuple: an integer is the exponent of that
        value, and a short tuple is left-padded, as in `Ordinal`."""
        if isinstance(exponent, int):
            exponent = (exponent,)
        exponent = tuple(exponent)
        if len(exponent) > self.n:
            raise ValueError(
                f"exponent {exponent} has {len(exponent)} coefficients, more "
                f"than the {self.n} of an ordinal below omega^{self.n}")
        if any(coefficient < 0 for coefficient in exponent):
            raise ValueError(f"exponent {exponent} is not an ordinal")
        return (0,) * (self.n - len(exponent)) + exponent

    def _nest(self, terms: Dict[Tuple[int, ...], int], level: int):
        """The blocks of `terms` as a spine, recursively; None is the zero
        ordinal, which is no spine at all."""
        if not terms:
            return None
        if not level:
            return self._chain(terms[()])
        blocks: Dict[int, Dict[Tuple[int, ...], int]] = {}
        for exponent, coefficient in terms.items():
            blocks.setdefault(exponent[0], {})[exponent[1:]] = coefficient
        spine = None
        for index in range(max(blocks), -1, -1):   # deepest block first
            spine = Tree(self.SPINE, spine,
                         self._nest(blocks.get(index, {}), level - 1))
        return spine

    def _unnest(self, node, level: int) -> Dict[Tuple[int, ...], int]:
        if node is None:
            return {}
        if not level:
            return {(): self._value(node)}
        terms, index = {}, 0
        while node is not None:
            for exponent, coefficient in self._unnest(node.right,
                                                      level - 1).items():
                terms[(index,) + exponent] = coefficient
            node, index = node.left, index + 1
        return terms

    @staticmethod
    def _chain(coefficient: int):
        """A positive natural as a binary chain, most significant bit deepest
        — so that deeper is more significant here too."""
        node = None
        for bit in format(coefficient, 'b'):
            node = Tree(bit, node, None)
        return node

    @staticmethod
    def _value(node) -> int:
        value, power = 0, 1
        while node is not None:
            value += power * int(node.label)
            power, node = power * 2, node.left
        return value

    # ---------------- the automata ----------------
    @classmethod
    def _enc(cls, letters) -> int:
        return encode_symbol(tuple(letters), cls.LETTERS)

    @classmethod
    def _universe(cls, n: int) -> SparseTreeAutomaton:
        """One tree per ordinal: chains carry no leading zero, spines no
        trailing zero block, and the nesting is exactly n deep."""
        chain = 0
        spine = {level: level for level in range(1, n + 1)}   # states 1..n
        root, dead = n + 1, n + 2
        bot = n + 3

        exc = [(bot, bot, ('1',), chain),        # deepest bit is the MSB
               (chain, bot, ('0',), chain),
               (chain, bot, ('1',), chain)]
        for level in range(1, n + 1):
            payload = chain if level == 1 else spine[level - 1]
            exc += [
                # the deepest block of a spine must be non-zero, or the spine
                # would encode an ordinal it is already too long for
                (bot, payload, (cls.SPINE,), spine[level]),
                (spine[level], payload, (cls.SPINE,), spine[level]),
                (spine[level], bot, (cls.SPINE,), spine[level]),
            ]
        exc += [(bot, bot, (cls.ROOT,), root),           # the ordinal zero
                (spine[n], bot, (cls.ROOT,), root)]

        accepting = [False] * (n + 3)
        accepting[root] = True
        return SparseTreeAutomaton(
            n + 3, dead,
            [e[0] for e in exc], [e[1] for e in exc],
            [cls._enc(e[2]) for e in exc], [e[3] for e in exc],
            accepting, 1, set(cls.LETTERS))

    @classmethod
    def _comparison(cls, accepting) -> SparseTreeAutomaton:
        """The order, decided at the deepest position where two ordinals
        differ.

        Every node combines its children the same way, whatever level of the
        nesting it sits at: a verdict from the left subtree stands, since left
        is deeper and deeper is more significant; failing that the right
        subtree, which is this block's payload; failing that the labels
        themselves, where a present position beats an absent one.
        """
        equal, less, greater = cls._EQUAL, cls._LESS, cls._GREATER
        bot = 3

        def verdict(one: str, other: str) -> int:
            if one == other:
                return equal
            if one == cls.PAD:                 # a term the other ordinal has
                return less
            if other == cls.PAD:
                return greater
            return less if one < other else greater      # '0' before '1'

        # both subtrees agree so far: the labels here decide
        exc = [(left, right, (one, other), verdict(one, other))
               for left in (bot, equal) for right in (bot, equal)
               for one in sorted(cls.LETTERS) for other in sorted(cls.LETTERS)
               if verdict(one, other) != equal]

        # a subtree that has already decided outranks everything above it
        defaults = [(left, right, left if left in (less, greater) else right)
                    for left in (bot, equal, less, greater)
                    for right in (bot, equal, less, greater)
                    if left in (less, greater) or right in (less, greater)]

        return SparseTreeAutomaton(
            3, equal,
            [e[0] for e in exc], [e[1] for e in exc],
            [cls._enc(e[2]) for e in exc], [e[3] for e in exc],
            [state in accepting for state in range(3)], 2, set(cls.LETTERS),
            pd_left=[d[0] for d in defaults], pd_right=[d[1] for d in defaults],
            pd_target=[d[2] for d in defaults])
