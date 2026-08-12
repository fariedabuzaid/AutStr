"""Ordinals below :math:`\\omega^\\omega`, as automatic structures.

Delhommé's theorem draws the line exactly here: the word-automatic ordinals are
precisely those below :math:`\\omega^\\omega`. Every one of them lives in some
:math:`\\omega^n`, so a factory for :math:`(\\omega^n, <)` covers all of them —
and past that boundary the string engine cannot go. (:math:`\\omega^{\\omega}`
itself and everything up to :math:`\\omega^{\\omega^\\omega}` needs the tree
engine.)

The structure is *derived, not authored*. By Cantor normal form an ordinal
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
successor, and ``Ordinal(n)`` has a definable least element but no greatest.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple, Union

from autstr.buildin.presentations import BuechiArithmetic
from autstr.interpretations import interpret
from autstr.symbolic import FunctionCodec, order_signature

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
