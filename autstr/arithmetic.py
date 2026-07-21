"""Büchi arithmetic over the integers, as a symbolic structure.

:math:`(\\mathbb{Z}, +, <, \\mid_2)` presented in base 2, with the generic
symbolic interface of `autstr.symbolic` on top: build terms and formulas with
Python operators, then evaluate, model check, test for emptiness or
finiteness, and enumerate solutions.

    >>> Z = integers()
    >>> x, y, z = Z.vars("x y z")
    >>> phi = (x + y).eq(z) & z.lt(100)
    >>> phi.check()
    True
    >>> phi.evaluate().contains(x=3, y=4, z=7)
    True

Integers are written directly wherever a term is expected -- ``x + 5``,
``x.lt(100)`` -- and solutions come back as Python integers.

The signature is that of the underlying presentation: ``A`` (addition graph),
``Lt``/``Gt``/``Eq``, ``Neg`` (the graph of negation), ``N0`` (non-negative),
``Z`` (zero), ``Pt`` (powers of two) and ``B``, where ``B(x, y)`` holds iff
``y`` is a power of two dividing ``x``. Anything the operators do not cover is
reachable through `SymbolicContext.rel`.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from autstr.buildin.presentations import BuechiArithmeticZ
from autstr.symbolic import FunctionCodec, Signature, SymbolicContext

#: Base-2 encoding: a sign symbol followed by the magnitude, least significant
#: bit first. This is the convention of `lsbf_Z_automaton`.
SIGN_POSITIVE = '0'
SIGN_NEGATIVE = '1'
PADDING = '*'


def encode(n: int) -> List[str]:
    """The word encoding an integer: sign symbol, then magnitude bits LSB
    first."""
    sign = SIGN_POSITIVE if n >= 0 else SIGN_NEGATIVE
    return [sign] + list(format(abs(n), 'b')[::-1])


def decode(word) -> int:
    """The integer encoded by a word, ignoring padding."""
    word = ''.join(word).replace(PADDING, '')
    if not word:
        raise ValueError("empty encoding")
    magnitude = int(word[1:][::-1] or '0', base=2)
    return magnitude if word[0] == SIGN_POSITIVE else -magnitude


def signature() -> Signature:
    """The signature of Büchi arithmetic: ``+`` and unary ``-`` as functions,
    the order and equality as relational methods, and the integer codec."""
    sig = Signature(codec=FunctionCodec(encode, decode))
    sig.function('+', graph='A', out=2)
    sig.function('neg', graph='Neg', out=1)
    sig.operator('+', '+')
    sig.operator('-', 'neg')
    sig.operator('eq', 'Eq')
    sig.operator('lt', 'Lt')
    sig.operator('gt', 'Gt')
    # B(x, y): y is a power of two dividing x. Spelled out as a method rather
    # than bound to `|`, which on formulas already means union.
    sig.operator('divided_by_power', 'B')
    return sig


@lru_cache(maxsize=1)
def integers() -> SymbolicContext:
    """The symbolic interface to Büchi arithmetic over :math:`\\mathbb{Z}`.

    The presentation is built once and shared; expressions are immutable, so
    sharing it is safe and avoids rebuilding the base automata per term.
    """
    return BuechiArithmeticZ().symbolic(signature())


def variables(names) -> tuple:
    """Shorthand for ``integers().vars(names)``."""
    return integers().vars(names)
