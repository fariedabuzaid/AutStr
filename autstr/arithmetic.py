"""Büchi arithmetic, over the naturals and over the integers.

:math:`(\\mathbb{N}, +, <, \\mid_2)` and :math:`(\\mathbb{Z}, +, <, \\mid_2)`
presented in base 2, where ``x \\mid_2 y`` says that `y` is a power of two
dividing `x`. Adding that predicate to Presburger arithmetic is what makes the
structure *Büchi* arithmetic: it can talk about the binary expansion of a
number, and it remains decidable.

    >>> Z = BuechiArithmeticZ().symbolic()
    >>> x, y, z = Z.vars("x y z")
    >>> ((x + y).eq(z) & z.lt(100)).check()
    True

Both presentations are compiled from a handful of small automata the first time
they are constructed; the derived relations (order, equality, negation) are
first-order definitions over those.
"""
from typing import List

import itertools as it

from autstr.presentations import (
    AutomaticPresentation, CompiledPresentation,
)
from autstr.utils.automata_tools import create_sparse_dfa


def _build_buechi_arithmetic() -> AutomaticPresentation:
    """Büchi arithmetic over the natural numbers, compiled from scratch."""
    # Universe automaton - only accepts valid binary numbers (no '*')
    universe = create_sparse_dfa(
        states={'i', '0', '0+', '1', '*'},
        input_symbols={('0',), ('1',), ('*',)},
        transitions={
            'i': {('0',): '0', ('1',): '1', ('*',): '*'},
            '0': {('0',): '0+', ('1',): '1', ('*',): '*'},
            '0+': {('0',): '0+', ('1',): '1', ('*',): '*'},
            '1': {('0',): '0+', ('1',): '1', ('*',): '*'},
            '*': {('0',): '*', ('1',): '*', ('*',): '*'},
        },
        initial_state='i',
        final_states={'0', '1'}
    )

    addition = create_sparse_dfa(
        states={0, 1, 2},
        input_symbols={('0', '0', '0'), ('0', '0', '1'), ('0', '0', '*'), ('0', '1', '0'), ('0', '1', '1'),
                       ('0', '1', '*'), ('0', '*', '0'), ('0', '*', '1'), ('0', '*', '*'),
                       ('1', '0', '0'), ('1', '0', '1'), ('1', '0', '*'), ('1', '1', '0'), ('1', '1', '1'),
                       ('1', '1', '*'), ('1', '*', '0'), ('1', '*', '1'), ('1', '*', '*'),
                       ('*', '0', '0'), ('*', '0', '1'), ('*', '0', '*'), ('*', '1', '0'), ('*', '1', '1'),
                       ('*', '1', '*'), ('*', '*', '0'), ('*', '*', '1'), ('*', '*', '*'),
                       },
        transitions={
            0: {
                ('0', '0', '0'): 0, ('0', '0', '1'): 2, ('0', '0', '*'): 2, ('0', '1', '0'): 2, ('0', '1', '1'): 0,
                ('0', '1', '*'): 2, ('0', '*', '0'): 0, ('0', '*', '1'): 2, ('0', '*', '*'): 2,
                ('1', '0', '0'): 2, ('1', '0', '1'): 0, ('1', '0', '*'): 2, ('1', '1', '0'): 1, ('1', '1', '1'): 2,
                ('1', '1', '*'): 2, ('1', '*', '0'): 2, ('1', '*', '1'): 0, ('1', '*', '*'): 2,
                ('*', '0', '0'): 0, ('*', '0', '1'): 2, ('*', '0', '*'): 2, ('*', '1', '0'): 2, ('*', '1', '1'): 0,
                ('*', '1', '*'): 2, ('*', '*', '0'): 2, ('*', '*', '1'): 2, ('*', '*', '*'): 2,
            },
            1: {
                ('0', '0', '0'): 2, ('0', '0', '1'): 0, ('0', '0', '*'): 2, ('0', '1', '0'): 1, ('0', '1', '1'): 2,
                ('0', '1', '*'): 2, ('0', '*', '0'): 2, ('0', '*', '1'): 0, ('0', '*', '*'): 2,
                ('1', '0', '0'): 1, ('1', '0', '1'): 2, ('1', '0', '*'): 2, ('1', '1', '0'): 2, ('1', '1', '1'): 1,
                ('1', '1', '*'): 2, ('1', '*', '0'): 1, ('1', '*', '1'): 2, ('1', '*', '*'): 2,
                ('*', '0', '0'): 2, ('*', '0', '1'): 0, ('*', '0', '*'): 2, ('*', '1', '0'): 1, ('*', '1', '1'): 2,
                ('*', '1', '*'): 2, ('*', '*', '0'): 2, ('*', '*', '1'): 0, ('*', '*', '*'): 2,
            },
            2: {
                ('0', '0', '0'): 2, ('0', '0', '1'): 2, ('0', '0', '*'): 2, ('0', '1', '0'): 2, ('0', '1', '1'): 2,
                ('0', '1', '*'): 2, ('0', '*', '0'): 2, ('0', '*', '1'): 2, ('0', '*', '*'): 2,
                ('1', '0', '0'): 2, ('1', '0', '1'): 2, ('1', '0', '*'): 2, ('1', '1', '0'): 2, ('1', '1', '1'): 2,
                ('1', '1', '*'): 2, ('1', '*', '0'): 2, ('1', '*', '1'): 2, ('1', '*', '*'): 2,
                ('*', '0', '0'): 2, ('*', '0', '1'): 2, ('*', '0', '*'): 2, ('*', '1', '0'): 2, ('*', '1', '1'): 2,
                ('*', '1', '*'): 2, ('*', '*', '0'): 2, ('*', '*', '1'): 2, ('*', '*', '*'): 2,
            },
        },
        initial_state=0,
        final_states={0}
    )

    input_symbols = {a for a in it.product(['0', '1', '*'], repeat=2)}
    weak_div = create_sparse_dfa(
        states={'0', '1', 'e'},
        input_symbols=input_symbols,
        transitions={
            '0': {
                a: '0' if a == ('0', '0') else '1' if a == ('0', '1') or a == ('1', '1') else 'e' for a in input_symbols
            },
            '1': {
                a: 'e' if a[1] != '*' else '1' for a in input_symbols
            },
            'e': {a: 'e' for a in input_symbols}
        },
        initial_state='0',
        final_states={'1'}
    )
    

    # Create presentation
    presentation = AutomaticPresentation({'U': universe, 'A': addition, 'B': weak_div})

    # Add bootstrap remaining relations
    presentation.update(Z='A(x,x,x)')
    presentation.update(Eq='exists z.(Z(z) and A(x,z,y))')
    presentation.update(Pt='B(x,x)')
    presentation.update(Lt='exists z.(not Z(z) and A(x, z, y))')
    presentation.update(Gt='exists z.(not Z(z) and A(y, z, x))')
    
    return presentation

def _build_buechi_arithmetic_Z() -> AutomaticPresentation:
    """Büchi arithmetic over the integers, compiled from scratch."""
    # Universe automaton
    universe_states = ['-1', 'i+', 'i', '0', '0+', '1', '*']
    universe_symbols = {('0',), ('1',), ('*',)}
    universe_trans = {
        '-1': {('0',): 'i', ('1',): 'i+', ('*',): '*'},
        'i+': {('0',): '0+', ('1',): '1', ('*',): '*'},
        'i': {('0',): '0', ('1',): '1', ('*',): '*'},
        '0': {('0',): '0+', ('1',): '1', ('*',): '*'},
        '0+': {('0',): '0+', ('1',): '1', ('*',): '*'},
        '1': {('0',): '0+', ('1',): '1', ('*',): '*'},
        '*': {('0',): '*', ('1',): '*', ('*',): '*'},
    }
    universe = create_sparse_dfa(
        universe_states, universe_symbols, universe_trans, '-1', {'0', '1'}
    )
    
    # Addition automaton (intermediate)
    add_states = [-1, 0, 1, 2]
    add_symbols = set(it.product(['0', '1', '*'], repeat=3))
    add_trans = {
        -1: {a: 0 if '*' not in a else 2 for a in add_symbols},
        0: {
            ('0','0','0'): 0, ('0','0','1'): 2, ('0','0','*'): 2,
            ('0','1','0'): 2, ('0','1','1'): 0, ('0','1','*'): 2,
            ('0','*','0'): 0, ('0','*','1'): 2, ('0','*','*'): 2,
            ('1','0','0'): 2, ('1','0','1'): 0, ('1','0','*'): 2,
            ('1','1','0'): 1, ('1','1','1'): 2, ('1','1','*'): 2,
            ('1','*','0'): 2, ('1','*','1'): 0, ('1','*','*'): 2,
            ('*','0','0'): 0, ('*','0','1'): 2, ('*','0','*'): 2,
            ('*','1','0'): 2, ('*','1','1'): 0, ('*','1','*'): 2,
            ('*','*','0'): 2, ('*','*','1'): 2, ('*','*','*'): 2,
        },
        1: {
            ('0','0','0'): 2, ('0','0','1'): 0, ('0','0','*'): 2,
            ('0','1','0'): 1, ('0','1','1'): 2, ('0','1','*'): 2,
            ('0','*','0'): 2, ('0','*','1'): 0, ('0','*','*'): 2,
            ('1','0','0'): 1, ('1','0','1'): 2, ('1','0','*'): 2,
            ('1','1','0'): 2, ('1','1','1'): 1, ('1','1','*'): 2,
            ('1','*','0'): 1, ('1','*','1'): 2, ('1','*','*'): 2,
            ('*','0','0'): 2, ('*','0','1'): 0, ('*','0','*'): 2,
            ('*','1','0'): 1, ('*','1','1'): 2, ('*','1','*'): 2,
            ('*','*','0'): 2, ('*','*','1'): 0, ('*','*','*'): 2,
        },
        2: {s: 2 for s in add_symbols}
    }
    addition_intermediate = create_sparse_dfa(add_states, add_symbols, add_trans, -1, {0})
    
    # Weak division automaton
    div_states = ['-1', '0', '1', 'e']
    div_symbols = set(it.product(['0', '1', '*'], repeat=2))
    div_trans = {
        '-1': {a: '0' if a[1] == '0' else 'e' for a in div_symbols},
        '0': {
            a: '0' if a == ('0','0') 
            else '1' if a in {('0','1'), ('1','1')} 
            else 'e' for a in div_symbols
        },
        '1': {
            a: 'e' if a[1] != '*' else '1' for a in div_symbols
        },
        'e': {a: 'e' for a in div_symbols}
    }
    weak_div = create_sparse_dfa(div_states, div_symbols, div_trans, '-1', {'1'})
    
    # N0 automaton
    n0_states = [-1, 0, 1]
    n0_symbols = {('0',), ('1',), ('*',)}
    n0_trans = {
        -1: {('0',): 1, ('1',): 0, ('*',): 0},
        0: {s: 0 for s in n0_symbols},
        1: {s: 1 for s in n0_symbols}
    }
    N0 = create_sparse_dfa(n0_states, n0_symbols, n0_trans, -1, {1})
    
    # Create presentation
    presentation = AutomaticPresentation({
        'U': universe, 
        'A0': addition_intermediate, 
        'B': weak_div, 
        'N0': N0
    })
    presentation.update(Z='A0(x,x,x)')
    
    # Define addition formula
    c000 = '(N0(x) and N0(y) and N0(z) and A0(x, y, z))'
    c001 = '(N0(x) and N0(y) and not N0(z) and exists a z0.(Z(z0) and A0(x,y,a) and A0(a,z,z0)))'
    c010 = '(N0(x) and not N0(y) and N0(z) and A0(z, y, x))'
    c011 = '(N0(x) and not N0(y) and not N0(z) and A0(z, x, y))'
    c100 = '(not N0(x) and N0(y) and N0(z) and A0(x, z, y))'
    c101 = '(not N0(x) and N0(y) and not N0(z) and A0(z, y, x))'
    c110 = '(not N0(x) and not N0(y) and N0(z) and exists a z0.(Z(z0) and A0(x,y,a) and A0(a,z,z0)))'
    c111 = '(not N0(x) and not N0(y) and not N0(z) and A0(x,y,z))'
    phi_A = ' or '.join([c000, c001, c010, c011, c100, c101, c110, c111])
    presentation.update(A=phi_A)
    
    presentation.update(Eq='exists z.(Z(z) and A(x,z,y))')
    presentation.update(Pt='B(x,x) and N0(x)')
    presentation.update(Lt='exists z.(N0(z) and not Z(z) and A(x, z, y))')
    presentation.update(Gt='exists z.(N0(z) and not Z(z) and A(y, z, x))')
    presentation.update(Neg='exists z.(Z(z) and A(x,y,z))')
    
    # Delete auxiliary relation
    del presentation.automata['A0']
    
    return presentation


# --------------------------------------------------------------------------
# The presentations. Each declares its own vocabulary, so `symbolic()` takes
# no argument; see `CompiledPresentation`.
# --------------------------------------------------------------------------

class BuechiArithmetic(CompiledPresentation):
    """Büchi arithmetic over the natural numbers: :math:`(\\mathbb{N}, +, <,
    \\mid_2)` in base 2.

        >>> N = BuechiArithmetic()
        >>> x, y = N.symbolic().vars("x y")
        >>> ((x + y).eq(12) & x.lt(y)).check()
        True

    ``B(x, y)`` holds iff `y` is a power of two dividing `x`; it is spelled
    ``.divided_by_power`` rather than bound to ``|``, which on formulas already
    means union.
    """

    _BUILD = staticmethod(_build_buechi_arithmetic)

    #: base-2 encoding: the magnitude, least significant bit first
    PADDING = '*'

    @staticmethod
    def encode(n: int) -> List[str]:
        """The word encoding a natural number: binary, least significant bit
        first."""
        if n < 0:
            raise ValueError(f"not a natural number: {n}")
        return list(format(n, 'b')[::-1])

    @staticmethod
    def decode(word) -> int:
        """The natural number encoded by a word, ignoring padding."""
        digits = ''.join(word).replace(BuechiArithmetic.PADDING, '')
        return int(digits[::-1] or '0', base=2)

    def default_signature(self):
        """``+`` as addition, with the order, equality and divisibility as
        methods, and naturals written as Python integers."""
        from autstr.symbolic import FunctionCodec, Signature
        signature = Signature(codec=FunctionCodec(self.encode, self.decode))
        signature.function('+', graph='A', out=2)
        signature.operator('+', '+')
        signature.operator('eq', 'Eq')
        signature.operator('lt', 'Lt')
        signature.operator('gt', 'Gt')
        signature.operator('divided_by_power', 'B')
        return signature


class BuechiArithmeticZ(CompiledPresentation):
    """Büchi arithmetic over the integers: :math:`(\\mathbb{Z}, +, <, \\mid_2)`
    in base 2.

        >>> Z = BuechiArithmeticZ()
        >>> x, y, z = Z.symbolic().vars("x y z")
        >>> ((x + y).eq(z) & z.lt(100)).check()
        True
        >>> (3, 4, 7) in (x + y).eq(z)
        True

    Integers are written directly wherever a term is expected -- ``x + 5``,
    ``x.lt(100)`` -- and solutions come back as Python integers. Anything the
    operators do not cover is reachable through `SymbolicContext.rel`.
    """

    _BUILD = staticmethod(_build_buechi_arithmetic_Z)

    #: base-2 encoding: a sign symbol, then the magnitude, least significant
    #: bit first
    SIGN_POSITIVE = '0'
    SIGN_NEGATIVE = '1'
    PADDING = '*'

    @staticmethod
    def encode(n: int) -> List[str]:
        """The word encoding an integer: sign symbol, then magnitude bits least
        significant first."""
        sign = (BuechiArithmeticZ.SIGN_POSITIVE if n >= 0
                else BuechiArithmeticZ.SIGN_NEGATIVE)
        return [sign] + list(format(abs(n), 'b')[::-1])

    @staticmethod
    def decode(word) -> int:
        """The integer encoded by a word, ignoring padding."""
        word = ''.join(word).replace(BuechiArithmeticZ.PADDING, '')
        if not word:
            raise ValueError("empty encoding")
        magnitude = int(word[1:][::-1] or '0', base=2)
        return (magnitude if word[0] == BuechiArithmeticZ.SIGN_POSITIVE
                else -magnitude)

    def default_signature(self):
        """``+`` as addition and unary ``-`` as negation, with the order,
        equality and divisibility as methods, and integers written as Python
        integers."""
        from autstr.symbolic import FunctionCodec, Signature
        signature = Signature(codec=FunctionCodec(self.encode, self.decode))
        signature.function('+', graph='A', out=2)
        signature.function('neg', graph='Neg', out=1)
        signature.operator('+', '+')
        signature.operator('-', 'neg')
        signature.operator('eq', 'Eq')
        signature.operator('lt', 'Lt')
        signature.operator('gt', 'Gt')
        # B(x, y): y is a power of two dividing x. Spelled out as a method
        # rather than bound to `|`, which on formulas already means union.
        signature.operator('divided_by_power', 'B')
        return signature


