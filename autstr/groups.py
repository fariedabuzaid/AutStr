"""Uniformly automatic classes of finite groups.

**Finite abelian groups** (`FiniteAbelianGroups`): every finite abelian group
is a direct sum of cyclic groups, so the advice is the '#'-separated list of
their orders in LSB-first binary and addition is blockwise. This is the
direct-product closure of the cyclic groups, made explicit -- compare
`autstr.composition.direct_product_closure`.


**Groups with a cyclic subgroup of index <= 2** (`IndexTwoCyclicGroups`):
every such group is <r, s | r^n, s^2 = r^w, s r s^-1 = r^u> with u^2 = 1
(mod n), and the classification yields six families over the cyclic part
Z_n:

    abelian       Z_2 x Z_n            u = 1        w = 0
    cyclic        C_2n                 u = 1        w = 1
    dihedral      D_n                  u = -1       w = 0
    dicyclic      Dic (Q_2n)           u = -1       w = n/2   (n even)
    semidihedral  SD_2n                u = n/2 - 1  w = 0     (n = 2^k >= 4)
    modular       M_2n                 u = n/2 + 1  w = 0     (n = 2^k >= 4)

The advice is one family symbol followed by the LSB-first binary digits of
n; an element r^a s^e is encoded as the twist bit e followed by the digits
of a. Multiplication obeys

    (r^a s^e)(r^b s^f) = r^{a + u^e b + [e and f] w} s^{e xor f},

and every ingredient is regular: modular addition is the usual carry
automaton, conjugation is the identity, negation, or +/-x + (x mod 2)*(n/2)
(the n/2-shift is advice-recognizable), and w is an advice-definable
constant. The relation M(x,y,z) is *defined* from these primitives by a
first-order formula — the uniform analog of the Büchi-arithmetic bootstrap.

**Extraspecial p-groups** (`ExtraspecialGroups(p)`, p a fixed prime): the
Heisenberg-type group of order p^(1+2n) has elements (c, a, b) with
c in Z_p, a, b in Z_p^n and multiplication

    (c, a, b)(c', a', b') = (c + c' + <a, b'>, a + a', b + b').

For fixed p the bilinear correction <a, b'> pairs digits *positionwise*, so
it is a running sum mod p in the automaton state — nilpotency class 2 with
growing rank is uniformly automatic, in contrast to growing modulus
(Heisenberg over Z_n interprets modular multiplication and is not).

**Class-2 groups of bounded linear cut-rank** (`CutRankGroups(p, k, r)`):
the common generalization. A member is a central extension of Z_p^n by
Z_p^k given by commutator labels [x_j, x_i] = y^B[j,i]; the advice spells
out, cut by cut, a rank-<=r factorization of the crossing block of B, and
the automaton carries r linear functionals of the digits read so far
instead of the digits themselves. Bounded pathwidth, bounded vertex cover,
the extraspecial matching and the complete graph (nothing commutes,
pathwidth n-1, cut-rank 1!) are all special layouts.
"""
import itertools as it
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

from autstr import chain_ring as cr
from autstr.presentations import AutomaticPresentation
from autstr.sparse_automata import SparseDFA
from autstr.uniform import (
    SymbolicClassWrapper, UniformlyAutomaticClass, dfa_from_delta,
)

PAD = '*'
SEP = '#'


def _lsbf_bits(n: int) -> List[str]:
    """LSB-first binary digits of n >= 1."""
    return list(bin(n)[2:][::-1])


# ====================================================================
# Groups with a cyclic subgroup of index <= 2
# ====================================================================

class IndexTwoCyclicGroups(SymbolicClassWrapper):
    """The uniformly automatic class of finite groups with a cyclic subgroup
    of index <= 2, in one advice format. Signature: M(x,y,z) [z = x*y],
    T(x) [x is a twisted element r^a s], Eq(x,y), plus the primitives the
    bootstrap is built from (CAdd, Conj, IsW, family predicates, ...)."""

    FAMILY_SYMBOLS = {
        'abelian': 'fA', 'cyclic': 'fC', 'dihedral': 'fD',
        'dicyclic': 'fQ', 'semidihedral': 'fS', 'modular': 'fM',
    }

    def __init__(self):
        self.families = dict(self.FAMILY_SYMBOLS)
        self.family_of = {sym: fam for fam, sym in self.families.items()}
        self.sigma = {PAD, '0', '1'} | set(self.families.values())

        automata = {
            'U': self._universe_automaton(),
            'T': self._twist_automaton(),
            'Odd': self._odd_automaton(),
            'CZero': self._czero_automaton(),
            'COne': self._cone_automaton(),
            'CHalf': self._chalf_automaton(),
            'CEq': self._ceq_automaton(),
            'Eq': self._eq_automaton(),
            'CAdd': self._cadd_automaton(),
        }
        for family, symbol in self.families.items():
            automata['F' + symbol[1]] = self._family_automaton(symbol)
        self.cls = UniformlyAutomaticClass(automata, padding_symbol=PAD)
        self.cls.element_alphabet = ['0', '1']

        # Bootstrap: negation, conjugation by u, the constant w, and finally
        # the group multiplication, all defined first-order from the primitives
        self.cls.define('CNeg', 'exists o.(CZero(o) and CAdd(x,y,o))')
        self.cls.define('Conj', (
            '((FA(x) or FC(x)) and CEq(x,y)) or '
            '((FD(x) or FQ(x)) and CNeg(x,y)) or '
            '(FS(x) and (((not Odd(x)) and CNeg(x,y)) or '
            '  (Odd(x) and (exists h.(CHalf(h) and '
            '    (exists m.(CNeg(x,m) and CAdd(h,m,y)))))))) or '
            '(FM(x) and (((not Odd(x)) and CEq(x,y)) or '
            '  (Odd(x) and (exists h.(CHalf(h) and CAdd(h,x,y))))))'
        ))
        self.cls.define('IsW', (
            '((FA(x) or FD(x) or FS(x) or FM(x)) and CZero(x)) or '
            '(FC(x) and COne(x)) or (FQ(x) and CHalf(x))'
        ))
        self.cls.define('M', (
            '((not T(x)) and ((T(y) and T(z)) or ((not T(y)) and (not T(z)))) '
            '  and CAdd(x,y,z)) or '
            '(T(x) and (not T(y)) and T(z) and '
            '  (exists a.(Conj(y,a) and CAdd(a,x,z)))) or '
            '(T(x) and T(y) and (not T(z)) and '
            '  (exists a.(Conj(y,a) and (exists w.(IsW(w) and '
            '    (exists s.(CAdd(a,x,s) and CAdd(s,w,z))))))))'
        ))

    # ---------------- presentation automata ----------------

    def _universe_automaton(self) -> SparseDFA:
        """U(p, x): p = [family][canonical LSB binary of n] with the family's
        side conditions (dicyclic: n even; semidihedral/modular: n a power of
        two >= 4); x = [twist bit][digits of a] with a < n, exact length."""
        families = set(self.families.values())
        states = ['start', 'dead'] + [
            ('u', fam, cmp, last, ones, first)
            for fam in families
            for cmp in ('EQ', 'LT', 'GT')
            for last in ('0', '1')
            for ones in (0, 1, 2)
            for first in ('-', '0', '1')
        ]

        def delta(q, sym):
            a, x = sym
            if q == 'dead':
                return 'dead'
            if q == 'start':
                if a in families and x in ('0', '1'):
                    return ('u', a, 'EQ', '0', 0, '-')
                return 'dead'
            _, fam, cmp, last, ones, first = q
            if a in ('0', '1') and x in ('0', '1'):
                if x != a:
                    cmp = 'LT' if x == '0' else 'GT'
                return ('u', fam, cmp, a, min(2, ones + int(a)),
                        a if first == '-' else first)
            return 'dead'

        def is_final(q):
            if q in ('start', 'dead') or q[5] == '-':
                return False
            _, fam, cmp, last, ones, first = q
            if last != '1' or cmp != 'LT':  # canonical n, element < n
                return False
            family = self.family_of[fam]
            if family == 'dicyclic':
                return first == '0'  # n even (and hence >= 2)
            if family in ('semidihedral', 'modular'):
                # n a power of two: exactly one 1-digit (the last); n >= 4
                return ones == 1 and first == '0'
            return True

        finals = {q for q in states if is_final(q)}
        return dfa_from_delta(self.sigma, states, 2, delta, 'start', finals)

    def _twist_automaton(self) -> SparseDFA:
        """T(p, x): the twist bit of x is 1."""
        families = set(self.families.values())
        states = ['start', 'yes', 'no', 'dead']

        def delta(q, sym):
            a, x = sym
            if q == 'start':
                if a in families and x in ('0', '1'):
                    return 'yes' if x == '1' else 'no'
                return 'dead'
            if q in ('yes', 'no') and a in ('0', '1') and x in ('0', '1'):
                return q
            return 'dead'

        return dfa_from_delta(self.sigma, states, 2, delta, 'start', {'yes'})

    def _odd_automaton(self) -> SparseDFA:
        """Odd(p, x): the least significant digit of a is 1."""
        families = set(self.families.values())
        states = ['start', 'head', 'yes', 'no', 'dead']

        def delta(q, sym):
            a, x = sym
            if q == 'start':
                return 'head' if a in families and x in ('0', '1') else 'dead'
            if a in ('0', '1') and x in ('0', '1'):
                if q == 'head':
                    return 'yes' if x == '1' else 'no'
                if q in ('yes', 'no'):
                    return q
            return 'dead'

        return dfa_from_delta(self.sigma, states, 2, delta, 'start', {'yes'})

    def _czero_automaton(self) -> SparseDFA:
        """CZero(p, x): a = 0."""
        families = set(self.families.values())
        states = ['start', 'zero', 'dead']

        def delta(q, sym):
            a, x = sym
            if q == 'start':
                return 'zero' if a in families and x in ('0', '1') else 'dead'
            if q == 'zero' and a in ('0', '1') and x == '0':
                return 'zero'
            return 'dead'

        return dfa_from_delta(self.sigma, states, 2, delta, 'start', {'zero'})

    def _cone_automaton(self) -> SparseDFA:
        """COne(p, x): a = 1 mod n (i.e. digits 100..0, or a = 0 when n = 1)."""
        families = set(self.families.values())
        states = ['start', 'head', 'is0', 'is1', 'dead']

        def delta(q, sym):
            a, x = sym
            if q == 'start':
                return 'head' if a in families and x in ('0', '1') else 'dead'
            if a in ('0', '1') and x in ('0', '1'):
                if q == 'head':
                    return 'is1' if x == '1' else 'is0'  # is0: a=0 so far
                if q in ('is0', 'is1') and x == '0':
                    return 'is1' if q == 'is1' else 'dead'
            return 'dead'

        # a = 0 with a single digit means n = 1, where 0 = 1 mod n
        return dfa_from_delta(self.sigma, states, 2, delta, 'start', {'is1', 'is0'})

    def _chalf_automaton(self) -> SparseDFA:
        """CHalf(p, x): 2a = n, i.e. n_0 = 0, n_{i+1} = a_i and the top digit
        of a is 0. Only satisfiable when n is even."""
        families = set(self.families.values())
        states = ['start', 'head', ('s', '0'), ('s', '1'), 'dead']

        def delta(q, sym):
            a, x = sym
            if q == 'start':
                return 'head' if a in families and x in ('0', '1') else 'dead'
            if a in ('0', '1') and x in ('0', '1'):
                if q == 'head':
                    return ('s', x) if a == '0' else 'dead'  # n_0 must be 0
                prev = q[1]
                return ('s', x) if a == prev else 'dead'  # n_i = a_{i-1}
            return 'dead'

        # top digit of a pending in the state must be 0 (2a has no overflow)
        return dfa_from_delta(self.sigma, states, 2, delta, 'start', {('s', '0')})

    def _ceq_automaton(self) -> SparseDFA:
        """CEq(p, x, y): the cyclic parts agree (twist bits are ignored)."""
        families = set(self.families.values())
        states = ['start', 'eq', 'dead']

        def delta(q, sym):
            a, x, y = sym
            if q == 'start':
                return 'eq' if a in families and x in ('0', '1') and y in ('0', '1') else 'dead'
            if q == 'eq' and a in ('0', '1') and x in ('0', '1') and x == y:
                return 'eq'
            return 'dead'

        return dfa_from_delta(self.sigma, states, 3, delta, 'start', {'eq'})

    def _eq_automaton(self) -> SparseDFA:
        """Eq(p, x, y): x and y are the same element (identical words)."""
        families = set(self.families.values())
        states = ['start', 'eq', 'dead']

        def delta(q, sym):
            a, x, y = sym
            if x != y:
                return 'dead'
            if q == 'start':
                return 'eq' if a in families and x in ('0', '1') else 'dead'
            if q == 'eq' and a in ('0', '1') and x in ('0', '1'):
                return 'eq'
            return 'dead'

        return dfa_from_delta(self.sigma, states, 3, delta, 'start', {'eq'})

    def _cadd_automaton(self) -> SparseDFA:
        """CAdd(p, x, y, z): a_x + a_y = a_z (mod n), twist bits ignored.
        Verified as a_x + a_y = a_z + q*n for a guessed wrap q in {0,1},
        digit by digit LSB-first with one signed carry per branch."""
        families = set(self.families.values())
        carries = [-2, -1, 0, 1, 2, 'F']
        states = ['start', 'dead'] + [('c', c0, c1) for c0 in carries for c1 in carries]

        def step(carry, n_bit, x, y, z, q):
            if carry == 'F':
                return 'F'
            s = carry + x + y - z - q * n_bit
            if s % 2 != 0:
                return 'F'
            s //= 2
            return s if -2 <= s <= 2 else 'F'

        def delta(q, sym):
            a, x, y, z = sym
            if q == 'dead':
                return 'dead'
            if q == 'start':
                if a in families and all(s in ('0', '1') for s in (x, y, z)):
                    return ('c', 0, 0)
                return 'dead'
            if a in ('0', '1') and all(s in ('0', '1') for s in (x, y, z)):
                bits = (int(a), int(x), int(y), int(z))
                c0, c1 = step(q[1], *bits, 0), step(q[2], *bits, 1)
                return ('c', c0, c1) if (c0, c1) != ('F', 'F') else 'dead'
            return 'dead'

        finals = {('c', c0, c1) for c0 in carries for c1 in carries if 0 in (c0, c1)}
        return dfa_from_delta(self.sigma, states, 4, delta, 'start', finals)

    def _family_automaton(self, symbol: str) -> SparseDFA:
        """F*(p, x): the advice belongs to the given family (x is ignored)."""
        families = set(self.families.values())
        states = ['start', 'yes', 'dead']

        def delta(q, sym):
            a, x = sym
            if q == 'start':
                if a in families and x in ('0', '1'):
                    return 'yes' if a == symbol else 'dead'
                return 'dead'
            if q == 'yes' and a in ('0', '1') and x in ('0', '1'):
                return 'yes'
            return 'dead'

        return dfa_from_delta(self.sigma, states, 2, delta, 'start', {'yes'})

    # ---------------- encodings and class operations ----------------

    def advice(self, family: str, n: int) -> List[str]:
        """Advice string of the group in the given family over Z_n."""
        if family not in self.families:
            raise ValueError(f"unknown family {family!r}; choose from {sorted(self.families)}")
        if n < 1:
            raise ValueError("cyclic part order must be >= 1")
        if family == 'dicyclic' and n % 2 != 0:
            raise ValueError("dicyclic groups need an even cyclic part")
        if family in ('semidihedral', 'modular') and (n < 4 or n & (n - 1) != 0):
            raise ValueError(f"{family} groups need n a power of two >= 4")
        return [self.families[family]] + _lsbf_bits(n)

    def dihedral(self, n: int) -> List[str]:
        """D_n of order 2n."""
        return self.advice('dihedral', n)

    def dicyclic(self, n: int) -> List[str]:
        """Dicyclic group of order 2n over Z_n (n even); n = 4 is Q_8."""
        return self.advice('dicyclic', n)

    def semidihedral(self, n: int) -> List[str]:
        """SD of order 2n over Z_n, n = 2^k >= 4."""
        return self.advice('semidihedral', n)

    def modular(self, n: int) -> List[str]:
        """Modular maximal-cyclic group of order 2n over Z_n, n = 2^k >= 4."""
        return self.advice('modular', n)

    def cyclic(self, n: int) -> List[str]:
        """C_2n presented over the index-2 subgroup Z_n."""
        return self.advice('cyclic', n)

    def abelian(self, n: int) -> List[str]:
        """Z_2 x Z_n."""
        return self.advice('abelian', n)

    def parameters(self, advice: Sequence[str]) -> Tuple[str, int]:
        """(family, n) described by an advice string."""
        family = self.family_of[advice[0]]
        n = sum(int(b) << i for i, b in enumerate(advice[1:]))
        return family, n

    def encode(self, element: Tuple[int, int], advice: Sequence[str]) -> List[str]:
        """Encode r^a s^e, given as (e, a), for the group of the advice."""
        e, a = element
        _, n = self.parameters(advice)
        if e not in (0, 1) or not 0 <= a < n:
            raise ValueError(f"({e}, {a}) is not an element (twist in {{0,1}}, 0 <= a < {n})")
        digits = _lsbf_bits(a) if a > 0 else ['0']
        return [str(e)] + digits + ['0'] * (len(advice) - 1 - len(digits))

    def multiply(self, advice: Sequence[str], g: Tuple[int, int], h: Tuple[int, int]) -> Tuple[int, int]:
        """Reference implementation of the group law (for testing/decoding)."""
        family, n = self.parameters(advice)
        u = {'abelian': 1, 'cyclic': 1, 'dihedral': -1, 'dicyclic': -1,
             'semidihedral': n // 2 - 1, 'modular': n // 2 + 1}[family]
        w = {'abelian': 0, 'cyclic': 1 % n, 'dihedral': 0, 'dicyclic': n // 2,
             'semidihedral': 0, 'modular': 0}[family]
        (e, a), (f, b) = g, h
        c = (a + (u * b if e else b) + (w if e and f else 0)) % n
        return (e ^ f, c)

    def evaluate(self, phi) -> Tuple[SparseDFA, List[str]]:
        return self.cls.evaluate(phi)

    def check(self, phi, advice: Sequence[str], **elements) -> bool:
        """Model check against one group; free variables can be assigned
        elements as (twist, exponent) pairs."""
        words = {name: self.encode(el, advice) for name, el in elements.items()}
        return self.cls.check(phi, advice, **words)

    def check_implicit(self, phi, advice: Sequence[str], **elements) -> bool:
        """Like `check`, evaluated implicitly (no query automaton)."""
        words = {name: self.encode(el, advice) for name, el in elements.items()}
        return self.cls.check_implicit(phi, advice, **words)

    def get_structure(self, advice: Sequence[str]) -> AutomaticPresentation:
        return self.cls.get_structure(advice)


# ====================================================================
# Extraspecial p-groups (fixed p, growing rank)
# ====================================================================

class ExtraspecialGroups(SymbolicClassWrapper):
    """For a fixed prime p, the uniformly automatic class of Heisenberg-type
    groups of order p^(1+2n): elements (c, a, b) with c in Z_p and
    a, b in Z_p^n, multiplied as (c,a,b)(c',a',b') =
    (c + c' + <a,b'>, a + a', b + b'). Advice: the unary word 1^(n+1).
    Signature: M(x,y,z), Cen(x) [x is central], Eq(x,y)."""

    def __init__(self, p: int):
        if p < 2 or any(p % d == 0 for d in range(2, int(p ** 0.5) + 1)):
            raise ValueError(f"p must be prime, got {p}")
        self.p = p
        self.digits = [str(d) for d in range(p)]
        self.pairs = {f"{d}_{e}": (d, e) for d in range(p) for e in range(p)}
        self.sigma = {PAD, '1'} | set(self.digits) | set(self.pairs)

        self.cls = UniformlyAutomaticClass({
            'U': self._universe_automaton(),
            'M': self._multiplication_automaton(),
            'Cen': self._center_automaton(),
            'Eq': self._eq_automaton(),
        }, padding_symbol=PAD)
        # element tape holds a c-digit then (a_i, b_i) pair symbols
        self.cls.element_alphabet = list(self.digits) + list(self.pairs)

    def _universe_automaton(self) -> SparseDFA:
        """U(p, x): x = [c-digit][(a_i, b_i) pairs], exactly advice length."""
        states = ['start', 'ok', 'dead']

        def delta(q, sym):
            a, x = sym
            if q == 'start':
                return 'ok' if a == '1' and x in self.digits else 'dead'
            if q == 'ok' and a == '1' and x in self.pairs:
                return 'ok'
            return 'dead'

        return dfa_from_delta(self.sigma, states, 2, delta, 'start', {'ok'})

    def _multiplication_automaton(self) -> SparseDFA:
        """M(p, x, y, z): componentwise addition mod p, with the symplectic
        correction <a_x, b_y> accumulated as a running deficit of the center
        digit: accept iff c_z - c_x - c_y - sum_i a_x,i * b_y,i = 0 (mod p)."""
        states = ['start', 'dead'] + [('d', r) for r in range(self.p)]

        def delta(q, sym):
            a, x, y, z = sym
            if q == 'dead':
                return 'dead'
            if q == 'start':
                if a == '1' and all(s in self.digits for s in (x, y, z)):
                    return ('d', (int(z) - int(x) - int(y)) % self.p)
                return 'dead'
            if a == '1' and all(s in self.pairs for s in (x, y, z)):
                (ax, bx), (ay, by), (az, bz) = self.pairs[x], self.pairs[y], self.pairs[z]
                if az != (ax + ay) % self.p or bz != (bx + by) % self.p:
                    return 'dead'
                return ('d', (q[1] - ax * by) % self.p)
            return 'dead'

        return dfa_from_delta(self.sigma, states, 4, delta, 'start', {('d', 0)})

    def _center_automaton(self) -> SparseDFA:
        """Cen(p, x): a = b = 0 (x is central)."""
        states = ['start', 'ok', 'dead']
        zero_pair = f"0_0"

        def delta(q, sym):
            a, x = sym
            if q == 'start':
                return 'ok' if a == '1' and x in self.digits else 'dead'
            if q == 'ok' and a == '1' and x == zero_pair:
                return 'ok'
            return 'dead'

        return dfa_from_delta(self.sigma, states, 2, delta, 'start', {'ok'})

    def _eq_automaton(self) -> SparseDFA:
        """Eq(p, x, y): identical elements."""
        states = ['start', 'eq', 'dead']

        def delta(q, sym):
            a, x, y = sym
            if x != y:
                return 'dead'
            if q == 'start':
                return 'eq' if a == '1' and x in self.digits else 'dead'
            if q == 'eq' and a == '1' and x in self.pairs:
                return 'eq'
            return 'dead'

        return dfa_from_delta(self.sigma, states, 3, delta, 'start', {'eq'})

    # ---------------- encodings and class operations ----------------

    def advice(self, n: int) -> List[str]:
        """Advice string of the extraspecial group of order p^(1+2n)."""
        if n < 0:
            raise ValueError("rank must be >= 0")
        return ['1'] * (n + 1)

    def encode(self, element, n: int) -> List[str]:
        """Encode (c, a_vector, b_vector) for the rank-n group."""
        c, a, b = element
        a, b = list(a), list(b)
        if len(a) != n or len(b) != n:
            raise ValueError(f"vectors must have length {n}")
        if not (0 <= c < self.p and all(0 <= v < self.p for v in a + b)):
            raise ValueError(f"components must lie in Z_{self.p}")
        return [str(c)] + [f"{ai}_{bi}" for ai, bi in zip(a, b)]

    def multiply(self, g, h) -> Tuple[int, Tuple[int, ...], Tuple[int, ...]]:
        """Reference implementation of the group law."""
        (c1, a1, b1), (c2, a2, b2) = g, h
        c = (c1 + c2 + sum(x * y for x, y in zip(a1, b2))) % self.p
        a = tuple((x + y) % self.p for x, y in zip(a1, a2))
        b = tuple((x + y) % self.p for x, y in zip(b1, b2))
        return (c, a, b)

    def evaluate(self, phi) -> Tuple[SparseDFA, List[str]]:
        return self.cls.evaluate(phi)

    def check(self, phi, n: int, **elements) -> bool:
        """Model check against the rank-n group; free variables can be
        assigned elements as (c, a_vector, b_vector) triples."""
        words = {name: self.encode(el, n) for name, el in elements.items()}
        return self.cls.check(phi, self.advice(n), **words)

    def check_implicit(self, phi, n: int, **elements) -> bool:
        """Like `check`, evaluated implicitly (no query automaton)."""
        words = {name: self.encode(el, n) for name, el in elements.items()}
        return self.cls.check_implicit(phi, self.advice(n), **words)

    def get_structure(self, n: int) -> AutomaticPresentation:
        return self.cls.get_structure(self.advice(n))


# ====================================================================
# Finite abelian groups
# ====================================================================

class FiniteAbelianGroups(SymbolicClassWrapper):
    """The uniformly automatic class of all finite abelian groups, presented
    by their cyclic decompositions: the group Z_{n_1} ⊕ ... ⊕ Z_{n_k} has
    advice bin(n_1)# ... bin(n_k)# (LSB-first binary per block)."""

    #: addition, not multiplication -- these members are abelian by
    #: construction, and the class presents the operation as A(x, y, z).
    GRAPH = 'A'
    OPERATOR = '+'
    #: the identity is the unique idempotent, and x + 0 = y exactly when x = y.
    #: The witness may not be called e0: nltk reads e-names as event variables.
    EQUALITY = 'exists z0.(A(z0,z0,z0) and A(x,z0,y))'

    def __init__(self, eager_equality: bool = False):
        self.sigma = {PAD, '0', '1', SEP}
        self.cls = UniformlyAutomaticClass({
            'U': self._universe_automaton(),
            'A': self._addition_automaton(),
        }, padding_symbol=PAD)
        self._declare_equality(eager_equality)
        self.cls.element_alphabet = ['0', '1', SEP]

    def _universe_automaton(self) -> SparseDFA:
        """U(p, x): p is a sequence of canonical nonempty binary blocks (MSB
        1 last, so each block denotes some n >= 1) and per block the element
        digits denote a value < n. LSB-first comparison: the last differing
        digit decides."""
        states = ['between', 'pad', 'dead']
        states += [('c', cmp, msb) for cmp in ('EQ', 'LT', 'GT') for msb in (0, 1)]

        def delta(q, sym):
            a, x = sym
            if q == 'dead':
                return 'dead'
            if q == 'pad':
                return 'pad' if (a, x) == (PAD, PAD) else 'dead'
            if a in ('0', '1') and x in ('0', '1'):
                cmp = q[1] if q != 'between' else 'EQ'
                if x != a:
                    cmp = 'LT' if x == '0' else 'GT'
                return ('c', cmp, int(a))
            if (a, x) == (SEP, SEP):
                # block ends: canonical (msb 1) and element value < n
                if q != 'between' and q[1] == 'LT' and q[2] == 1:
                    return 'between'
                return 'dead'
            if (a, x) == (PAD, PAD) and q == 'between':
                return 'pad'
            return 'dead'

        return dfa_from_delta(self.sigma, states, 2, delta, 'between', {'between', 'pad'})

    def _addition_automaton(self) -> SparseDFA:
        """A(p, x, y, z): per block, x + y = z + b*n for some b in {0,1}
        (i.e. z = x + y mod n), checked digit by digit LSB-first with one
        signed carry per branch b."""
        carries = list(range(-2, 3)) + ['F']
        states = [('a', c0, c1) for c0 in carries for c1 in carries]
        states += ['pad', 'dead']

        def step(carry, n_bit, x, y, z, b):
            if carry == 'F':
                return 'F'
            s = carry + x + y - z - b * n_bit
            if s % 2 != 0:
                return 'F'
            s //= 2
            return s if -2 <= s <= 2 else 'F'

        def delta(q, sym):
            a, x, y, z = sym
            if q == 'dead':
                return 'dead'
            if q == 'pad':
                return 'pad' if all(s == PAD for s in sym) else 'dead'
            c0, c1 = q[1], q[2]
            if a in ('0', '1') and all(s in ('0', '1') for s in (x, y, z)):
                bits = (int(a), int(x), int(y), int(z))
                return ('a', step(c0, *bits, 0), step(c1, *bits, 1))
            if all(s == SEP for s in sym):
                # block ends: some branch must close with carry 0
                return ('a', 0, 0) if 0 in (c0, c1) else 'dead'
            if all(s == PAD for s in sym) and (c0, c1) == (0, 0):
                return 'pad'
            return 'dead'

        return dfa_from_delta(self.sigma, states, 4, delta, ('a', 0, 0),
                              {('a', 0, 0), 'pad'})

    # ---------------- encodings and class operations ----------------

    def advice(self, orders: Sequence[int]) -> List[str]:
        """Advice string of Z_{n_1} ⊕ ... ⊕ Z_{n_k}."""
        word = []
        for n in orders:
            if n < 1:
                raise ValueError("cyclic factor orders must be >= 1")
            word += _lsbf_bits(n) + [SEP]
        return word

    def encode(self, element: Union[int, Sequence[int]], orders: Sequence[int]) -> List[str]:
        """Encode a group element (one value per cyclic factor)."""
        if isinstance(element, int):
            element = (element,)
        if len(element) != len(orders):
            raise ValueError(f"element has {len(element)} components, group has {len(orders)}")
        word = []
        for a, n in zip(element, orders):
            if not 0 <= a < n:
                raise ValueError(f"component {a} not in Z_{n}")
            block_len = len(_lsbf_bits(n))
            digits = _lsbf_bits(a) if a > 0 else ['0']
            word += digits + ['0'] * (block_len - len(digits)) + [SEP]
        return word

    def evaluate(self, phi) -> Tuple[SparseDFA, List[str]]:
        return self.cls.evaluate(phi)

    def check(self, phi, orders: Sequence[int], **elements) -> bool:
        """Model check against Z_{n_1} ⊕ ... ⊕ Z_{n_k}; free variables can be
        assigned group elements (tuples with one value per factor, or a
        single int for a cyclic group)."""
        words = {name: self.encode(e, orders) for name, e in elements.items()}
        return self.cls.check(phi, self.advice(orders), **words)

    def check_implicit(self, phi, orders: Sequence[int], **elements) -> bool:
        """Like `check`, evaluated implicitly (no query automaton)."""
        words = {name: self.encode(e, orders) for name, e in elements.items()}
        return self.cls.check_implicit(phi, self.advice(orders), **words)

    def get_structure(self, orders: Sequence[int]) -> AutomaticPresentation:
        return self.cls.get_structure(self.advice(orders))


# ====================================================================
# Class-2 groups of bounded linear cut-rank (linear rank-width)
# ====================================================================

CMARK = 'c'


def _rref_mod(A: np.ndarray, p: int) -> Tuple[np.ndarray, List[int]]:
    """Reduced row echelon form over Z_p: (nonzero rows, pivot columns)."""
    A = A.copy() % p
    m, n = A.shape
    pivots = []
    row = 0
    for col in range(n):
        if row == m:
            break
        sel = next((i for i in range(row, m) if A[i, col]), None)
        if sel is None:
            continue
        A[[row, sel]] = A[[sel, row]]
        A[row] = (A[row] * pow(int(A[row, col]), p - 2, p)) % p
        for i in range(m):
            if i != row and A[i, col]:
                A[i] = (A[i] - A[i, col] * A[row]) % p
        pivots.append(col)
        row += 1
    return A[:row], pivots


def _solve_xa_eq_b(A: np.ndarray, B: np.ndarray, p: int) -> np.ndarray:
    """Any X over Z_p with X A = B (mod p); raises if the system is
    inconsistent. A: (a, c), B: (m, c), X: (m, a); free variables are 0."""
    a, c = A.shape
    m = B.shape[0]
    X = np.zeros((m, a), dtype=np.int64)
    if c == 0:
        return X
    reduced, pivots = _rref_mod(np.concatenate([A.T % p, B.T % p], axis=1), p)
    for row, col in enumerate(pivots):
        if col >= a:
            raise ValueError("inconsistent linear system over Z_p")
        X[:, col] = reduced[row, a:]
    return X


class CutRankGroups(SymbolicClassWrapper):
    """For a fixed prime p, center dimension k, width r and ring depth d, the
    uniformly automatic class of class-2 groups over R = Z/p^d whose commutation
    form admits a linear layout of module cut-rank <= r (bounded *linear
    rank-width* over R). With d = 1 (the default) R is the field F_p and this is
    the original construction; d > 1 is the exponent-p^d ("Idea 2") case, where
    the form is R-valued and the width is the module cut-rank of the crossing
    blocks (the streaming lifts unconditionally, carrying row-module
    generator interfaces).

    A member is a form with labels B[j,i] in R^k for i < j, presenting
    x_j x_i = x_i x_j y^B[j,i]. Elements are (b, a) with b in R^k, a in R^n and

        (b, a)(b', a') = (b + b' + C(a, a'), a + a'),
        C(a, a') = sum_{i<j} B[j,i] a_j a'_i,

    a 2-cocycle because C is bilinear. The multiplication automaton never
    needs the digits it has read — only what the future consumes from them,
    which is the crossing block of B at the current cut. If every crossing
    block factors through rank r, the automaton can carry the r linear
    functionals w = V a' instead of the digits. The advice letter at
    position t spells the factorization out — three matrices over Z_p:

        T (r x r): basis change between cuts,   w <- T w + v a'_t,
        v (r):     coefficient of the new digit,
        R (k x r): read-off,   sum_{i<t} B[t,i] a'_i = R w,

    so the state is (deficit in Z_p^k, w in Z_p^r): p^(k+r) states however
    long the word. Every well-shaped advice presents *some* group in the
    class (the streamed cocycle is bilinear by construction), and
    `advice` compiles a form into letters, failing precisely when a cut
    exceeds rank r; `linear_cut_rank` measures the width a form needs.

    Special layouts: a perfect matching is extraspecial (`matching_form`,
    cut-rank 1, compare `ExtraspecialGroups`); the complete graph — nothing
    commutes, pathwidth n-1 — also has cut-rank 1, its crossing blocks are
    all-ones (`clique_form`); a vertex cover of size c needs rank <= c.
    Signature: M(x,y,z), Eq(x,y); the center is first-order definable,
    Cen(x) := all y,z,w (M(x,y,z) and M(y,x,w) -> z = w).
    """

    #: marker letter opening each position's entry stretch in factored mode
    MARKER = 'n'

    def __init__(self, p: int, k: int = 1, r: int = 1, d: int = 1,
                 factored: bool = None):
        """`factored=None` (the default) enumerates one advice letter per
        (T, v, R) triple when that alphabet fits under 20000 letters (the
        original encoding, byte-identical) and otherwise switches to
        *factored* letters: each position becomes a marker 'n' followed by
        one letter per ring entry (T row-major, then v, then R row-major),
        with the element digit repeated along the stretch. The factored
        alphabet has q+1 advice letters however large r, k and d are --
        this is what makes width r >= 2 over the ring representable."""
        if p < 2 or p > 9 or any(p % f == 0 for f in range(2, int(p ** 0.5) + 1)):
            raise ValueError(f"p must be a prime in 2..9 (single digits), got {p}")
        if k < 1 or r < 1:
            raise ValueError("need k >= 1 central and r >= 1 state dimensions")
        if d < 1:
            raise ValueError(f"center ring depth d must be >= 1, got {d}")
        n_flat = p ** (d * (r * r + r + k * r))
        if factored is None:
            factored = n_flat > 20000
        if not factored and n_flat > 20000:
            raise ValueError(
                f"the flat advice alphabet would have {n_flat} letters; "
                f"choose smaller p, k, r or d, or use factored=True")
        self.p, self.k, self.r, self.d = p, k, r, d
        self.q = p ** d                       # center/quotient ring R = Z/p^d
        self.factored = factored
        self.n_entries = r * r + r + k * r    # ring entries per position
        self.digits = [str(x) for x in range(self.q)]
        self._digitset = set(self.digits)
        self.letters: Dict[str, tuple] = {}
        self.entry_letters: Dict[str, int] = {}
        if factored:
            self.entry_letters = {f'e{self._digits_of(c)}': c
                                  for c in range(self.q)}
            advice = {self.MARKER} | set(self.entry_letters)
        else:
            for entries in it.product(range(self.q), repeat=self.n_entries):
                T = tuple(tuple(entries[i * r + j] for j in range(r))
                          for i in range(r))
                v = tuple(entries[r * r: r * r + r])
                R = tuple(tuple(entries[r * r + r + l * r + s]
                                for s in range(r)) for l in range(k))
                self.letters[self._letter_name(entries)] = (T, v, R)
            advice = set(self.letters)
        self.sigma = {PAD, CMARK} | set(self.digits) | advice
        self._advice_tape = {PAD, CMARK} | advice
        self._element_tape = {PAD} | set(self.digits)
        self._cls = None

    def _digits_of(self, c: int) -> str:
        """A ring entry as its d base-p digits, least significant first."""
        return ''.join(str(dig)
                       for dig in cr.to_digits(int(c) % self.q, self.p, self.d))

    #: cap on the transition enumerations of the lazy `cls` build; beyond it
    #: the build would run for minutes to hours, so `cls` raises instead
    _CLS_ENUMERATION_CAP = 20_000_000

    def _cls_cost(self) -> int:
        """Transition enumerations the lazy `cls` build performs (dominated by
        the 4-tape multiplication automaton: states x advice x digits^3)."""
        q, k, r = self.q, self.k, self.r
        n_states = 1 + sum(q ** j for j in range(k))
        if self.factored:
            n_states += q ** (k + 2 * r) * (self.n_entries + 1)
        else:
            n_states += q ** (k + r)
        return n_states * (len(self._advice_tape)) * (q + 1) ** 3

    @property
    def cls(self) -> UniformlyAutomaticClass:
        """The uniformly automatic presentation, built lazily: the multiplication
        automaton is a 4-tape product over the full advice alphabet, so its
        construction is O(sigma^4) and only feasible for a small ring alphabet
        (essentially Z/4, width 1). The reference law, `advice` compiler, width
        measure and `simulate` do not need it and stay cheap for any q."""
        if self._cls is None:
            cost = self._cls_cost()
            if cost > self._CLS_ENUMERATION_CAP:
                raise ValueError(
                    f"advice alphabet too large to build the explicit "
                    f"presentation automata: {len(self.letters)} letters, "
                    f"~{cost:.0e} transition enumerations "
                    f"(cap {self._CLS_ENUMERATION_CAP:.0e}); "
                    f"use check_implicit or simulate instead")
            self._cls = UniformlyAutomaticClass({
                'U': self._universe_automaton(),
                'M': self._multiplication_automaton(),
                'Eq': self._eq_automaton(),
            }, padding_symbol=PAD)
            self._cls.element_alphabet = list(self.digits)
        return self._cls

    def _letter_name(self, entries: Sequence[int]) -> str:
        """Advice-letter key for the flat entry list [T (r*r), v (r), R (k*r)],
        each ring entry spelled as its d base-p digits (fixed width d, so the
        d = 1 field encoding is the original single-digit form)."""
        return 'a' + ''.join(
            str(dig) for e in entries
            for dig in cr.to_digits(int(e) % self.q, self.p, self.d))

    # ---------------- the automata ----------------

    def _shape_states(self):
        """Shape states shared by U and Eq: the k center markers, then either
        one letter per position (flat) or marker + entry stretches with the
        element digit repeated (factored; the digit memory lives only in U)."""
        states = ['dead', 'ok'] + [('c', j) for j in range(self.k)]
        if self.factored:
            states += [('s', ph, dig) for ph in range(self.n_entries)
                       for dig in self.digits]
        return states

    def _shape_step(self, q, a, x):
        """One shape step (state, advice letter, element digit) shared by U
        and Eq; the caller has already checked the digit condition."""
        if a == CMARK and q not in ('ok', 'dead') and q[0] == 'c':
            return 'ok' if q[1] + 1 == self.k else ('c', q[1] + 1)
        if not self.factored:
            return 'ok' if a in self.letters and q == 'ok' else 'dead'
        if a == self.MARKER and q == 'ok':
            return ('s', 0, x) if self.n_entries else 'ok'
        if a in self.entry_letters and q != 'ok' and q[0] == 's' \
                and q[2] == x:
            return 'ok' if q[1] + 1 == self.n_entries else ('s', q[1] + 1, x)
        return 'dead'

    def _universe_automaton(self) -> SparseDFA:
        """U(p, x): k digits under the center markers, then one digit per
        advice letter (repeated along the stretch in factored mode)."""
        def delta(q, sym):
            a, x = sym
            if q != 'dead' and x in self.digits:
                return self._shape_step(q, a, x)
            return 'dead'

        return dfa_from_delta(self.sigma, self._shape_states(), 2, delta,
                              ('c', 0), {'ok'},
                              tapes=[self._advice_tape, self._element_tape],
                              dead='dead')

    def _multiplication_automaton(self) -> SparseDFA:
        """M(p, x, y, z): z = x·y via the shared `_m_step` transition. The
        center digits seed the deficit d = b_z - b_x - b_y; each x-position
        checks the digit sum, subtracts its commutator contribution
        x_t * (R w) from the deficit and updates the carried functionals
        w <- T w + v * y_t (streamed through an accumulator in factored
        mode); accept iff d = 0."""
        k, r, q = self.k, self.r, self.q
        states = ['dead']
        for j in range(k):
            for d in it.product(range(q), repeat=j):
                states.append(('c', j, d + (0,) * (k - j)))
        if self.factored:
            for d in it.product(range(q), repeat=k):
                for w in it.product(range(q), repeat=r):
                    for acc in it.product(range(q), repeat=r):
                        for ph in range(self.n_entries + 1):
                            states.append(('x', d, w, acc, ph))
        else:
            for d in it.product(range(q), repeat=k):
                for w in it.product(range(q), repeat=r):
                    states.append(('x', d, w))

        def delta(state, sym):
            return self._m_step(state, *sym)

        finals = [s for s in states if self._m_accepting(s)]
        return dfa_from_delta(self.sigma, states, 4, delta, ('c', 0, (0,) * k),
                              set(finals),
                              tapes=[self._advice_tape] + [self._element_tape] * 3,
                              dead='dead')

    def _eq_automaton(self) -> SparseDFA:
        """Eq(p, x, y): identical well-shaped elements."""
        def delta(q, sym):
            a, x, y = sym
            if q != 'dead' and x == y and x in self.digits:
                return self._shape_step(q, a, x)
            return 'dead'

        return dfa_from_delta(self.sigma, self._shape_states(), 3, delta,
                              ('c', 0), {'ok'},
                              tapes=[self._advice_tape] + [self._element_tape] * 2,
                              dead='dead')

    # ---------------- forms, layouts and advice ----------------

    def _check_form(self, n: int, form: Dict[Tuple[int, int], Sequence[int]]):
        for (j, i), label in form.items():
            if not 1 <= i < j <= n:
                raise ValueError(f"label position {(j, i)} needs 1 <= i < j <= {n}")
            if len(label) != self.k:
                raise ValueError(f"label {label} at {(j, i)} must have length k={self.k}")

    def _crossing(self, n: int, form: Dict, t: int) -> np.ndarray:
        """Crossing block of the form at the cut after position t, flattened
        to one row per (later position j, center coordinate)."""
        M = np.zeros(((n - t) * self.k, t), dtype=np.int64)
        for (j, i), label in form.items():
            if i <= t < j:
                for l in range(self.k):
                    M[(j - t - 1) * self.k + l, i - 1] = label[l] % self.q
        return M

    def linear_cut_rank(self, n: int, form: Dict) -> int:
        """The width the given layout of the form needs: the maximal module
        cut-rank over R = Z/p^d of its crossing blocks (the free rank of the
        saturated interface; the ordinary F_p rank when d = 1)."""
        self._check_form(n, form)
        return max((cr.module_cut_rank(self._crossing(n, form, t), self.p, self.d)
                    for t in range(1, n)), default=0)

    def advice(self, n: int, form: Dict[Tuple[int, int], Sequence[int]]) -> List[str]:
        """Compile a form — {(j, i): label in Z_p^k, i < j} — into advice.
        Maintains a row basis V of each crossing block's row space and
        expresses the next basis and the current read-off row over it;
        raises if some cut exceeds rank r."""
        self._check_form(n, form)
        p, k, r, d, q = self.p, self.k, self.r, self.d, self.q
        word = [CMARK] * k
        V_prev = np.zeros((r, 0), dtype=np.int64)
        for t in range(1, n + 1):
            Bt = np.zeros((k, t - 1), dtype=np.int64)
            for i in range(1, t):
                label = form.get((t, i))
                if label:
                    Bt[:, i - 1] = [c % q for c in label]
            # read-off of the new position over the carried interface:
            # R V_prev = Bt (rows of Bt lie in the crossing row module)
            R = cr.solve_left(V_prev, Bt, p, d)
            V = np.zeros((r, t), dtype=np.int64)
            if t < n:
                # carry a minimal GENERATING set of the crossing block's row
                # module (Smith with the p-power factors kept). Restricting a
                # row of M_t to the first t-1 columns gives a row of M_{t-1},
                # so row spaces restrict into row spaces and the transition T
                # below always solves. A saturated basis would NOT work here:
                # pure closures are non-unique over Z/p^d and need not nest
                # under restriction (e.g. the width-1 form {(3,1): 2,
                # (3,2): 1, (4,2): 2} over Z/4, where the saturated cut-3
                # interface (0,1,0) restricts outside rowsp{(2,1)}).
                basis, exps = cr.saturate(self._crossing(n, form, t), p, d)
                if basis.shape[0] > r:
                    raise ValueError(
                        f"cut {t} has module cut-rank {basis.shape[0]} > r = {r}; "
                        f"this form needs width {self.linear_cut_rank(n, form)}")
                for row, e in enumerate(exps):
                    V[row] = (basis[row] * p ** e) % q
            # basis change between cuts: T V_prev = V restricted to earlier cols
            T = cr.solve_left(V_prev, V[:, :t - 1], p, d)
            entries = [T[i, j] for i in range(r) for j in range(r)]
            entries += [V[s, t - 1] for s in range(r)]
            entries += [R[l, s] for l in range(k) for s in range(r)]
            if self.factored:
                word.append(self.MARKER)
                word += [f'e{self._digits_of(int(c))}' for c in entries]
            else:
                word.append(self._letter_name(entries))
            V_prev = V
        return word

    def clique_form(self, n: int, label: Sequence[int] = None) -> Dict:
        """Nothing commutes: [x_j, x_i] = y^label for every pair. Pathwidth
        n-1, but every crossing block is all-ones — cut-rank 1."""
        label = tuple(label) if label is not None else (1,) + (0,) * (self.k - 1)
        return {(j, i): label for j in range(2, n + 1) for i in range(1, j)}

    def matching_form(self, n: int) -> Dict:
        """Disjoint commutator pairs hitting the first center coordinate:
        the extraspecial layout, cut-rank 1."""
        e1 = (1,) + (0,) * (self.k - 1)
        return {(2 * t, 2 * t - 1): e1 for t in range(1, n // 2 + 1)}

    # ---------------- encodings and class operations ----------------

    def multiply(self, n: int, form: Dict, g, h):
        """Reference implementation of the group law given by the form, over
        R = Z/p^d (both quotient and center coordinates range over R)."""
        (b1, a1), (b2, a2) = g, h
        b = [(u + v) % self.q for u, v in zip(b1, b2)]
        for (j, i), label in form.items():
            c = a1[j - 1] * a2[i - 1]
            for l in range(self.k):
                b[l] = (b[l] + label[l] * c) % self.q
        return tuple(b), tuple((u + v) % self.q for u, v in zip(a1, a2))

    def simulate(self, advice: Sequence[str], gx, gy, gz) -> bool:
        """Run the multiplication automaton directly over the tapes: True iff
        the advice accepts gx * gy = gz. Mirrors the M transition without
        building the 4-tape product DFA, so the streamed compile can be checked
        against the reference law for any ring alphabet (the full automaton is
        only buildable for small q). gx, gy, gz are (b, a) elements."""
        n = self._n_of_advice(advice)
        ex, ey, ez = (self.encode(g, n) for g in (gx, gy, gz))
        state = self._m_initial()
        for i in range(len(advice)):
            state = self._m_step(state, advice[i], ex[i], ey[i], ez[i])
        return self._m_accepting(state)

    # ---------------- the multiplication transition (shared) ----------------

    def _m_initial(self):
        return ('c', 0, (0,) * self.k)

    def _m_accepting(self, state) -> bool:
        if state == 'dead' or state[0] != 'x' or state[1] != (0,) * self.k:
            return False
        return not self.factored or state[4] == self.n_entries

    def _m_step(self, state, a, x, y, z):
        """One step of the multiplication automaton over R = Z/p^d, shared by
        `simulate`, the automaton builder and the implicit M atom. `a` is the
        advice symbol, `x, y, z` the element symbols at this position. In
        factored mode the state additionally carries the accumulator of the
        streamed w <- T w + v*y update and the phase inside the entry
        stretch; the marker commits the accumulator."""
        k, r, q = self.k, self.r, self.q
        if state == 'dead':
            return 'dead'
        if not (x in self._digitset and y in self._digitset and z in self._digitset):
            return 'dead'
        xi, yi, zi = int(x), int(y), int(z)
        if a == CMARK and state[0] == 'c':
            j, d = state[1], state[2]
            d = d[:j] + ((zi - xi - yi) % q,) + d[j + 1:]
            if j + 1 < k:
                return ('c', j + 1, d)
            return (('x', d, (0,) * r, (0,) * r, self.n_entries)
                    if self.factored else ('x', d, (0,) * r))
        if state[0] != 'x':
            return 'dead'
        if not self.factored:
            if a in self.letters:
                if (xi + yi - zi) % q:
                    return 'dead'
                T, v, R = self.letters[a]
                d, w = state[1], state[2]
                d = tuple((d[l] - xi * sum(R[l][s] * w[s] for s in range(r)))
                          % q for l in range(k))
                w = tuple((sum(T[s][u] * w[u] for u in range(r)) + v[s] * yi)
                          % q for s in range(r))
                return ('x', d, w)
            return 'dead'
        if (xi + yi - zi) % q:
            return 'dead'
        d, w, acc, ph = state[1], state[2], state[3], state[4]
        if a == self.MARKER:
            # commit the streamed update; legal only on a complete stretch
            if ph != self.n_entries:
                return 'dead'
            return ('x', d, acc, (0,) * r, 0)
        val = self.entry_letters.get(a)
        if val is None or ph >= self.n_entries:
            return 'dead'
        if ph < r * r:                       # T entry: acc += T[i][j] * w[j]
            i, j = divmod(ph, r)
            acc = acc[:i] + ((acc[i] + val * w[j]) % q,) + acc[i + 1:]
        elif ph < r * r + r:                 # v entry: acc += v[i] * y
            i = ph - r * r
            acc = acc[:i] + ((acc[i] + val * yi) % q,) + acc[i + 1:]
        else:                                # R entry: d -= x * R[l][s] * w[s]
            l, s = divmod(ph - r * r - r, r)
            d = d[:l] + ((d[l] - xi * val * w[s]) % q,) + d[l + 1:]
        return ('x', d, w, acc, ph + 1)

    def identity(self, n: int):
        return (0,) * self.k, (0,) * n

    def encode(self, element, n: int) -> List[str]:
        """Encode (b, a) with b in R^k, a in R^n, R = Z/p^d. In factored
        mode each position's digit is repeated along its entry stretch."""
        b, a = element
        if len(b) != self.k or len(a) != n:
            raise ValueError(f"element must be (R^{self.k}, R^{n}), R = Z/{self.q}")
        if not all(0 <= x < self.q for x in tuple(b) + tuple(a)):
            raise ValueError(f"components must lie in Z/{self.q}")
        stretch = 1 + self.n_entries if self.factored else 1
        return [str(x) for x in b] + [str(x) for x in a for _ in range(stretch)]

    def _n_of_advice(self, advice: Sequence[str]) -> int:
        """The member size n presented by an advice word."""
        stretch = 1 + self.n_entries if self.factored else 1
        return (len(advice) - self.k) // stretch

    def decode(self, word: Sequence[str]):
        """Inverse of `encode`: an element word back to its (b, a) tuple."""
        stretch = 1 + self.n_entries if self.factored else 1
        b = tuple(int(x) for x in word[:self.k])
        rest = list(word[self.k:])
        a = tuple(int(rest[t * stretch]) for t in range(len(rest) // stretch))
        return b, a

    def evaluate(self, phi) -> Tuple[SparseDFA, List[str]]:
        return self.cls.evaluate(phi)

    def check(self, phi, advice: Sequence[str], **elements) -> bool:
        """Model check against the member presented by the advice; free
        variables can be assigned elements as (b, a) tuples."""
        n = self._n_of_advice(advice)
        words = {name: self.encode(el, n) for name, el in elements.items()}
        return self.cls.check(phi, advice, **words)

    def _implicit_atoms(self) -> Dict:
        """Functional base automata (Dom, Adv, M, Eq) built straight from the
        transition deltas -- no explicit product automaton, so this works for the
        large-q members whose `cls` cannot be built. Each entry is a builder
        ``args -> ImplicitDFA`` over the formula's argument tapes."""
        from autstr.implicit import ImplicitDFA
        digitset = self._digitset

        def shape(args, extra_ok, tracked=None):
            # advice-shape run shared by Dom/Adv/Eq: k center markers then
            # letters (with the element digit repeated along each stretch in
            # factored mode -- `tracked` names the tape whose digit repeats)
            adv = args[0]

            def step(state, s):
                if state == 'dead' or not extra_ok(s):
                    return 'dead'
                return self._shape_step(
                    state, s[adv], s[tracked] if tracked is not None else None)
            return ImplicitDFA(args, lambda: ('c', 0), step,
                               lambda st: st == 'ok')

        def Dom(args):
            xv = args[1]
            return shape(args, lambda s: s[xv] in digitset, tracked=args[1])

        def Adv(args):
            return shape(args, lambda s: True)

        def Eq(args):
            xv, yv = args[1], args[2]
            return shape(args, lambda s: s[xv] in digitset and s[xv] == s[yv],
                         tracked=args[1])

        def M(args):
            adv, xv, yv, zv = args
            return ImplicitDFA(
                args, self._m_initial,
                lambda state, s: self._m_step(state, s[adv], s[xv], s[yv], s[zv]),
                self._m_accepting)

        return {'Dom': Dom, 'Adv': Adv, 'M': M, 'Eq': Eq}

    @property
    def implicit_cls(self):
        """The fully implicit presentation of this class (functional atoms
        only, nothing compiled): an `autstr.implicit.ImplicitClass` over raw
        element words. `check_implicit`/`evaluate_implicit` add the
        (b, a)-tuple encoding on top of it."""
        from autstr.implicit import ImplicitClass
        return ImplicitClass(self._implicit_atoms(), list(self.digits))

    def check_implicit(self, phi, advice: Sequence[str], **elements) -> bool:
        """Like `check`, but evaluated implicitly (no query or base automaton) --
        the only viable model checker for the large-alphabet ring members whose
        `cls` cannot be built. See `autstr.implicit`."""
        n = self._n_of_advice(advice)
        words = {name: self.encode(el, n) for name, el in elements.items()}
        return self.implicit_cls.check(phi, advice, **words)

    def evaluate_implicit(self, phi, advice: Sequence[str], **elements):
        """The satisfying set of phi on the member presented by the advice,
        computed implicitly: unassigned free variables stay open and are
        solved for. Yields assignments {var: (b, a)}; `len` is the exact
        solution count without enumeration. Works for members whose automata
        cannot be built."""
        from autstr.implicit import MappedSolutions
        n = self._n_of_advice(advice)
        words = {name: self.encode(el, n) for name, el in elements.items()}
        sols = self.implicit_cls.evaluate(phi, advice, **words)
        return MappedSolutions(sols, self.decode)

    def get_structure(self, advice: Sequence[str]) -> AutomaticPresentation:
        return self.cls.get_structure(advice)


class InfiniteExtraspecialGroup:
    """The infinite extraspecial p-group: finitely supported vectors over
    F_p with a one-digit centre.

    An element is a triple (a, b, c) with a, b finitely supported sequences
    over F_p and c in F_p, multiplied by

        (a, b, c) * (a', b', c') = (a + a', b + b', c + c' + <a, b'>),

    the central extension of F_p^(w) + F_p^(w) by F_p along the standard
    symplectic form. It is non-abelian -- the commutator of (a, b, c) and
    (a', b', c') is <a, b'> - <a', b> -- and unlike `ExtraspecialGroups` it is
    a single infinite structure rather than a class of finite ones, so its
    elements have an advice-free encoding and can be written as constants.

    Encoding: the first letter carries the centre, and letter i + 1 carries
    the pair (a_i, b_i); trailing (0, 0) pairs are trimmed, so encodings are
    unique and equality is the diagonal.
    """

    PAD = '*'

    def __init__(self, p: int = 3):
        if not 2 <= p <= 9:
            raise ValueError("p must be a single digit prime power base")
        self.p = p
        self.centre_letters = [f'c{d}' for d in range(p)]
        self.pair_letters = [f'{i}{j}' for i in range(p) for j in range(p)]
        self.sigma = {self.PAD, *self.centre_letters, *self.pair_letters}
        self.presentation = AutomaticPresentation(
            {'U': self._universe(), 'M': self._multiplication(),
             'Eq': self._equality()},
            padding_symbol=self.PAD)

    # ---------------- encoding ----------------

    def encode(self, element) -> List[str]:
        """(a, b, c) -> word. `a` and `b` are sequences of digits."""
        a, b, c = element
        a, b = list(a), list(b)
        n = max(len(a), len(b))
        a += [0] * (n - len(a))
        b += [0] * (n - len(b))
        while a and a[-1] % self.p == 0 and b[-1] % self.p == 0:
            a.pop()
            b.pop()
        return [f'c{c % self.p}'] + [f'{a[i] % self.p}{b[i] % self.p}'
                                     for i in range(len(a))]

    def decode(self, word):
        letters = [s for s in word if s != self.PAD]
        centre = int(letters[0][1:])
        a = tuple(int(s[0]) for s in letters[1:])
        b = tuple(int(s[1]) for s in letters[1:])
        return a, b, centre

    # ---------------- automata ----------------

    def _digits(self, letter):
        """The (a, b) digits a letter contributes; padding contributes zeros."""
        return (0, 0) if letter == self.PAD else (int(letter[0]),
                                                 int(letter[1]))

    def _universe(self) -> SparseDFA:
        states = ['i', 'ok', 'z', 'd']

        def delta(q, sym):
            (letter,) = sym
            if q == 'i':
                return 'ok' if letter in self.centre_letters else 'd'
            if q in ('ok', 'z'):
                if letter == self.PAD:
                    return q                      # trailing padding only
                if letter in self.pair_letters:
                    return 'z' if letter == '00' else 'ok'
            return 'd'

        return dfa_from_delta(self.sigma, states, 1, delta, 'i', {'ok'})

    def _multiplication(self) -> SparseDFA:
        p = self.p
        states = ['i', 'd'] + [f'm{r}{s}' for r in range(p) for s in range(p)]

        def delta(q, sym):
            lx, ly, lz = sym
            if q == 'i':
                if not all(l in self.centre_letters for l in (lx, ly, lz)):
                    return 'd'
                cx, cy, cz = (int(l[1:]) for l in (lx, ly, lz))
                return f'm{(cz - cx - cy) % p}0'   # required carry, sum so far
            if q == 'd':
                return 'd'
            if any(l in self.centre_letters for l in (lx, ly, lz)):
                return 'd'
            required, running = int(q[1]), int(q[2])
            ax, bx = self._digits(lx)
            ay, by = self._digits(ly)
            az, bz = self._digits(lz)
            if az != (ax + ay) % p or bz != (bx + by) % p:
                return 'd'
            return f'm{required}{(running + ax * by) % p}'

        finals = {f'm{r}{r}' for r in range(p)}
        return dfa_from_delta(self.sigma, states, 3, delta, 'i', finals)

    def _equality(self) -> SparseDFA:
        """Encodings are unique, so equality is the diagonal."""
        def delta(q, sym):
            return 'ok' if q == 'ok' and sym[0] == sym[1] else 'd'

        return dfa_from_delta(self.sigma, ['ok', 'd'], 2, delta, 'ok', {'ok'})

    # ---------------- interface ----------------

    def multiply(self, x, y):
        """The reference product, for checking the automaton against."""
        (ax, bx, cx), (ay, by, cy) = x, y
        n = max(len(ax), len(bx), len(ay), len(by))
        pad = lambda v: list(v) + [0] * (n - len(v))
        ax, bx, ay, by = map(pad, (ax, bx, ay, by))
        pairing = sum(ax[i] * by[i] for i in range(n)) % self.p
        return (tuple((ax[i] + ay[i]) % self.p for i in range(n)),
                tuple((bx[i] + by[i]) % self.p for i in range(n)),
                (cx + cy + pairing) % self.p)

    def default_signature(self):
        from autstr.symbolic import FunctionCodec, operation_signature
        return operation_signature(
            self.presentation.get_relation_symbols(), graph='M', operator='*',
            codec=FunctionCodec(self.encode, self.decode))

    def symbolic(self, signature=None):
        """A symbolic interface with ``*`` bound to the group product."""
        return self.presentation.symbolic(signature or self.default_signature())
