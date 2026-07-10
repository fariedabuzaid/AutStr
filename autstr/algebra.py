"""Finite boolean algebras and the localizations Z[1/p] as automatic
structures.

**Finite boolean algebras** (`FiniteBooleanAlgebras`): the algebra with n
atoms is (up to isomorphism) the powerset algebra of [n]. The advice is
simply the unary word 1^n; elements are subsets of [n] as bitvectors of
length n, and all operations are positionwise. Signature:

    Leq(x,y)      x <= y            Meet(x,y,z)   z = x AND y
    Join(x,y,z)   z = x OR y        Compl(x,y)    y = NOT x
    Atom(x)       x is an atom

**The localizations Z[1/p]** (`Z1pLocalization`): a single automatic structure,
not a class. An element is written with a sign and a radix-aligned pair of
integer and fractional digits, and addition is recognized by a Buechi-style
carry automaton.

The finite abelian groups live in `autstr.groups` beside the other group
classes.
"""
import itertools as it
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union

from nltk.sem import logic

from autstr.presentations import AutomaticPresentation
from autstr.sparse_automata import SparseDFA
from autstr.uniform import UniformlyAutomaticClass, dfa_from_delta
from autstr.utils.logic import get_free_elementary_vars

PAD = '*'


def _is_prime(n: int) -> bool:
    """Return True iff n is a prime number >= 2."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


@dataclass(frozen=True)
class Z1pElement:
    """Canonical element of Z[1/p] represented as num / p**exp.

    In canonical form, either num == 0 and exp == 0, or p does not divide num.
    """
    num: int
    exp: int


class Z1pLocalization:
    """Factory-backed arithmetic model for the fixed localization Z[1/p].

    This class provides a canonical representation and exact arithmetic for a
    fixed prime p. It is designed as the API layer that a dedicated automatic
    presentation can be attached to.
    """

    def __init__(self, p: int):
        if not _is_prime(p):
            raise ValueError(f"p must be prime, got {p}")
        self.p = int(p)
        self._presentation = None

    def normalize(self, num: int, exp: int) -> Z1pElement:
        """Return the canonical representative of num / p**exp."""
        if exp < 0:
            raise ValueError("exponent must be >= 0")
        num = int(num)
        exp = int(exp)
        if num == 0:
            return Z1pElement(0, 0)
        while exp > 0 and num % self.p == 0:
            num //= self.p
            exp -= 1
        return Z1pElement(num, exp)

    def element(self, num: int, exp: int = 0) -> Z1pElement:
        """Create a canonical element from num / p**exp."""
        return self.normalize(num, exp)

    def from_fraction(self, num: int, den: int) -> Z1pElement:
        """Create an element from a reduced or unreduced fraction num/den.

        The denominator must be a positive power of p.
        """
        if den <= 0:
            raise ValueError("denominator must be > 0")
        exp = 0
        d = int(den)
        while d % self.p == 0:
            d //= self.p
            exp += 1
        if d != 1:
            raise ValueError(f"denominator {den} is not a power of {self.p}")
        return self.normalize(int(num), exp)

    def add(self, x: Z1pElement, y: Z1pElement) -> Z1pElement:
        """Return x + y in canonical form."""
        m = max(x.exp, y.exp)
        nx = x.num * (self.p ** (m - x.exp))
        ny = y.num * (self.p ** (m - y.exp))
        return self.normalize(nx + ny, m)

    def neg(self, x: Z1pElement) -> Z1pElement:
        """Return -x in canonical form."""
        return self.normalize(-x.num, x.exp)

    def sub(self, x: Z1pElement, y: Z1pElement) -> Z1pElement:
        """Return x - y in canonical form."""
        return self.add(x, self.neg(y))

    def equals(self, x: Z1pElement, y: Z1pElement) -> bool:
        """Semantic equality in Z[1/p] (canonical reps compare directly)."""
        return x == y

    # ---------------- automatic presentation ----------------
    #
    # An element x = ±(A + C) with integer part A and fractional part
    # C in [0,1) (finite base-p expansion) is encoded as
    #
    #     [sign]  (a_1,c_1) (a_2,c_2) ... (a_L,c_L)
    #
    # where a_i is the digit of A at p^(i-1) and c_i the digit of C at
    # p^(-i): both expansions grow rightward from the radix point, so
    # convolution padding stays trailing. The last pair is nonzero (unique
    # encodings); zero is the bare sign '+'.
    #
    # Magnitude addition A0 is regular: the integer track adds LSB-first
    # (carry runs forward); the fractional track adds MSB-first, so the
    # overflow g in {0,1} of C_x + C_y is guessed at the radix point and
    # both guesses are tracked deterministically — g is simultaneously the
    # carry into the integer units digit. Full signed addition A is then
    # defined by first-order case analysis over the signs (Büchi-Z style),
    # and Z / Eq are defined from A.

    def _pair_symbol(self, d: int, e: int) -> str:
        return f"{d}_{e}"

    @property
    def sigma(self) -> set:
        pairs = {self._pair_symbol(d, e)
                 for d in range(self.p) for e in range(self.p)}
        return {PAD, '+', '-'} | pairs

    def _pair_digits(self):
        return {self._pair_symbol(d, e): (d, e)
                for d in range(self.p) for e in range(self.p)}

    def _universe_automaton(self) -> SparseDFA:
        """U(x): a sign followed by digit pairs whose last pair is nonzero;
        zero is the bare '+' (no '-0')."""
        pair_of = self._pair_digits()
        states = ['start', 'zero_plus', 'minus', 'nz', 'zz', 'dead']

        def delta(q, sym):
            a = sym[0]
            if q == 'dead':
                return 'dead'
            if q == 'start':
                return {'+': 'zero_plus', '-': 'minus'}.get(a, 'dead')
            pair = pair_of.get(a)
            if pair is None:
                return 'dead'
            return 'zz' if pair == (0, 0) else 'nz'

        return dfa_from_delta(self.sigma, states, 1, delta, 'start', {'zero_plus', 'nz'})

    def _sign_automaton(self) -> SparseDFA:
        """N0(x): x >= 0 (sign '+'; zero is nonnegative)."""
        pair_of = self._pair_digits()
        states = ['start', 'yes', 'no', 'dead']

        def delta(q, sym):
            a = sym[0]
            if q == 'start':
                return {'+': 'yes', '-': 'no'}.get(a, 'dead')
            if q in ('yes', 'no') and a in pair_of:
                return q
            return 'dead'

        return dfa_from_delta(self.sigma, states, 1, delta, 'start', {'yes'})

    def _magnitude_addition_automaton(self) -> SparseDFA:
        """A0(x, y, z): |x| + |y| = |z| (signs are ignored). Branch g guesses
        the fractional overflow; each branch tracks (integer carry, expected
        fractional carry) and must close at (0, 0)."""
        p = self.p
        pair_of = self._pair_digits()
        branch_states = [(k, g) for k in (0, 1) for g in (0, 1)] + ['F']
        states = ['start', 'dead'] + [('c', b0, b1)
                                      for b0 in branch_states for b1 in branch_states]

        def digits(component):
            if component == PAD:
                return (0, 0)  # this tape already ended
            return pair_of.get(component)

        def step(branch, dx, dy, dz):
            if branch == 'F':
                return 'F'
            kappa, gamma = branch
            (ax, cx), (ay, cy), (az, cz) = dx, dy, dz
            # integer track, LSB-first: a_x + a_y + kappa = a_z + p*kappa'
            kappa_next, remainder = divmod(ax + ay + kappa - az, p)
            if remainder != 0 or not 0 <= kappa_next <= 1:
                return 'F'
            # fractional track, MSB-first: c_x + c_y + gamma' = c_z + p*gamma
            gamma_next = cz + p * gamma - cx - cy
            if not 0 <= gamma_next <= 1:
                return 'F'
            return (kappa_next, gamma_next)

        def delta(q, sym):
            if q == 'dead':
                return 'dead'
            if q == 'start':
                if all(s in ('+', '-') for s in sym):
                    # branch g: fractional overflow g is also the carry into
                    # the integer units digit
                    return ('c', (0, 0), (1, 1))
                return 'dead'
            dx, dy, dz = (digits(s) for s in sym)
            if None in (dx, dy, dz):
                return 'dead'
            b0, b1 = step(q[1], dx, dy, dz), step(q[2], dx, dy, dz)
            return ('c', b0, b1) if (b0, b1) != ('F', 'F') else 'dead'

        finals = {('c', b0, b1) for b0 in branch_states for b1 in branch_states
                  if (0, 0) in (b0, b1)}
        return dfa_from_delta(self.sigma, states, 3, delta, 'start', finals)

    @property
    def presentation(self) -> AutomaticPresentation:
        """The automatic presentation of (Z[1/p], +), built on first use.
        Signature: A(x,y,z) [z = x + y], N0(x) [x >= 0], Z(x) [x = 0],
        Eq(x,y)."""
        if self._presentation is None:
            pres = AutomaticPresentation({
                'U': self._universe_automaton(),
                'A0': self._magnitude_addition_automaton(),
                'N0': self._sign_automaton(),
            }, padding_symbol=PAD)
            pres.update(Z='A0(x,x,x)')
            # Sign case analysis; the (+,+,-) and (-,-,+) cases are impossible
            cases = [
                '(N0(x) and N0(y) and N0(z) and A0(x,y,z))',
                '(N0(x) and (not N0(y)) and N0(z) and A0(z,y,x))',
                '(N0(x) and (not N0(y)) and (not N0(z)) and A0(z,x,y))',
                '((not N0(x)) and N0(y) and N0(z) and A0(x,z,y))',
                '((not N0(x)) and N0(y) and (not N0(z)) and A0(z,y,x))',
                '((not N0(x)) and (not N0(y)) and (not N0(z)) and A0(x,y,z))',
            ]
            pres.update(A=' or '.join(cases))
            pres.update(Eq='exists z.(Z(z) and A(x,z,y))')
            del pres.automata['A0']
            self._presentation = pres
        return self._presentation

    def _coerce(self, value) -> Z1pElement:
        if isinstance(value, Z1pElement):
            return self.normalize(value.num, value.exp)
        if isinstance(value, int):
            return self.normalize(value, 0)
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return self.normalize(*value)
        raise ValueError(f"cannot interpret {value!r} as an element of Z[1/{self.p}]")

    def encode(self, value) -> List[str]:
        """Encode an element as a word of the presentation."""
        x = self._coerce(value)
        sign = '+' if x.num >= 0 else '-'
        magnitude, k = abs(x.num), x.exp
        integer_part, frac = divmod(magnitude, self.p ** k)

        int_digits = []
        while integer_part:
            integer_part, d = divmod(integer_part, self.p)
            int_digits.append(d)
        frac_digits = [(frac // self.p ** (k - i)) % self.p for i in range(1, k + 1)]

        length = max(len(int_digits), k)
        pairs = [
            self._pair_symbol(
                int_digits[i] if i < len(int_digits) else 0,
                frac_digits[i] if i < k else 0,
            )
            for i in range(length)
        ]
        return [sign] + pairs

    def evaluate(self, phi) -> SparseDFA:
        """Evaluate a first-order query; see AutomaticPresentation.evaluate.
        The result's tapes are the free variables in sorted order."""
        return self.presentation.evaluate(phi)

    def check(self, phi, **elements) -> bool:
        """Model check a formula against (Z[1/p], +). Free variables can be
        assigned elements (Z1pElement, int, or (num, exp) pairs); unassigned
        free variables are existentially quantified."""
        if not elements:
            return self.presentation.check(phi)
        if isinstance(phi, str):
            phi = logic.Expression.fromstring(phi)
        free_vars = get_free_elementary_vars(phi)
        unknown = set(elements) - set(free_vars)
        if unknown:
            raise ValueError(f"assignments for non-free variables: {unknown}")
        for v in free_vars:
            if v not in elements:
                phi = logic.ExistsExpression(logic.Variable(v), phi)

        dfa = self.presentation.evaluate(phi)
        order = sorted(elements)
        words = {name: self.encode(value) for name, value in elements.items()}
        length = max(len(w) for w in words.values())
        for w in words.values():
            w += [PAD] * (length - len(w))
        return dfa.accepts([
            tuple(words[name][i] for name in order) for i in range(length)
        ])


def z1p_localization(p: int) -> Z1pLocalization:
    """Return a fixed-prime localization factory for Z[1/p]."""
    return Z1pLocalization(p)


# ====================================================================
# Finite boolean algebras
# ====================================================================

class FiniteBooleanAlgebras:
    """The uniformly automatic class of all finite boolean algebras. The
    member with n atoms has advice 1^n; its elements are the subsets of
    {0, ..., n-1}, encoded as bitvectors of length exactly n."""

    def __init__(self):
        self.sigma = {PAD, '0', '1'}
        self.cls = UniformlyAutomaticClass({
            'U': self._universe_automaton(),
            'Leq': self._pointwise_automaton(2, lambda x, y: x <= y),
            'Meet': self._pointwise_automaton(3, lambda x, y, z: z == (x & y)),
            'Join': self._pointwise_automaton(3, lambda x, y, z: z == (x | y)),
            'Compl': self._pointwise_automaton(2, lambda x, y: y == 1 - x),
            'Atom': self._atom_automaton(),
        }, padding_symbol=PAD)

    def _universe_automaton(self) -> SparseDFA:
        """U(p, x): p in 1* and x a {0,1}-word of exactly the same length
        (unique encodings: no early padding)."""
        states = ['ok', 'pad', 'dead']

        def delta(q, sym):
            a, x = sym
            if q == 'dead':
                return 'dead'
            if q == 'pad':
                return 'pad' if (a, x) == (PAD, PAD) else 'dead'
            if a == '1' and x in ('0', '1'):
                return 'ok'
            if (a, x) == (PAD, PAD):
                return 'pad'
            return 'dead'

        return dfa_from_delta(self.sigma, states, 2, delta, 'ok', {'ok', 'pad'})

    def _pointwise_automaton(self, arity: int, relation) -> SparseDFA:
        """A relation that holds iff `relation` holds on the bits of every
        position (advice digit must be 1)."""
        states = ['ok', 'pad', 'dead']

        def delta(q, sym):
            a, bits = sym[0], sym[1:]
            if q == 'dead':
                return 'dead'
            if q == 'pad':
                return 'pad' if all(s == PAD for s in sym) else 'dead'
            if a == '1' and all(b in ('0', '1') for b in bits):
                return 'ok' if relation(*(int(b) for b in bits)) else 'dead'
            if all(s == PAD for s in sym):
                return 'pad'
            return 'dead'

        return dfa_from_delta(self.sigma, states, 1 + arity, delta, 'ok', {'ok', 'pad'})

    def _atom_automaton(self) -> SparseDFA:
        """Atom(p, x): x contains exactly one 1."""
        states = ['zero', 'one', 'pad', 'dead']

        def delta(q, sym):
            a, x = sym
            if q == 'dead':
                return 'dead'
            if q == 'pad':
                return 'pad' if (a, x) == (PAD, PAD) else 'dead'
            if a == '1' and x == '0':
                return q
            if a == '1' and x == '1':
                return 'one' if q == 'zero' else 'dead'
            if (a, x) == (PAD, PAD) and q == 'one':
                return 'pad'
            return 'dead'

        return dfa_from_delta(self.sigma, states, 2, delta, 'zero', {'one', 'pad'})

    # ---------------- encodings and class operations ----------------

    def advice(self, n: int) -> List[str]:
        """Advice string of the boolean algebra with n atoms."""
        return ['1'] * n

    def encode(self, subset, n: int) -> List[str]:
        """Encode a subset of {0, ..., n-1} as an element word."""
        subset = set(subset)
        if not subset <= set(range(n)):
            raise ValueError(f"not a subset of range({n}): {subset}")
        return ['1' if i in subset else '0' for i in range(n)]

    def evaluate(self, phi) -> Tuple[SparseDFA, List[str]]:
        return self.cls.evaluate(phi)

    def check(self, phi, n: int, **subsets) -> bool:
        """Model check against the algebra with n atoms; free variables can
        be assigned subsets of {0, ..., n-1}."""
        words = {name: self.encode(s, n) for name, s in subsets.items()}
        return self.cls.check(phi, self.advice(n), **words)

    def get_structure(self, n: int) -> AutomaticPresentation:
        return self.cls.get_structure(self.advice(n))


# ====================================================================
# Finite abelian groups
# ====================================================================
