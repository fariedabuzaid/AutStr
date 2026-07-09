"""Built-in tree-automatic presentations.

The flagship example is Skolem arithmetic, the multiplicative monoid
(N_{>0}, ·): multiplication of naturals is not string-automatic, but the
prime-factorization encoding makes it tree-automatic — the multiplicative
structure is the direct sum of countably many copies of (N, +), one per
prime, and a tree bundles the finitely many nonzero summands.
"""
from itertools import product as iter_product
from typing import List, Optional

import numpy as np

from autstr.sparse_automata import SparseDFA
from autstr.sparse_tree_automata import SparseTreeAutomaton, Tree
from autstr.tree_presentations import TreeAutomaticPresentation
from autstr.utils.misc import encode_symbol
from autstr.utils.tree_automata_tools import from_string_dfa


_PRIMES: List[int] = [2, 3]


def _prime(i: int) -> int:
    """The i-th prime (0-indexed), extending a shared trial-division cache."""
    while len(_PRIMES) <= i:
        c = _PRIMES[-1] + 2
        while not all(c % p for p in _PRIMES if p * p <= c):
            c += 2
        _PRIMES.append(c)
    return _PRIMES[i]


class SkolemArithmetic(TreeAutomaticPresentation):
    """(N_{>0}, ·, =) presented by tree automata.

    Encoding: n = prod_i p_i^{e_i} is a left spine of 'p'-labelled nodes,
    one per prime index up to the largest with e_i > 0 (so no trailing
    zero exponents); spine node i carries e_i as a right-hanging chain of
    bits with the least significant bit at the top and the most significant
    (always '1') at the bottom; e_i = 0 is an absent chain. n = 1 is the
    single node 'p'.

    Relations: M(x, y, z) iff x·y = z, and equality E. Multiplication is
    positionwise addition of the exponent vectors, so its automaton runs a
    binary addition automaton independently on every exponent branch of the
    3-tape convolution; a single further state rides down the spine and
    checks that every branch accepted.
    """

    LETTERS = frozenset({'*', '0', '1', 'p'})
    PAD = '*'

    def __init__(self, max_states: Optional[int] = None):
        super().__init__(
            {'U': self._universe(), 'M': self._multiplication(),
             'E': self._equality()},
            padding_symbol=self.PAD, max_states=max_states)

    # ---------------- encoding elements ----------------

    @classmethod
    def encode(cls, n: int) -> Tree:
        if n < 1:
            raise ValueError("Skolem arithmetic encodes positive integers")
        if n == 1:
            return Tree('p')
        exponents = []
        i = 0
        while n > 1:
            p, e = _prime(i), 0
            while n % p == 0:
                n //= p
                e += 1
            exponents.append(e)
            i += 1
        spine = None
        for e in reversed(exponents):
            chain = None                    # e = 0: no chain
            if e:
                for bit in format(e, 'b'):  # MSB placed deepest, LSB at top
                    chain = Tree(bit, chain, None)
            spine = Tree('p', spine, chain)
        return spine

    @classmethod
    def decode(cls, tree: Tree) -> int:
        n, i, node = 1, 0, tree
        while node is not None:
            if node.label != 'p':
                raise ValueError("not a Skolem arithmetic element tree")
            e, power, c = 0, 1, node.right
            while c is not None:
                e += power * int(c.label)
                power *= 2
                c = c.left
            n *= _prime(i) ** e
            node, i = node.left, i + 1
        return n

    # ---------------- the automata ----------------

    @classmethod
    def _enc(cls, letters) -> int:
        return encode_symbol(tuple(letters), cls.LETTERS)

    @classmethod
    def _universe(cls) -> SparseTreeAutomaton:
        CH, ONE, SP, DEAD = range(4)            # chain, the tree "1", spine
        BOT = 4
        exc = [
            (BOT, BOT, ('1',), CH),             # deepest bit is the MSB '1'
            (CH, BOT, ('0',), CH),
            (CH, BOT, ('1',), CH),
            (BOT, BOT, ('p',), ONE),            # n = 1
            (BOT, CH, ('p',), SP),              # deepest spine node: e_i > 0
            (SP, BOT, ('p',), SP),
            (SP, CH, ('p',), SP),
        ]
        return SparseTreeAutomaton(
            4, DEAD,
            [e[0] for e in exc], [e[1] for e in exc],
            [cls._enc(e[2]) for e in exc], [e[3] for e in exc],
            [False, True, True, False], 1, set(cls.LETTERS))

    @classmethod
    def _addition_dfa(cls) -> SparseDFA:
        """String DFA over letter triples, LSB first: x + y = z with '*'
        read as 0 (a tape past its end contributes nothing)."""
        C0, C1, SINK = range(3)
        val = {'*': 0, '0': 0, '1': 1}
        rows = [[], [], []]                     # (symbol, target) per state
        for carry in (C0, C1):
            for a, b, c in iter_product(('*', '0', '1'), repeat=3):
                s = val[a] + val[b] + carry
                if s % 2 == val[c]:
                    rows[carry].append((cls._enc((a, b, c)), s // 2))
        width = max(len(r) for r in rows)
        ex_syms = np.full((3, width), -1, dtype=np.int32)
        ex_states = np.full((3, width), -1, dtype=np.int32)
        for q, row in enumerate(rows):
            for j, (s, t) in enumerate(row):
                ex_syms[q, j], ex_states[q, j] = s, t
        return SparseDFA(3, np.full(3, SINK, dtype=np.int32),
                         ex_syms, ex_states, [True, False, False], C0,
                         symbol_arity=3, base_alphabet=set(cls.LETTERS))

    @classmethod
    def _multiplication(cls) -> SparseTreeAutomaton:
        chain = from_string_dfa(cls._addition_dfa())
        n = chain.num_states
        S_OK, BOT = n, n + 1
        left = np.where(chain.exc_left == n, BOT, chain.exc_left)
        right = np.where(chain.exc_right == n, BOT, chain.exc_right)
        exc = list(zip(left.tolist(), right.tolist(),
                       chain.exc_symbol.tolist(), chain.exc_target.tolist()))

        good_chain = np.flatnonzero(chain.is_accepting).tolist()
        spine_syms = [cls._enc(t) for t in iter_product(('p', '*'), repeat=3)
                      if t != ('*', '*', '*')]
        for s in spine_syms:
            for lc in (BOT, S_OK):              # deeper spine or none
                for rc in [BOT] + good_chain:   # branch adds up (or empty)
                    exc.append((lc, rc, s, S_OK))

        return SparseTreeAutomaton(
            n + 1, chain.default_state,
            [e[0] for e in exc], [e[1] for e in exc],
            [e[2] for e in exc], [e[3] for e in exc],
            [False] * n + [True], 3, set(cls.LETTERS))

    @classmethod
    def _equality(cls) -> SparseTreeAutomaton:
        OK, DEAD, BOT = 0, 1, 2
        exc = [(lc, rc, cls._enc((a, a)), OK)
               for a in ('p', '0', '1')
               for lc in (OK, BOT) for rc in (OK, BOT)]
        return SparseTreeAutomaton(
            2, DEAD,
            [e[0] for e in exc], [e[1] for e in exc],
            [e[2] for e in exc], [e[3] for e in exc],
            [True, False], 2, set(cls.LETTERS))


def skolem_arithmetic(max_states: Optional[int] = None) -> SkolemArithmetic:
    return SkolemArithmetic(max_states=max_states)
