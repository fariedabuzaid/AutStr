import random
from math import gcd

import pytest

from autstr.buildin.tree_presentations import SkolemArithmetic, _prime
from autstr.sparse_tree_automata import Tree

enc = SkolemArithmetic.encode


@pytest.fixture(scope="module")
def sk():
    return SkolemArithmetic(max_states=500_000)


class TestEncoding:
    def test_prime_helper(self):
        assert [_prime(i) for i in range(10)] == \
            [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

    def test_roundtrip(self):
        for n in range(1, 301):
            assert SkolemArithmetic.decode(enc(n)) == n

    def test_universe_accepts_encodings(self, sk):
        U = sk.automata['U']
        for n in range(1, 150):
            assert U.accepts(enc(n)), n

    def test_universe_rejects_malformed(self, sk):
        U = sk.automata['U']
        bad = [
            Tree('0'),                                    # bare bit
            Tree('1'),
            Tree('p', Tree('p'), None),                   # trailing e_i = 0
            Tree('p', None, Tree('0')),                   # exponent chain "0"
            Tree('p', None, Tree('1', Tree('0'), None)),  # leading zero (MSB 0)
            Tree('p', None, Tree('1', None, Tree('1'))),  # bit with right child
            Tree('p', None, Tree('p')),                   # 'p' inside a chain
            Tree('1', Tree('p'), None),                   # bit above spine
        ]
        for t in bad:
            assert not U.accepts(t), t


class TestMultiplication:
    def test_products_pointwise(self, sk):
        M = sk.evaluate('M(x,y,z)')
        rng = random.Random(0)
        for _ in range(150):
            x, y = rng.randint(1, 100), rng.randint(1, 100)
            z = x * y if rng.random() < 0.5 else rng.randint(1, 10000)
            assert M.accepts(enc(x), enc(y), enc(z)) == (x * y == z), (x, y, z)

    def test_divisibility_definable(self, sk):
        D = sk.evaluate('exists z.(M(x,z,y))')
        for x in range(1, 25):
            for y in range(1, 25):
                assert D.accepts(enc(x), enc(y)) == (y % x == 0), (x, y)

    def test_equality_relation(self, sk):
        E = sk.evaluate('E(x,y)')
        for x in range(1, 20):
            for y in range(1, 20):
                assert E.accepts(enc(x), enc(y)) == (x == y), (x, y)

    def test_coprimality_definable(self, sk):
        """gcd(x, y) = 1 iff every common divisor is 1 (note 1 = the unique
        idempotent, M(d,d,d))."""
        C = sk.evaluate(
            'all d.((exists a.(M(d,a,x)) and exists b.(M(d,b,y)))'
            ' -> M(d,d,d))')
        for x in range(1, 15):
            for y in range(1, 15):
                assert C.accepts(enc(x), enc(y)) == (gcd(x, y) == 1), (x, y)


SENTENCES = [
    ('all x.(all y.(exists z.(M(x,y,z))))', True),        # totality
    # nb: 'e' would be an nltk *event* variable, hence 'u' for the unit
    ('exists u.(all x.(M(x,u,x)))', True),                 # neutral element
    ('exists x.(M(x,x,x))', True),                         # an idempotent (1)
    ('all x.(exists y.(M(y,y,x)))', False),                # 2 is not a square
    ('all x.(exists y.(M(x,y,x) and not M(y,y,y)))', False),  # x·y=x forces y=1
    ('all x.(all y.(all z.(M(x,y,z) -> M(y,x,z))))', True),   # commutativity
    ('all z.(all w.((exists x.(exists y.(M(x,y,z) and M(x,y,w)))'
     ' and E(z,z)) or E(z,z)))', True),                    # sanity: tautology
    ('exists x.((not M(x,x,x)) and all d.(all q.'
     '(M(d,q,x) -> (M(d,d,d) or E(d,x)))))', True),        # a prime exists
]


class TestSentences:
    @pytest.mark.parametrize("phi,expected", SENTENCES)
    def test_sentence(self, sk, phi, expected):
        assert sk.check(phi) == expected, phi

    def test_functionality(self, sk):
        """x·y is single-valued (4 tapes: the heaviest sentence here)."""
        phi = ('all x.(all y.(all z.(all w.'
               '((M(x,y,z) and M(x,y,w)) -> E(z,w)))))')
        assert sk.check(phi)
