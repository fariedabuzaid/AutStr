import random

import numpy as np
import pytest

from autstr.buildin.presentations import BuechiArithmeticZ
from autstr.sparse_automata import SparseDFA
from autstr.sparse_tree_automata import Tree
from autstr.tree_presentations import TreeAutomaticPresentation
from autstr.utils.tree_automata_tools import from_string_dfa, string_chain
from test_tree_automata import random_sta  # noqa: F401 (import path check)


# ====================================================================
# The embedding: chains of a string language
# ====================================================================

def random_sdfa(rng, max_states=5, max_symbols=4, max_exc=8) -> SparseDFA:
    n = rng.randint(1, max_states)
    m = rng.randint(2, max_symbols)
    ex_syms = np.full((n, 2), -1, dtype=np.int32)
    ex_states = np.full((n, 2), -1, dtype=np.int32)
    for q in range(n):
        syms = rng.sample(range(m), 2)
        for j, s in enumerate(syms):
            if rng.random() < 0.7:
                ex_syms[q, j] = s
                ex_states[q, j] = rng.randrange(n)
    return SparseDFA(
        n, np.array([rng.randrange(n) for _ in range(n)], dtype=np.int32),
        ex_syms, ex_states,
        np.array([rng.random() < 0.5 for _ in range(n)]),
        rng.randrange(n), 1, set(range(m)))


class TestStringEmbedding:
    def test_chain_acceptance_matches_dfa(self):
        rng = random.Random(0)
        for trial in range(30):
            dfa = random_sdfa(rng)
            tree = from_string_dfa(dfa)
            m = len(dfa.base_alphabet)
            for _ in range(30):
                word = [rng.randrange(m) for _ in range(rng.randint(1, 10))]
                want = bool(dfa.is_accepting[dfa.compute(word)])
                got = tree.accepts(string_chain(word))
                assert got == want, (trial, word)

    def test_non_chain_trees_rejected(self):
        rng = random.Random(1)
        dfa = random_sdfa(rng)
        tree = from_string_dfa(dfa)
        # a tree with a right child is never a chain
        bushy = Tree(0, Tree(0), Tree(0))
        assert not tree.accepts(bushy)


# ====================================================================
# Cross-validation: Büchi arithmetic through both engines
# ====================================================================

def encode_int(n: int):
    """The library's integer encoding: sign letter, then LSB-first bits."""
    return [str(int(n < 0))] + list(format(abs(n), 'b')[::-1])


def convolve_words(words, pad='*'):
    """Padded positionwise convolution (the string convention)."""
    length = max(len(w) for w in words)
    padded = [list(w) + [pad] * (length - len(w)) for w in words]
    return [tuple(w[i] for w in padded) for i in range(length)]


@pytest.fixture(scope="module")
def engines():
    string_pres = BuechiArithmeticZ()
    tree_automata = {
        'U': from_string_dfa(string_pres.automata['U']),
        'A': from_string_dfa(string_pres.automata['A']),
        'Lt': from_string_dfa(string_pres.automata['Lt']),
    }
    tree_pres = TreeAutomaticPresentation(
        tree_automata, padding_symbol='*', enforce_consistency=False,
        max_states=200_000)
    return string_pres, tree_pres


SENTENCES = [
    ('all x.(exists y.(Lt(x,y)))', True),          # no greatest integer
    ('exists x.(all y.(Lt(x,y)))', False),         # no least integer (Lt irrefl.)
    ('exists x.(all y.(A(x,y,y)))', True),         # a neutral element exists
    ('all x.(all y.(exists z.(A(x,z,y))))', True), # subtraction is total
    ('all x.(exists y.(Lt(y,x) and Lt(x,y)))', False),
    ('exists x.(exists y.(Lt(x,y) and Lt(y,x)))', False),
]


class TestBuechiCrossValidation:
    def test_sentences_agree(self, engines):
        string_pres, tree_pres = engines
        for phi, expected in SENTENCES:
            s = string_pres.check(phi)
            t = tree_pres.check(phi)
            assert s == expected, f"string engine wrong on {phi}"
            assert t == expected, f"tree engine wrong on {phi}"

    def test_binary_relation_memberships_agree(self, engines):
        """Evaluate the same definable relation in both engines and compare
        memberships pointwise: 'some z lies strictly between x and y'."""
        string_pres, tree_pres = engines
        phi = 'exists z.(Lt(x,z) and Lt(z,y))'
        s_rel = string_pres.evaluate(phi)
        t_rel = tree_pres.evaluate(phi)

        for x in range(-4, 5):
            for y in range(-4, 5):
                word = convolve_words([encode_int(x), encode_int(y)])
                want = y - x >= 2
                s_got = bool(s_rel.is_accepting[s_rel.compute(
                    [s_rel.encode_symbol(c) for c in word])])
                chain = string_chain(word)
                t_got = t_rel.accepts(chain)
                assert s_got == want, (x, y)
                assert t_got == want, (x, y)

    def test_addition_memberships_agree(self, engines):
        string_pres, tree_pres = engines
        t_rel = tree_pres.evaluate('A(x,y,z)')
        s_rel = string_pres.evaluate('A(x,y,z)')
        rng = random.Random(2)
        for _ in range(40):
            x, y = rng.randint(-6, 6), rng.randint(-6, 6)
            z = x + y if rng.random() < 0.6 else rng.randint(-13, 13)
            word = convolve_words([encode_int(x), encode_int(y),
                                   encode_int(z)])
            want = (x + y == z)
            s_got = bool(s_rel.is_accepting[s_rel.compute(
                [s_rel.encode_symbol(c) for c in word])])
            t_got = t_rel.accepts(string_chain(word))
            assert s_got == want, (x, y, z)
            assert t_got == want, (x, y, z)
