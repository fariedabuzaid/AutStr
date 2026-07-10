"""Closure operations on automatic presentations.

The building block is `recode`: re-expressing an automaton over a different
base alphabet. Every closure construction needs it -- a disjoint union
re-expresses both factors over the union plus its tags, a direct product
re-expresses each factor over the pair alphabet.
"""
import itertools as it
import random

import numpy as np

from autstr.sparse_automata import SparseDFA, recode
from autstr.utils.misc import encode_symbol
from test_automata_tools import random_dfa, words


def run(dfa, word):
    state = dfa.start_state
    for symbol in word:
        state = dfa.transition(state, symbol)
    return bool(dfa.is_accepting[state])


def encode(dfa, letters):
    return [encode_symbol(tuple(letter), dfa.base_alphabet_frozen)
            for letter in letters]


class TestRecode:
    def test_identity_on_the_same_alphabet(self):
        rng = random.Random(1)
        for _ in range(20):
            dfa = random_dfa(rng, rng.randint(1, 4), 3, 1)
            wider = recode(dfa, dfa.base_alphabet)
            for word in words(3, 4):
                assert run(wider, word) == run(dfa, word)

    def test_letters_outside_the_image_are_rejected(self):
        """The point of the fresh dead state: a widened automaton must reject
        the letters it was never defined on."""
        rng = random.Random(2)
        for trial in range(20):
            m = rng.randint(2, 3)
            dfa = random_dfa(rng, rng.randint(1, 4), m, 1)
            # old letters 0..m-1 are embedded as 0..m-1 of a wider alphabet
            wider = recode(dfa, set(range(m + 2)))
            assert wider.num_states == dfa.num_states + 1
            for word in words(m, 3):
                assert run(wider, word) == run(dfa, word), trial
            # any word touching a new letter must be rejected
            for word in it.product(range(m + 2), repeat=2):
                if max(word) >= m:
                    assert not run(wider, word), (trial, word)

    def test_permuting_the_letters(self):
        """A bijective letter_map relabels the language."""
        rng = random.Random(3)
        m = 3
        for trial in range(20):
            dfa = random_dfa(rng, rng.randint(1, 4), m, 1)
            perm = {0: 2, 1: 0, 2: 1}
            relabelled = recode(dfa, set(range(m)), perm)
            for word in words(m, 4):
                assert run(relabelled, [perm[a] for a in word]) == run(dfa, word), \
                    (trial, word)

    def test_multi_tape_recoding(self):
        """Every tape is recoded, so a k-tape automaton keeps its relation."""
        rng = random.Random(4)
        m = 2
        for trial in range(20):
            dfa = random_dfa(rng, rng.randint(1, 4), m, 2)
            wider = recode(dfa, set(range(m + 1)))
            for word in words(m * m, 3):
                digits = [(s // m, s % m) for s in word]
                widened = [a * (m + 1) + b for a, b in digits]
                assert run(wider, widened) == run(dfa, word), (trial, word)

    def test_rejects_a_non_injective_map(self):
        dfa = random_dfa(random.Random(5), 2, 3, 1)
        import pytest
        with pytest.raises(ValueError):
            recode(dfa, {0, 1, 2}, {0: 0, 1: 0, 2: 1})
