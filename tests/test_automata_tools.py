"""Brute-force checks of the string pipeline against its defining semantics.

Each operation is compared, on random small automata, with the set of words it
is *defined* to accept — enumerated directly rather than derived from the
implementation.
"""
import itertools as it
import random

import numpy as np

from autstr.sparse_automata import SparseDFA
from autstr.utils.automata_tools import (
    expand, pad, permute_tapes, projection, unpad,
)


def random_dfa(rng: random.Random, num_states: int, m: int,
               arity: int) -> SparseDFA:
    num_symbols = m ** arity
    width = rng.randint(0, min(4, num_symbols))
    symbols = np.full((num_states, width), -1, dtype=np.int32)
    targets = np.full((num_states, width), -1, dtype=np.int32)
    for q in range(num_states):
        for j, s in enumerate(rng.sample(range(num_symbols), width)):
            symbols[q, j] = s
            targets[q, j] = rng.randrange(num_states)
    defaults = np.array([rng.randrange(num_states) for _ in range(num_states)],
                        dtype=np.int32)
    return SparseDFA(num_states, defaults, symbols, targets,
                     [rng.random() < 0.5 for _ in range(num_states)],
                     rng.randrange(num_states), arity, set(range(m)))


def run(dfa: SparseDFA, word) -> bool:
    state = dfa.start_state
    for symbol in word:
        state = dfa.transition(state, symbol)
    return bool(dfa.is_accepting[state])


def words(num_symbols: int, max_length: int):
    return it.chain.from_iterable(
        it.product(range(num_symbols), repeat=length)
        for length in range(max_length + 1))


TRIALS = 20
MAX_LENGTH = 3


class TestExpand:
    def test_places_the_tape_and_ignores_the_others(self):
        rng = random.Random(7)
        for trial in range(TRIALS):
            m = rng.randint(2, 3)
            dfa = random_dfa(rng, rng.randint(1, 4), m, 1)
            for position in (0, 1):
                wide = expand(dfa, 2, [position])
                for word in words(m * m, MAX_LENGTH):
                    tape = [s // m if position == 0 else s % m for s in word]
                    assert run(wide, word) == run(dfa, tape), (trial, position)

    def test_repeated_position_is_the_diagonal(self):
        rng = random.Random(8)
        for trial in range(TRIALS):
            m = rng.randint(2, 3)
            dfa = random_dfa(rng, rng.randint(1, 4), m, 2)
            diagonal = expand(dfa, 1, [0, 0])
            for word in words(m, MAX_LENGTH):
                assert run(diagonal, word) == run(dfa, [a * m + a for a in word]), \
                    trial


class TestProjection:
    def test_is_the_existential_over_the_dropped_tape(self):
        rng = random.Random(9)
        for trial in range(TRIALS):
            m = rng.randint(2, 3)
            dfa = random_dfa(rng, rng.randint(1, 4), m, 2)
            for tape in (0, 1):
                projected = projection(dfa, tape)
                for word in words(m, MAX_LENGTH):
                    witness = any(
                        run(dfa, [(a * m + b) if tape == 0 else (b * m + a)
                                  for a, b in zip(fill, word)])
                        for fill in it.product(range(m), repeat=len(word)))
                    assert run(projected, word) == witness, (trial, tape, word)


class TestPadding:
    def test_pad_accepts_the_language_followed_by_padding(self):
        rng = random.Random(10)
        for trial in range(TRIALS):
            m = rng.randint(2, 3)
            dfa = random_dfa(rng, rng.randint(1, 4), m, 1)
            padded = pad(dfa, 0)
            for word in words(m, MAX_LENGTH):
                expected = any(
                    run(dfa, word[:cut]) and all(s == 0 for s in word[cut:])
                    for cut in range(len(word) + 1))
                assert run(padded, word) == expected, (trial, word)

    def test_unpad_accepts_words_whose_padding_extension_is_accepted(self):
        rng = random.Random(11)
        for trial in range(TRIALS):
            m = rng.randint(2, 3)
            dfa = random_dfa(rng, rng.randint(1, 4), m, 1)
            trimmed = unpad(dfa, 0)
            for word in words(m, MAX_LENGTH):
                expected = any(run(dfa, list(word) + [0] * k)
                               for k in range(dfa.num_states + 2))
                assert run(trimmed, word) == expected, (trial, word)


class TestPermuteTapes:
    def test_swaps_the_digits(self):
        rng = random.Random(12)
        for trial in range(TRIALS):
            m = rng.randint(2, 3)
            dfa = random_dfa(rng, rng.randint(1, 4), m, 2)
            swapped = permute_tapes(dfa, [1, 0])
            for word in words(m * m, MAX_LENGTH):
                original = [(s % m) * m + s // m for s in word]
                assert run(swapped, word) == run(dfa, original), trial
