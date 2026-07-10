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


# ====================================================================
# Disjoint union of two automatic structures
# ====================================================================

import pytest

from autstr.closure import disjoint_union, prefix
from autstr.presentations import AutomaticPresentation


def chain(letters, name='Lt'):
    """A finite linear order: one element per letter, in the given order."""
    alphabet = {'*'} | set(letters)
    n = len(letters)

    # U: a single letter from `letters`
    sym = lambda t: encode_symbol(t, frozenset(alphabet))
    ex_s = np.full((3, n), -1, dtype=np.int32)
    ex_t = np.full((3, n), -1, dtype=np.int32)
    for j, a in enumerate(letters):
        ex_s[0, j], ex_t[0, j] = sym((a,)), 1
    universe = SparseDFA(3, np.array([2, 2, 2], dtype=np.int32), ex_s, ex_t,
                         [False, True, False], 0, 1, alphabet)

    # Lt(x, y): x before y in `letters`
    pairs = [(a, b) for i, a in enumerate(letters) for b in letters[i + 1:]]
    width = max(len(pairs), 1)
    ex_s = np.full((3, width), -1, dtype=np.int32)
    ex_t = np.full((3, width), -1, dtype=np.int32)
    for j, (a, b) in enumerate(pairs):
        ex_s[0, j], ex_t[0, j] = sym((a, b)), 1
    less = SparseDFA(3, np.array([2, 2, 2], dtype=np.int32), ex_s, ex_t,
                     [False, True, False], 0, 2, alphabet)
    return AutomaticPresentation({'U': universe, name: less},
                                 padding_symbol='*')


class TestPrefix:
    def test_prepends_one_symbol(self):
        rng = random.Random(6)
        dfa = random_dfa(rng, 3, 3, 1)
        tagged = prefix(dfa, (2,))
        for word in words(3, 3):
            assert run(tagged, [2] + list(word)) == run(dfa, word), word
            if word and word[0] != 2:
                assert not run(tagged, list(word))


class TestDisjointUnion:
    def test_domain_is_the_tagged_union(self):
        a, b = chain(['p', 'q']), chain(['r'])
        u = disjoint_union(a, b)
        universe = u.automata['U']
        for tag, letter, expected in [('<l>', 'p', True), ('<l>', 'q', True),
                                      ('<r>', 'r', True), ('<l>', 'r', False),
                                      ('<r>', 'p', False)]:
            assert universe.accepts([(tag,), (letter,)]) == expected, (tag, letter)

    def test_relations_do_not_cross_the_union(self):
        a, b = chain(['p', 'q']), chain(['r', 's'])
        u = disjoint_union(a, b)
        less = u.automata['Lt']

        def holds(tx, x, ty, y):
            return less.accepts([(tx, ty), (x, y)])

        assert holds('<l>', 'p', '<l>', 'q')       # inside the left factor
        assert holds('<r>', 'r', '<r>', 's')       # inside the right factor
        assert not holds('<l>', 'q', '<l>', 'p')   # wrong way round
        assert not holds('<l>', 'p', '<r>', 's')   # never across the union
        assert not holds('<r>', 'r', '<l>', 'q')

    def test_first_order_theory_of_the_union(self):
        a, b = chain(['p', 'q']), chain(['r'])
        u = disjoint_union(a, b)
        # some element is below another (true in the left factor)
        assert u.check('exists x.(exists y.(Lt(x,y)))')
        # but not every element has something above it: q and r are maximal
        assert not u.check('all x.(exists y.(Lt(x,y)))')
        # the order is not total across the union
        assert not u.check('all x.(all y.(Lt(x,y) or Lt(y,x)))')

    def test_a_two_element_union_of_singletons_has_no_relation(self):
        u = disjoint_union(chain(['p']), chain(['r']))
        assert not u.check('exists x.(exists y.(Lt(x,y)))')

    def test_rejects_mismatched_signatures(self):
        a = chain(['p'], name='Lt')
        b = chain(['r'], name='Below')
        with pytest.raises(ValueError):
            disjoint_union(a, b)

    def test_rejects_a_tag_already_in_the_alphabet(self):
        a = chain(['p', '<l>'])
        with pytest.raises(ValueError):
            disjoint_union(a, chain(['r']))
