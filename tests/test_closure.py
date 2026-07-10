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


# ====================================================================
# Direct products
# ====================================================================

from autstr.closure import direct_product

TRANSITIVE = 'all x.(all y.(all z.((Lt(x,y) and Lt(y,z)) -> Lt(x,z))))'


def element(pairs):
    """A one-letter word over the pair alphabet, per tape."""
    return [tuple(pairs)]


class TestDirectProduct:
    def test_domain_is_the_product_of_the_domains(self):
        p = direct_product(chain(['p', 'q']), chain(['r', 's']))
        universe = p.automata['U']
        for a in ('p', 'q'):
            for b in ('r', 's'):
                assert universe.accepts(element([(a, b)])), (a, b)
        # a pair letter exists for every (a, b), but a padded half is no element
        assert not universe.accepts(element([('p', '*')]))
        assert not universe.accepts(element([('*', 'r')]))

    def test_sync_moves_both_coordinates(self):
        p = direct_product(chain(['p', 'q']), chain(['r', 's']), kind='sync')
        less = p.automata['Lt']

        def holds(x, y):
            return less.accepts(element([x, y]))

        assert holds(('p', 'r'), ('q', 's'))        # both coordinates advance
        assert not holds(('p', 'r'), ('q', 'r'))    # only the left one moves
        assert not holds(('p', 'r'), ('p', 's'))    # only the right one moves

    def test_async_moves_exactly_one_coordinate(self):
        p = direct_product(chain(['p', 'q']), chain(['r', 's']), kind='async')
        less = p.automata['Lt']

        def holds(x, y):
            return less.accepts(element([x, y]))

        assert holds(('p', 'r'), ('q', 'r'))        # left moves, right fixed
        assert holds(('p', 'r'), ('p', 's'))        # right moves, left fixed
        assert not holds(('p', 'r'), ('q', 's'))    # both move: not async
        assert not holds(('p', 'r'), ('p', 'r'))    # neither moves

    def test_the_two_products_differ_in_their_theory(self):
        """The product of two strict orders is transitive; the asynchronous
        product is the grid's covering relation, which is not."""
        left, right = chain(['p', 'q', 't']), chain(['r', 's', 'u'])
        assert direct_product(left, right, kind='sync').check(TRANSITIVE)
        assert not direct_product(left, right, kind='async').check(TRANSITIVE)

    def test_a_relation_is_restricted_to_the_domain(self):
        """An embedded factor leaves the other half unconstrained, so the
        product must intersect it back with the domain: a tuple whose right
        half is padding satisfies the left factor's order, but is no element."""
        p = direct_product(chain(['p', 'q']), chain(['r', 's']), kind='async')
        assert not p.automata['Lt'].accepts(element([('p', '*'), ('q', '*')]))

    def test_rejects_an_unknown_kind(self):
        with pytest.raises(ValueError):
            direct_product(chain(['p']), chain(['r']), kind='weak')


# ====================================================================
# Union of two uniformly automatic classes
# ====================================================================

from autstr.closure import class_union, tagged_advice
from autstr.uniform import UniformlyAutomaticClass, dfa_from_delta


def prefix_chain_class(letter):
    """The class of finite linear orders: advice `letter^n` presents the chain
    of length n, whose elements are the nonempty prefixes of the advice."""
    sigma = {'*', letter}

    def universe(q, sym):
        a, x = sym
        if a != letter:
            return 'dead'
        if q == 'init':
            return 'in' if x == letter else 'dead'
        if q == 'in':
            return 'in' if x == letter else 'done'
        if q == 'done':
            return 'done' if x == '*' else 'dead'
        return 'dead'

    def less(q, sym):
        a, x, y = sym
        if a != letter:
            return 'dead'
        if q == 'init':
            return 'both' if (x, y) == (letter, letter) else 'dead'
        if q == 'both':
            if (x, y) == (letter, letter):
                return 'both'
            if (x, y) == ('*', letter):
                return 'yonly'          # x ran out first: x is shorter
            return 'dead'
        if q == 'yonly':
            if (x, y) == ('*', letter):
                return 'yonly'
            if (x, y) == ('*', '*'):
                return 'done'
            return 'dead'
        if q == 'done':
            return 'done' if (x, y) == ('*', '*') else 'dead'
        return 'dead'

    u = dfa_from_delta(sigma, ['init', 'in', 'done', 'dead'], 2, universe,
                       'init', {'in', 'done'})
    lt = dfa_from_delta(sigma, ['init', 'both', 'yonly', 'done', 'dead'], 3,
                        less, 'init', {'yonly', 'done'})
    return UniformlyAutomaticClass({'U': u, 'Lt': lt}, padding_symbol='*')


SOME_EDGE = 'exists x.(exists y.(Lt(x,y)))'
NO_MAXIMUM = 'all x.(exists y.(Lt(x,y)))'
TOTAL = 'all x.(all y.(Lt(x,y) or Lt(y,x) or (not Lt(x,y))))'


class TestClassUnion:
    def test_the_factor_classes_are_recovered(self):
        """The union must decide every sentence on a tagged advice exactly as
        the factor decides it on the bare advice."""
        left, right = prefix_chain_class('a'), prefix_chain_class('b')
        both = class_union(left, right)
        for phi in (SOME_EDGE, NO_MAXIMUM):
            for n in (1, 2, 3):
                assert both.check(phi, tagged_advice(['a'] * n, '<l>')) == \
                    left.check(phi, ['a'] * n), (phi, n, 'left')
                assert both.check(phi, tagged_advice(['b'] * n, '<r>')) == \
                    right.check(phi, ['b'] * n), (phi, n, 'right')

    def test_a_chain_of_length_one_has_no_edge(self):
        both = class_union(prefix_chain_class('a'), prefix_chain_class('b'))
        assert not both.check(SOME_EDGE, tagged_advice(['a'], '<l>'))
        assert both.check(SOME_EDGE, tagged_advice(['a', 'a'], '<l>'))
        assert not both.check(SOME_EDGE, tagged_advice(['b'], '<r>'))
        assert both.check(SOME_EDGE, tagged_advice(['b', 'b'], '<r>'))

    def test_a_wrongly_tagged_advice_presents_the_empty_structure(self):
        """The advice languages are disjoint: the right class never reads a
        left-tagged advice."""
        both = class_union(prefix_chain_class('a'), prefix_chain_class('b'))
        assert not both.check(SOME_EDGE, tagged_advice(['b', 'b'], '<l>'))

    def test_rejects_mismatched_signatures(self):
        left = prefix_chain_class('a')
        right = prefix_chain_class('b')
        right.class_automata['Other'] = right.class_automata['Lt']
        with pytest.raises(ValueError):
            class_union(left, right)


# ====================================================================
# Direct product closure of a uniformly automatic class
# ====================================================================

from autstr.closure import blocks, direct_product_closure


class TestDirectProductClosure:
    def test_a_single_block_is_the_original_member(self):
        chains = prefix_chain_class('a')
        products = direct_product_closure(chains)
        for n in (1, 2, 3):
            assert products.check(SOME_EDGE, blocks(['a'] * n)) == \
                chains.check(SOME_EDGE, ['a'] * n), n

    def test_the_product_relation_is_componentwise(self):
        """Lt holds in a product only when it holds in *every* component, so a
        product with a one-element factor has no edge at all."""
        products = direct_product_closure(prefix_chain_class('a'))
        assert not products.check(SOME_EDGE, blocks(['a'], ['a', 'a']))
        assert not products.check(SOME_EDGE, blocks(['a', 'a'], ['a']))
        assert products.check(SOME_EDGE, blocks(['a', 'a'], ['a', 'a']))

    def test_three_factors(self):
        products = direct_product_closure(prefix_chain_class('a'))
        assert products.check(SOME_EDGE, blocks(['a', 'a'], ['a', 'a'],
                                                ['a', 'a']))
        assert not products.check(SOME_EDGE, blocks(['a', 'a'], ['a', 'a'],
                                                    ['a']))

    def test_a_product_of_orders_is_transitive(self):
        products = direct_product_closure(prefix_chain_class('a'))
        transitive = ('all x.(all y.(all z.((Lt(x,y) and Lt(y,z)) '
                      '-> Lt(x,z))))')
        assert products.check(transitive, blocks(['a', 'a', 'a'],
                                                 ['a', 'a', 'a']))

    def test_mixing_two_classes(self):
        """The point of the whole exercise: products of members drawn from
        either of two different classes."""
        both = class_union(prefix_chain_class('a'), prefix_chain_class('b'))
        products = direct_product_closure(both)
        left2 = tagged_advice(['a', 'a'], '<l>')
        right2 = tagged_advice(['b', 'b'], '<r>')
        right1 = tagged_advice(['b'], '<r>')
        assert products.check(SOME_EDGE, blocks(left2, right2))
        assert not products.check(SOME_EDGE, blocks(left2, right1))

    def test_rejects_a_separator_already_in_the_alphabet(self):
        chains = prefix_chain_class('a')
        with pytest.raises(ValueError):
            direct_product_closure(chains, separator='a')


# ====================================================================
# The point of the exercise: mixing two families of finite groups
# ====================================================================

from autstr.groups import ExtraspecialGroups, IndexTwoCyclicGroups
from autstr.uniform import UniformlyAutomaticClass

IDENTITY = 'exists u.(all x.(M(x,u,x)))'
ABELIAN = 'all x.(all y.(all z.(M(x,y,z) -> M(y,x,z))))'


def _reduct(uniform):
    """The common signature of the two group classes: domain and product."""
    return UniformlyAutomaticClass(
        {'U': uniform.class_automata['U'], 'M': uniform.class_automata['M']},
        padding_symbol='*')


@pytest.fixture(scope="module")
def mixed_groups():
    """All finite direct products of index-<=2 cyclic groups and extraspecial
    3-groups, drawn from either family."""
    cyclic, extra = IndexTwoCyclicGroups(), ExtraspecialGroups(3)
    both = class_union(_reduct(cyclic.cls), _reduct(extra.cls))
    return direct_product_closure(both), cyclic, extra


class TestMixedGroupProducts:
    def test_a_product_of_groups_has_an_identity(self, mixed_groups):
        products, cyclic, extra = mixed_groups
        z4 = tagged_advice(cyclic.cyclic(4), '<l>')
        heis = tagged_advice(extra.advice(1), '<r>')
        assert products.check(IDENTITY, blocks(z4, heis))

    def test_a_product_is_abelian_exactly_when_its_factors_are(self, mixed_groups):
        products, cyclic, extra = mixed_groups
        z4 = tagged_advice(cyclic.cyclic(4), '<l>')
        d4 = tagged_advice(cyclic.advice('dihedral', 4), '<l>')
        heis = tagged_advice(extra.advice(1), '<r>')

        assert products.check(ABELIAN, blocks(z4))            # Z4
        assert not products.check(ABELIAN, blocks(heis))      # extraspecial
        assert products.check(ABELIAN, blocks(z4, z4))        # Z4 x Z4
        # one nonabelian factor is enough, from either family
        assert not products.check(ABELIAN, blocks(z4, heis))
        assert not products.check(ABELIAN, blocks(d4, z4))
