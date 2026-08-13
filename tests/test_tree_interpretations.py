"""First-order interpretations over the tree engine.

The source is mostly Skolem arithmetic, (N_{>0}, ·) — multiplication is not
string-automatic, so these interpretations exist only over trees. Python's own
arithmetic is the oracle throughout. The quotient tests add a purpose-built
structure whose classes are large and whose representatives can be written
down by hand.

An element of a k-dimensional interpretation is a k-tuple of trees, which *is*
one tree over k-tuples, since a tree convolution already overlays the shapes.
So the fold that regroups the tapes is the same operation as in the string
engine, and these tests check that the structure it produces means what it
should.
"""
import itertools

import pytest

from autstr.tree_arithmetic import SkolemArithmetic
from autstr.interpretations import _EQUIV, _tree_representatives, interpret
from autstr.sparse_tree_automata import Tree, convolve_trees
from autstr.tree_presentations import TreeAutomaticPresentation
from autstr.utils.tree_automata_tools import (
    attach_padding, equivalent, minimize, partial_tree_automaton,
    tree_automaton, tree_order,
)

PAD = SkolemArithmetic.PAD
LETTERS = SkolemArithmetic.LETTERS


@pytest.fixture(scope="module")
def skolem():
    return SkolemArithmetic()


def tuple_tree(*numbers):
    """A k-tuple of naturals as one tree over k-tuples — an element of a
    k-dimensional interpretation of Skolem arithmetic."""
    return convolve_trees([SkolemArithmetic.encode(n) for n in numbers],
                          LETTERS, PAD)


def as_one_tape(tree):
    """An element tree wrapped for a *unary* relation, whose single tape reads
    whole elements."""
    if tree is None:
        return None
    return Tree((tree.label,), as_one_tape(tree.left), as_one_tape(tree.right))


class TestOneDimensional:
    """A definable reduct with a restricted domain: elements keep the source's
    trees, so the oracle is plain arithmetic on the numbers they encode."""

    @pytest.fixture(scope="class")
    def squares(self, skolem):
        return interpret(skolem, domain=('exists r.(M(r,r,x))', ['x']),
                         relations={'M': ('M(x,y,z)', ['x', 'y', 'z'])})

    def test_the_result_is_a_tree_presentation(self, squares):
        assert isinstance(squares, TreeAutomaticPresentation)
        assert sorted(squares.get_relation_symbols()) == ['M', 'U']

    def test_the_domain_is_the_squares(self, squares):
        universe = squares.automata['U']
        for n in range(1, 30):
            is_square = round(n ** 0.5) ** 2 == n
            assert universe.accepts(SkolemArithmetic.encode(n)) == is_square, n

    def test_multiplication_restricted_to_the_domain(self, squares):
        product = squares.automata['M']
        squares_only = [1, 4, 9, 16, 25, 36, 100]
        for a, b in itertools.product(squares_only, repeat=2):
            for c in squares_only:
                want = a * b == c
                assert product.accepts(*map(SkolemArithmetic.encode,
                                            (a, b, c))) == want, (a, b, c)
        # a product of the source whose factors leave the domain is gone,
        # even though it holds in the source: 2 · 2 = 4 and 2 is no square
        assert product.accepts(*map(SkolemArithmetic.encode, (4, 9, 36)))
        assert not product.accepts(*map(SkolemArithmetic.encode, (2, 2, 4)))

    def test_a_first_order_fact_about_the_squares(self, squares):
        # the squares are closed under multiplication, and every square is a
        # product of two squares (itself and 1)
        assert squares.check('all x.(all y.(exists z.(M(x,y,z))))')
        assert squares.check('all x.(exists y.(exists z.(M(y,z,x))))')


class TestTwoDimensional:
    """Elements are pairs of naturals under componentwise multiplication."""

    @pytest.fixture(scope="class")
    def pairs(self, skolem):
        return interpret(
            skolem,
            domain=('Eq(x0,x0) & Eq(x1,x1)', ['x0', 'x1']),
            relations={'M': ('M(x0,y0,z0) & M(x1,y1,z1)',
                             ['x0', 'x1', 'y0', 'y1', 'z0', 'z1'])},
            dimension=2)

    def test_the_alphabet_is_the_square_of_the_sources(self, pairs, skolem):
        assert len(pairs.base_alphabet) == len(skolem.base_alphabet) ** 2
        assert pairs.padding_symbol == (PAD, PAD)
        assert pairs.automata['U'].symbol_arity == 1
        assert pairs.automata['M'].symbol_arity == 3

    def test_the_universe_is_every_pair(self, pairs):
        universe = pairs.automata['U']
        for a, b in itertools.product([1, 2, 6, 10, 27], repeat=2):
            assert universe.accepts(as_one_tape(tuple_tree(a, b))), (a, b)

    def test_multiplication_is_componentwise(self, pairs):
        product = pairs.automata['M']
        space = [(1, 1), (2, 3), (5, 7), (10, 21), (4, 9), (2, 2)]
        for x, y in itertools.product(space, repeat=2):
            for z in space:
                want = (x[0] * y[0], x[1] * y[1]) == z
                assert product.accepts(tuple_tree(*x), tuple_tree(*y),
                                       tuple_tree(*z)) == want, (x, y, z)

    def test_first_order_facts_about_the_product_monoid(self, pairs):
        assert pairs.check('exists u.(all x.(M(u,x,x)))')          # identity
        assert pairs.check('all x.(all y.(exists z.(M(x,y,z))))')  # total
        assert pairs.check(                                        # commutes
            'all x.(all y.(all z.(M(x,y,z) -> M(y,x,z))))')

    def test_a_sparse_domain_does_not_blow_up(self, skolem):
        """The diagonal {(a, a)} realizes only a sparse subset of the pairs,
        yet minimization keeps multiplication the size of the one-dimensional
        encoding — the tree counterpart of the same measurement on words."""
        diagonal = interpret(
            skolem, domain=('Eq(x0,x1)', ['x0', 'x1']),
            relations={'M': ('Eq(x0,x1) & Eq(y0,y1) & Eq(z0,z1) & M(x0,y0,z0)',
                             ['x0', 'x1', 'y0', 'y1', 'z0', 'z1'])},
            dimension=2)
        native = interpret(skolem, domain=('Eq(x,x)', ['x']),
                           relations={'M': ('M(x,y,z)', ['x', 'y', 'z'])})
        assert diagonal.automata['M'].num_states == \
            native.automata['M'].num_states == \
            skolem.automata['M'].num_states


#: a two-letter alphabet for the purpose-built structure below
MARKS = ['a', 'b']
MARK_ALPHABET = set(MARKS) | {PAD}


def any_tree():
    """Every tree over ``{a, b}``."""
    return partial_tree_automaton(MARK_ALPHABET, 1, {
        (left, right, (letter,)): 'ok' for letter in MARKS
        for left in (None, 'ok') for right in (None, 'ok')}, {'ok'})


def root_pairs(accepted):
    """Pairs of trees whose root letters are one of the given pairs — the
    rest of the two trees is free, which is what makes the classes big."""
    states = [f'{a}{b}' for a in MARKS for b in MARKS] + ['past']
    table = {}
    for left in [None] + states:
        for right in [None] + states:
            for a in MARKS:
                for b in MARKS:
                    table[(left, right, (a, b))] = f'{a}{b}'
                # below the root one tree may stop before the other
                table[(left, right, (a, PAD))] = 'past'
                table[(left, right, (PAD, a))] = 'past'
    return partial_tree_automaton(MARK_ALPHABET, 2, table, set(accepted))


def complete_tree(depth, root='a', filler='a'):
    """The complete binary tree of the given depth."""
    def build(level):
        return None if level > depth else \
            Tree(filler, build(level + 1), build(level + 1))
    return Tree(root, build(1), build(1))


def one_per_class(raw, representatives):
    """The specification of a complete system of representatives, checked in
    the engine itself: every class has a representative, and no class has two.

    Quantifiers here range over the *elements*, so this says exactly what the
    construction promises, whatever the classes happen to be.
    """
    scratch = TreeAutomaticPresentation(
        {'U': raw.automata['U'], 'Eqv': raw.relation(_EQUIV),
         'Rep': representatives,
         'Le': tree_order(raw.base_alphabet, raw.padding_symbol)},
        padding_symbol=raw.padding_symbol)
    return (scratch.check('all x.(exists y.(Eqv(x,y) & Rep(y)))'),
            # antisymmetry of a linear order is equality
            scratch.check('all x.(all y.((Rep(x) & Rep(y) & Eqv(x,y))'
                          ' -> (Le(x,y) & Le(y,x))))'))


class TestQuotients:
    """Elements as classes of a definable equivalence.

    Over words the representative is the shortlex-least element of a class.
    Over trees no automatic order is well-founded, so a class need have no
    least element, and the representative is instead the least *description* —
    Kuske & Weidner's construction. These check that what comes out really is
    one element per class, and that it is the one the construction promises.
    """

    @pytest.fixture(scope="class")
    def marked(self):
        """Trees over ``{a, b}``, with "same root letter" as the equivalence
        and one edge between the two classes."""
        return TreeAutomaticPresentation(
            {'U': any_tree(), 'Same': root_pairs({'aa', 'bb'}),
             'Edge': root_pairs({'ab'})}, padding_symbol=PAD)

    def test_quotienting_by_equality_gives_the_structure_back(self, skolem):
        """Every class is a singleton, so its only description is its one
        member — the strongest check that nothing is lost on the way."""
        same = interpret(skolem, domain=('Eq(x,x)', ['x']),
                         relations={'M': ('M(x,y,z)', ['x', 'y', 'z'])},
                         quotient=('Eq(x,y)', ['x', 'y']))
        assert equivalent(same.automata['U'], skolem.automata['U'])
        assert equivalent(same.automata['M'], skolem.automata['M'])
        assert same.automata['M'].accepts(*[SkolemArithmetic.encode(n)
                                            for n in (5, 7, 35)])

    def test_a_class_becomes_one_element(self, marked):
        quotient = interpret(marked, 'U(x)', {'Edge': 'Edge(x,y)'},
                             quotient='Same(x,y)')
        raw = interpret(marked, 'U(x)',
                        {'Edge': 'Edge(x,y)', _EQUIV: 'Same(x,y)'})
        depth = raw.relation(_EQUIV).num_states

        # every class shares only the root, so a description is any tree of
        # depth at most the bound — and the least of those, in an order where
        # growing a tree makes it smaller, is the full one with least letters
        universe = quotient.automata['U']
        for root in MARKS:
            assert universe.accepts(complete_tree(depth, root))
        assert not universe.accepts(Tree('a'))
        assert not universe.accepts(complete_tree(depth - 1))
        assert not universe.accepts(complete_tree(depth, 'a', 'b'))

        # and those two trees are the whole universe
        only_those = minimize(
            attach_padding(tree_automaton(complete_tree(depth, 'a'),
                                          MARK_ALPHABET), PAD).union(
                attach_padding(tree_automaton(complete_tree(depth, 'b'),
                                              MARK_ALPHABET), PAD)))
        assert equivalent(universe, only_those)

    def test_the_relations_come_along(self, marked):
        quotient = interpret(marked, 'U(x)', {'Edge': 'Edge(x,y)'},
                             quotient='Same(x,y)')
        assert quotient.check('exists x.(exists y.(Edge(x,y)))')
        assert quotient.check('all x.(all y.(Edge(x,y) -> (not Edge(y,x))))')
        assert not quotient.check('exists x.(Edge(x,x))')

    @pytest.mark.parametrize("case", ['equality', 'root letter'])
    def test_exactly_one_representative_per_class(self, skolem, marked, case):
        if case == 'equality':
            raw = interpret(skolem, domain=('Eq(x,x)', ['x']),
                            relations={_EQUIV: ('Eq(x,y)', ['x', 'y'])})
        else:
            raw = interpret(marked, 'U(x)', {_EQUIV: 'Same(x,y)'})
        exists, unique = one_per_class(raw, _tree_representatives(raw))
        assert exists and unique

    def test_a_two_dimensional_quotient(self, skolem):
        """Elements are pairs, and the equivalence forgets the second
        coordinate — so the quotient is the first coordinate's structure."""
        pairs = interpret(
            skolem,
            domain=('Eq(x0,x0) & Eq(x1,x1)', ['x0', 'x1']),
            relations={'First': ('Eq(x0,y0) & Eq(x1,x1) & Eq(y1,y1)',
                                 ['x0', 'x1', 'y0', 'y1'])},
            dimension=2,
            quotient=('Eq(x0,y0) & Eq(x1,x1) & Eq(y1,y1)',
                      ['x0', 'x1', 'y0', 'y1']))
        # one element per first coordinate, and First is then equality on them
        assert pairs.check('all x.(all y.(First(x,y) -> First(y,x)))')
        assert pairs.check('all x.(First(x,x))')
        assert pairs.check('exists x.(exists y.(not First(x,y)))')

    def test_the_scaffolding_names_are_reserved(self, skolem):
        for name in ('_Equiv', '_Le', '_Rep'):
            with pytest.raises(ValueError, match="reserved"):
                interpret(skolem, domain=('Eq(x,x)', ['x']),
                          relations={name: ('Eq(x,y)', ['x', 'y'])},
                          quotient=('Eq(x,y)', ['x', 'y']))


class TestValidationCarriesOverToTrees:
    def test_domain_needs_one_element(self, skolem):
        with pytest.raises(ValueError, match="defines 2 element"):
            interpret(skolem, domain=('M(x,y,y)', ['x', 'y']), relations={})

    def test_free_variables_must_be_a_multiple_of_the_dimension(self, skolem):
        with pytest.raises(ValueError, match="multiple of the dimension"):
            interpret(skolem,
                      domain=('Eq(x0,x0) & Eq(x1,x1)', ['x0', 'x1']),
                      relations={'R': ('M(x0,x1,x2)', ['x0', 'x1', 'x2'])},
                      dimension=2)

    def test_universe_symbol_is_reserved(self, skolem):
        with pytest.raises(ValueError, match="reserved universe"):
            interpret(skolem, domain=('Eq(x,x)', ['x']),
                      relations={'U': ('Eq(x,y)', ['x', 'y'])})
