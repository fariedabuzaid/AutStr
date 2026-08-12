"""First-order interpretations over the tree engine.

The source is Skolem arithmetic, (N_{>0}, ·) — multiplication is not
string-automatic, so these interpretations exist only over trees. Python's own
arithmetic is the oracle throughout.

An element of a k-dimensional interpretation is a k-tuple of trees, which *is*
one tree over k-tuples, since a tree convolution already overlays the shapes.
So the fold that regroups the tapes is the same operation as in the string
engine, and these tests check that the structure it produces means what it
should.
"""
import itertools

import pytest

from autstr.buildin.tree_presentations import SkolemArithmetic
from autstr.interpretations import interpret
from autstr.sparse_tree_automata import Tree, convolve_trees
from autstr.tree_presentations import TreeAutomaticPresentation

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


class TestQuotientsAreNotBuiltForTreesYet:
    """Representatives exist — every tree-automatic equivalence has a regular
    complete system of them — but not by the string engine's route of taking
    the least element of a class, since no tree-automatic order is
    well-founded. Reaching them needs the shadow construction of Kuske &
    Weidner (MFCS 2011), which is not built here, so the construction refuses
    rather than quietly picking a non-representative.
    """

    def test_the_quotient_is_refused_with_an_explanation(self, skolem):
        with pytest.raises(NotImplementedError, match="well-founded"):
            interpret(skolem, domain=('Eq(x,x)', ['x']),
                      relations={'M': ('M(x,y,z)', ['x', 'y', 'z'])},
                      quotient=('Eq(x,y)', ['x', 'y']))

    def test_naming_the_representatives_is_the_way_round_it(self, skolem):
        """The same construction with the choice made explicit: restrict the
        domain to one element per class. Here the classes are "same square",
        and the representative is the square itself."""
        representatives = interpret(
            skolem, domain=('exists r.(M(r,r,x))', ['x']),
            relations={'M': ('M(x,y,z)', ['x', 'y', 'z'])})
        assert representatives.automata['U'].accepts(
            SkolemArithmetic.encode(9))
        assert not representatives.automata['U'].accepts(
            SkolemArithmetic.encode(3))


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
