"""The countable atomless boolean algebra.

The oracle is Python's set algebra. A clopen subset of Cantor space is decided
by finitely many bits of a point, so restricting to points of a fixed depth
turns every element of the sample space into an ordinary finite set of binary
strings, and meet, join, complement and containment become set operations on
those.
"""
import itertools

import pytest

from autstr.algebra import FiniteBooleanAlgebras
from autstr.sparse_tree_automata import Tree
from autstr.tree_algebra import AtomlessBooleanAlgebra

#: deep enough to separate every element of SPACE below
DEPTH = 4

#: clopen sets, written as the cylinders they contain
SPACE = [set(), {''}, {'0'}, {'1'}, {'00'}, {'01'}, {'0', '10'},
         {'00', '11'}, {'000', '01', '1'}, {'10'}, {'0', '11'}]


@pytest.fixture(scope="module")
def algebra():
    return AtomlessBooleanAlgebra()


def points(clopen, depth: int = DEPTH):
    """The points of the given depth the clopen set contains — its extension,
    as a plain finite set."""
    return {word for word in map(''.join, itertools.product('01', repeat=depth))
            if any(word.startswith(cylinder) for cylinder in clopen)}


EVERYTHING = points({''})


class TestCodec:
    def test_roundtrip(self, algebra):
        for clopen in SPACE:
            assert points(algebra.decode(algebra.encode(clopen))) == \
                points(clopen), clopen

    def test_the_empty_set_and_the_whole_space(self, algebra):
        assert algebra.encode(set()) == Tree('0')
        assert algebra.encode({''}) == Tree('1')
        assert algebra.decode(Tree('0')) == frozenset()
        assert algebra.decode(Tree('1')) == frozenset([''])

    def test_any_covering_set_is_canonicalized(self, algebra):
        # the two halves are the whole space, and a split that decides nothing
        # collapses
        assert algebra.encode({'0', '1'}) == algebra.encode({''})
        assert algebra.encode({'110', '111'}) == algebra.encode({'11'})
        assert algebra.decode(algebra.encode({'0', '1'})) == frozenset([''])

    def test_a_redundant_cylinder_is_absorbed(self, algebra):
        assert algebra.encode({'0', '01'}) == algebra.encode({'0'})

    def test_a_non_binary_string_is_rejected(self, algebra):
        with pytest.raises(ValueError, match="cylinder"):
            algebra.encode({'02'})


class TestUniverse:
    """One tree per clopen set: the reduced ones."""

    def test_accepts_every_encoding(self, algebra):
        for clopen in SPACE:
            assert algebra.automata['U'].accepts(algebra.encode(clopen))

    def test_rejects_a_split_that_decides_nothing(self, algebra):
        # both sides out, and both sides in: each is a second encoding of a
        # constant, so neither is an element
        assert not algebra.automata['U'].accepts(
            Tree('n', Tree('0'), Tree('0')))
        assert not algebra.automata['U'].accepts(
            Tree('n', Tree('1'), Tree('1')))
        assert algebra.automata['U'].accepts(Tree('n', Tree('0'), Tree('1')))


class TestOperationsAgainstSetAlgebra:
    def test_containment(self, algebra):
        order = algebra.automata['Leq']
        for x, y in itertools.product(SPACE, repeat=2):
            assert order.accepts(algebra.encode(x), algebra.encode(y)) == \
                (points(x) <= points(y)), (x, y)

    @pytest.mark.parametrize("name,operation",
                             [('Meet', lambda a, b: a & b),
                              ('Join', lambda a, b: a | b)])
    def test_meet_and_join(self, algebra, name, operation):
        algebra.materialize()
        relation = algebra.automata[name]
        for x, y, z in itertools.product(SPACE, repeat=3):
            want = operation(points(x), points(y)) == points(z)
            assert relation.accepts(*map(algebra.encode, (x, y, z))) == want, \
                (name, x, y, z)

    def test_complement(self, algebra):
        algebra.materialize()
        relation = algebra.automata['Compl']
        for x, y in itertools.product(SPACE, repeat=2):
            want = (EVERYTHING - points(x)) == points(y)
            assert relation.accepts(algebra.encode(x),
                                    algebra.encode(y)) == want, (x, y)


class TestTermSurface:
    """Meet, join and complement are terms — ``*``, ``+`` and unary ``-`` —
    because ``&``, ``|`` and ``~`` already mean the connectives of the logic."""

    def test_operations_as_terms(self, algebra):
        x, y = algebra.symbolic().vars('x y')
        meet = (x * y).eq({'00'}).evaluate()
        assert meet.contains(x={'0'}, y={'00', '11'})
        assert not meet.contains(x={'0'}, y={'1'})

        join = (x + y).eq({''}).evaluate()
        assert join.contains(x={'0'}, y={'1'})
        assert not join.contains(x={'0'}, y={'00'})

        complement = (-x).eq(y).evaluate()
        assert complement.contains(x={'0'}, y={'1'})
        assert not complement.contains(x={'0'}, y={'0'})

    def test_the_order_as_a_method(self, algebra):
        x, y = algebra.symbolic().vars('x y')
        order = x.leq(y).evaluate()
        assert order.contains(x={'00'}, y={'0'})
        assert not order.contains(x={'0'}, y={'00'})

    @pytest.mark.parametrize("law", ['idempotent', 'commutative', 'absorption',
                                     'distributive', 'involution', 'de morgan'])
    def test_the_axioms_hold(self, algebra, law):
        x, y, z = algebra.symbolic().vars('x y z')
        laws = {
            'idempotent': (x * x).eq(x).all('x'),
            'commutative': (x * y).eq(y * x).all('x y'),
            'absorption': (x * (x + y)).eq(x).all('x y'),
            'distributive': (x * (y + z)).eq((x * y) + (x * z)).all('x y z'),
            'involution': (-(-x)).eq(x).all('x'),
            'de morgan': (-(x * y)).eq((-x) + (-y)).all('x y'),
        }
        assert laws[law].check()


class TestAtomless:
    """The property the algebra is named for, and the one that pins it down:
    a countable atomless boolean algebra is unique up to isomorphism."""

    def test_it_is_atomless(self, algebra):
        assert algebra.is_atomless()

    def test_the_atom_relation_is_empty(self, algebra):
        assert not algebra.check('exists x.(Atom(x))')

    def test_the_same_sentence_separates_it_from_the_finite_algebras(
            self, algebra):
        """`Atom` is in both signatures, so one sentence tells them apart —
        every finite boolean algebra has an atom and this one has none."""
        finite = FiniteBooleanAlgebras()
        assert finite.check('exists x.(Atom(x))', 4)
        assert not algebra.check('exists x.(Atom(x))')

    def test_it_has_a_bottom_and_a_top(self, algebra):
        assert algebra.check('exists x.(all y.(Leq(x,y)))')
        assert algebra.check('exists x.(all y.(Leq(y,x)))')

    def test_every_element_has_a_complement(self, algebra):
        assert algebra.check('all x.(exists y.(Compl(x,y)))')

    def test_the_order_is_a_partial_order(self, algebra):
        assert algebra.check('all x.(Leq(x,x))')
        assert algebra.check(
            'all x.(all y.(all z.((Leq(x,y) & Leq(y,z)) -> Leq(x,z))))')
        assert algebra.check(
            'all x.(all y.((Leq(x,y) & Leq(y,x)) -> Eq(x,y)))')


class TestConstruction:
    def test_repr(self, algebra):
        assert repr(algebra) == "<AtomlessBooleanAlgebra>"

    def test_the_signature_matches_the_finite_algebras(self, algebra):
        assert sorted(algebra.get_relation_symbols()) == \
            ['Atom', 'Compl', 'Eq', 'Join', 'Leq', 'Meet', 'U']
