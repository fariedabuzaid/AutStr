"""The infinite k-ary tree with its successors and prefix order.

The oracle is Python's own word operations: a vertex is a tuple of child
indices, ``y = x·i`` is a tuple append, and the prefix order is a slice
comparison. Every relation is checked exhaustively against those over the
vertices of depth ≤ 3.
"""
import itertools

import pytest

from autstr.infinite_graphs import RegularTree


@pytest.fixture(scope="module")
def binary():
    return RegularTree(2)


def vertices(k: int, depth: int):
    """Every vertex of depth at most `depth`, as a tuple of child indices."""
    return [word
            for length in range(depth + 1)
            for word in itertools.product(range(k), repeat=length)]


class TestCodec:
    def test_roundtrip(self, binary):
        for vertex in vertices(2, 3):
            assert binary.decode(binary.encode(vertex)) == vertex

    def test_the_root_is_the_empty_word(self, binary):
        assert binary.encode(()) == []

    def test_a_string_of_digits_is_accepted(self, binary):
        assert binary.encode('011') == binary.encode((0, 1, 1))

    def test_an_index_outside_the_branching_is_rejected(self, binary):
        with pytest.raises(ValueError, match="child index"):
            binary.encode((0, 2))


class TestSuccessors:
    def test_each_successor_appends_its_letter(self, binary):
        S = binary.symbolic()
        x, y = S.vars('x y')
        for i in range(2):
            relation = S.rel(f'S{i}')(x, y).evaluate()
            for a, b in itertools.product(vertices(2, 3), repeat=2):
                expected = (b == a + (i,))
                assert relation.contains(x=a, y=b) == expected, (i, a, b)

    def test_child_is_the_union_of_the_successors(self, binary):
        S = binary.symbolic()
        x, y = S.vars('x y')
        relation = S.rel('Child')(x, y).evaluate()
        for a, b in itertools.product(vertices(2, 3), repeat=2):
            assert relation.contains(x=a, y=b) == (b[:-1] == a and len(b) > 0)


class TestPrefixOrder:
    def test_agrees_with_the_slice_comparison(self, binary):
        S = binary.symbolic()
        x, y = S.vars('x y')
        relation = S.rel('Prefix')(x, y).evaluate()
        for a, b in itertools.product(vertices(2, 3), repeat=2):
            assert relation.contains(x=a, y=b) == (b[:len(a)] == a), (a, b)

    def test_is_a_partial_order_with_a_least_element(self, binary):
        assert binary.check('all x.(Prefix(x,x))')
        assert binary.check('all x.(all y.((Prefix(x,y) & Prefix(y,x)) '
                            '-> Eq(x,y)))')
        assert binary.check('exists r.(all x.(Prefix(r,x)))')     # the root


class TestGraph:
    def test_the_edge_is_the_parent_child_relation(self, binary):
        S = binary.symbolic()
        x, y = S.vars('x y')
        edge = x.adj(y).evaluate()
        assert edge.contains(x=(), y=(0,))
        assert edge.contains(x=(0,), y=())            # undirected
        assert edge.contains(x=(0, 1), y=(0, 1, 0))
        assert not edge.contains(x=(), y=(0, 1))      # not a grandchild
        assert not edge.contains(x=(0,), y=(1,))      # not siblings
        assert not edge.contains(x=(0,), y=(0,))      # no self-loop

    def test_is_symmetric(self, binary):
        assert binary.is_symmetric()

    def test_every_vertex_but_the_root_has_exactly_one_parent(self, binary):
        assert binary.check(
            'all y.((exists p.(Child(p,y))) -> '
            'all p.(all q.((Child(p,y) & Child(q,y)) -> Eq(p,q))))')
        # the root is the one vertex without a parent, and it is unique
        assert binary.check(
            'exists r.((not exists p.(Child(p,r))) & '
            'all x.((not exists p.(Child(p,x))) -> Eq(x,r)))')

    def test_every_vertex_has_a_child_for_each_letter(self, binary):
        for i in range(2):
            assert binary.check(f'all x.(exists y.(S{i}(x,y)))')

    def test_distinct_letters_give_distinct_children(self, binary):
        assert binary.check(
            'all x.(all y.(all z.((S0(x,y) & S1(x,z)) -> (not Eq(y,z)))))')


class TestConstruction:
    def test_branching_must_be_positive(self):
        with pytest.raises(ValueError, match=">= 1"):
            RegularTree(0)

    def test_repr(self, binary):
        assert repr(binary) == "<RegularTree branching=2>"

    def test_a_ray_is_the_one_ary_tree(self):
        ray = RegularTree(1)
        # exactly one child each: the tree degenerates to a chain
        assert ray.check('all x.(exists y.(S0(x,y)))')
        assert ray.check('all x.(all y.(all z.((Child(x,y) & Child(x,z)) '
                         '-> Eq(y,z))))')

    def test_a_wider_tree(self):
        wide = RegularTree(4)
        assert sorted(s for s in wide.get_relation_symbols()
                      if s.startswith('S')) == ['S0', 'S1', 'S2', 'S3']
        S = wide.symbolic()
        x, y = S.vars('x y')
        assert x.adj(y).evaluate().contains(x=(3,), y=(3, 2))
