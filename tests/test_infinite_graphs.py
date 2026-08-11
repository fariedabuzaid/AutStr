"""The generic InfiniteGraph wrapper.

These tests exercise the wrapper's mechanics against a real presentation
(Büchi arithmetic over ℤ) rather than a bespoke automaton: the edge relation is
defined by a formula, so no automata are authored here. The concrete graph
factories — ordinals, integer grids, … — arrive in later commits, each with its
own ground-truth oracle.
"""
import pytest

from autstr.arithmetic import decode, encode
from autstr.buildin.presentations import BuechiArithmeticZ
from autstr.infinite_graphs import InfiniteGraph
from autstr.symbolic import FunctionCodec


@pytest.fixture(scope="module")
def complete_graph():
    """The complete graph on ℤ: x—y iff x ≠ y. Symmetric, and definable
    straight from the order, so it needs no hand-built automaton. Vertices are
    integers via Büchi's own codec, so they can be written as Python values."""
    z = BuechiArithmeticZ()
    z.update(Adj='Lt(x,y) | Lt(y,x)')
    return InfiniteGraph(z, edge='Adj', codec=FunctionCodec(encode, decode))


class TestConstruction:
    def test_missing_edge_relation_is_rejected(self):
        z = BuechiArithmeticZ()
        with pytest.raises(ValueError, match="no edge relation 'Adj'"):
            InfiniteGraph(z, edge='Adj')

    def test_non_binary_edge_is_rejected(self):
        z = BuechiArithmeticZ()          # A is the ternary addition graph
        with pytest.raises(ValueError, match="arity 3, not 2"):
            InfiniteGraph(z, edge='A')

    def test_repr_names_the_edge(self, complete_graph):
        assert "edge='Adj'" in repr(complete_graph)


class TestSymbolicSurface:
    def test_adjacency_binds_to_the_edge(self, complete_graph):
        G = complete_graph.symbolic()
        x, y = G.vars('x y')
        relation = x.adj(y).evaluate()
        assert relation.contains(x=3, y=5)
        assert relation.contains(x=5, y=3)
        assert not relation.contains(x=4, y=4)     # no self-loop

    def test_equality_is_bound(self, complete_graph):
        G = complete_graph.symbolic()
        x, y = G.vars('x y')
        relation = x.eq(y).evaluate()
        assert relation.contains(x=7, y=7)
        assert not relation.contains(x=7, y=8)

    def test_a_first_order_sentence(self, complete_graph):
        G = complete_graph.symbolic()
        x, y = G.vars('x y')
        # every vertex has a neighbour (the complete graph on ℤ is infinite)
        assert x.adj(y).drop('y').all('x').check()
        # no vertex is adjacent to itself
        assert not x.adj(x).drop('x').check()


class TestSymmetry:
    def test_undirected_edge_is_symmetric(self, complete_graph):
        assert complete_graph.is_symmetric()

    def test_a_directed_edge_is_not(self):
        z = BuechiArithmeticZ()
        order = InfiniteGraph(z, edge='Lt', directed=True)
        assert not order.is_symmetric()


class TestPassthrough:
    def test_check_and_evaluate_reach_the_presentation(self, complete_graph):
        assert complete_graph.check('exists x.(exists y.(Adj(x,y)))')
        assert not complete_graph.check('exists x.(Adj(x,x))')
        assert 'Adj' in complete_graph.get_relation_symbols()
