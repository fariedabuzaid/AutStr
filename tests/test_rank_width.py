"""Graphs of bounded rank-width (RankWidthGraph / RankWidthClass)."""
import itertools as it
import random

import pytest

from autstr.sparse_tree_automata import Tree
from autstr.tree_graphs import RankWidthClass, RankWidthGraph


_XA = '(Subset(x,a) and (not Subset(y,a)))'
_XB = '(Subset(y,a) and (not Subset(x,a)))'
TWO_COL = f'exists a.(all x.(all y.((not E(x,y)) or ({_XA} or {_XB}))))'
HAS_EDGE = 'exists x.(exists y.(E(x,y)))'


@pytest.fixture(scope="module")
def rw1():
    return RankWidthClass(1)


@pytest.fixture(scope="module")
def rw2():
    return RankWidthClass(2)


class TestRankWidthGraph:
    def test_family_widths(self):
        """Known rank-widths on the caterpillar decompositions: cliques,
        paths and complete bipartite graphs are width 1 (all are
        distance-hereditary), cycles need 2."""
        assert RankWidthGraph.clique(5).width == 1
        assert RankWidthGraph.path(6).width == 1
        assert RankWidthGraph.complete_bipartite(2, 3).width == 1
        assert RankWidthGraph.cycle(5).width == 2
        assert RankWidthGraph.cycle(6).width == 2
        assert RankWidthGraph(RankWidthGraph.caterpillar(3), []).width == 0

    def test_leaves_are_the_vertices(self):
        g = RankWidthGraph.path(4)
        assert g.n == 4 and g.vertices == [0, 1, 2, 3]
        assert g.edges == {frozenset((i, i + 1)) for i in range(3)}

    def test_edge_validation(self):
        with pytest.raises(ValueError):
            RankWidthGraph(RankWidthGraph.caterpillar(3), [(0, 0)])
        with pytest.raises(ValueError):
            RankWidthGraph(RankWidthGraph.caterpillar(3), [(0, 5)])

    def test_encode_decode_sets(self):
        g = RankWidthGraph.path(5)
        for subset in ({0, 3}, set(), {4}, {0, 1, 2, 3, 4}):
            assert g.decode_set(g.encode_set(subset)) == subset
            assert g.decode_set(g.encode_set_padded(subset)) == subset

    def test_to_networkx(self):
        nx = pytest.importorskip("networkx")
        g = RankWidthGraph.cycle(5)
        assert nx.is_isomorphic(g.to_networkx(), nx.cycle_graph(5))


class TestEdgeRelation:
    """E agrees with the edge set, explicitly and implicitly."""

    @pytest.mark.parametrize("build", [
        lambda: RankWidthGraph.clique(4),
        lambda: RankWidthGraph.path(5),
        lambda: RankWidthGraph.complete_bipartite(2, 3),
    ])
    def test_r1_families(self, rw1, build):
        g = build()
        for u, v in it.combinations(range(g.n), 2):
            expected = frozenset((u, v)) in g.edges
            assert rw1.check('E(x,y)', g, x={u}, y={v}) == expected
            assert rw1.check('E(x,y)', g, x={v}, y={u}) == expected
            assert rw1.check_implicit('E(x,y)', g, x={u}, y={v}) == expected
        assert not rw1.check('E(x,y)', g, x={0}, y={0})

    def test_r2_cycle_implicit(self, rw2):
        """C_5 needs width 2; decided through the functional atoms."""
        g = RankWidthGraph.cycle(5)
        for u, v in it.combinations(range(5), 2):
            expected = frozenset((u, v)) in g.edges
            assert rw2.check_implicit('E(x,y)', g, x={u}, y={v}) == expected

    def test_random_graphs(self, rw1, rw2):
        """Random graphs on the caterpillar decomposition, r = the measured
        width: E must reproduce the edge set exactly."""
        rng = random.Random(5)
        tested = 0
        while tested < 15:
            n = rng.randint(4, 6)
            edges = [e for e in it.combinations(range(n), 2)
                     if rng.random() < 0.4]
            g = RankWidthGraph(RankWidthGraph.caterpillar(n), edges)
            if not 1 <= g.width <= 2:
                continue
            tested += 1
            cls = rw1 if g.width == 1 else rw2
            for u, v in it.combinations(range(n), 2):
                assert cls.check_implicit('E(x,y)', g, x={u}, y={v}) \
                    == (frozenset((u, v)) in g.edges), (edges, (u, v))

    def test_width_guard(self, rw1):
        g = RankWidthGraph.cycle(5)
        assert g.width == 2
        with pytest.raises(ValueError):
            rw1.advice(g)
        with pytest.raises(ValueError):
            RankWidthClass(3)          # flat letters exceed the cap

    def test_sing_and_subset(self, rw1):
        g = RankWidthGraph.path(4)
        assert rw1.check('Sing(x)', g, x={2})
        assert not rw1.check('Sing(x)', g, x={1, 2})
        assert not rw1.check('Sing(x)', g, x=set())
        assert rw1.check('Subset(x,y)', g, x={1}, y={1, 3})
        assert not rw1.check('Subset(x,y)', g, x={0, 2}, y={2})


class TestMSO:
    def test_has_edge(self, rw1):
        assert rw1.check(HAS_EDGE, RankWidthGraph.path(3))
        empty = RankWidthGraph(RankWidthGraph.caterpillar(3), [])
        assert not rw1.check(HAS_EDGE, empty)

    def test_two_colourability_across_the_class(self, rw1):
        """One automaton decides 2-colourability for every rank-width-1
        member (Courcelle over the rank decomposition)."""
        automaton, variables = rw1.evaluate(TWO_COL)
        assert variables == ['advice']
        cases = [(RankWidthGraph.path(4), True),
                 (RankWidthGraph.clique(2), True),
                 (RankWidthGraph.clique(3), False),
                 (RankWidthGraph.complete_bipartite(2, 3), True)]
        for graph, expected in cases:
            assert rw1.check(TWO_COL, graph) == expected, graph.edges


class TestImplicitPaths:
    def test_implicit_matches_explicit(self, rw1):
        g = RankWidthGraph.complete_bipartite(2, 2)
        for phi in ('E(x,y)', 'exists y.(E(x,y))', HAS_EDGE):
            sets = {}
            if 'x' in phi.split('.')[-1] and 'exists x' not in phi:
                sets['x'] = {0}
            if phi == 'E(x,y)':
                sets['y'] = {2}
            assert rw1.check(phi, g, **sets) \
                == rw1.check_implicit(phi, g, **sets)

    def test_neighbor_solution_sets(self, rw2):
        """evaluate_implicit: the satisfying set of E(x, ·) is exactly the
        neighborhood, with the exact count known without enumeration."""
        g = RankWidthGraph.cycle(5)
        for v in range(5):
            nb = rw2.evaluate_implicit('E(x,y)', g, x={v})
            expected = sorted({(v - 1) % 5} | {(v + 1) % 5})
            assert len(nb) == 2
            assert sorted(tuple(s['y'])[0] for s in nb) == expected

    def test_domain_solution_count(self, rw1):
        """The satisfying set of Sing(x) has exactly n members."""
        g = RankWidthGraph.path(4)
        singles = rw1.evaluate_implicit('Sing(x)', g)
        assert len(singles) == 4
        assert sorted(tuple(s['x'])[0] for s in singles) == [0, 1, 2, 3]
