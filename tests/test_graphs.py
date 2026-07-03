import itertools as it

import networkx as nx
import pytest

from autstr.graphs import (
    PathWidthClass,
    PathWidthGraph,
    TreeDepthClass,
    TreeDepthGraph,
)


@pytest.fixture(scope="module")
def depth2():
    return TreeDepthClass(2)


@pytest.fixture(scope="module")
def depth3():
    return TreeDepthClass(3)


@pytest.fixture(scope="module")
def width2():
    return PathWidthClass(2)


class TestTreeDepthGraph:
    def test_roundtrip_star(self):
        g = nx.star_graph(3)  # center 0, leaves 1..3; tree-depth 2
        tdg = TreeDepthGraph.from_networkx(g)
        assert tdg.height == 2
        back = tdg.to_networkx()
        assert set(back.nodes) == set(g.nodes)
        assert {frozenset(e) for e in back.edges} == {frozenset(e) for e in g.edges}

    def test_roundtrip_path(self):
        g = nx.path_graph(4)  # tree-depth 3
        tdg = TreeDepthGraph.from_networkx(g)
        assert tdg.height == 3
        back = tdg.to_networkx()
        assert {frozenset(e) for e in back.edges} == {frozenset(e) for e in g.edges}

    def test_explicit_forest(self):
        g = nx.path_graph(3)  # 0-1-2, td 2 with elimination root 1
        forest = {1: None, 0: 1, 2: 1}
        tdg = TreeDepthGraph.from_networkx(g, forest=forest)
        assert tdg.height == 2

    def test_invalid_forest_rejected(self):
        g = nx.path_graph(3)
        # 0 and 2 are siblings, but the edge 1-2 connects... use a bad forest:
        # root 0 with children 1 and 2: edge (1,2) is not ancestor-descendant
        g2 = nx.Graph([(0, 1), (0, 2), (1, 2)])
        forest = {0: None, 1: 0, 2: 0}
        with pytest.raises(ValueError):
            TreeDepthGraph.from_networkx(g2, forest=forest)

    def test_invalid_letters_rejected(self):
        with pytest.raises(ValueError):
            TreeDepthGraph([(2, (0,))])  # first letter must have depth 1
        with pytest.raises(ValueError):
            TreeDepthGraph([(1, ()), (3, (0, 1))])  # depth jump > +1
        with pytest.raises(ValueError):
            TreeDepthGraph([(1, (1,))])  # profile length mismatch

    def test_encode_set(self):
        g = nx.star_graph(2)
        tdg = TreeDepthGraph.from_networkx(g)
        bits = tdg.encode_set({tdg.nodes[0], tdg.nodes[2]})
        assert bits == ('1', '0', '1')


class TestTreeDepthClass:
    def test_adjacency_star(self, depth2):
        g = nx.star_graph(3)
        tdg = TreeDepthGraph.from_networkx(g)
        for u, v in it.combinations(g.nodes, 2):
            expected = g.has_edge(u, v)
            assert depth2.check('E(x,y)', tdg, x={u}, y={v}) == expected
            assert depth2.check('E(x,y)', tdg, x={v}, y={u}) == expected  # symmetric

    def test_no_self_loops(self, depth2):
        g = nx.star_graph(2)
        tdg = TreeDepthGraph.from_networkx(g)
        for v in g.nodes:
            assert not depth2.check('E(x,y)', tdg, x={v}, y={v})

    def test_subset_and_sing(self, depth2):
        g = nx.star_graph(2)
        tdg = TreeDepthGraph.from_networkx(g)
        assert depth2.check('Subset(x,y)', tdg, x={0}, y={0, 1})
        assert not depth2.check('Subset(x,y)', tdg, x={0, 2}, y={0, 1})
        assert depth2.check('Sing(x)', tdg, x={1})
        assert not depth2.check('Sing(x)', tdg, x={0, 1})
        assert not depth2.check('Sing(x)', tdg, x=set())

    def test_depth_bound_enforced(self, depth2):
        g = nx.path_graph(4)  # td 3
        tdg = TreeDepthGraph.from_networkx(g)
        with pytest.raises(ValueError):
            depth2.check('E(x,y)', tdg, x={0}, y={1})

    def test_uniformity_one_query_many_graphs(self, depth2):
        """The same query automaton decides all member structures."""
        dfa, variables = depth2.evaluate('exists x.(exists y.(E(x,y)))')
        assert variables == ['advice']
        has_edge = {True: nx.star_graph(2), False: nx.empty_graph(3)}
        for expected, g in has_edge.items():
            advice = depth2.advice(TreeDepthGraph.from_networkx(g))
            assert dfa.accepts([(s,) for s in advice]) == expected

    def test_bipartiteness_mso(self, depth3):
        """One MSO query automaton decides bipartiteness for every graph of
        tree-depth <= 3: evaluate once, then model checking is just running
        each graph's advice string through the DFA."""
        phi = ('exists c.(all x.(all y.((not E(x,y)) or '
               '((Subset(x,c) and (not Subset(y,c))) or '
               '((not Subset(x,c)) and Subset(y,c))))))')
        dfa, variables = depth3.evaluate(phi)
        assert variables == ['advice']

        cases = [
            (nx.cycle_graph(3), False),  # triangle: odd cycle
            (nx.path_graph(4), True),
            (nx.complete_bipartite_graph(1, 3), True),  # star
            (nx.empty_graph(3), True),
        ]
        for g, expected in cases:
            advice = depth3.advice(TreeDepthGraph.from_networkx(g))
            assert dfa.accepts([(s,) for s in advice]) == expected

    def test_get_structure(self, depth2):
        g = nx.star_graph(2)
        tdg = TreeDepthGraph.from_networkx(g)
        structure = depth2.get_structure(tdg)
        edge_dfa = structure.evaluate('E(x,y)')
        for u, v in it.permutations(g.nodes, 2):
            word = list(zip(tdg.encode_set({u}), tdg.encode_set({v})))
            assert edge_dfa.accepts(word) == g.has_edge(u, v)


class TestPathWidthGraph:
    def test_roundtrip_cycle(self):
        g = nx.cycle_graph(5)  # pathwidth 2
        pwg = PathWidthGraph.from_networkx(g)
        assert pwg.width == 2
        back = pwg.to_networkx()
        assert set(back.nodes) == set(g.nodes)
        assert {frozenset(e) for e in back.edges} == {frozenset(e) for e in g.edges}

    def test_roundtrip_path(self):
        g = nx.path_graph(6)  # pathwidth 1
        pwg = PathWidthGraph.from_networkx(g)
        assert pwg.width == 1
        back = pwg.to_networkx()
        assert {frozenset(e) for e in back.edges} == {frozenset(e) for e in g.edges}

    def test_complete_graph(self):
        g = nx.complete_graph(4)  # pathwidth 3
        pwg = PathWidthGraph.from_networkx(g)
        assert pwg.width == 3
        back = pwg.to_networkx()
        assert {frozenset(e) for e in back.edges} == {frozenset(e) for e in g.edges}

    def test_explicit_order(self):
        g = nx.path_graph(4)
        pwg = PathWidthGraph.from_networkx(g, order=[0, 1, 2, 3])
        assert pwg.width == 1
        assert pwg.nodes == [0, 1, 2, 3]

    def test_invalid_letters_rejected(self):
        with pytest.raises(ValueError):
            PathWidthGraph([(0, (1,))])  # profile references unoccupied register
        with pytest.raises(ValueError):
            PathWidthGraph([(0, ()), (1, (1,))])  # own register in profile


class TestPathWidthClass:
    def test_adjacency_cycle(self, width2):
        g = nx.cycle_graph(5)
        pwg = PathWidthGraph.from_networkx(g)
        for u, v in it.combinations(g.nodes, 2):
            expected = g.has_edge(u, v)
            assert width2.check('E(x,y)', pwg, x={u}, y={v}) == expected
            assert width2.check('E(x,y)', pwg, x={v}, y={u}) == expected

    def test_subset_and_sing(self, width2):
        pwg = PathWidthGraph.from_networkx(nx.path_graph(4))
        assert width2.check('Sing(x)', pwg, x={2})
        assert not width2.check('Sing(x)', pwg, x={1, 2})
        assert width2.check('Subset(x,y)', pwg, x={0, 3}, y={0, 1, 3})
        assert not width2.check('Subset(x,y)', pwg, x={0, 2}, y={0, 1, 3})

    def test_width_bound_enforced(self, width2):
        pwg = PathWidthGraph.from_networkx(nx.complete_graph(4))  # width 3
        with pytest.raises(ValueError):
            width2.check('E(x,y)', pwg, x={0}, y={1})

    def test_uniformity_one_query_many_graphs(self, width2):
        dfa, variables = width2.evaluate('exists x.(exists y.(E(x,y)))')
        assert variables == ['advice']
        cases = [(nx.empty_graph(3), False), (nx.cycle_graph(4), True),
                 (nx.path_graph(5), True)]
        for g, expected in cases:
            advice = width2.advice(PathWidthGraph.from_networkx(g))
            assert dfa.accepts([(s,) for s in advice]) == expected

    def test_bipartiteness_mso(self, width2):
        """Odd vs even cycles, decided by one automaton for all graphs of
        pathwidth <= 2."""
        phi = ('exists c.(all x.(all y.((not E(x,y)) or '
               '((Subset(x,c) and (not Subset(y,c))) or '
               '((not Subset(x,c)) and Subset(y,c))))))')
        dfa, variables = width2.evaluate(phi)
        assert variables == ['advice']
        for n in (3, 4, 5, 6):
            advice = width2.advice(PathWidthGraph.from_networkx(nx.cycle_graph(n)))
            assert dfa.accepts([(s,) for s in advice]) == (n % 2 == 0), n
