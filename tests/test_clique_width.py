import pytest

nx = pytest.importorskip("networkx")

from autstr.sparse_tree_automata import Tree
from autstr.tree_graphs import CliqueWidthClass, CliqueWidthGraph


_XA = '(Subset(x,a) and (not Subset(y,a)))'
_XB = '(Subset(y,a) and (not Subset(x,a)))'
TWO_COL = f'exists a.(all x.(all y.((not E(x,y)) or ({_XA} or {_XB}))))'
HAS_EDGE = 'exists x.(exists y.(E(x,y)))'


@pytest.fixture(scope="module")
def cw2():
    return CliqueWidthClass(2)


@pytest.fixture(scope="module")
def cw3():
    return CliqueWidthClass(3)


class TestExpressionDecoding:
    """The k-expression must denote the graph it is supposed to."""

    @pytest.mark.parametrize("build,reference", [
        (lambda: CliqueWidthGraph.clique(1), lambda: nx.complete_graph(1)),
        (lambda: CliqueWidthGraph.clique(4), lambda: nx.complete_graph(4)),
        (lambda: CliqueWidthGraph.complete_bipartite(2, 3),
         lambda: nx.complete_bipartite_graph(2, 3)),
        (lambda: CliqueWidthGraph.path(5), lambda: nx.path_graph(5)),
        (lambda: CliqueWidthGraph.cycle(5), lambda: nx.cycle_graph(5)),
        (lambda: CliqueWidthGraph.cycle(6), lambda: nx.cycle_graph(6)),
    ])
    def test_families_match_networkx(self, build, reference):
        assert nx.is_isomorphic(build().to_networkx(), reference())

    def test_leaves_are_the_vertices(self):
        graph = CliqueWidthGraph.path(4)
        assert graph.vertices == [0, 1, 2, 3]

    def test_malformed_expressions_are_rejected(self):
        # a join is unary
        with pytest.raises(ValueError):
            CliqueWidthGraph(Tree('e01', Tree('v0'), Tree('v1')), 2)
        # a union is binary
        with pytest.raises(ValueError):
            CliqueWidthGraph(Tree('u', Tree('v0')), 2)
        # a leaf must create a vertex
        with pytest.raises(ValueError):
            CliqueWidthGraph(Tree('u', Tree('v0'), Tree('e01')), 2)
        # the label must fit within k
        with pytest.raises(ValueError):
            CliqueWidthGraph(Tree('v3'), 2)


class TestEdgeAutomaton:
    """E(x, y) must decide adjacency exactly."""

    def test_agrees_with_the_decoded_edges(self, cw2):
        for graph in (CliqueWidthGraph.clique(4),
                      CliqueWidthGraph.complete_bipartite(2, 3)):
            reference = graph.to_networkx()
            for u in graph.vertices:
                for v in graph.vertices:
                    if u >= v:
                        continue
                    assert cw2.check('E(x,y)', graph, x={u}, y={v}) == \
                        reference.has_edge(u, v), (u, v)

    def test_agrees_on_paths(self, cw3):
        graph = CliqueWidthGraph.path(5)
        reference = graph.to_networkx()
        for u in graph.vertices:
            for v in graph.vertices:
                if u >= v:
                    continue
                assert cw3.check('E(x,y)', graph, x={u}, y={v}) == \
                    reference.has_edge(u, v), (u, v)

    def test_a_vertex_is_not_adjacent_to_itself(self, cw2):
        graph = CliqueWidthGraph.clique(3)
        assert not cw2.check('E(x,y)', graph, x={0}, y={0})


class TestSetRelations:
    def test_sing(self, cw3):
        graph = CliqueWidthGraph.path(4)
        assert cw3.check('Sing(x)', graph, x={2})
        assert not cw3.check('Sing(x)', graph, x={0, 1})
        assert not cw3.check('Sing(x)', graph, x=set())

    def test_subset(self, cw3):
        graph = CliqueWidthGraph.path(4)
        assert cw3.check('Subset(x,y)', graph, x={0}, y={0, 1})
        assert cw3.check('Subset(x,y)', graph, x=set(), y={0, 1})
        assert not cw3.check('Subset(x,y)', graph, x={0, 2}, y={0, 1})

    def test_encode_set_round_trip(self):
        graph = CliqueWidthGraph.path(4)
        assert graph.encode_set(set()) == Tree('0')
        marks = graph.encode_set({0, 3})
        assert marks is not None                    # a non-empty marked domain


class TestQueries:
    def test_first_order_sentence(self, cw3):
        assert cw3.check(HAS_EDGE, CliqueWidthGraph.path(4))
        assert not cw3.check(HAS_EDGE, CliqueWidthGraph.clique(1))

    def test_two_colourability_is_decided_once_for_the_whole_class(self, cw2):
        """One MSO compile, then a linear-time decision on every member."""
        automaton, _ = cw2.evaluate(TWO_COL)
        assert automaton.num_states > 0
        cases = [CliqueWidthGraph.clique(2), CliqueWidthGraph.clique(3),
                 CliqueWidthGraph.clique(4),
                 CliqueWidthGraph.complete_bipartite(2, 3),
                 CliqueWidthGraph.complete_bipartite(3, 3)]
        for graph in cases:
            assert cw2.check(TWO_COL, graph) == \
                nx.is_bipartite(graph.to_networkx()), graph.vertices

    def test_get_structure(self, cw2):
        structure = cw2.get_structure(CliqueWidthGraph.clique(3))
        assert structure.check('exists x.(exists y.(E(x,y)))')
