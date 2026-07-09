import os
import random

import networkx as nx
import pytest

from autstr.sparse_tree_automata import Tree
from autstr.tree_graphs import TreeWidthClass, TreeWidthGraph

# Set-quantifier (MSO) queries and w>=2 multi-variable queries currently
# exceed the flat-symbol engine's practical envelope (see the module
# docstring of autstr.tree_graphs); they are opt-in until the factored-
# symbol transition representation lands.
heavy = pytest.mark.skipif(not os.environ.get('AUTSTR_HEAVY'),
                           reason="needs >6GB RAM and minutes of compile "
                                  "(set AUTSTR_HEAVY=1)")


@pytest.fixture(scope="module")
def tw1():
    return TreeWidthClass(1, max_states=500_000)


@pytest.fixture(scope="module")
def tw2():
    return TreeWidthClass(2, max_states=500_000)


def chain_graph(letters) -> TreeWidthGraph:
    tree = None
    for label in reversed(letters):
        tree = Tree(label, tree, None)
    return TreeWidthGraph(tree)


def random_graph(rng: random.Random, w: int, max_size=7) -> TreeWidthGraph:
    """A random valid tree representation of width <= w (>= 1 vertex)."""
    def build(budget, occupied):
        if rng.random() < 0.2 and occupied:
            label = 'n'
        else:
            r = rng.randrange(w + 1)
            pool = sorted(occupied - {r})
            profile = tuple(sorted(rng.sample(
                pool, rng.randint(0, len(pool)))))
            label = (r, profile)
            occupied = occupied | {r}
        used = 1
        left = right = None
        if budget - used >= 1 and rng.random() < 0.7:
            left, u = build(budget - used, occupied)
            used += u
        if budget - used >= 1 and rng.random() < 0.4:
            right, u = build(budget - used, occupied)
            used += u
        return Tree(label, left, right), used

    while True:
        tree, _ = build(rng.randint(1, max_size), frozenset())
        g = TreeWidthGraph(tree)
        if g.num_nodes >= 1:
            return g


# C5 and C6 as explicit width-2 chain representations
C5 = chain_graph([(0, ()), (1, (0,)), (2, (1,)), (1, (2,)), (2, (0, 1))])
C6 = chain_graph([(0, ()), (1, (0,)), (2, (1,)), (1, (2,)), (2, (1,)),
                  (1, (0, 2))])


HAS_EDGE = 'exists x.(exists y.(E(x,y)))'
CONNECTED = ('all c.(((exists x.(Sing(x) and Subset(x,c))) and '
             '(all x.(all y.((Subset(x,c) and E(x,y)) -> Subset(y,c))))) '
             '-> (all x.(Sing(x) -> Subset(x,c))))')
TWO_COL = ('exists c.(all x.(all y.(E(x,y) -> '
           '(not ((Subset(x,c) and Subset(y,c)) or '
           '((not Subset(x,c)) and (not Subset(y,c))))))))')


class TestGraphObject:
    def test_validation(self):
        with pytest.raises(ValueError):        # unoccupied register
            TreeWidthGraph(Tree((0, (1,))))
        with pytest.raises(ValueError):        # own register in profile
            TreeWidthGraph(Tree((0, ()), Tree((0, (0,)), None, None), None))

    def test_explicit_cycles_decode(self):
        for g, n in ((C5, 5), (C6, 6)):
            G = g.to_networkx()
            assert G.number_of_nodes() == n
            assert G.number_of_edges() == n
            assert nx.is_connected(G)
            assert all(d == 2 for _, d in G.degree())

    def test_from_networkx_roundtrip(self):
        rng = random.Random(0)
        graphs = [
            nx.path_graph(5), nx.cycle_graph(5), nx.star_graph(4),
            nx.complete_graph(4), nx.random_labeled_tree(8, seed=1),
            nx.disjoint_union(nx.path_graph(3), nx.cycle_graph(4)),
        ]
        for _ in range(4):
            graphs.append(nx.gnp_random_graph(7, 0.3, seed=rng.randrange(99)))
        for G in graphs:
            g = TreeWidthGraph.from_networkx(G)
            H = g.to_networkx()
            assert set(H.nodes) == set(G.nodes)
            assert {frozenset(e) for e in H.edges} == \
                {frozenset(e) for e in G.edges}

    def test_random_representations_decode_consistently(self):
        rng = random.Random(1)
        for _ in range(20):
            g = random_graph(rng, 2)
            G = g.to_networkx()
            # every declared edge joins two existing vertices, no loops
            for u, v in g.edges():
                assert u != v and G.has_edge(u, v)


class TestRelations:
    def test_edges_pointwise(self, tw1):
        rng = random.Random(2)
        for _ in range(3):
            g = random_graph(rng, 1, max_size=6)
            G = g.to_networkx()
            for u in g.nodes:
                for v in g.nodes:
                    want = G.has_edge(u, v) and u != v
                    assert tw1.check('E(x,y)', g, x={u}, y={v}) == want, \
                        (g.tree, u, v)

    def test_sing_and_subset(self, tw1):
        rng = random.Random(3)
        g = random_graph(rng, 1, max_size=6)
        nodes = g.nodes
        u = nodes[0]
        assert tw1.check('Sing(x)', g, x={u})
        assert not tw1.check('Sing(x)', g, x=set(nodes)) or len(nodes) == 1
        assert not tw1.check('Sing(x)', g, x=set())
        assert tw1.check('Subset(x,y)', g, x={u}, y=set(nodes))
        assert tw1.check('Subset(x,y)', g, x=set(), y={u})
        if len(nodes) >= 2:
            assert not tw1.check('Subset(x,y)', g, x=set(nodes),
                                 y={nodes[1]})


class TestQueries:
    def test_has_edge_against_networkx(self, tw1):
        rng = random.Random(4)
        sta, _ = tw1.evaluate(HAS_EDGE)
        for _ in range(10):
            g = random_graph(rng, 1)
            assert nx.is_forest(g.to_networkx())   # width 1 = forests
            want = g.to_networkx().number_of_edges() > 0
            assert sta.accepts(tw1.advice(g)) == want, g.tree

    def test_get_structure(self, tw1):
        rng = random.Random(6)
        g = random_graph(rng, 1, max_size=6)
        S = tw1.get_structure(g)
        want = g.to_networkx().number_of_edges() > 0
        assert S.check(HAS_EDGE) == want


class TestHeavyMSO:
    """Set-quantifier queries: correct on the engine, but flat-symbol
    determinization of set projections is dense — opt-in until the
    factored-symbol representation lands."""

    @heavy
    def test_odd_even_cycle_two_colorability(self, tw2):
        sta, _ = tw2.evaluate(TWO_COL)
        assert not sta.accepts(tw2.advice(C5))     # odd cycle
        assert sta.accepts(tw2.advice(C6))         # even cycle

    @heavy
    def test_sentences_against_networkx(self, tw2):
        rng = random.Random(4)
        queries = [(HAS_EDGE, lambda G: G.number_of_edges() > 0),
                   (CONNECTED, nx.is_connected),
                   (TWO_COL, nx.is_bipartite)]
        compiled = [(tw2.evaluate(phi)[0], truth) for phi, truth in queries]
        graphs = [C5, C6] + [random_graph(rng, 2, max_size=7)
                             for _ in range(6)]
        for g in graphs:
            G = g.to_networkx()
            for sta, truth in compiled:
                assert sta.accepts(tw2.advice(g)) == truth(G), g.tree

    @heavy
    def test_forests_two_colorable(self, tw1):
        rng = random.Random(5)
        sta, _ = tw1.evaluate(TWO_COL)
        for _ in range(5):
            g = random_graph(rng, 1)
            assert sta.accepts(tw1.advice(g))      # forests: always
