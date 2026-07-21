"""Equality across the presentations.

Every family now declares an equality relation. Most define it from something
they already have, and the definition costs an automaton construction that
many queries never need -- so it is registered and built on first use, with an
``eager_equality`` flag for precompiling.
"""
import pytest

from autstr.algebra import FiniteBooleanAlgebras
from autstr.graphs import TreeDepthClass, TreeDepthGraph
from autstr.tree_graphs import CliqueWidthClass, CliqueWidthGraph
from autstr.groups import FiniteAbelianGroups


class TestDeferredConstruction:
    def test_declared_before_it_is_built(self):
        a = FiniteAbelianGroups()
        assert 'Eq' in a.cls.get_relation_symbols()
        assert 'Eq' not in a.cls.class_automata

    def test_built_on_first_use(self):
        a = FiniteAbelianGroups()
        a.cls.check('exists x.(Eq(x,x))', a.advice([2]))
        assert 'Eq' in a.cls.class_automata

    def test_materialize_forces_construction(self):
        a = FiniteAbelianGroups()
        assert a.cls.materialize() is a.cls
        assert 'Eq' in a.cls.class_automata

    def test_eager_flag_builds_at_construction(self):
        assert 'Eq' in FiniteAbelianGroups(eager_equality=True).cls.class_automata


class TestAbelianGroupEquality:
    """Defined from the operation: the identity is the unique idempotent, and
    x + 0 = y exactly when x = y."""

    @pytest.fixture(scope="class")
    def group(self):
        return FiniteAbelianGroups()

    def test_pointwise_against_python(self, group):
        advice = group.advice([4])
        for i in range(4):
            for j in range(4):
                got = group.cls.check(
                    'Eq(x,y)', advice,
                    x=group.encode([i], [4]), y=group.encode([j], [4]))
                assert got == (i == j), (i, j)

    def test_the_operator_form_agrees(self, group):
        K = group.symbolic()
        x, y, z = K.vars('x y z')
        advice = group.advice([4])
        sums = (x + y).eq(z)
        for i in range(4):
            for j in range(4):
                got = K.check_member(
                    sums, advice, x=group.encode([i], [4]),
                    y=group.encode([j], [4]),
                    z=group.encode([(i + j) % 4], [4]))
                assert got, (i, j)


class TestLatticeEquality:
    """Antisymmetry of the order is equality."""

    def test_pointwise_on_a_four_element_algebra(self):
        algebra = FiniteBooleanAlgebras()
        advice = ['1'] * 2                      # the algebra with 2 atoms
        elements = ['00', '01', '10', '11']
        for a in elements:
            for b in elements:
                got = algebra.cls.check('Eq(x,y)', advice,
                                        x=list(a), y=list(b))
                assert got == (a == b), (a, b)


class TestSetEquality:
    """Elements are vertex sets, so equality is mutual inclusion -- and 'E' is
    the edge relation here, which is exactly why equality is never guessed
    from the relation names."""

    def test_edge_and_equality_are_different_relations(self):
        cls = TreeDepthClass(2)
        assert {'E', 'Eq'} <= set(cls.cls.get_relation_symbols())

    def test_pointwise_on_a_small_graph(self):
        cls = TreeDepthClass(2)
        graph = TreeDepthGraph([(1, ()), (2, (1,))])
        sets = [{0}, {1}, {0, 1}]
        for i, x in enumerate(sets):
            for j, y in enumerate(sets):
                assert cls.check('Eq(x,y)', graph, x=x, y=y) == (i == j)

    def test_equality_is_not_adjacency(self):
        """Why equality is named explicitly rather than guessed: on this graph
        the two vertices are adjacent, and adjacency is what 'E' means here."""
        cls = TreeDepthClass(2)
        graph = TreeDepthGraph([(1, ()), (2, (1,))])
        u, v = graph.nodes[0], graph.nodes[1]
        assert (u, v) in graph.edges() or (v, u) in graph.edges()
        assert cls.check('E(x,y)', graph, x={u}, y={v})
        assert not cls.check('Eq(x,y)', graph, x={u}, y={v})


class TestTreeGraphClassEquality:
    """The tree-engine graph classes register the same set equality. They are
    the expensive ones to build, so clique-width 2 stands in for the family."""

    def test_declared_lazily(self):
        cls = CliqueWidthClass(2)
        assert 'Eq' in cls.cls.get_relation_symbols()
        assert 'Eq' not in cls.cls.class_automata

    def test_equality_is_not_adjacency(self):
        cls = CliqueWidthClass(2)
        graph = CliqueWidthGraph.clique(2)
        u, v = sorted(graph.vertices)[:2]
        assert cls.check('E(x,y)', graph, x={u}, y={v})
        assert not cls.check('Eq(x,y)', graph, x={u}, y={v})
        assert cls.check('Eq(x,y)', graph, x={u}, y={u})
