"""The integer grid ℤⁿ (Cayley graph of ℤⁿ) as an automatic structure.

Adjacency is checked against the ground truth ``sum(|Δ_i|) == 1`` — differ by
±1 in exactly one coordinate — which is what the asynchronous product of n
integer paths should compute.
"""
import itertools

import pytest

from autstr.infinite_graphs import IntegerGrid


def adjacent(p, q):
    """Ground truth: two points of ℤⁿ are grid-adjacent iff they differ by ±1
    in exactly one coordinate."""
    return sum(abs(a - b) for a, b in zip(p, q)) == 1


@pytest.fixture(scope="module")
def grids():
    return {n: IntegerGrid(n) for n in (1, 2, 3)}


class TestCodec:
    def test_roundtrips(self, grids):
        for n, grid in grids.items():
            for point in itertools.product(range(-3, 4), repeat=n):
                assert grid.decode(grid.encode(point)) == point, (n, point)

    def test_wrong_dimension_is_rejected(self, grids):
        with pytest.raises(ValueError, match="2-tuple"):
            grids[2].encode((1, 2, 3))


class TestAdjacency:
    def test_matches_ground_truth(self, grids):
        for n, grid in grids.items():
            relation = grid.symbolic()
            x, y = relation.vars('x y')
            edge = x.adj(y).evaluate()
            points = list(itertools.product(range(-2, 3), repeat=n))
            # a representative sample of pairs, both directions
            for p in points[::3]:
                for q in points:
                    assert edge.contains(x=p, y=q) == adjacent(p, q), (n, p, q)

    def test_symmetric(self, grids):
        for grid in grids.values():
            assert grid.is_symmetric()

    def test_no_self_loops(self, grids):
        for grid in grids.values():
            G = grid.symbolic()
            x = G.vars('x')[0]
            assert not x.adj(x).drop('x').check()


class TestFirstOrder:
    def test_the_path_is_2_regular(self):
        """Every vertex of ℤ¹ has exactly two neighbours — a first-order
        statement, counting via equality: ∃y∃z distinct, both adjacent, and
        every neighbour is one of them."""
        G = IntegerGrid(1).symbolic()
        x, y, z, w = G.vars('x y z w')
        exactly_two = (
            ~y.eq(z) & x.adj(y) & x.adj(z)
            & x.adj(w).implies(w.eq(y) | w.eq(z)).all('w')
        ).drop('y z').all('x')
        assert exactly_two.check()

    def test_dimension_one_is_the_path(self, grids):
        grid = grids[1]
        edge = grid.symbolic()
        x, y = edge.vars('x y')
        rel = x.adj(y).evaluate()
        assert rel.contains(x=(5,), y=(6,))
        assert rel.contains(x=(5,), y=(4,))
        assert not rel.contains(x=(5,), y=(7,))
