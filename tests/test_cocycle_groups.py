import itertools as it
import random

import pytest

from autstr.cocycle_groups import (CocycleSites, FLATTENINGS, fixed_k_sites,
                                   laminar_sites, point_target_sites,
                                   scattered_sites)
from autstr.sparse_tree_automata import Tree
from autstr.tree_groups import CutRankTreeGroups


def mixed_instance():
    """A small site tree with z-sites both above and *below* commutator
    pairs (the latter force inward-flowing claims)."""
    shape = Tree('z', Tree('x', Tree('x', Tree('z'), None), Tree('x')), None)
    sites = CocycleSites(2, shape)
    # positions: z=1, x=2 (over the z), x=3, x=4, z=5 (root)
    T = {(4, 2, 1): 1, (4, 3, 1): 1, (3, 2, 5): 1, (4, 2, 5): 1}
    return sites, T


def elements(sites):
    p = sites.p
    return [(b, a)
            for b in it.product(range(p), repeat=len(sites.Z))
            for a in it.product(range(p), repeat=len(sites.X))]


BUSHY = Tree('s', Tree('s', Tree('s'), Tree('s')),
             Tree('s', Tree('s'), Tree('s')))
SPINEY = Tree('s', Tree('s', Tree('s'), None), None)


class TestReferenceLaw:
    def test_group_axioms_exhaustive(self):
        sites, T = mixed_instance()
        elems = elements(sites)
        mul = lambda g, h: sites.multiply(T, g, h)
        one = sites.identity()
        for g, h, f in it.product(elems, repeat=3):
            assert mul(mul(g, h), f) == mul(g, mul(h, f)), (g, h, f)
        for g in elems:
            assert mul(one, g) == g and mul(g, one) == g
            assert any(mul(g, h) == one for h in elems)

    def test_group_axioms_sampled_p3(self):
        shape = Tree('z', Tree('x', Tree('x', Tree('z'), None), Tree('x')), None)
        sites = CocycleSites(3, shape)
        T = {(4, 2, 1): 2, (4, 3, 1): 1, (3, 2, 5): 1}
        rng = random.Random(1)
        elems = elements(sites)
        mul = lambda g, h: sites.multiply(T, g, h)
        for _ in range(2000):
            g, h, f = (rng.choice(elems) for _ in range(3))
            assert mul(mul(g, h), f) == mul(g, mul(h, f))

    def test_tensor_validation(self):
        sites, _ = mixed_instance()
        with pytest.raises(ValueError):
            sites.check_tensor({(2, 4, 1): 1})   # needs i < j
        with pytest.raises(ValueError):
            sites.check_tensor({(4, 2, 3): 1})   # target must be a z-site


class TestFixedKCorner:
    """Theorem 2 instances embed with the expected width profile and the
    same group law."""

    def test_law_agrees_with_cutrank_tree_groups(self):
        crt = CutRankTreeGroups(2)
        n = 4
        for form in (crt.clique_form(n), crt.matching_form(n),
                     {(j, 1): (1,) for j in range(2, n + 1)}):
            sites, T = fixed_k_sites(2, crt.balanced(n), form, k=1)
            for g, h in it.product(elements(sites), repeat=2):
                assert sites.multiply(T, g, h) == crt.multiply(n, form, g, h)

    def test_width_and_profile(self):
        crt = CutRankTreeGroups(2)
        n = 6
        sites, T = fixed_k_sites(2, crt.balanced(n), crt.clique_form(n), k=1)
        assert sites.cut_width(T) == 1
        for t, ranks in sites.cut_profile(T).items():
            if t <= n:   # layout cuts: no z-site inside
                assert ranks['F_g'] == ranks['F_py'] == ranks['F_px'] == 0

    def test_width_k2(self):
        n = 4
        form = {(2, 1): (1, 0), (4, 3): (0, 1), (3, 1): (1, 1)}
        sites, T = fixed_k_sites(2, CutRankTreeGroups.balanced(n), form, k=2)
        assert sites.cut_width(T) <= 2


class TestLaminarCorner:
    """TreeExtraspecialGroups instances embed with width exactly 1."""

    @staticmethod
    def their_law(p, maps, g1, g2):
        """The tree-extraspecial law by addresses: c(v) gains
        sum over inner w above v of a1(w) * b2(w)."""
        (a1, b1, c1), (a2, b2, c2) = g1, g2
        a = {w: (a1[w] + a2[w]) % p for w in a1}
        b = {w: (b1[w] + b2[w]) % p for w in b1}
        c = {v: (c1[v] + c2[v] + sum(a1[w] * b2[w] for w in a1
                                     if v != w and v.startswith(w))) % p
             for v in c1}
        return a, b, c

    def embed(self, sites, maps, g):
        a_addr, b_addr, c_addr = g
        avec = [0] * len(sites.X)
        bvec = [0] * len(sites.Z)
        xi = {t: idx for idx, t in enumerate(sites.X)}
        zi = {v: idx for idx, v in enumerate(sites.Z)}
        for w, (pos_x, pos_y) in maps['inner'].items():
            avec[xi[pos_x]] = a_addr[w]
            avec[xi[pos_y]] = b_addr[w]
        for v, pos in maps['leaf'].items():
            bvec[zi[pos]] = c_addr[v]
        return tuple(bvec), tuple(avec)

    @pytest.mark.parametrize("shape", [BUSHY, SPINEY])
    def test_width_is_one(self, shape):
        sites, T, _ = laminar_sites(2, shape)
        assert sites.cut_width(T) == 1
        for ranks in sites.cut_profile(T).values():
            assert ranks['F_y'] == ranks['F_x'] == ranks['F_m'] == 0

    @pytest.mark.parametrize("shape", [BUSHY, SPINEY])
    def test_law_agrees(self, shape):
        p = 3
        sites, T, maps = laminar_sites(p, shape)
        inners = sorted(maps['inner'])
        leaves = sorted(maps['leaf'])
        rng = random.Random(7)
        for _ in range(300):
            gs = []
            for _ in range(2):
                gs.append(({w: rng.randrange(p) for w in inners},
                           {w: rng.randrange(p) for w in inners},
                           {v: rng.randrange(p) for v in leaves}))
            expected = self.their_law(p, maps, gs[0], gs[1])
            got = sites.multiply(T, self.embed(sites, maps, gs[0]),
                                 self.embed(sites, maps, gs[1]))
            assert got == self.embed(sites, maps, expected)


class TestBeyondBothCorners:
    def test_point_targets_width_one(self):
        """Growing center, non-laminar law, still width 1: in neither
        corner class."""
        sites, T = point_target_sites(2, BUSHY)
        assert sites.cut_width(T) == 1

    def test_point_targets_differ_from_laminar(self):
        lam_sites, lam_T, maps = laminar_sites(2, BUSHY)
        pt_sites, pt_T = point_target_sites(2, BUSHY)
        one = lam_sites.identity()
        root_x, root_y = maps['inner']['']
        xi = {t: idx for idx, t in enumerate(lam_sites.X)}
        g = list(one[1]); g[xi[root_x]] = 1
        h = list(one[1]); h[xi[root_y]] = 1
        g = (one[0], tuple(g)); h = (one[0], tuple(h))
        assert lam_sites.multiply(lam_T, g, h) != pt_sites.multiply(pt_T, g, h)

    def test_scattered_targets_have_growing_width(self):
        """Private targets force unbounded claim rank: the width definition
        correctly charges inward-flowing traffic."""
        for m in (2, 3, 4):
            sites, T = scattered_sites(2, m)
            assert sites.cut_width(T) == m
