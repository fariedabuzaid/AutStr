import itertools as it
import os
import random

import pytest

heavy = pytest.mark.skipif(not os.environ.get('AUTSTR_HEAVY'),
                           reason="builds the sub-alphabet presentation "
                                  "automata: several GB, run under a "
                                  "memory cap (set AUTSTR_HEAVY=1)")

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


def ring_elements(sites):
    """Elements with both center and quotient coords over R = Z/p^d."""
    return [(b, a)
            for b in it.product(range(sites.q), repeat=len(sites.Z))
            for a in it.product(range(sites.q), repeat=len(sites.X))]


class TestChainRingCenter:
    """Idea 2: an exponent-p^d center, R = Z/p^d (d > 1)."""

    def small_ring_instance(self, p, d):
        # x-sites at 1,2 with a single z-site at the root (position 3)
        shape = Tree('z', Tree('x', Tree('x'), None), None)
        return CocycleSites(p, shape, d=d)

    def test_group_axioms_over_z4_exhaustive(self):
        sites = self.small_ring_instance(2, 2)     # center over Z/4
        T = {(2, 1, 3): 2}                          # valuation-1 coefficient
        assert sites.q == 4
        elems = ring_elements(sites)
        mul = lambda g, h: sites.multiply(T, g, h)
        one = sites.identity()
        for g, h, f in it.product(elems, repeat=3):
            assert mul(mul(g, h), f) == mul(g, mul(h, f)), (g, h, f)
        for g in elems:
            assert mul(one, g) == g and mul(g, one) == g
            assert any(mul(g, h) == one for h in elems)

    def test_center_has_exponent_p_squared(self):
        """A commutator with a unit coefficient generates a center element of
        order p^d = 4, not p -- the point of Idea 2."""
        sites = self.small_ring_instance(2, 2)
        T = {(2, 1, 3): 1}
        x = ((0,), (0, 1))       # a with a[2]=1
        y = ((0,), (1, 0))       # a with a[1]=1
        # [x, y] lands the commutator coefficient in the center coordinate;
        # x*y and y*x differ by that central element, of additive order 4.
        xy = sites.multiply(T, x, y)
        yx = sites.multiply(T, y, x)
        diff = (xy[0][0] - yx[0][0]) % sites.q
        assert diff % 2 == 1                         # a unit -> order 4
        assert sites.q == 4

    def test_sampled_axioms_z9(self):
        shape = Tree('z', Tree('x', Tree('x', Tree('z'), None), Tree('x')), None)
        sites = CocycleSites(3, shape, d=2)          # center over Z/9
        T = {(4, 2, 1): 3, (4, 3, 1): 1, (3, 2, 5): 6}
        rng = random.Random(2)
        elems = ring_elements(sites)
        mul = lambda g, h: sites.multiply(T, g, h)
        for _ in range(3000):
            g, h, f = (rng.choice(elems) for _ in range(3))
            assert mul(mul(g, h), f) == mul(g, mul(h, f))

    def test_width_counts_valuation_carrying_generators(self):
        """A lone valuation-1 coefficient has module cut-rank 1 over Z/4 (it
        would be dropped by a naive mod-p reduction, giving a wrong 0)."""
        sites = self.small_ring_instance(2, 2)
        assert sites.cut_width({(2, 1, 3): 2}) == 1
        assert sites.cut_width({(2, 1, 3): 1}) == 1
        assert sites.cut_width({}) == 0

    def test_width_matches_saturate_free_rank(self):
        """cut_width equals the free rank of the saturated flattening for a
        two-pair block over Z/4."""
        shape = Tree('z', Tree('x', Tree('x', Tree('x'), None), None), None)
        sites = CocycleSites(2, shape, d=2)          # x at 1,2,3; z at 4
        T = {(2, 1, 4): 2, (3, 1, 4): 1}             # mixed valuations
        prof = sites.cut_profile(T)
        # the cut below x-position 2 (subtree {1,2}) crosses one pair upward
        assert sites.cut_width(T) == max(max(r.values()) for r in prof.values())
        assert sites.cut_width(T) >= 1

    def test_d1_default_is_the_field_case(self):
        """d = 1 reproduces the original F_p law and widths byte-for-byte."""
        shape = Tree('z', Tree('x', Tree('x', Tree('z'), None), Tree('x')), None)
        T = {(4, 2, 1): 1, (4, 3, 1): 1, (3, 2, 5): 1, (4, 2, 5): 1}
        default = CocycleSites(2, shape)             # d defaults to 1
        explicit = CocycleSites(2, shape, d=1)
        assert default.q == explicit.q == 2
        elems = ring_elements(default)
        for g, h in it.product(elems, repeat=2):
            assert default.multiply(T, g, h) == explicit.multiply(T, g, h)
        assert default.cut_profile(T) == explicit.cut_profile(T)
        assert default.cut_width(T) == explicit.cut_width(T)


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


# ====================================================================
# The claim-and-verify automaton (CocycleRankWidthGroups, width 1)
# ====================================================================

from autstr.cocycle_groups import CocycleRankWidthGroups


@pytest.fixture(scope="module")
def crw2():
    """Simulator-only instance: the presentation automata are never built
    (the full merge-letter alphabet is too large for the flat builder)."""
    return CocycleRankWidthGroups(2)


def fpx_instance():
    """An instance with a nonzero mixed-x flattening: subtree [2, 3]
    contains the larger endpoint and the target but not the smaller."""
    shape = Tree('x', Tree('x'), Tree('x', Tree('z'), None))
    sites = CocycleSites(2, shape)
    T = {(3, 1, 2): 1, (4, 1, 2): 1}
    return sites, T


def two_center_instance():
    """A fixed-k embedding with k = 2: both center digits consume the
    pair-sum register along the z-chain."""
    form = {(2, 1): (1, 1), (3, 1): (1, 1)}
    return fixed_k_sites(2, CutRankTreeGroups.balanced(3), form, 2)


def random_site_shape(rng, budget):
    label = ZSITE if rng.random() < 0.4 else XSITE
    if budget <= 1:
        return Tree(label)
    left = right = None
    used = 1
    if rng.random() < 0.8:
        left = random_site_shape(rng, (budget - 1) // 2 + 1)
        used += len(CocycleSites(2, left).seq) if False else 0
    if rng.random() < 0.5:
        right = random_site_shape(rng, (budget - 1) // 2)
    return Tree(label, left, right)


from autstr.cocycle_groups import XSITE, ZSITE


class TestClaimAndVerifyProtocol:
    """The transition function and the compiler, driven directly over the
    convolved trees (`simulate` runs exactly the automaton's delta)."""

    def sim_check(self, crw, sites, T, exhaustive=True, rounds=80, seed=1):
        advice = crw.advice(sites, T)
        elems = elements(sites)
        rng = random.Random(seed)
        pairs = (it.product(elems, repeat=2) if exhaustive else
                 ((rng.choice(elems), rng.choice(elems))
                  for _ in range(rounds)))
        for g, h in pairs:
            expected = sites.multiply(T, g, h)
            tx = crw.encode(g, sites, advice)
            ty = crw.encode(h, sites, advice)
            assert crw.simulate(advice, tx, ty,
                                crw.encode(expected, sites, advice)), (g, h)
            if sites.Z:
                wb = list(expected[0])
                v = rng.randrange(len(wb))
                wb[v] = (wb[v] + 1) % sites.p
                wrong = (tuple(wb), expected[1])
                assert not crw.simulate(advice, tx, ty,
                                        crw.encode(wrong, sites, advice)), \
                    (g, h)

    def test_fixed_k_corner(self, crw2):
        n = 3
        for form in ({(j, i): (1,) for j in range(2, n + 1)
                      for i in range(1, j)},
                     {(j, 1): (1,) for j in range(2, n + 1)}):
            sites, T = fixed_k_sites(2, CutRankTreeGroups.balanced(n), form, 1)
            self.sim_check(crw2, sites, T)

    def test_two_center_chain(self, crw2):
        sites, T = two_center_instance()
        self.sim_check(crw2, sites, T)

    @pytest.mark.parametrize("shape", [SPINEY, BUSHY])
    def test_laminar_corner(self, crw2, shape):
        sites, T, _ = laminar_sites(2, shape)
        self.sim_check(crw2, sites, T, exhaustive=False)

    def test_point_targets(self, crw2):
        sites, T = point_target_sites(2, BUSHY)
        self.sim_check(crw2, sites, T, exhaustive=False)

    def test_mixed_claims(self, crw2):
        sites, T = mixed_instance()
        self.sim_check(crw2, sites, T)

    def test_fpx_path(self, crw2):
        sites, T = fpx_instance()
        assert sites.cut_width(T) == 1
        self.sim_check(crw2, sites, T)

    def test_zero_tensor(self, crw2):
        sites, _ = mixed_instance()
        self.sim_check(crw2, sites, {})

    def test_width_guard(self, crw2):
        sites, T = scattered_sites(2, 2)
        with pytest.raises(ValueError):
            crw2.advice(sites, T)

    def test_random_width1_instances(self, crw2):
        """Fuzz: random site trees with sparse random tensors of cut-width
        one; the compiler's lemma assertions and the simulator must agree
        with the reference law on all of them."""
        rng = random.Random(42)
        found = 0
        attempts = 0
        while found < 12 and attempts < 400:
            attempts += 1
            shape = random_site_shape(rng, rng.randint(3, 7))
            sites = CocycleSites(2, shape)
            if len(sites.X) < 2 or not sites.Z or sites.n_sites > 8:
                continue
            triples = [(j, i, v) for i in sites.X for j in sites.X if i < j
                       for v in sites.Z]
            rng.shuffle(triples)
            T = {t: 1 for t in triples[:rng.randint(1, 3)]}
            if sites.cut_width(T) > 1:
                continue
            found += 1
            self.sim_check(crw2, sites, T, exhaustive=False, rounds=40,
                           seed=attempts)
        assert found >= 8, f"only {found} width-1 instances found"


class TestRingEmbeddings:
    """The embedding constructors with a ring depth d: the same layouts,
    coefficients and widths, over R = Z/p^d."""

    def test_fixed_k_matches_cutrank_tree_groups_z4(self):
        crt = CutRankTreeGroups(2, d=2)
        n = 4
        rng = random.Random(3)
        for form in (crt.clique_form(n),
                     {(j, i): (2,) for j in range(2, n + 1)
                      for i in range(1, j)}):        # valuation-1 labels
            sites, T = fixed_k_sites(2, crt.balanced(n), form, k=1, d=2)
            assert sites.q == 4
            elems = ring_elements(sites)
            for _ in range(400):
                g, h = rng.choice(elems), rng.choice(elems)
                assert sites.multiply(T, g, h) == crt.multiply(n, form, g, h)

    def test_fixed_k_ring_width(self):
        crt = CutRankTreeGroups(2, d=2)
        n = 6
        sites, T = fixed_k_sites(2, crt.balanced(n), crt.clique_form(n), 1,
                                 d=2)
        assert sites.cut_width(T) == 1
        val1 = {(j, i): (2,) for j in range(2, n + 1) for i in range(1, j)}
        sites2, T2 = fixed_k_sites(2, crt.balanced(n), val1, 1, d=2)
        assert sites2.cut_width(T2) == 1                 # not 0: valuation

    def test_laminar_and_point_targets_ring_width_one(self):
        for build in (lambda: laminar_sites(2, BUSHY, d=2)[:2],
                      lambda: point_target_sites(2, BUSHY, d=2)):
            sites, T = build()
            assert sites.q == 4 and sites.d == 2
            assert sites.cut_width(T) == 1

    def test_scattered_ring_width_grows(self):
        for m in (2, 3):
            sites, T = scattered_sites(2, m, d=2)
            assert sites.q == 4
            assert sites.cut_width(T) == m

    def test_d1_default_unchanged(self):
        a = fixed_k_sites(2, CutRankTreeGroups.balanced(3),
                          {(2, 1): (1,)}, 1)
        b = fixed_k_sites(2, CutRankTreeGroups.balanced(3),
                          {(2, 1): (1,)}, 1, d=1)
        assert a[0].q == b[0].q == 2 and a[1] == b[1]

    def test_ring_embeddings_through_the_microcode(self, crw4):
        """Integration of the embeddings with the ring claim-and-verify
        automaton: laminar and point-target Z/4 members compile at width 1
        and simulate the reference law."""
        ring = TestClaimAndVerifyChainRing()
        sites, T, _ = laminar_sites(2, SPINEY, d=2)
        ring.sim_check(crw4, sites, T, rounds=200)
        pt_sites, pt_T = point_target_sites(2, BUSHY, d=2)
        ring.sim_check(crw4, pt_sites, pt_T, rounds=200)
        crt = CutRankTreeGroups(2, d=2)
        fk_sites, fk_T = fixed_k_sites(2, crt.balanced(3),
                                       {(j, 1): (2,) for j in (2, 3)}, 1,
                                       d=2)
        ring.sim_check(crw4, fk_sites, fk_T, rounds=200)


def _tree_eq(a, b):
    if a is None and b is None:
        return True
    if a is None or b is None or a.label != b.label:
        return False
    return _tree_eq(a.left, b.left) and _tree_eq(a.right, b.right)


def _letters(tree):
    out = []
    stack = [tree]
    while stack:
        node = stack.pop()
        if node is None:
            continue
        out.append(node.label)
        stack.append(node.left)
        stack.append(node.right)
    return out


@pytest.fixture(scope="module")
def crw4():
    """Simulator-only Z/4 instance (d = 2); its automata are never built."""
    return CocycleRankWidthGroups(2, d=2)


class TestClaimAndVerifyChainRing:
    """The ring generalisation of the claim-and-verify microcode: saturated
    interfaces, R-valued registers and the truncated claim invariant
    (compiler-tracked valuations, `zr` re-anchoring). Everything checked
    against the reference law via `simulate`."""

    Z4_SPINE = Tree('z', Tree('x', Tree('x'), None), None)
    Z4_MIXED = Tree('z', Tree('x', Tree('x', Tree('z'), None), Tree('x')),
                    None)
    # two z-sites below the pair: claim vector with mixed valuations
    Z4_DEEP = Tree('x', Tree('x', Tree('z', Tree('z'), None), None), None)

    def sim_check(self, crw, sites, T, rounds=400, seed=1):
        advice = crw.advice(sites, T)
        rng = random.Random(seed)
        q = sites.q
        elems = [(tuple(rng.randrange(q) for _ in sites.Z),
                  tuple(rng.randrange(q) for _ in sites.X))
                 for _ in range(60)]
        for _ in range(rounds):
            g, h = rng.choice(elems), rng.choice(elems)
            expected = sites.multiply(T, g, h)
            tx = crw.encode(g, sites, advice)
            ty = crw.encode(h, sites, advice)
            assert crw.simulate(advice, tx, ty,
                                crw.encode(expected, sites, advice)), (g, h)
            if sites.Z:
                wb = list(expected[0])
                v = rng.randrange(len(wb))
                wb[v] = (wb[v] + rng.randrange(1, q)) % q
                wrong = (tuple(wb), expected[1])
                assert not crw.simulate(advice, tx, ty,
                                        crw.encode(wrong, sites, advice)), \
                    (g, h, wrong)

    def test_unit_and_valuation_coefficients_z4(self, crw4):
        sites = CocycleSites(2, self.Z4_SPINE, d=2)
        self.sim_check(crw4, sites, {(2, 1, 3): 1})
        self.sim_check(crw4, sites, {(2, 1, 3): 2})   # valuation 1

    def test_mixed_claims_z4(self, crw4):
        sites = CocycleSites(2, self.Z4_MIXED, d=2)
        T = {(4, 2, 1): 2, (4, 3, 1): 1, (3, 2, 5): 1, (4, 2, 5): 2}
        assert sites.cut_width(T) == 1
        self.sim_check(crw4, sites, T)
        self.sim_check(crw4, sites, {})

    def test_reanchor_z4(self, crw4):
        """A claim vector {v1: 2, v2: 1} over Z/4: position v1 determines
        only 2P, v2 later pins P down -- the compiler must emit the ring
        re-anchor op `zr` and the protocol must still match the law."""
        sites = CocycleSites(2, self.Z4_DEEP, d=2)
        T = {(4, 3, 1): 2, (4, 3, 2): 1}
        advice = crw4.advice(sites, T)
        assert any(l.startswith('zr') for l in _letters(advice))
        self.sim_check(crw4, sites, T)
        # anchor-first and both-valuation variants (no re-anchor needed)
        self.sim_check(crw4, sites, {(4, 3, 1): 1, (4, 3, 2): 2})
        self.sim_check(crw4, sites, {(4, 3, 1): 2, (4, 3, 2): 2})

    def test_z9(self):
        crw = CocycleRankWidthGroups(3, d=2)
        sites = CocycleSites(3, self.Z4_MIXED, d=2)
        self.sim_check(crw, sites, {(4, 2, 1): 3, (4, 3, 1): 1, (3, 2, 5): 6},
                       rounds=250)
        deep = CocycleSites(3, self.Z4_DEEP, d=2)
        self.sim_check(crw, deep, {(4, 3, 1): 3, (4, 3, 2): 1}, rounds=250)

    def test_random_width1_ring_instances(self, crw4):
        """Fuzz: random site trees with random Z/4 tensors of module
        cut-width one; compiler lemma assertions and the simulator must
        agree with the reference law on all of them."""
        rng = random.Random(11)
        found = 0
        attempts = 0
        while found < 12 and attempts < 600:
            attempts += 1
            shape = random_site_shape(rng, rng.randint(3, 7))
            sites = CocycleSites(2, shape, d=2)
            if len(sites.X) < 2 or not sites.Z or sites.n_sites > 7:
                continue
            triples = [(j, i, v) for i in sites.X for j in sites.X if i < j
                       for v in sites.Z]
            rng.shuffle(triples)
            T = {t: rng.randrange(1, 4) for t in triples[:rng.randint(1, 3)]}
            if sites.cut_width(T) > 1:
                continue
            found += 1
            self.sim_check(crw4, sites, T, rounds=150, seed=attempts)
        assert found >= 8, f"only {found} width-1 ring instances found"

    def test_d1_is_the_field_class(self):
        """d = 1 (default) compiles byte-identical advice to an explicit
        d=1 instance, over the field letter format."""
        sites, T = mixed_instance()
        default = CocycleRankWidthGroups(2)
        explicit = CocycleRankWidthGroups(2, d=1)
        assert default.q == explicit.q == 2
        assert _tree_eq(default.advice(sites, T), explicit.advice(sites, T))
        assert not any(l.startswith('zr')
                       for l in _letters(default.advice(sites, T)))

    def test_guards(self, crw4):
        # width over 1 rejected over the ring
        sites, T = scattered_sites(2, 2)
        ring_sites = CocycleSites(2, sites.shape, d=2)
        with pytest.raises(ValueError):
            crw4.advice(ring_sites, T)
        # p/d mismatch between class and sites
        field_sites, field_T = mixed_instance()
        with pytest.raises(ValueError):
            crw4.advice(field_sites, field_T)
        # cls build gated with a pointer to the implicit path
        with pytest.raises(ValueError, match="check_implicit"):
            crw4.evaluate('M(x,y,z)')

    def test_check_implicit_ring(self, crw4):
        """Functional atoms: FO on ring members without building any
        automaton (the microcode `cls` is far beyond the enumeration cap)."""
        sites = CocycleSites(2, self.Z4_SPINE, d=2)
        T = {(2, 1, 3): 2}
        advice = crw4.advice(sites, T)
        one = sites.identity()
        g = ((1,), (3, 2))
        z = sites.multiply(T, g, g)
        assert crw4.check_implicit('M(x,x,z)', sites, advice, x=g, z=z)
        wrong = (((z[0][0] + 1) % 4,), z[1])
        assert not crw4.check_implicit('M(x,x,z)', sites, advice, x=g,
                                       z=wrong)
        assert crw4.check_implicit('Eq(x,x)', sites, advice, x=g)
        assert crw4.check_implicit('exists y.(M(x,y,u))', sites, advice,
                                   x=g, u=one)
        noncomm = ('exists x.(exists y.(exists z.'
                   '(M(x,y,z) and (not M(y,x,z)))))')
        assert crw4.check_implicit(noncomm, sites, advice)
        assert not crw4.check_implicit(noncomm, sites, crw4.advice(sites, {}))

    def test_check_implicit_field(self, crw2):
        """The implicit path decides field members identically to the
        (validated) simulate/reference law."""
        sites, T = mixed_instance()
        advice = crw2.advice(sites, T)
        g = ((0, 0), (1, 0, 1))
        z = sites.multiply(T, g, g)
        assert crw2.check_implicit('M(x,x,z)', sites, advice, x=g, z=z)
        wb = ((z[0][0] + 1) % 2, z[0][1])
        assert not crw2.check_implicit('M(x,x,z)', sites, advice, x=g,
                                       z=(wb, z[1]))


@heavy
class TestClaimAndVerifyAutomaton:
    """End to end through the real presentation automata, instantiated over
    the sub-alphabet of merge letters that the test instances use. Heavy:
    the micro-op letters act totally on the register states, so the flat
    per-pair transition diagrams are dense -- unlike every other class in
    the package (factored letters are the real fix, see
    paper/theorem3-notes.md)."""

    @pytest.fixture(scope="class")
    def auto(self, crw2):
        instances = []
        sites, T = mixed_instance()
        instances.append(("mixed", sites, T))
        instances.append(("zero", sites, {}))
        lam_sites, lam_T, _ = laminar_sites(2, SPINEY)
        instances.append(("laminar", lam_sites, lam_T))
        merges = set()
        for _, s, t in instances:
            merges |= crw2.used_merge_letters(crw2.advice(s, t))
        crw = CocycleRankWidthGroups(2, merge_letters=merges)
        sta, variables = crw.evaluate('M(x,y,z)')
        return crw, (sta, variables), instances

    def test_multiplication_end_to_end(self, auto):
        """Measured 2026-07-11: passes within a 4G memory cap."""
        crw, (sta, variables), instances = auto
        rng = random.Random(9)
        for name, sites, T in instances:
            advice = crw.advice(sites, T)
            elems = elements(sites)
            for g, h in it.product(elems, repeat=2):
                expected = sites.multiply(T, g, h)
                trees = {'advice': advice,
                         'x': crw.encode(g, sites, advice),
                         'y': crw.encode(h, sites, advice),
                         'z': crw.encode(expected, sites, advice)}
                assert sta.accepts(*[trees[v] for v in variables]), \
                    (name, g, h)
                wb = list(expected[0])
                v = rng.randrange(len(wb))
                wb[v] = (wb[v] + 1) % 2
                trees['z'] = crw.encode((tuple(wb), expected[1]), sites,
                                        advice)
                assert not sta.accepts(*[trees[v] for v in variables]), \
                    (name, g, h)

    def test_nonabelian_uniform(self, auto):
        """One automaton decides commutativity across distributed-center
        members. Measured 2026-07-11: exceeds a 4G memory cap (the nested
        existential projections over the dense per-pair diagrams); needs a
        larger machine or factored transition letters."""
        crw, _, _ = auto
        phi = 'exists x.(exists y.(exists z.(M(x,y,z) and (not M(y,x,z)))))'
        sta, variables = crw.evaluate(phi)
        assert variables == ['advice']
        lam_sites, lam_T, _ = laminar_sites(2, SPINEY)
        mixed_s, mixed_T = mixed_instance()
        cases = [(crw.advice(lam_sites, lam_T), True),
                 (crw.advice(mixed_s, mixed_T), True),
                 (crw.advice(mixed_s, {}), False)]
        for advice, expected in cases:
            assert sta.accepts(advice) == expected
