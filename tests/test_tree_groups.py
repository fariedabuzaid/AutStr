import itertools as it
import os
import random

import pytest

from autstr.sparse_tree_automata import Tree
from autstr.tree_groups import TreeExtraspecialGroups

heavy = pytest.mark.skipif(not os.environ.get('AUTSTR_HEAVY'),
                           reason="exhaustive ring sweep; run with AUTSTR_HEAVY=1")


# ====================================================================
# Reference implementation: the central-extension law (validated
# computationally against the group axioms before the class was built)
# ====================================================================

def addresses(shape: Tree):
    inners, leaves = [], []
    stack = [(shape, '')]
    while stack:
        node, addr = stack.pop()
        if node.left is None and node.right is None:
            leaves.append(addr)
        else:
            inners.append(addr)
            if node.left is not None:
                stack.append((node.left, addr + '0'))
            if node.right is not None:
                stack.append((node.right, addr + '1'))
    return sorted(inners), sorted(leaves)


def law(p, shape, g1, g2):
    """(a1,b1,c1)·(a2,b2,c2) in G_shape."""
    inners, leaves = addresses(shape)
    a1, b1, c1 = g1
    a2, b2, c2 = g2
    a = {w: (a1[w] + a2[w]) % p for w in inners}
    b = {w: (b1[w] + b2[w]) % p for w in inners}
    c = {v: (c1[v] + c2[v] + sum(a1[w] * b2[w] for w in inners
                                 if w != v and v.startswith(w))) % p
         for v in leaves}
    return a, b, c


def random_shape(rng: random.Random, max_size=6) -> Tree:
    def build(budget):
        if budget <= 1 or rng.random() < 0.3:
            return Tree('s'), 1
        used = 1
        left = right = None
        if rng.random() < 0.75:
            left, u = build(budget - used)
            used += u
        if rng.random() < 0.5 and budget - used >= 1:
            right, u = build(budget - used)
            used += u
        return Tree('s', left, right), used
    return build(rng.randint(1, max_size))[0]


def random_element(rng, p, shape):
    inners, leaves = addresses(shape)
    return ({w: rng.randrange(p) for w in inners},
            {w: rng.randrange(p) for w in inners},
            {v: rng.randrange(p) for v in leaves})


@pytest.fixture(scope="module", params=[2, 3])
def grp(request):
    return TreeExtraspecialGroups(request.param, max_states=200_000)


def mul_oracle(grp):
    sta, variables = grp.evaluate('M(x,y,z)')

    def holds(shape, gx, gy, gz):
        trees = {'advice': grp.advice(shape),
                 'x': grp.encode(shape, *gx),
                 'y': grp.encode(shape, *gy),
                 'z': grp.encode(shape, *gz)}
        return sta.accepts(*[trees[v] for v in variables])
    return holds


class TestMultiplication:
    def test_pointwise_against_reference_law(self, grp):
        p = grp.p
        rng = random.Random(p)
        holds = mul_oracle(grp)
        for _ in range(12):
            shape = random_shape(rng)
            inners, leaves = addresses(shape)
            for _ in range(12):
                gx = random_element(rng, p, shape)
                gy = random_element(rng, p, shape)
                gz = law(p, shape, gx, gy)
                assert holds(shape, gx, gy, gz), (shape, gx, gy)
                # perturb one coordinate: must be rejected
                a, b, c = ({**gz[0]}, {**gz[1]}, {**gz[2]})
                if leaves and rng.random() < 0.5:
                    v = rng.choice(leaves)
                    c[v] = (c[v] + rng.randrange(1, p)) % p
                elif inners:
                    w = rng.choice(inners)
                    a[w] = (a[w] + rng.randrange(1, p)) % p
                else:
                    v = leaves[0]
                    c[v] = (c[v] + rng.randrange(1, p)) % p
                assert not holds(shape, gx, gy, (a, b, c)), (shape, gx, gy)

    def test_spine_commutators_hit_all_leaves_below(self, grp):
        """[x_w, y_w] = prod of z_v below w, checked through the automaton
        on a spine and on a bushy shape."""
        p = grp.p
        holds = mul_oracle(grp)
        shapes = [TreeExtraspecialGroups.spine(3),
                  Tree('s', Tree('s', Tree('s'), Tree('s')), Tree('s'))]
        for shape in shapes:
            inners, leaves = addresses(shape)
            zero = ({w: 0 for w in inners}, {w: 0 for w in inners},
                    {v: 0 for v in leaves})
            for w in inners:
                x_w = ({**zero[0], w: 1}, dict(zero[1]), dict(zero[2]))
                y_w = (dict(zero[0]), {**zero[1], w: 1}, dict(zero[2]))
                # commutator [x,y] = (xy)(x^-1 y^-1) by the reference law
                t = law(p, shape, x_w, y_w)

                def inverse(g):
                    a, b, c = g
                    na = {k: -v % p for k, v in a.items()}
                    nb = {k: -v % p for k, v in b.items()}
                    nc = {k: -v % p for k, v in c.items()}
                    for wi in inners:
                        for v in leaves:
                            if wi != v and v.startswith(wi):
                                nc[v] = (nc[v] + a[wi] * b[wi]) % p
                    return na, nb, nc
                comm = law(p, shape, t, law(p, shape, inverse(x_w),
                                            inverse(y_w)))
                want = {v: (1 if (v.startswith(w) and v != w) else 0)
                        for v in leaves}
                assert comm == (zero[0], zero[1], want), (shape, w)
                # and the automaton agrees on the defining products
                assert holds(shape, x_w, y_w, t)
                assert holds(shape, t, law(p, shape, inverse(x_w),
                                           inverse(y_w)), comm)


IDENTITY_AND_INVERSES = ('exists u.(all x.(M(x,u,x)) and '
                         'all x.(exists y.(M(x,y,u))))')
COMMUTATIVE = 'all x.(all y.(all z.(M(x,y,z) -> M(y,x,z))))'
SQUARES_TRIVIAL = 'exists u.(all x.(M(x,u,x)) and all x.(M(x,x,u)))'
FUNCTIONAL = ('all x.(all y.(all z.(all w.'
              '((M(x,y,z) and M(x,y,w)) -> E(z,w)))))')


class TestSentences:
    def test_group_axioms(self, grp):
        rng = random.Random(7)
        sta, _ = grp.evaluate(IDENTITY_AND_INVERSES)
        for _ in range(5):
            shape = random_shape(rng)
            assert sta.accepts(grp.advice(shape)), shape

    def test_commutative_iff_no_inner_node(self, grp):
        rng = random.Random(8)
        sta, _ = grp.evaluate(COMMUTATIVE)
        shapes = [Tree('s'), TreeExtraspecialGroups.spine(1),
                  TreeExtraspecialGroups.spine(2)]
        shapes += [random_shape(rng) for _ in range(4)]
        for shape in shapes:
            inners, _ = addresses(shape)
            want = len(inners) == 0            # single node: G = Z_p
            assert sta.accepts(grp.advice(shape)) == want, shape

    def test_exponent(self, grp):
        """p odd: exponent p (squaring... cubing is trivial); p = 2: the
        extraspecial 2-groups have exponent 4, so squares are trivial only
        in the abelian (single-node) case."""
        sta, _ = grp.evaluate(SQUARES_TRIVIAL)
        single, spine2 = Tree('s'), TreeExtraspecialGroups.spine(2)
        if grp.p == 2:
            assert sta.accepts(grp.advice(single))        # Z_2
            assert not sta.accepts(grp.advice(spine2))    # exponent 4
        else:
            # doubling is invertible mod an odd p: x^2 = e only at e
            assert not sta.accepts(grp.advice(single))
            assert not sta.accepts(grp.advice(spine2))

    def test_functionality(self, grp):
        """Arity 5: p = 3 means 14**5 = 537824 flat symbols, but a diagram
        only tests the digits its transition depends on."""
        sta, _ = grp.evaluate(FUNCTIONAL)
        assert sta.accepts(grp.advice(TreeExtraspecialGroups.spine(2)))
        assert sta.accepts(grp.advice(Tree('s', Tree('s'), Tree('s'))))

    def test_get_structure(self, grp):
        S = grp.get_structure(TreeExtraspecialGroups.spine(2))
        assert S.check('exists u.(all x.(M(x,u,x)))')
        assert not S.check('all x.(all y.(all z.(M(x,y,z) -> M(y,x,z))))')


# ====================================================================
# CutRankTreeGroups: tree layouts of bounded cut-rank
# ====================================================================

from autstr.groups import CutRankGroups
from autstr.tree_groups import CutRankTreeGroups


@pytest.fixture(scope="module")
def crt2():
    return CutRankTreeGroups(2)


@pytest.fixture(scope="module")
def crt2_mult(crt2):
    sta, variables = crt2.evaluate('M(x,y,z)')
    assert set(variables) == {'advice', 'x', 'y', 'z'}
    return sta, variables


class TestCutRankTreeGroups:
    def elements(self, crt, n):
        return [(b, a)
                for b in it.product(range(crt.p), repeat=crt.k)
                for a in it.product(range(crt.p), repeat=n)]

    def holds(self, crt, oracle, advice, gx, gy, gz):
        sta, variables = oracle
        trees = {'advice': advice, 'x': crt.encode(gx, advice),
                 'y': crt.encode(gy, advice), 'z': crt.encode(gz, advice)}
        return sta.accepts(*[trees[v] for v in variables])

    def test_multiplication_automaton_exhaustive(self, crt2, crt2_mult):
        """M agrees with the reference law on every product, on spine and
        balanced layouts, for forms of tree-cut-rank <= 1."""
        n = 4
        star = {(j, 1): (1,) for j in range(2, n + 1)}
        cases = [(crt2.spine(n), {}), (crt2.spine(n), crt2.clique_form(n)),
                 (crt2.spine(n), crt2.matching_form(n)),
                 (crt2.balanced(n), crt2.clique_form(n)),
                 (crt2.balanced(n), crt2.matching_form(n)),
                 (crt2.balanced(n), star)]
        elems = self.elements(crt2, n)
        for shape, form in cases:
            advice = crt2.advice(shape, form)
            for g, h in it.product(elems, repeat=2):
                expected = crt2.multiply(n, form, g, h)
                assert self.holds(crt2, crt2_mult, advice, g, h, expected), \
                    (form, g, h)
                wrong = (((expected[0][0] + 1) % 2,), expected[1])
                assert not self.holds(crt2, crt2_mult, advice, g, h, wrong), \
                    (form, g, h)

    def test_cross_engine_agreement(self, crt2, crt2_mult):
        """A spine layout is the word class: both engines decide the same
        products for the same form."""
        rng = random.Random(3)
        n = 5
        crg = CutRankGroups(2)
        form = crt2.clique_form(n)
        word_advice = crg.advice(n, form)
        tree_advice = crt2.advice(crt2.spine(n), form)
        word_dfa, word_vars = crg.evaluate('M(x,y,z)')
        for _ in range(60):
            g = ((rng.randrange(2),), tuple(rng.randrange(2) for _ in range(n)))
            h = ((rng.randrange(2),), tuple(rng.randrange(2) for _ in range(n)))
            expected = crt2.multiply(n, form, g, h)
            for z in (expected, (((expected[0][0] + 1) % 2,), expected[1])):
                columns = {'advice': word_advice, 'x': crg.encode(g, n),
                           'y': crg.encode(h, n), 'z': crg.encode(z, n)}
                word_word = [tuple(columns[v][i] for v in word_vars)
                             for i in range(len(word_advice))]
                assert (word_dfa.accepts(word_word)
                        == self.holds(crt2, crt2_mult, tree_advice, g, h, z))

    def test_clique_scales_on_balanced(self, crt2, crt2_mult):
        """n = 14 on a balanced layout: pathwidth n-1, tree-cut-rank 1."""
        rng = random.Random(5)
        n = 14
        form = crt2.clique_form(n)
        advice = crt2.advice(crt2.balanced(n), form)
        for _ in range(50):
            g = ((rng.randrange(2),), tuple(rng.randrange(2) for _ in range(n)))
            h = ((rng.randrange(2),), tuple(rng.randrange(2) for _ in range(n)))
            expected = crt2.multiply(n, form, g, h)
            assert self.holds(crt2, crt2_mult, advice, g, h, expected)
            wrong = (((expected[0][0] + 1) % 2,), expected[1])
            assert not self.holds(crt2, crt2_mult, advice, g, h, wrong)

    def test_pauli_triangle_balanced(self, crt2):
        """The 1-qubit Pauli group on a genuinely branching layout."""
        n = 3
        form = crt2.clique_form(n)
        advice = crt2.advice(crt2.balanced(n), form)
        x1 = ((0,), (1, 0, 0))
        g = ((0,), (1, 1, 1))
        y = ((1,), (0, 0, 0))
        gg = crt2.multiply(n, form, g, g)
        assert gg == y
        assert crt2.check('M(x,x,z)', advice, x=g, z=y)
        assert not crt2.check('M(x,x,z)', advice, x=g, z=crt2.identity(n))
        central = ('all u.(all v.(all w.((not M(x,u,v)) or (not M(u,x,w)) '
                   'or Eq(v,w))))')
        assert crt2.check(central, advice, x=g)
        assert crt2.check(central, advice, x=y)
        assert not crt2.check(central, advice, x=x1)

    def test_width_guard(self, crt2):
        """The interleaved matching splits across the balanced layout's
        first subtree: tree-cut-rank 2, rejected at r = 1; r = 2 exceeds
        the flat-alphabet budget and switches to factored letters (forcing
        flat still raises)."""
        n = 4
        form = {(3, 1): (1,), (4, 2): (1,)}
        shape = crt2.balanced(n)
        assert crt2.tree_cut_rank(shape, form) == 2
        with pytest.raises(ValueError):
            crt2.advice(shape, form)
        assert CutRankTreeGroups(2, r=2).factored      # auto-factored now
        with pytest.raises(ValueError):
            CutRankTreeGroups(2, r=2, factored=False)

    def test_tree_cut_rank(self, crt2):
        assert crt2.tree_cut_rank(crt2.balanced(7), crt2.clique_form(7)) == 1
        assert crt2.tree_cut_rank(crt2.spine(6), crt2.matching_form(6)) == 1
        assert crt2.tree_cut_rank(crt2.balanced(4), crt2.matching_form(4)) == 1
        assert crt2.tree_cut_rank(crt2.spine(3), {}) == 0
        star = {(j, 1): (1,) for j in range(2, 8)}
        assert crt2.tree_cut_rank(crt2.balanced(7), star) == 1

    def test_nonabelian_uniform(self, crt2):
        """One tree automaton decides commutativity across all layouts."""
        phi = 'exists x.(exists y.(exists z.(M(x,y,z) and (not M(y,x,z)))))'
        sta, variables = crt2.evaluate(phi)
        assert variables == ['advice']
        cases = [(crt2.advice(crt2.spine(3), {}), False),
                 (crt2.advice(crt2.balanced(3), crt2.clique_form(3)), True),
                 (crt2.advice(crt2.spine(4), crt2.matching_form(4)), True)]
        for advice, expected in cases:
            assert sta.accepts(advice) == expected

    def test_validation(self, crt2):
        with pytest.raises(ValueError):
            CutRankTreeGroups(4)
        with pytest.raises(ValueError):
            crt2.advice(crt2.spine(3), {(1, 2): (1,)})
        with pytest.raises(ValueError):
            crt2.encode(((0,), (0, 0)), crt2.spine(3))  # wrong length
        with pytest.raises(ValueError):
            crt2.encode(((0,), (2, 0, 0)), crt2.spine(3))  # bad digit


class TestCutRankTreeGroupsChainRing:
    """The exponent-p^d ("Idea 2") tree layout: the saturated tree merge over
    the chain ring R = Z/p^d, whose crux is the two-sided sibling factorisation
    (chain_ring.factor_two_sided)."""

    def elements(self, crt, shape):
        seq, _, _ = crt._layout(shape)
        n = len(seq)
        return [((b,), a) for b in range(crt.q)
                for a in it.product(range(crt.q), repeat=n)] if crt.k == 1 else [
            (b, a) for b in it.product(range(crt.q), repeat=crt.k)
            for a in it.product(range(crt.q), repeat=n)]

    def cases(self, crt, n):
        star = {(j, 1): (1,) for j in range(2, n + 1)}
        val1 = {(j, i): (2,) for j in range(2, n + 1) for i in range(1, j)}  # valuation 1
        return [(crt.spine(n), crt.clique_form(n)),
                (crt.spine(n), crt.matching_form(n)),
                (crt.balanced(n), crt.clique_form(n)),
                (crt.balanced(n), crt.matching_form(n)),
                (crt.balanced(n), star),
                (crt.balanced(n), val1)]         # exercises factor_two_sided

    def test_d1_encoding_is_unchanged(self):
        from autstr.tree_groups import CutRankTreeGroups
        c = CutRankTreeGroups(2)
        assert c.q == 2 and c.d == 1
        assert c.digits == ['0', '1']
        assert c.leaf_letters and all(name[0] == 'a' for name in c.leaf_letters)

    def test_simulate_matches_built_automaton_field(self):
        """At d = 1 the tree automaton is buildable; simulate reproduces it
        exactly, tying the transition-level check to the real DFA."""
        from autstr.tree_groups import CutRankTreeGroups
        c = CutRankTreeGroups(2)
        M = c.cls.class_automata['M']
        variables = ['advice', 'x', 'y', 'z']    # raw M tape order
        n = 4
        rng = random.Random(1)
        for shape, form in self.cases(c, n):
            advice = c.advice(shape, form)
            elems = self.elements(c, shape)
            for _ in range(60):
                g, h = rng.choice(elems), rng.choice(elems)
                z = c.multiply(n, form, g, h)
                tx, ty, tz = (c.encode(e, advice) for e in (g, h, z))
                tapes = {'advice': advice, 'x': tx, 'y': ty, 'z': tz}
                word = _convolve_trees(variables, tapes)
                assert M.accepts(word) == c.simulate(advice, g, h, z)
                assert c.simulate(advice, g, h, z)

    def test_reference_law_group_axioms_z4(self):
        from autstr.tree_groups import CutRankTreeGroups
        c = CutRankTreeGroups(2, d=2)
        shape = c.balanced(3)
        seq, _, _ = c._layout(shape); n = len(seq)
        form = c.clique_form(n)
        elems = self.elements(c, shape)
        mul = lambda g, h: c.multiply(n, form, g, h)
        one = c.identity(n)
        rng = random.Random(3)
        for g in elems:
            assert mul(one, g) == g and mul(g, one) == g
        for _ in range(6000):
            g, h, f = (rng.choice(elems) for _ in range(3))
            assert mul(mul(g, h), f) == mul(g, mul(h, f))

    def test_tree_merge_matches_reference_z4(self):
        """The saturated streaming compile agrees with the reference law over
        Z/4 on spine and balanced layouts, including a valuation-1 form whose
        sibling blocks force the ring two-sided factorisation."""
        from autstr.tree_groups import CutRankTreeGroups
        c = CutRankTreeGroups(2, d=2)
        n = 4
        rng = random.Random(5)
        for shape, form in self.cases(c, n):
            advice = c.advice(shape, form)
            elems = self.elements(c, shape)
            for _ in range(250):
                g, h = rng.choice(elems), rng.choice(elems)
                z = c.multiply(n, form, g, h)
                assert c.simulate(advice, g, h, z), (form, g, h)
                wrong = (((z[0][0] + 1) % c.q,), z[1])
                assert not c.simulate(advice, g, h, wrong)

    def test_tree_cut_rank_counts_valuation_over_z4(self):
        from autstr.tree_groups import CutRankTreeGroups
        c = CutRankTreeGroups(2, d=2)
        n = 4
        val1 = {(j, i): (2,) for j in range(2, n + 1) for i in range(1, j)}
        assert c.tree_cut_rank(c.balanced(n), val1) == 1     # not 0
        assert c.tree_cut_rank(c.balanced(n), c.clique_form(n)) == 1
        assert c.tree_cut_rank(c.balanced(n), {}) == 0

    def test_width_guard_over_z4(self):
        from autstr.tree_groups import CutRankTreeGroups
        c = CutRankTreeGroups(2, d=2)
        # a spine layout whose middle cut needs rank 2
        form = {(2, 1): (1,), (4, 3): (1,), (4, 1): (1,), (3, 2): (1,)}
        if c.tree_cut_rank(c.spine(4), form) > 1:
            with pytest.raises(ValueError):
                c.advice(c.spine(4), form)

    def test_encode_range_is_the_ring(self):
        from autstr.tree_groups import CutRankTreeGroups
        c = CutRankTreeGroups(2, d=2)
        c.encode(((3,), (2, 1)), c.spine(2))
        with pytest.raises(ValueError):
            c.encode(((4,), (0, 0)), c.spine(2))

    def test_cls_guard_raises_instead_of_hanging(self):
        """The Z/4 tree member's product automaton is infeasible: check/
        evaluate/get_structure raise a clear error pointing at
        check_implicit/simulate instead of hanging in the lazy build."""
        from autstr.tree_groups import CutRankTreeGroups
        c = CutRankTreeGroups(2, d=2)
        advice = c.advice(c.spine(2), {})
        with pytest.raises(ValueError, match="check_implicit"):
            c.evaluate('M(x,y,z)')
        with pytest.raises(ValueError, match="check_implicit"):
            c.get_structure(advice)
        with pytest.raises(ValueError, match="check_implicit"):
            c.check('Eq(x,x)', advice, x=c.identity(2))
        # the implicit paths still decide the same member
        assert c.check_implicit('Eq(x,x)', advice, x=c.identity(2))
        z = c.multiply(2, {}, c.identity(2), c.identity(2))
        assert c.simulate(advice, c.identity(2), c.identity(2), z)
        # the field alphabet still builds
        assert CutRankTreeGroups(2).cls is not None

    def test_factored_letters_lift_the_width_cap(self):
        """Factored tree letters (bare 'a'/'b'/'d' markers + one letter per
        ring entry): width r = 2 over Z/4 -- far beyond the flat cap --
        compiles on spine and balanced layouts and matches the reference
        law, including a balanced(6) form whose sibling block itself has
        module rank 2 (the two-sided factorisation with r = 2 interfaces)."""
        from autstr.tree_groups import CutRankTreeGroups
        crt = CutRankTreeGroups(2, r=2, d=2)
        assert crt.factored and len(crt.advice_letters) == crt.q + 4
        rng = random.Random(23)

        def sweep(shape, form, rounds=150):
            seq, _, _ = crt._layout(shape)
            n = len(seq)
            advice = crt.advice(shape, form)
            for _ in range(rounds):
                g = ((rng.randrange(4),),
                     tuple(rng.randrange(4) for _ in range(n)))
                h = ((rng.randrange(4),),
                     tuple(rng.randrange(4) for _ in range(n)))
                z = crt.multiply(n, form, g, h)
                assert crt.simulate(advice, g, h, z), (form, g, h)
                wrong = (((z[0][0] + 1) % 4,), z[1])
                assert not crt.simulate(advice, g, h, wrong), (form, g, h)

        form = {(3, 1): (1,), (4, 2): (1,), (2, 1): (2,)}
        assert crt.tree_cut_rank(crt.balanced(4), form) == 2
        sweep(crt.balanced(4), form)
        sweep(crt.spine(4), form)
        # balanced(6): left subtree {1,2,3}, right {4,5} -- the sibling
        # block of the root carries two independent pairs (rank 2)
        form6 = {(4, 1): (1,), (5, 2): (2,), (5, 3): (1,)}
        shape6 = crt.balanced(6)
        assert crt.tree_cut_rank(shape6, form6) == 2
        sweep(shape6, form6, rounds=120)

    def test_factored_agrees_with_flat(self):
        """Where both encodings exist (the field), factored simulate decides
        exactly as flat simulate; d > 1 forces factored letters (the ring
        merge carries a pairing table, which has no flat encoding)."""
        from autstr.tree_groups import CutRankTreeGroups
        rng = random.Random(29)
        flat = CutRankTreeGroups(2)
        fac = CutRankTreeGroups(2, factored=True)
        assert not flat.factored and fac.factored
        n = 4
        for shape in (flat.spine(n), flat.balanced(n)):
            for form in (flat.clique_form(n), flat.matching_form(n)):
                a_flat = flat.advice(shape, form)
                a_fac = fac.advice(shape, form)
                for _ in range(80):
                    g = ((rng.randrange(2),),
                         tuple(rng.randrange(2) for _ in range(n)))
                    h = ((rng.randrange(2),),
                         tuple(rng.randrange(2) for _ in range(n)))
                    z = rng.choice([
                        flat.multiply(n, form, g, h),
                        ((rng.randrange(2),),
                         tuple((x + y) % 2 for x, y in zip(g[1], h[1])))])
                    assert flat.simulate(a_flat, g, h, z) \
                        == fac.simulate(a_fac, g, h, z), (form, g, h, z)
        assert CutRankTreeGroups(2, d=2).factored
        with pytest.raises(ValueError):
            CutRankTreeGroups(2, d=2, factored=False)

    def test_factored_check_implicit(self):
        """FO on a factored ring width-2 tree member via the implicit path."""
        from autstr.tree_groups import CutRankTreeGroups
        crt = CutRankTreeGroups(2, r=2, d=2)
        form = {(3, 1): (1,), (4, 2): (1,), (2, 1): (2,)}
        advice = crt.advice(crt.balanced(4), form)
        one = crt.identity(4)
        g = ((3,), (1, 0, 2, 3))
        z = crt.multiply(4, form, g, g)
        assert crt.check_implicit('M(x,x,z)', advice, x=g, z=z)
        wrong = (((z[0][0] + 1) % 4,), z[1])
        assert not crt.check_implicit('M(x,x,z)', advice, x=g, z=wrong)
        assert crt.check_implicit('Eq(x,x)', advice, x=g)
        assert crt.check_implicit('exists y.(M(x,y,u))', advice, x=g, u=one)

    def test_ring_interfaces_and_pairing_tables(self):
        """Regression for the ring-interface finding: over Z/p^d the
        interface must generate the row module (pure closures do not nest
        under restriction) and the sibling merge must carry a pairing
        table (the block need not factor as a Q matrix through valuation-
        carrying interfaces). Both failure modes, previously fatal, now
        compile and match the reference law."""
        from autstr.tree_groups import CutRankTreeGroups
        crt = CutRankTreeGroups(2, d=2)
        rng = random.Random(11)
        cases = [
            # restriction failure (saturated interfaces): fuzz-found forms
            (Tree('s', Tree('s', Tree('s', Tree('s'), None), None), None),
             {(4, 2): (1,), (5, 2): (2,), (4, 3): (2,)}),
            # merge failure (row-module interfaces without a table): the
            # scratch counterexample B_{31} = 2 on ({1,2})({3,4})
            (Tree('s', Tree('s', Tree('s'), None), Tree('s', Tree('s'), None)),
             {(3, 1): (2,)}),
        ]
        for shape, raw in cases:
            seq, _, _ = crt._layout(shape)
            n = len(seq)
            form = {k_: v for k_, v in raw.items() if k_[0] <= n}
            assert crt.tree_cut_rank(shape, form) <= 1
            advice = crt.advice(shape, form)
            for _ in range(150):
                g = ((rng.randrange(4),),
                     tuple(rng.randrange(4) for _ in range(n)))
                h = ((rng.randrange(4),),
                     tuple(rng.randrange(4) for _ in range(n)))
                z = crt.multiply(n, form, g, h)
                assert crt.simulate(advice, g, h, z), (form, g, h)
                wrong = (((z[0][0] + 1) % 4,), z[1])
                assert not crt.simulate(advice, g, h, wrong), (form, g, h)

    @heavy
    def test_tree_merge_exhaustive_z4(self):
        """Exhaustive streaming-vs-reference over Z/4 on every layout/form."""
        from autstr.tree_groups import CutRankTreeGroups
        c = CutRankTreeGroups(2, d=2)
        n = 4
        for shape, form in self.cases(c, n):
            advice = c.advice(shape, form)
            elems = self.elements(c, shape)
            for g, h in it.product(elems, repeat=2):
                z = c.multiply(n, form, g, h)
                assert c.simulate(advice, g, h, z), (form, g, h)
                wrong = (((z[0][0] + 1) % c.q,), z[1])
                assert not c.simulate(advice, g, h, wrong)


def _convolve_trees(variables, tapes):
    """Convolve equally-shaped labelled trees into a single symbol-tuple tree
    (order given by `variables`)."""
    def rec(nodes):
        node0 = next(n for n in nodes if n is not None)
        label = tuple(n.label for n in nodes)
        left = rec([n.left for n in nodes]) if node0.left is not None else None
        right = rec([n.right for n in nodes]) if node0.right is not None else None
        return Tree(label, left, right)
    return rec([tapes[v] for v in variables])
