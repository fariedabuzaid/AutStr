"""Implicit first-order evaluation (autstr.implicit).

Three layers: the combinators agree with the built automata; `check_implicit`
agrees with the explicit `check` on small buildable classes; and the functional
path decides formulas on the large-alphabet ring members whose query/base
automata cannot be built, matching the reference law.
"""
import itertools as it
import random
import re

import pytest
from nltk.sem import logic

from autstr.groups import CutRankGroups
from autstr.tree_groups import CutRankTreeGroups
from autstr import implicit as im


BATTERY = [
    'M(x,y,z)', 'Eq(x,y)', 'not Eq(x,y)',
    'exists z.(M(x,y,z))',
    'exists x.(exists y.(exists z.(M(x,y,z) and (not M(y,x,z)))))',  # nonabelian
    'all x.(all y.(all z.((M(x,y,z)) -> (M(y,x,z)))))',              # commutative
    'exists u.(all x.(M(x,u,x)))',                                  # identity exists
    'all u.((all x.(M(x,u,x))) -> Eq(u,x))',                        # free x, bound u
]


def free_vars(phi):
    bound = set(re.findall(r'(?:all|exists) (\w+)', phi))
    used = set(v.strip() for g in re.findall(r'[MEq]\w*\(([^)]*)\)', phi)
                for v in g.split(','))
    return sorted(used - bound - {''})


def word_elements(c, n):
    return [((b,), a) for b in range(c.q)
            for a in it.product(range(c.q), repeat=n)]


# ---------------------------------------------- core combinators vs automata

class TestCombinatorsMatchAutomata:
    def test_string_atom_and_projection(self):
        c = CutRankGroups(2); n = 3; form = c.clique_form(n)
        advice = c.advice(n, form); M = c.cls.class_automata['M']
        Eq = c.cls.class_automata['Eq']
        elems = word_elements(c, n); enc = {g: c.encode(g, n) for g in elems}
        alpha = lambda v: c.digits
        rng = random.Random(0)
        phi = logic.Expression.fromstring('M(a,x,y,z)')
        phi3 = logic.Expression.fromstring('exists z.(M(a,x,y,z) and Eq(a,z,w))')
        for _ in range(200):
            g, h = rng.choice(elems), rng.choice(elems)
            z = c.multiply(n, form, g, h)
            inp = {'a': advice, 'x': enc[g], 'y': enc[h], 'z': enc[z]}
            assert im.check_string(phi, {'M': M}, inp, len(advice), alpha)
            wrong = (((z[0][0] + 1) % 2,), z[1])
            assert not im.check_string(phi, {'M': M},
                                       dict(inp, z=c.encode(wrong, n)),
                                       len(advice), alpha)
            # exists z.(M and Eq(z,w)) is true iff w == g*h
            inpw = {'a': advice, 'x': enc[g], 'y': enc[h], 'w': enc[z]}
            assert im.check_string(phi3, {'M': M, 'Eq': Eq}, inpw, len(advice), alpha)
            assert not im.check_string(phi3, {'M': M, 'Eq': Eq},
                                       dict(inpw, w=c.encode(wrong, n)),
                                       len(advice), alpha)

    def test_tree_atom_and_projection(self):
        c = CutRankTreeGroups(2); n = 4; shape = c.balanced(n)
        form = c.clique_form(n); advice = c.advice(shape, form)
        M = c.cls.class_automata['M']; Eq = c.cls.class_automata['Eq']
        elems = word_elements(c, n); enc = {g: c.encode(g, advice) for g in elems}
        alpha = lambda v: c.digits
        rng = random.Random(0)
        phi = logic.Expression.fromstring('M(a,x,y,z)')
        phi3 = logic.Expression.fromstring('exists z.(M(a,x,y,z) and Eq(a,z,w))')
        for _ in range(120):
            g, h = rng.choice(elems), rng.choice(elems)
            z = c.multiply(n, form, g, h)
            inp = {'a': advice, 'x': enc[g], 'y': enc[h], 'z': enc[z]}
            assert im.check_tree(phi, {'M': M}, inp, alpha)
            wrong = (((z[0][0] + 1) % 2,), z[1])
            assert not im.check_tree(phi, {'M': M},
                                     dict(inp, z=c.encode(wrong, advice)), alpha)
            inpw = {'a': advice, 'x': enc[g], 'y': enc[h], 'w': enc[z]}
            assert im.check_tree(phi3, {'M': M, 'Eq': Eq}, inpw, alpha)
            assert not im.check_tree(phi3, {'M': M, 'Eq': Eq},
                                     dict(inpw, w=c.encode(wrong, advice)), alpha)


# ---------------------------------------------- check_implicit vs explicit

class TestImplicitMatchesExplicit:
    def test_word_field(self):
        c = CutRankGroups(2); n = 3
        elems = word_elements(c, n)
        advices = [c.advice(n, {}), c.advice(n, c.clique_form(n)),
                   c.advice(n, c.matching_form(n))]
        rng = random.Random(7); checks = 0
        for adv in advices:
            for phi in BATTERY:
                for _ in range(6):
                    asn = {v: rng.choice(elems) for v in free_vars(phi)}
                    assert c.check(phi, adv, **asn) == c.check_implicit(phi, adv, **asn), \
                        (phi, asn)
                    checks += 1
        assert checks > 0

    def test_tree_field(self):
        c = CutRankTreeGroups(2); n = 3; shape = c.balanced(n)
        elems = word_elements(c, n)
        advices = [c.advice(shape, {}), c.advice(shape, c.clique_form(n)),
                   c.advice(shape, c.matching_form(n))]
        rng = random.Random(8)
        for adv in advices:
            for phi in BATTERY:
                for _ in range(6):
                    asn = {v: rng.choice(elems) for v in free_vars(phi)}
                    assert c.check(phi, adv, **asn) == c.check_implicit(phi, adv, **asn), \
                        (phi, asn)


# ---------------------------------------------- satisfying sets

class TestEvaluateImplicit:
    """The satisfying-set primitive: exact counts without enumeration, lazy
    enumeration matching brute force, on buildable and unbuildable members."""

    def test_word_solutions_match_brute_force(self):
        c = CutRankGroups(2); n = 3; form = c.clique_form(n)
        adv = c.advice(n, form)
        elems = word_elements(c, n)
        g, h = ((1,), (1, 0, 1)), ((0,), (0, 1, 1))
        z = c.multiply(n, form, g, h)
        sols = c.evaluate_implicit('M(x,y,z)', adv, x=g, y=h)
        assert len(sols) == 1 and list(sols) == [{'z': z}]
        # centralizer of g, brute-forced
        cent = c.evaluate_implicit('exists z.(M(x,y,z) and M(y,x,z))',
                                   adv, x=g)
        brute = sorted(e for e in elems
                       if c.multiply(n, form, g, e) == c.multiply(n, form, e, g))
        assert sorted(s['y'] for s in cent) == brute
        assert len(cent) == len(brute)          # count without enumeration
        # the whole domain, and two open variables (inverse pairs)
        assert len(c.evaluate_implicit('Eq(x,x)', adv)) == len(elems)
        one = c.identity(n)
        pairs = c.evaluate_implicit('M(x,y,u)', adv, u=one)
        assert len(pairs) == len(elems)
        assert all(c.multiply(n, form, s['x'], s['y']) == one for s in pairs)

    def test_word_heavy_ring_member(self):
        """Solution sets on Z/8 and factored width-2 Z/4 members, whose
        automata cannot be built."""
        c8 = CutRankGroups(2, d=3)
        form = c8.clique_form(3)
        adv = c8.advice(3, form)
        g = ((5,), (3, 1, 6))
        inv = c8.evaluate_implicit('M(x,y,u)', adv, x=g, u=c8.identity(3))
        assert len(inv) == 1
        assert c8.multiply(3, form, g, next(iter(inv))['y']) == c8.identity(3)
        cf = CutRankGroups(2, r=2, d=2)
        formf = {(3, 1): (1,), (4, 2): (1,), (2, 1): (2,)}
        advf = cf.advice(4, formf)
        gf = ((3,), (1, 0, 2, 3))
        invf = cf.evaluate_implicit('M(x,y,u)', advf, x=gf, u=cf.identity(4))
        assert len(invf) == 1
        assert cf.multiply(4, formf, gf, next(iter(invf))['y']) \
            == cf.identity(4)

    def test_tree_solutions_and_decode(self):
        c = CutRankTreeGroups(2, d=2)
        shape = c.balanced(4); form = c.clique_form(4)
        adv = c.advice(shape, form)
        g = ((3,), (1, 2, 0, 3))
        assert c.decode(c.encode(g, adv), adv) == g
        inv = c.evaluate_implicit('M(x,y,u)', adv, x=g, u=c.identity(4))
        assert len(inv) == 1
        assert c.multiply(4, form, g, next(iter(inv))['y']) == c.identity(4)
        assert len(c.evaluate_implicit('Eq(x,x)', adv)) == 4 ** 5
        # factored decode roundtrip
        cf = CutRankTreeGroups(2, r=2, d=2)
        advf = cf.advice(cf.balanced(4), {(3, 1): (1,), (4, 2): (1,)})
        assert cf.decode(cf.encode(g, advf), advf) == g

    def test_tree_solutions_match_brute_force(self):
        c = CutRankTreeGroups(2); n = 3; shape = c.balanced(n)
        form = c.matching_form(n)
        adv = c.advice(shape, form)
        elems = word_elements(c, n)
        g = ((1,), (1, 0, 1))
        cent = c.evaluate_implicit('exists z.(M(x,y,z) and M(y,x,z))',
                                   adv, x=g)
        brute = sorted(e for e in elems
                       if c.multiply(n, form, g, e) == c.multiply(n, form, e, g))
        assert sorted(s['y'] for s in cent) == brute

    def test_cocycle_ring_member(self):
        from autstr.cocycle_groups import CocycleRankWidthGroups, CocycleSites
        from autstr.sparse_tree_automata import Tree
        crw = CocycleRankWidthGroups(2, d=2)
        sites = CocycleSites(2, Tree('z', Tree('x', Tree('x'), None), None),
                             d=2)
        T = {(2, 1, 3): 2}
        adv = crw.advice(sites, T)
        g = ((1,), (3, 2))
        sq = sites.multiply(T, g, g)
        sols = crw.evaluate_implicit('M(x,x,z)', sites, adv, x=g)
        assert len(sols) == 1 and list(sols) == [{'z': sq}]
        assert len(crw.evaluate_implicit('Eq(x,x)', sites, adv)) == 4 ** 3

    def test_implicit_class_is_first_class(self):
        """`ImplicitClass`/`ImplicitTreeClass`: a presentation given purely
        functionally, checked and evaluated without compiling anything."""
        c8 = CutRankGroups(2, d=3)
        adv = c8.advice(3, c8.clique_form(3))
        icls = im.ImplicitClass(c8._implicit_atoms(), list(c8.digits))
        g = ((5,), (3, 1, 6))
        w = c8.encode(g, 3)
        assert icls.check('Eq(x,x)', adv, x=w)
        assert icls.check(
            'exists x.(exists y.(exists z.(M(x,y,z) and (not M(y,x,z)))))',
            adv)
        raw = icls.evaluate('Eq(x,x)', adv, x=w)
        assert len(raw) == 1 and list(raw) == [{}]
        c4 = CutRankTreeGroups(2, d=2)
        advt = c4.advice(c4.balanced(3), c4.clique_form(3))
        tcls = im.ImplicitTreeClass(c4._implicit_atoms(), list(c4.digits))
        gt = ((3,), (1, 2, 0))
        assert tcls.check('Eq(x,x)', advt, x=c4.encode(gt, advt))
        assert len(tcls.evaluate('Eq(x,x)', advt)) == 4 ** 4

    def test_uniform_class_raw_solutions(self):
        """`evaluate_implicit` on the uniform class layer yields raw element
        words."""
        c = CutRankGroups(2); n = 3; form = c.clique_form(n)
        adv = c.advice(n, form)
        g, h = ((1,), (1, 0, 1)), ((0,), (0, 1, 1))
        z = c.multiply(n, form, g, h)
        raw = c.cls.evaluate_implicit('M(x,y,z)', adv,
                                      x=c.encode(g, n), y=c.encode(h, n))
        assert len(raw) == 1 and list(raw)[0]['z'] == c.encode(z, n)


# ------------------------------------- the functional path on heavy members

class TestFunctionalReachesHeavyMembers:
    """The query/base automata are infeasible for these rings; only the
    functional implicit evaluator can decide the formulas."""

    @pytest.mark.parametrize("p,d", [(2, 3), (3, 2)])   # Z/8, Z/9
    def test_word_multiplication_and_sentence(self, p, d):
        c = CutRankGroups(p, d=d); n = 3; q = c.q
        form = c.clique_form(n); adv = c.advice(n, form)
        elems = word_elements(c, n); rng = random.Random(p)
        for _ in range(120):
            g, h = rng.choice(elems), rng.choice(elems)
            z = c.multiply(n, form, g, h)
            assert c.check_implicit('M(x,y,z)', adv, x=g, y=h, z=z)
            assert c.check_implicit('M(x,y,z)', adv, x=g, y=h, z=z) == c.simulate(adv, g, h, z)
            wrong = (((z[0][0] + 1) % q,), z[1])
            assert not c.check_implicit('M(x,y,z)', adv, x=g, y=h, z=wrong)
        # the clique is non-commutative
        assert c.check_implicit(
            'exists x.(exists y.(exists z.(M(x,y,z) and (not M(y,x,z)))))', adv)
        # the zero form is commutative
        adv0 = c.advice(n, {})
        assert c.check_implicit(
            'all x.(all y.(all z.((M(x,y,z)) -> (M(y,x,z)))))', adv0)

    def test_tree_multiplication_and_sentence_z4(self):
        c = CutRankTreeGroups(2, d=2); n = 4; q = c.q
        val1 = {(j, i): (2,) for j in range(2, n + 1) for i in range(1, j)}
        for shape in (c.balanced(n), c.spine(n)):
            adv = c.advice(shape, val1)
            elems = word_elements(c, n); rng = random.Random(3)
            for _ in range(80):
                g, h = rng.choice(elems), rng.choice(elems)
                z = c.multiply(n, val1, g, h)
                assert c.check_implicit('M(x,y,z)', adv, x=g, y=h, z=z)
                assert c.check_implicit('M(x,y,z)', adv, x=g, y=h, z=z) \
                    == c.simulate(adv, g, h, z)
                wrong = (((z[0][0] + 1) % q,), z[1])
                assert not c.check_implicit('M(x,y,z)', adv, x=g, y=h, z=wrong)
            assert c.check_implicit(
                'exists x.(exists y.(exists z.(M(x,y,z) and (not M(y,x,z)))))', adv)
