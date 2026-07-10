import random

import pytest

from autstr.sparse_tree_automata import Tree
from autstr.tree_groups import TreeExtraspecialGroups


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
