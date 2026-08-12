"""First-order interpretations (Phase 1: one-dimensional, no quotient)."""
import itertools

import pytest

from autstr.arithmetic import decode, encode
from autstr.buildin.presentations import BuechiArithmeticZ
from autstr.infinite_graphs import IntegerGrid, InfiniteGraph
from autstr.interpretations import interpret
from autstr.symbolic import FunctionCodec

# the ±1 edge on ℤ, as a first-order formula over (ℤ, +)
_ONE = 'Pt(o) & (all p.(Pt(p) -> (not Lt(p,o))))'
_STEP = f'exists o.(({_ONE}) & (A(x,o,y) | A(y,o,x)))'
_TRUE = 'Eq(x,x)'


@pytest.fixture(scope="module")
def integer_codec():
    return FunctionCodec(encode, decode)


class TestReconstructsTheGrid:
    """Interpreting the path in (ℤ, +) must reproduce the hand-built grid path
    exactly — same relation, and, because the engine minimizes, the same
    canonical automaton. This is the efficiency claim as a test."""

    def test_same_canonical_automaton_as_the_handbuilt_path(self):
        path = interpret(BuechiArithmeticZ(), domain=_TRUE,
                         relations={'E': (_STEP, ['x', 'y'])})
        handbuilt = IntegerGrid(1).presentation
        assert path.automata['E'].num_states == handbuilt.automata['E'].num_states == 15

    def test_adjacency_semantics(self, integer_codec):
        path = interpret(BuechiArithmeticZ(), domain=_TRUE,
                         relations={'E': (_STEP, ['x', 'y'])})
        graph = InfiniteGraph(path, edge='E', codec=integer_codec)
        G = graph.symbolic()
        x, y = G.vars('x y')
        edge = x.adj(y).evaluate()
        assert edge.contains(x=3, y=4)
        assert edge.contains(x=4, y=3)
        assert not edge.contains(x=3, y=5)
        assert graph.is_symmetric()


class TestDomainRestriction:
    """The domain formula carves out a sub-universe; relations restrict to it."""

    def test_even_integers_with_the_induced_order(self, integer_codec):
        # x is even iff x = y + y for some y
        evens = interpret(BuechiArithmeticZ(),
                          domain='exists y.(A(y,y,x))',
                          relations={'Lt': ('Lt(x,y)', ['x', 'y'])})
        order = InfiniteGraph(evens, edge='Lt', directed=True,
                              codec=integer_codec)
        G = order.symbolic()
        x, y = G.vars('x y')
        rel = x.adj(y).evaluate()
        assert rel.contains(x=2, y=4)          # both even, 2 < 4
        assert not rel.contains(x=4, y=2)
        # odd numbers are not in the universe: 3 < 4 is not in the relation
        assert not rel.contains(x=3, y=4)
        assert not order.is_symmetric()        # a strict order


class TestArgumentOrder:
    def test_order_permutes_the_tapes(self):
        # R(a, b) defined by Lt(a, b) but declared with reversed argument order
        greater = interpret(BuechiArithmeticZ(), domain=_TRUE,
                            relations={'R': ('Lt(a,b)', ['b', 'a'])})
        rel = greater.evaluate('R(x,y)')

        def holds(a, b):
            conv = [tuple(t) for t in
                    itertools.zip_longest(encode(a), encode(b), fillvalue='*')]
            return rel.accepts(conv)

        # R(x, y) reads as "y < x", i.e. x > y
        assert holds(5, 3)
        assert not holds(3, 5)


def _step(a, b):
    return f'exists o.(({_ONE}) & (A({a},o,{b}) | A({b},o,{a})))'


class TestTwoDimensional:
    """Elements are pairs of integers. The k-dim interpretation of the grid
    must reproduce the composition-built one exactly — same canonical
    automaton — a cross-check of two independent constructions."""

    def test_reproduces_the_product_grid(self):
        edge = (f'(({_step("x0","y0")}) & Eq(x1,y1)) '
                f'| (({_step("x1","y1")}) & Eq(x0,y0))')
        grid = interpret(
            BuechiArithmeticZ(),
            domain=('Eq(x0,x0) & Eq(x1,x1)', ['x0', 'x1']),
            relations={'E': (edge, ['x0', 'x1', 'y0', 'y1'])},
            dimension=2)
        from_product = IntegerGrid(2).presentation
        assert grid.automata['E'].num_states == \
            from_product.automata['E'].num_states == 51
        assert grid.check('all x.(all y.(E(x,y) -> E(y,x)))')

    def test_adjacency(self):
        edge = (f'(({_step("x0","y0")}) & Eq(x1,y1)) '
                f'| (({_step("x1","y1")}) & Eq(x0,y0))')
        grid = interpret(
            BuechiArithmeticZ(),
            domain=('Eq(x0,x0) & Eq(x1,x1)', ['x0', 'x1']),
            relations={'E': (edge, ['x0', 'x1', 'y0', 'y1'])},
            dimension=2)
        rel = grid.evaluate('E(x,y)')

        def enc(a, b):
            wa, wb = encode(a), encode(b)
            n = max(len(wa), len(wb))
            wa += ['*'] * (n - len(wa)); wb += ['*'] * (n - len(wb))
            return list(zip(wa, wb))

        def adjacent(p, q):
            wp, wq = enc(*p), enc(*q)
            n = max(len(wp), len(wq))
            wp += [('*', '*')] * (n - len(wp))
            wq += [('*', '*')] * (n - len(wq))
            return rel.accepts(list(zip(wp, wq)))

        assert adjacent((0, 0), (1, 0))
        assert adjacent((2, 3), (2, 4))
        assert not adjacent((0, 0), (1, 1))
        assert not adjacent((0, 0), (0, 0))


class TestSparseDomainDoesNotBlowUp:
    """The user's concern, as a regression: the diagonal {(a, a)} ≅ ℤ realizes
    only a sparse subset of the pairs, yet minimization keeps the successor
    relation the same size as the one-dimensional native encoding."""

    def test_diagonal_matches_native_state_count(self):
        succ = f'exists o.(({_ONE}) & A(x0,o,y0))'   # y0 = x0 + 1
        diagonal = interpret(
            BuechiArithmeticZ(),
            domain=('Eq(x0,x1)', ['x0', 'x1']),
            relations={'S': (f'Eq(x0,x1) & Eq(y0,y1) & ({succ})',
                             ['x0', 'x1', 'y0', 'y1'])},
            dimension=2)
        native = interpret(BuechiArithmeticZ(), domain=_TRUE,
                           relations={'S': (succ.replace('x0', 'x').replace(
                               'y0', 'y'), ['x', 'y'])})
        assert diagonal.automata['S'].num_states == \
            native.automata['S'].num_states


class TestQuotient:
    """One-dimensional quotient interpretations: the universe becomes the
    shortlex-least representative of each equivalence class."""

    def test_integers_mod_two(self):
        # quotient of Z by "x - y is even" has exactly two classes
        even = 'exists u.(exists w.(A(u,u,w) & A(y,w,x)))'   # x = y + 2u
        Q = interpret(BuechiArithmeticZ(), domain=_TRUE,
                      relations={'Eq': (even, ['x', 'y'])},
                      dimension=1, quotient=(even, ['x', 'y']))
        assert Q.check('exists x.(exists y.(not Eq(x,y)))')          # >= 2
        assert not Q.check('exists x.(exists y.(exists z.('          # not >= 3
                           '(not Eq(x,y)) & (not Eq(x,z)) & (not Eq(y,z)))))')

    def test_a_trivial_equivalence_keeps_every_element(self):
        # x ~ y iff x = y: each class is a singleton, so the quotient is the
        # whole structure
        Q = interpret(BuechiArithmeticZ(), domain=_TRUE,
                      relations={'Lt': ('Lt(x,y)', ['x', 'y'])},
                      dimension=1, quotient=('Eq(x,y)', ['x', 'y']))
        assert Q.check('exists x.(exists y. Lt(x,y))')
        assert Q.check('all x.(exists y. Lt(x,y))')      # no maximum, like Z

    def test_integers_as_difference_pairs_of_naturals(self):
        """The textbook two-dimensional quotient: ℤ = ℕ² / "same difference",
        with the order read off the representatives. The result must be a
        discrete unbounded total order — ℤ, built from ℕ alone."""
        pairs = ('N0(x0) & N0(x1)', ['x0', 'x1'])
        # (x0,x1) ~ (y0,y1)  iff  x0 + y1 = y0 + x1  (same difference)
        same = ('exists s.(A(x0,y1,s) & A(y0,x1,s))',
                ['x0', 'x1', 'y0', 'y1'])
        # x0 - x1 < y0 - y1  iff  x0 + y1 < y0 + x1
        less = ('exists s.(exists t.(A(x0,y1,s) & A(y0,x1,t) & Lt(s,t)))',
                ['x0', 'x1', 'y0', 'y1'])
        Z = interpret(BuechiArithmeticZ(), domain=pairs,
                      relations={'L': less}, dimension=2, quotient=same)

        assert Z.check('all x.(not L(x,x))')                    # irreflexive
        assert Z.check('all x.(all y.(all z.('                  # transitive
                       '(L(x,y) & L(y,z)) -> L(x,z))))')
        assert Z.check('all x.(exists y. L(y,x))')              # no least
        assert Z.check('all x.(exists y. L(x,y))')              # no greatest
        assert Z.check('all x.(exists y.(L(x,y) & (not exists z.('
                       'L(x,z) & L(z,y)))))')                   # has successors
        assert not Z.check('all x.(all y.(L(x,y) -> exists z.('
                           'L(x,z) & L(z,y))))')                # so: not dense

    def test_reserved_scaffolding_names(self):
        with pytest.raises(ValueError, match="reserved"):
            interpret(BuechiArithmeticZ(), domain=_TRUE,
                      relations={'_Equiv': 'Eq(x,y)'},
                      quotient=('Eq(x,y)', ['x', 'y']))


class TestValidation:
    def test_domain_needs_one_free_variable(self):
        # dimension 1 (default), so a two-free-variable domain defines 2 elements
        with pytest.raises(ValueError, match="defines 2 element"):
            interpret(BuechiArithmeticZ(), domain='Lt(x,y)', relations={})

    def test_argument_order_must_match_free_variables(self):
        with pytest.raises(ValueError, match="does not match"):
            interpret(BuechiArithmeticZ(), domain=_TRUE,
                      relations={'R': ('Lt(a,b)', ['a', 'c'])})

    def test_universe_symbol_is_reserved(self):
        with pytest.raises(ValueError, match="reserved universe"):
            interpret(BuechiArithmeticZ(), domain=_TRUE,
                      relations={'U': 'Eq(x,x)'})

    def test_domain_dimension_must_match(self):
        # dimension 2 but the domain has only one free variable
        with pytest.raises(ValueError, match="free variable"):
            interpret(BuechiArithmeticZ(), domain=_TRUE, relations={},
                      dimension=2)

    def test_free_variables_must_be_a_multiple_of_the_dimension(self):
        # dimension 2 but the relation has three free variables
        with pytest.raises(ValueError, match="multiple of the dimension"):
            interpret(BuechiArithmeticZ(),
                      domain=('Eq(x0,x0) & Eq(x1,x1)', ['x0', 'x1']),
                      relations={'R': ('A(x0,x1,x2)', ['x0', 'x1', 'x2'])},
                      dimension=2)
