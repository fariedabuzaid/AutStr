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


class TestValidation:
    def test_domain_needs_one_free_variable(self):
        with pytest.raises(ValueError, match="one free variable"):
            interpret(BuechiArithmeticZ(), domain='Lt(x,y)', relations={})

    def test_argument_order_must_match_free_variables(self):
        with pytest.raises(ValueError, match="does not match"):
            interpret(BuechiArithmeticZ(), domain=_TRUE,
                      relations={'R': ('Lt(a,b)', ['a', 'c'])})

    def test_universe_symbol_is_reserved(self):
        with pytest.raises(ValueError, match="reserved universe"):
            interpret(BuechiArithmeticZ(), domain=_TRUE,
                      relations={'U': 'Eq(x,x)'})
