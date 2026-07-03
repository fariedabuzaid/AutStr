import networkx as nx
import pytest

from autstr.graphs import TreeDepthClass, TreeDepthGraph
from autstr.utils.automata_tools import word_automaton


class TestWordAutomaton:
    def test_exact_word_with_padding(self):
        alphabet = {'*', 'a', 'b'}
        dfa = word_automaton(['a', 'b', 'a'], alphabet, padding_symbol='*')
        assert dfa.accepts([('a',), ('b',), ('a',)])
        assert dfa.accepts([('a',), ('b',), ('a',), ('*',), ('*',)])
        assert not dfa.accepts([('a',), ('b',)])
        assert not dfa.accepts([('a',), ('b',), ('b',)])
        assert not dfa.accepts([('a',), ('b',), ('a',), ('a',)])
        assert not dfa.accepts([('*',), ('a',), ('b',), ('a',)])

    def test_empty_word(self):
        dfa = word_automaton([], {'*', 'a'}, padding_symbol='*')
        assert dfa.accepts([])
        assert dfa.accepts([('*',)])
        assert not dfa.accepts([('a',)])


class TestUniformlyAutomaticClass:
    @pytest.fixture(scope="class")
    def depth2(self):
        return TreeDepthClass(2)

    def test_check_relativizes_quantifiers(self, depth2):
        """Quantifiers range over the member structure only."""
        g_edge = TreeDepthGraph.from_networkx(nx.star_graph(1))
        g_empty = TreeDepthGraph.from_networkx(nx.empty_graph(2))
        # "every singleton has a neighbor"
        phi = 'all x.((not Sing(x)) or exists y.(E(x,y)))'
        assert depth2.check(phi, g_edge)
        assert not depth2.check(phi, g_empty)

    def test_negated_quantifier(self, depth2):
        g_empty = TreeDepthGraph.from_networkx(nx.empty_graph(2))
        assert depth2.check('not (exists x.(exists y.(E(x,y))))', g_empty)
        assert not depth2.check('not (all x.(Subset(x,x)))', g_empty)

    def test_invalid_advice_rejected(self, depth2):
        dfa, _ = depth2.evaluate('exists x.(Sing(x))')
        # depth-2 letter without a preceding depth-1 letter is not valid advice
        bad = [depth2.symbol_of[(2, (1,))]]
        assert not dfa.accepts([(s,) for s in bad])

    def test_get_structure_universe(self, depth2):
        tdg = TreeDepthGraph.from_networkx(nx.star_graph(2))
        structure = depth2.get_structure(tdg)
        universe = structure.automata['U']
        assert universe.accepts([(b,) for b in tdg.encode_set({0, 2})])
        assert universe.accepts([(b,) for b in tdg.encode_set(set())])
        # a bitvector longer than the graph is not an element
        assert not universe.accepts([('1',), ('1',), ('1',), ('1',)])
