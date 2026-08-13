"""Level 2 collapsible pushdown systems and their configuration graphs.

The oracle is the Python implementation of the stacks themselves: a stack is a
list of words of letters carrying links, and every operation on it is a few
lines of tuple slicing. The encoding, the domain automaton and each operation
relation are checked against it — on the configurations of an actual run, and
on stacks from a random walk through the operations, which reach shapes no
short run does.
"""
import itertools
import random

import pytest

from autstr.collapsible import (
    Configuration, Letter, Level2CPS, Operation, PAD, SEP, Stack,
    decode_configuration, decode_stack, encode_configuration, encode_stack,
    initial_stack,
)
from autstr.sparse_tree_automata import Tree, convolve_trees
from autstr.utils.misc import decode_symbol
from autstr.utils.tree_automata_tools import iterate_trees

#: every operation of the level 2 signature, over a two-symbol alphabet
OPERATIONS = ([Operation('clone')]
              + [Operation('push', symbol, level)
                 for symbol in ('a', 'b') for level in (1, 2)]
              + [Operation('pop', level=1), Operation('pop', level=2),
                 Operation('collapse')])


def walk(seed, steps=30):
    """Stacks from a random walk through the operations, restarting whenever
    the walk runs into an undefined one."""
    rng, stack, seen = random.Random(seed), initial_stack(), []
    for _ in range(steps):
        seen.append(stack)
        stack = stack.apply(rng.choice(OPERATIONS)) or initial_stack()
    return seen


@pytest.fixture(scope="module")
def sample():
    """A few hundred stacks, deep and wide enough to exercise the encoding."""
    return [stack for seed in range(40) for stack in walk(seed)]


@pytest.fixture(scope="module")
def system():
    """The level 2 collapsible pushdown system of Hague et al., which Kartzow
    draws as his running example: clone, push a linked to the stack below,
    then pop or collapse."""
    return Level2CPS(transitions=[
        ('0', None, 'Cl', '1', 'clone'),
        ('1', None, 'A', '0', 'push a 2'),
        ('1', None, 'B', '2', 'push a 2'),
        ('2', 'a', 'P', '2', 'pop 1'),
        ('2', 'a', 'Co', '0', 'collapse'),
    ])


@pytest.fixture(scope="module")
def graph(system):
    return system.configuration_graph()


@pytest.fixture(scope="module")
def operations():
    """A system whose rules mention every operation, over both stack symbols,
    so that its graph presents them all."""
    return Level2CPS(transitions=[
        ('q', None, 'c', 'q', 'clone'),
        ('q', None, 'p1', 'q', 'push a 1'),
        ('q', None, 'p2', 'q', 'push a 2'),
        ('q', None, 'o1', 'q', 'pop 1'),
        ('q', None, 'o2', 'q', 'pop 2'),
        ('q', None, 'co', 'q', 'collapse'),
    ], symbols=('b',)).configuration_graph()


class TestStacks:
    """The oracle itself, against the definitions."""

    def test_the_initial_stack(self):
        assert initial_stack().width == 1
        assert initial_stack().top() == Letter('⊥', 1)

    def test_clone_copies_links_too(self):
        stack = initial_stack().push('a', 2).clone()
        assert stack.words[0] == stack.words[1]
        # the copy still points where the original did, which is what makes
        # collapse jump further than a pop
        assert stack.top() == Letter('a', 2, 0)

    def test_a_level_2_link_records_the_width_below(self):
        stack = initial_stack().clone().clone().push('a', 2)
        assert stack.top().link == 2
        assert stack.collapse().width == 2

    def test_collapse_on_a_level_1_link_is_pop_1(self):
        stack = initial_stack().push('a').push('b')
        assert stack.collapse() == stack.pop1()

    def test_the_undefined_operations(self):
        assert initial_stack().pop1() is None       # nothing above the bottom
        assert initial_stack().pop2() is None       # only one word
        assert initial_stack().push('a', 2).collapse() is None   # link of 0

    def test_collapse_undoes_a_whole_run_of_pops(self):
        stack = initial_stack().clone().push('a', 2).clone().clone()
        assert stack.width == 4
        # the letter was pushed onto a stack of width 2, and every clone since
        # carried its link along, so one collapse undoes all three words
        assert stack.collapse().width == 1


class TestEncoding:
    """The bijection between configurations and encoding trees."""

    def test_the_paper_s_figure(self):
        """Kartzow's Figure 2: a stack of five words and its encoding."""
        bottom, first, later = Letter('⊥'), Letter('a', 2, 0), Letter('a', 2, 2)
        stack = Stack((
            (bottom, first, Letter('b', 2, 0)),
            (bottom, first, Letter('b', 2, 0), Letter('c', 2, 1)),
            (bottom, later, Letter('c')),
            (bottom, later, Letter('d', 2, 3), Letter('e')),
            (bottom, later),
        ))
        assert encode_stack(stack) == Tree(
            '⊥:1',
            Tree('a:2', Tree('b:2', None, Tree(SEP, Tree('c:2')))),
            Tree(SEP, Tree('a:2', Tree('c:1'),
                           Tree(SEP, Tree('d:2', Tree('e:1')),
                                Tree(SEP)))))
        assert decode_stack(encode_stack(stack)) == stack

    def test_the_initial_stack_is_a_single_node(self):
        assert encode_stack(initial_stack()) == Tree('⊥:1')
        assert encode_stack(initial_stack().clone()) == \
            Tree('⊥:1', None, Tree(SEP))

    def test_round_trip(self, sample):
        for stack in sample:
            assert decode_stack(encode_stack(stack)) == stack

    def test_distinct_stacks_get_distinct_trees(self, sample):
        seen = {}
        for stack in sample:
            key = repr(encode_stack(stack))
            assert seen.setdefault(key, stack) == stack

    def test_two_stacks_that_differ_only_in_a_link(self):
        """The links are not stored, but they are still visible: the position
        of a letter in the tree is what says where its link points."""
        cloned = initial_stack().push('a', 2).clone()          # both links 0
        pushed = initial_stack().push('a', 2).clone().pop1().push('a', 2)
        assert cloned.words[1][1] != pushed.words[1][1]
        assert encode_stack(cloned) != encode_stack(pushed)

    def test_a_configuration_carries_its_state(self):
        configuration = Configuration('q', initial_stack())
        assert encode_configuration(configuration).label == '<q>'
        assert decode_configuration(
            encode_configuration(configuration)) == configuration


class TestDomain:
    """The domain automaton accepts exactly the encoding trees."""

    def test_accepts_every_encoding(self, operations, sample):
        universe = operations.presentation.automata['U']
        for stack in sample:
            assert universe.accepts(
                encode_configuration(Configuration('q', stack)))

    def test_accepts_nothing_but_encodings(self, operations):
        """The other direction, and the one that makes this a presentation
        rather than a superset: enumerate what the automaton accepts, smallest
        trees first, and decode each back. Every accepted tree has to be the
        encoding of the configuration it decodes to."""
        alphabet = frozenset(operations.encoding.alphabet)

        def relabel(tree):
            """Enumeration yields symbol codes; put the labels back."""
            return Tree(decode_symbol(tree.label, 1, alphabet)[0],
                        relabel(tree.left) if tree.left is not None else None,
                        relabel(tree.right) if tree.right is not None else None)

        # the authored automaton, not the padding-saturated one the
        # presentation stores: a region of pure padding encodes nothing
        trees = iterate_trees(operations.encoding.universe())
        for tree in map(relabel, itertools.islice(trees, 2000)):
            assert encode_configuration(decode_configuration(tree)) == tree

    @pytest.mark.parametrize("tree", [
        Tree('<q>'),                                   # no stack at all
        Tree('<q>', Tree('a:1')),                      # no bottom symbol
        Tree('⊥:1', Tree('a:1')),                      # no control state
        Tree('<q>', Tree('⊥:1'), Tree(SEP)),           # a word beside the stack
        Tree('<q>', Tree('⊥:1', Tree('⊥:1'))),         # two bottoms
        Tree('<q>', Tree('⊥:1', None, Tree('a:1'))),   # a letter as separator
    ])
    def test_rejects_malformed_trees(self, operations, tree):
        assert not operations.presentation.automata['U'].accepts(tree)

    def test_adjacent_blocks_may_not_begin_with_the_same_level_1_letter(
            self, operations):
        """Two neighbouring blocks whose words agree on the next letter are
        really one block, so that tree encodes nothing — with a level 1 link,
        where the position fixes the link. With a level 2 link the two letters
        differ in where they point, so the same shape is an encoding."""
        universe = operations.presentation.automata['U']
        assert not universe.accepts(
            Tree('<q>', Tree('⊥:1', Tree('a:1'), Tree(SEP, Tree('a:1')))))
        assert universe.accepts(
            Tree('<q>', Tree('⊥:1', Tree('a:2'), Tree(SEP, Tree('a:2')))))


class TestOperations:
    """Each operation relation, against the stacks themselves."""

    #: relation name -> the operation it should present
    RELATIONS = {'Clone': Operation('clone'),
                 'Pop1': Operation('pop', level=1),
                 'Pop2': Operation('pop', level=2),
                 'Collapse': Operation('collapse'),
                 'Pusha1': Operation('push', 'a', 1),
                 'Pusha2': Operation('push', 'a', 2)}

    @pytest.mark.parametrize("name", sorted(RELATIONS))
    def test_against_the_oracle(self, operations, name, sample):
        operation = self.RELATIONS[name]
        automaton = operations.presentation.relation(name)
        alphabet = frozenset(operations.encoding.alphabet)
        for stack in sample[:120]:
            expected = stack.apply(operation)
            # the result, the other operations' results, and the stack itself
            candidates = [other.apply(op) for other in [stack]
                          for op in OPERATIONS] + [stack]
            for candidate in candidates:
                if candidate is None:
                    continue
                convolution = convolve_trees(
                    [encode_configuration(Configuration('q', stack)),
                     encode_configuration(Configuration('q', candidate))],
                    alphabet, PAD)
                assert automaton.accepts(convolution) == \
                    (candidate == expected), (name, stack, candidate)

    def test_collapse_is_more_than_the_pops(self, operations):
        """The point of the level: some collapse is neither a pop_1 nor a
        single pop_2, because it undoes a whole run of clones at once."""
        assert operations.check('exists x.(exists y.(Collapse(x,y) & '
                             '(not Pop1(x,y)) & (not Pop2(x,y))))')

    def test_collapse_on_a_level_1_link_is_pop_1(self, operations):
        assert operations.check(
            'all x.(all y.((Level1(x) & Collapse(x,y)) -> Pop1(x,y)))')
        assert operations.check(
            'all x.(all y.((Level1(x) & Pop1(x,y)) -> Collapse(x,y)))')

    def test_the_operations_are_functions(self, operations):
        """Each operation has at most one result — the system is what is
        nondeterministic, not the stack machinery."""
        for name in ('Clone', 'Pop1', 'Pop2', 'Collapse', 'Pusha2'):
            assert operations.check(
                f'all x.(all y.(all z.(({name}(x,y) & {name}(x,z))'
                f' -> Eq(y,z))))'), name

    def test_clone_and_a_push_never_agree(self, operations):
        assert not operations.check('exists x.(exists y.(Clone(x,y) '
                                 '& Pusha2(x,y)))')


class TestConfigurationGraph:
    """The graph of the system, against running it."""

    def test_edges_agree_with_the_step_relation(self, system, graph):
        configurations = system.reachable(bound=7)
        assert len(configurations) > 10
        alphabet = frozenset(graph.encoding.alphabet)
        edges = graph.presentation.automata['E']
        for source in configurations:
            successors = {target for _, target in system.step(source)}
            for candidate in configurations:
                convolution = convolve_trees(
                    [encode_configuration(source),
                     encode_configuration(candidate)], alphabet, PAD)
                assert edges.accepts(convolution) == \
                    (candidate in successors), (source, candidate)

    def test_the_labelled_edges_split_the_edge_relation(self, system, graph):
        names = [graph.label_names[label] for label in system.labels]
        assert graph.check(
            'all x.(all y.(E(x,y) <-> (' +
            ' | '.join(f'{name}(x,y)' for name in names) + ')))')

    def test_the_system_branches(self, graph):
        """Two rules apply in state 1, so the graph is not deterministic —
        a first-order question, and so decidable here."""
        assert not graph.is_deterministic()

    def test_some_configuration_is_stuck(self, graph):
        """Over all configurations, not only the reachable ones: in state 2
        with the bottom symbol on top, no rule applies."""
        assert graph.check('exists x.(not exists y.(E(x,y)))')
        assert graph.check(
            'all x.((State2(x) & TopBottom(x)) -> (not exists y.(E(x,y))))')

    def test_a_control_state_is_definable(self, graph):
        assert graph.check('all x.(State0(x) | State1(x) | State2(x))')
        assert not graph.check('exists x.(State0(x) & State1(x))')

    def test_evaluate_gives_back_the_step_relation(self, graph, system):
        initial = system.initial()
        successors = graph.evaluate('E(x,y)')
        assert successors.accepts(convolve_trees(
            [encode_configuration(initial),
             encode_configuration(system.step(initial)[0][1])],
            frozenset(graph.encoding.alphabet), PAD))

    def test_symbolic_interface(self, graph, system):
        initial = system.initial()
        x = graph.symbolic().var("x")
        assert x.adj(system.step(initial)[0][1]).check()
        # something does step into the initial configuration — a collapse all
        # the way back — but nothing steps into itself: every operation moves
        assert x.adj(initial).check()
        assert not (x.eq(initial) & x.adj(initial)).check()


class TestSystem:
    """The system's own bookkeeping."""

    def test_operations_parse(self):
        assert Operation.parse('push a') == Operation('push', 'a', 1)
        assert Operation.parse('push a 2') == Operation('push', 'a', 2)
        assert Operation.parse('pop 2') == Operation('pop', level=2)
        assert Operation.parse('clone') == Operation('clone')
        with pytest.raises(ValueError, match="no stack operation"):
            Operation.parse('push a 3')
        with pytest.raises(ValueError, match="no stack operation"):
            Operation.parse('collapse 2')

    def test_a_dash_matches_any_symbol(self):
        system = Level2CPS([('q', '-', 'x', 'q', 'clone')])
        assert system.rules[0].symbol is None
        assert system.step(Configuration('q', initial_stack()))

    def test_the_bottom_symbol_cannot_be_pushed(self):
        with pytest.raises(ValueError, match="bottom"):
            Level2CPS([('q', None, 'x', 'q', 'push ⊥ 1')])

    def test_reserved_characters_are_rejected(self):
        with pytest.raises(ValueError, match="reserved"):
            Level2CPS([('q', None, 'x', 'q', 'push a:1 1')])

    def test_relations_are_named_after_the_system(self, graph):
        assert graph.symbol_names == {'⊥': 'TopBottom', 'a': 'Topa'}
        assert graph.label_names['Cl'] == 'EdgeCl'
        assert graph.push_names == {('a', 2): 'Pusha2'}

    def test_names_fall_back_to_positions(self):
        """Symbols that would name the same relation — here a stack symbol
        literally called Bottom, beside the bottom symbol itself — are told
        apart by position, rather than one of them quietly replacing the
        other."""
        graph = Level2CPS([('q', None, 'x', 'q', 'clone')],
                          symbols=('Bottom',)).configuration_graph()
        assert sorted(graph.symbol_names.values()) == ['Top0', 'Top1']

    def test_the_run_of_the_paper_s_example(self, system):
        """The first steps of Kartzow's Figure 1: clone, push, clone, push."""
        configuration = system.initial()
        for label in ('Cl', 'A', 'Cl', 'A'):
            step = [target for read, target in system.step(configuration)
                    if read == label]
            assert len(step) == 1
            configuration = step[0]
        assert configuration.state == '0'
        # the paper draws this stack as ⊥ : ⊥a : ⊥aa; the links are what the
        # drawing leaves out, and each a points below the word it was pushed on
        assert repr(configuration.stack) == '⊥ : ⊥a[1] : ⊥a[1]a[2]'
