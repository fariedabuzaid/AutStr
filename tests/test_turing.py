"""Turing-machine configuration graphs.

The oracle is the machine itself: `TuringMachine.step` runs one step in Python,
and the edge relation must agree with it on every pair of configurations up to
a bounded tape length. Three machines exercise the three directions, including
the two cases where the encoding changes length — a right move off the written
tape, and a left move that writes a blank into the last cell.
"""
import itertools

import pytest

from autstr.turing import Configuration, TuringMachine


#: walks right over 1s, writes a 1 into the first blank and stops
SCANNER = TuringMachine({('q', '1'): ('q', '1', 'R'),
                         ('q', '_'): ('h', '1', 'S')}, blank='_')

#: walks left erasing as it goes — the encoding shrinks at every step
ERASER = TuringMachine({('q', '1'): ('q', '_', 'L'),
                        ('q', '_'): ('q', '_', 'L')}, blank='_')

#: flips a bit in place, then moves right; two states, two symbols
FLIPPER = TuringMachine({('a', '0'): ('b', '1', 'R'),
                         ('a', '1'): ('b', '0', 'R'),
                         ('b', '0'): ('a', '0', 'L'),
                         ('b', '1'): ('a', '1', 'L')}, blank='0')


def configurations(machine, length: int):
    """Every canonical configuration with a tape of at most `length` cells."""
    found = []
    for size in range(1, length + 1):
        for tape in itertools.product(machine.tape_alphabet, repeat=size):
            for head in range(size):
                for state in machine.states:
                    configuration = Configuration(state, tape, head)
                    if machine.canonical(configuration) == configuration:
                        found.append(configuration)
    return found


@pytest.fixture(scope="module")
def scanner_graph():
    return SCANNER.configuration_graph()


class TestMachine:
    def test_step_runs_the_machine(self):
        run = list(SCANNER.run(Configuration('q', ('1', '1'), 0)))
        assert run[-1] == Configuration('h', ('1', '1', '1'), 2)

    def test_a_missing_transition_halts(self):
        assert SCANNER.step(Configuration('h', ('1',), 0)) is None

    def test_a_left_move_off_the_tape_halts(self):
        assert ERASER.step(Configuration('q', ('1',), 0)) is None

    def test_trailing_blanks_are_dropped(self):
        canonical = SCANNER.canonical(Configuration('q', ('1', '_', '_'), 0))
        assert canonical == Configuration('q', ('1',), 0)

    def test_the_head_cell_is_never_dropped(self):
        configuration = Configuration('q', ('1', '_'), 1)
        assert SCANNER.canonical(configuration) == configuration

    def test_an_unknown_direction_is_rejected(self):
        with pytest.raises(ValueError, match="not a direction"):
            TuringMachine({('q', '1'): ('q', '1', 'up')})

    def test_the_padding_symbol_is_reserved(self):
        with pytest.raises(ValueError, match="reserved"):
            TuringMachine({('q', '*'): ('q', '*', 'R')})

    def test_the_head_is_on_the_tape(self):
        with pytest.raises(ValueError, match="off a tape"):
            Configuration('q', ('1',), 3)


class TestCodec:
    def test_roundtrip(self):
        for configuration in configurations(SCANNER, 3):
            assert SCANNER.decode(SCANNER.encode(configuration)) == configuration

    def test_the_word_marks_the_head_cell(self):
        assert SCANNER.encode(Configuration('q', ('1', '_'), 1)) == ['1', 'q|_']

    def test_a_word_without_a_head_is_no_configuration(self):
        with pytest.raises(ValueError, match="no head"):
            SCANNER.decode(['1', '1'])


class TestStepRelation:
    """The edge relation, checked exhaustively against the Python oracle."""

    @pytest.mark.parametrize("machine", [SCANNER, ERASER, FLIPPER],
                             ids=['scanner', 'eraser', 'flipper'])
    def test_agrees_with_the_oracle(self, machine):
        graph = machine.configuration_graph()
        x, y = graph.symbolic().vars('x y')
        edge = x.adj(y).evaluate()

        space = configurations(machine, 3)
        for source in space:
            successor = machine.step(source)
            for target in space:
                assert edge.contains(x=source, y=target) == \
                    (successor == target), (source, target)

    def test_the_tape_may_grow(self, scanner_graph):
        x, y = scanner_graph.symbolic().vars('x y')
        edge = x.adj(y).evaluate()
        # the head steps off the written tape, and the word gains a cell
        assert edge.contains(x=Configuration('q', ('1',), 0),
                             y=Configuration('q', ('1', '_'), 1))

    def test_the_tape_may_shrink(self):
        graph = ERASER.configuration_graph()
        x, y = graph.symbolic().vars('x y')
        edge = x.adj(y).evaluate()
        # erasing the last cell drops it from the encoding
        assert edge.contains(x=Configuration('q', ('1', '1'), 1),
                             y=Configuration('q', ('1',), 0))


class TestFirstOrderTheory:
    def test_the_machine_is_deterministic(self, scanner_graph):
        assert scanner_graph.is_deterministic()

    def test_halting_configurations_are_definable(self, scanner_graph):
        S = scanner_graph.symbolic()
        halting = S.rel('Halt')(S.vars('x')[0]).evaluate()
        assert halting.contains(x=Configuration('h', ('1',), 0))
        assert not halting.contains(x=Configuration('q', ('1',), 0))

    def test_a_stuck_left_move_halts(self):
        graph = ERASER.configuration_graph()
        S = graph.symbolic()
        halting = S.rel('Halt')(S.vars('x')[0]).evaluate()
        # the head cannot leave cell 0, so this configuration has no successor
        assert halting.contains(x=Configuration('q', ('1',), 0))
        assert not halting.contains(x=Configuration('q', ('1', '1'), 1))

    def test_some_configuration_halts_and_some_does_not(self, scanner_graph):
        assert scanner_graph.check('exists x.(Halt(x))')
        assert scanner_graph.check('exists x.(not Halt(x))')

    def test_every_configuration_has_at_most_one_successor(self):
        assert FLIPPER.configuration_graph().is_deterministic()


class TestTheBoundary:
    """Reachability is the halting problem, so it is not first-order over this
    graph and the construction does not offer it. What is offered is exactly
    what the engine can decide."""

    def test_no_reachability_relation_is_provided(self, scanner_graph):
        assert sorted(scanner_graph.get_relation_symbols()) == \
            ['E', 'Eq', 'Halt', 'U']

    def test_one_step_reachability_is_fine(self, scanner_graph):
        # bounded step counts stay first-order: they are just nested edges
        assert scanner_graph.check(
            'exists x.(exists y.(exists z.(E(x,y) & E(y,z) & Halt(z))))')
