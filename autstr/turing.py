"""Turing machines and their configuration graphs.

The configuration graph of a Turing machine is the standard example of an
automatic structure that is genuinely *about* computation: a configuration is a
word, and one step of the machine rewrites that word locally, at the head, so
the step relation is recognized by a small synchronous automaton — one whose
size depends only on the transition table, not on the tape.

That makes the first-order theory of the graph decidable, and this module lets
you ask it::

    >>> machine = TuringMachine(
    ...     transitions={('q', '1'): ('q', '1', 'R'),
    ...                  ('q', '_'): ('halt', '1', 'S')},
    ...     blank='_')
    >>> graph = machine.configuration_graph()
    >>> graph.check('all x.(all y.(all z.((E(x,y) & E(x,z)) -> Eq(y,z))))')
    True

**The boundary lesson.** Reachability — "does some sequence of steps lead from
this configuration to a halting one?" — is exactly the halting problem, so it
is undecidable, and therefore *not first-order definable* over this graph.
Nothing here provides it, and nothing can: the transitive closure of ``E`` is
outside FO. A decidable first-order theory is not a decidable graph. It is the
same lesson as the random graph in reverse — there, a decidable theory without
an automatic presentation; here, an automatic presentation whose decidable
theory still cannot express the one question you would most like to ask.

**Encoding.** The tape is one-way infinite, to the right. A configuration is
the word ``left · (state, symbol) · right``: the tape contents, with the cell
under the head replaced by a symbol naming both the state and what is written
there. Trailing blanks are dropped, so each configuration has exactly one word.
A step changes the word only at the head and, at most, by one cell at the right
end, which is why a synchronous automaton can read it. A left move off cell 0
has no successor, as does a configuration whose ``(state, symbol)`` pair the
transition table does not list — those are the halting configurations, and
``Halt`` is defined from ``E`` rather than authored.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from autstr.infinite_graphs import InfiniteGraph
from autstr.symbolic import FunctionCodec
from autstr.utils.automata_tools import partial_dfa

#: the padding symbol of the configuration encoding
_PAD = '*'

#: how a transition may move the head
DIRECTIONS = ('L', 'R', 'S')


@dataclass(frozen=True)
class Configuration:
    """A machine configuration: the control state, the tape, and where the
    head is.

    :param state: the control state.
    :param tape: the tape contents from cell 0, as a tuple of tape symbols.
    :param head: the index of the cell under the head.
    """
    state: str
    tape: Tuple[str, ...]
    head: int = 0

    def __post_init__(self):
        if not 0 <= self.head < max(len(self.tape), 1):
            raise ValueError(
                f"the head is at cell {self.head}, off a tape of "
                f"{len(self.tape)} cell(s)")


class TuringMachine:
    """A deterministic single-tape Turing machine with a one-way infinite
    tape.

    :param transitions: ``{(state, read): (state, write, direction)}``, with
        direction one of ``'L'``, ``'R'``, ``'S'``. A pair the table omits is
        halting.
    :param blank: the blank symbol, which fills the tape beyond what has been
        written.
    :param states: the control states. Read off the table when omitted.
    :param tape_alphabet: the tape symbols. Read off the table when omitted;
        the blank always belongs.
    """

    def __init__(self, transitions: Dict[Tuple[str, str], Tuple[str, str, str]],
                 blank: str = '_', states: Optional[Sequence[str]] = None,
                 tape_alphabet: Optional[Sequence[str]] = None) -> None:
        self.blank = blank
        self.transitions = dict(transitions)

        mentioned_states, mentioned_symbols = set(), {blank}
        for (state, read), (target, write, direction) in self.transitions.items():
            if direction not in DIRECTIONS:
                raise ValueError(
                    f"{direction!r} is not a direction; use one of "
                    f"{', '.join(DIRECTIONS)}")
            mentioned_states |= {state, target}
            mentioned_symbols |= {read, write}

        self.states = sorted(mentioned_states | set(states or ()))
        self.tape_alphabet = sorted(mentioned_symbols | set(tape_alphabet or ()))
        if blank not in self.tape_alphabet:
            raise ValueError(f"the blank {blank!r} is not a tape symbol")
        for symbol in self.tape_alphabet + self.states:
            if symbol == _PAD:
                raise ValueError(f"{_PAD!r} is reserved as the padding symbol")

        #: the head letters: one per (state, symbol) pair
        self._head_letters = {(q, a): f'{q}|{a}'
                              for q in self.states for a in self.tape_alphabet}
        self.alphabet = (list(self.tape_alphabet)
                         + sorted(self._head_letters.values()) + [_PAD])

    # -- the machine itself --------------------------------------------
    def canonical(self, configuration: Configuration) -> Configuration:
        """The configuration with trailing blanks past the head dropped — the
        one word that stands for it."""
        tape = list(configuration.tape) or [self.blank]
        while len(tape) - 1 > configuration.head and tape[-1] == self.blank:
            tape.pop()
        return Configuration(configuration.state, tuple(tape),
                             configuration.head)

    def step(self, configuration: Configuration) -> Optional[Configuration]:
        """The successor configuration, or None if there is none — the
        transition table does not cover this ``(state, symbol)`` pair, or the
        head would move off the left end of the tape.

        This is the Python oracle the automaton is checked against, and the
        obvious way to run the machine.
        """
        configuration = self.canonical(configuration)
        tape = list(configuration.tape)
        key = (configuration.state, tape[configuration.head])
        if key not in self.transitions:
            return None
        state, write, direction = self.transitions[key]

        tape[configuration.head] = write
        head = configuration.head + {'L': -1, 'R': 1, 'S': 0}[direction]
        if head < 0:
            return None
        if head == len(tape):
            tape.append(self.blank)
        return self.canonical(Configuration(state, tuple(tape), head))

    def run(self, configuration: Configuration, limit: int = 1000):
        """The run from a configuration: it, then its successors, stopping when
        the machine halts or after `limit` steps."""
        current = self.canonical(configuration)
        yield current
        for _ in range(limit):
            current = self.step(current)
            if current is None:
                return
            yield current

    # -- the configuration graph ---------------------------------------
    def configuration_graph(self) -> 'ConfigurationGraph':
        """The graph of configurations under one step of the machine."""
        return ConfigurationGraph(self)

    # -- element codec: a configuration <-> its word --------------------
    def encode(self, configuration: Configuration) -> list:
        """The word encoding a configuration."""
        configuration = self.canonical(configuration)
        word = list(configuration.tape)
        head = configuration.head
        word[head] = self._head_letters[(configuration.state, word[head])]
        return word

    def decode(self, word) -> Configuration:
        """The configuration a word encodes."""
        tape, state, head = [], None, None
        for position, symbol in enumerate(word):
            if symbol == _PAD:
                break
            if '|' in symbol:
                state, _, cell = symbol.partition('|')
                head = position
                tape.append(cell)
            else:
                tape.append(symbol)
        if head is None:
            raise ValueError(f"{word!r} carries no head, so it is no "
                             f"configuration")
        return Configuration(state, tuple(tape), head)


class ConfigurationGraph:
    """The configuration graph of a `TuringMachine`: configurations as
    vertices, one machine step as a directed edge.

    The presentation carries ``E`` (one step), ``Eq``, and ``Halt`` — the
    configurations with no successor, *defined* as ``not exists y. E(x,y)``
    rather than authored, since the engine can compute it.

    Reachability is deliberately absent; see the module docstring.
    """

    def __init__(self, machine: TuringMachine) -> None:
        from autstr.presentations import AutomaticPresentation
        self.machine = machine

        presentation = AutomaticPresentation(
            {'U': _configurations(machine),
             'Eq': _identity(machine),
             'E': _step(machine)},
            padding_symbol=_PAD)
        presentation.update(Halt='not exists y.(E(x,y))')
        self.presentation = presentation
        self.graph = InfiniteGraph(
            presentation, edge='E', directed=True,
            codec=FunctionCodec(machine.encode, machine.decode))

    def symbolic(self, signature=None):
        """A symbolic interface to the graph; write one step as ``x.adj(y)``
        and configurations as `Configuration` values."""
        return self.graph.symbolic(signature)

    def check(self, phi) -> bool:
        return self.graph.check(phi)

    def evaluate(self, phi):
        return self.graph.evaluate(phi)

    def get_relation_symbols(self):
        return self.presentation.get_relation_symbols()

    def is_deterministic(self) -> bool:
        """Whether every configuration has at most one successor — a
        first-order question, and so decidable here, unlike reachability."""
        return self.presentation.check(
            'all x.(all y.(all z.((E(x,y) & E(x,z)) -> Eq(y,z))))')

    def __repr__(self):
        return (f"<ConfigurationGraph states={len(self.machine.states)} "
                f"alphabet={len(self.machine.tape_alphabet)}>")


# ----------------------------------------------------------------------
# the automata
# ----------------------------------------------------------------------

def _configurations(machine: TuringMachine):
    """The universe: words with exactly one head letter, not ending in a
    blank — so each configuration has exactly one encoding."""
    blank, heads = machine.blank, machine._head_letters.values()
    plain = [a for a in machine.tape_alphabet if a != blank]

    table = {
        # before the head, a blank is an ordinary cell: what follows keeps it
        # from being trailing
        'before': {(blank,): 'before', **{(a,): 'before' for a in plain},
                   **{(h,): 'after' for h in heads}},
        # after the head, a word may not stop on a blank — that encoding is
        # the same configuration as the word without it
        'after': {(blank,): 'trailing', **{(a,): 'after' for a in plain}},
        'trailing': {(blank,): 'trailing', **{(a,): 'after' for a in plain}},
    }
    return partial_dfa(set(machine.alphabet), 1, table, 'before', {'after'})


def _identity(machine: TuringMachine):
    """``x = y`` on configuration words."""
    letters = [a for a in machine.alphabet if a != _PAD]
    return partial_dfa(set(machine.alphabet), 2,
                       {'s': {(a, a): 's' for a in letters}}, 's', {'s'})


def _step(machine: TuringMachine):
    """One machine step, as a synchronous two-tape automaton.

    Reading the two words in lockstep, the automaton walks the agreeing prefix,
    applies the local rewrite where the head is, and walks the agreeing suffix.
    A left move is the one case needing lookahead — the head lands on the cell
    *before* the one being read — and is handled by remembering, for one
    symbol, which control state the target word's head announced.
    """
    blank, pad = machine.blank, _PAD
    head_letter = machine._head_letters
    cells = machine.tape_alphabet

    # the agreeing prefix, and the two ways a rewrite can start
    prefix = {(a, a): 'prefix' for a in cells}
    for (state, read), (target, write, direction) in machine.transitions.items():
        source = head_letter[(state, read)]
        if direction == 'R':
            # the head letter becomes what was written; the head reappears on
            # the next cell, which the target word announces
            prefix[(source, write)] = f'right:{target}'
        elif direction == 'S':
            prefix[(source, head_letter[(target, write)])] = 'suffix'
        else:                                   # 'L': the head moves back one
            for cell in cells:
                # the target word already carries its head on the cell just
                # read — unchanged, which this pair checks — so remember the
                # state it announced and check the rewrite on the next symbol
                prefix[(cell, head_letter[(target, cell)])] = f'left:{target}'

    table = {'prefix': prefix,
             'suffix': {(a, a): 'suffix' for a in cells},
             'end': {}}

    for state in machine.states:
        # a right move: the target's head sits on the next cell, which holds
        # whatever the source had there — or a blank, when the tape grows
        table[f'right:{state}'] = {
            **{(a, head_letter[(state, a)]): 'suffix' for a in cells},
            (pad, head_letter[(state, blank)]): 'end',
        }

    for (state, read), (target, write, direction) in machine.transitions.items():
        if direction != 'L':
            continue
        source = head_letter[(state, read)]
        row = table.setdefault(f'left:{target}', {})
        row[(source, write)] = 'suffix'
        if write == blank:
            # the written blank would now be the last cell: it is dropped, and
            # the target word is one shorter
            row[(source, pad)] = 'end'

    return partial_dfa(set(machine.alphabet), 2, table, 'prefix',
                       {'suffix', 'end'})
