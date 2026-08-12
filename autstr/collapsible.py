"""Level 2 collapsible pushdown systems and their configuration graphs.

A collapsible pushdown stack of level 2 is a stack of stacks in which every
letter additionally carries a *collapse link* to some part of the stack lying
below it — a record of what the stack looked like when the letter was pushed.
The ``collapse`` operation throws the stack back to that recorded point in one
step, which is what makes these systems strictly more expressive than ordinary
higher-order pushdown systems: they are the operational counterpart of
higher-order recursion schemes.

**Why the tree engine.** Kartzow (2010) proved that level 2 collapsible
pushdown graphs are *tree*-automatic, and that is the only automatic route to
them: their MSO theory is undecidable, so the infinite-tree machinery that
serves ordinary pushdown graphs (Muller-Schupp, Caucal) has nothing to offer
here, while a finite-tree presentation gives the whole first-order theory. The
tradeoff is exactly the opposite of the one for ordinary pushdown graphs, whose
point is decidable MSO. It is also tight: Broadbent showed that at level 3 even
first-order model checking becomes undecidable.

**The encoding.** A stack ``w_1 : w_2 : … : w_n`` is a list of words, and
consecutive words share long prefixes, because that is the only way ``clone_2``
can make new ones. So the words are laid into one tree: a *block* is a maximal
run of consecutive words sharing their first two letters, blocks of a blockline
hang off each other as right children (``1``-successors), and the blockline a
block induces — the same words with their shared first letter removed — hangs
below it as a left child (``0``-successor). Every initial left-closed path of
the tree is then one word of the stack. Collapse links are *not* stored: a
level 1 link always points to the preceding letter, and a level 2 link on a
node ``d`` points to the substack of width ``|{d' a right child : d' ≤ d}|``,
which the position of ``d`` already determines. That is what makes the encoding
a bijection between configurations and a regular set of trees, and so the whole
structure tree-automatic without a quotient.

    >>> system = Level2CPS(                       # Hague et al.'s example
    ...     transitions=[('0', None, 'Cl', '1', 'clone'),
    ...                  ('1', None, 'A', '0', 'push a 2'),
    ...                  ('1', None, "A'", '2', 'push a 2'),
    ...                  ('2', 'a', 'P', '2', 'pop 1'),
    ...                  ('2', 'a', 'Co', '0', 'collapse')])
    >>> graph = system.configuration_graph()
    >>> graph.is_deterministic()               # two rules fire in state 1
    False
    >>> graph.check('exists x.(not exists y.(E(x,y)))')    # some are stuck
    True

**Reachability, and the contrast with Turing machines.** For a level 2
collapsible pushdown graph the reachability relation is itself tree-automatic
(Kartzow, Prop. 5.1), so first-order logic *with* reachability is decidable
here — the very question that `autstr.turing` cannot answer, since for a
configuration graph of a Turing machine reachability is the halting problem.
That construction is not implemented: it is exponential in the number of
control states and takes up the bulk of the paper, whereas everything below is
elementary. This module presents the one-step relation, over *all*
configurations rather than only those reachable from an initial one — the same
reading Kartzow's result takes, and the only one available without the
reachability construction.

References:

* A. Kartzow, *Collapsible Pushdown Graphs of Level 2 are Tree-Automatic*,
  Logical Methods in Computer Science 9(1), 2013 (STACS 2010).
* M. Hague, A. S. Murawski, C.-H. L. Ong, O. Serre, *Collapsible Pushdown
  Automata and Recursion Schemes*, LICS 2008.
* C. Broadbent, *The Limits of Decidability for First Order Logic on CPDA
  Graphs*, STACS 2012.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from autstr.infinite_graphs import InfiniteGraph
from autstr.sparse_tree_automata import Tree
from autstr.utils.tree_automata_tools import (
    attach_padding, expand, minimize, partial_tree_automaton,
)

#: the padding symbol of the tree convolutions; it must sort before every
#: other letter of the alphabet
PAD = '*'

#: the label of a separator node — a node that splits one block from the next,
#: written ``ε`` in the literature because it repeats no letter
SEP = '.'

#: the default bottom-of-stack symbol
BOTTOM = '⊥'


@dataclass(frozen=True)
class Letter:
    """A stack letter: a symbol plus its collapse link.

    :param symbol: the stack symbol.
    :param level: the level of the collapse link, 1 or 2.
    :param link: at level 2, the width of the substack the link points to; a
        level 1 link always points to the preceding letter, so it carries no
        value of its own.
    """
    symbol: str
    level: int = 1
    link: Optional[int] = None

    def label(self) -> str:
        """The tree label of this letter — the symbol and the link level, the
        only parts the encoding stores."""
        return f'{self.symbol}:{self.level}'

    def __repr__(self):
        return self.symbol if self.level == 1 \
            else f'{self.symbol}[{self.link}]'


@dataclass(frozen=True)
class Stack:
    """A level 2 collapsible stack: a nonempty list of nonempty words.

    The operations return ``None`` where they are undefined — popping the last
    letter of a word, popping the last word, or collapsing on a link that
    points nowhere — rather than raising, so that a configuration with no
    successor is an ordinary answer rather than an error.
    """
    words: Tuple[Tuple[Letter, ...], ...]

    @property
    def width(self) -> int:
        """The number of words — the ``n`` of ``w_1 : … : w_n``."""
        return len(self.words)

    def top(self) -> Letter:
        """The topmost letter: the last letter of the last word."""
        return self.words[-1][-1]

    def clone(self) -> 'Stack':
        """``clone_2``: duplicate the topmost word, links and all."""
        return Stack(self.words + (self.words[-1],))

    def push(self, symbol: str, level: int = 1) -> 'Stack':
        """``push_{symbol,level}``: write a letter onto the topmost word. A
        level 2 link points at the stack below the topmost word, whose width is
        recorded; a level 1 link needs no record."""
        link = None if level == 1 else self.width - 1
        letter = Letter(symbol, level, link)
        return Stack(self.words[:-1] + (self.words[-1] + (letter,),))

    def pop1(self) -> Optional['Stack']:
        """``pop_1``: drop the topmost letter, if the topmost word has one to
        spare."""
        if len(self.words[-1]) < 2:
            return None
        return Stack(self.words[:-1] + (self.words[-1][:-1],))

    def pop2(self) -> Optional['Stack']:
        """``pop_2``: drop the topmost word, if it is not the only one."""
        if self.width < 2:
            return None
        return Stack(self.words[:-1])

    def collapse(self) -> Optional['Stack']:
        """``collapse``: jump to the stack the topmost letter's link points to.

        At level 1 that is the preceding letter, so the operation coincides
        with `pop1`; at level 2 it is the substack of the recorded width, so it
        is a whole run of `pop2` at once.
        """
        top = self.top()
        if top.level == 1:
            return self.pop1()
        return Stack(self.words[:top.link]) if top.link else None

    def apply(self, operation: 'Operation') -> Optional['Stack']:
        """The stack this operation produces, or None where it is undefined."""
        if operation.kind == 'clone':
            return self.clone()
        if operation.kind == 'push':
            return self.push(operation.symbol, operation.level)
        if operation.kind == 'pop':
            return self.pop1() if operation.level == 1 else self.pop2()
        return self.collapse()

    def __repr__(self):
        return ' : '.join(''.join(repr(letter) for letter in word)
                          for word in self.words)


@dataclass(frozen=True)
class Configuration:
    """A configuration: a control state and a stack."""
    state: str
    stack: Stack

    def __repr__(self):
        return f'{self.state} {self.stack!r}'


@dataclass(frozen=True)
class Operation:
    """A stack operation, as it appears in a transition rule.

    :param kind: ``'clone'``, ``'push'``, ``'pop'`` or ``'collapse'``.
    :param symbol: the symbol a push writes.
    :param level: the level of a push's link, or of a pop.
    """
    kind: str
    symbol: Optional[str] = None
    level: Optional[int] = None

    @staticmethod
    def parse(spec) -> 'Operation':
        """An operation from its spelling: ``'clone'``, ``'push a'``,
        ``'push a 2'``, ``'pop 1'``, ``'pop 2'`` or ``'collapse'``."""
        if isinstance(spec, Operation):
            return spec
        parts = str(spec).split()
        if parts == ['clone'] or parts == ['clone', '2']:
            return Operation('clone')
        if parts == ['collapse']:
            return Operation('collapse')
        if len(parts) == 2 and parts[0] == 'pop' and parts[1] in ('1', '2'):
            return Operation('pop', level=int(parts[1]))
        if parts[:1] == ['push'] and len(parts) in (2, 3):
            level = int(parts[2]) if len(parts) == 3 else 1
            if level in (1, 2):
                return Operation('push', symbol=parts[1], level=level)
        raise ValueError(
            f"{spec!r} is no stack operation; write one of "
            f"'clone', 'push <symbol> [1|2]', 'pop 1', 'pop 2', 'collapse'")

    def __repr__(self):
        if self.kind == 'push':
            return f'push {self.symbol} {self.level}'
        if self.kind == 'pop':
            return f'pop {self.level}'
        return self.kind


def initial_stack(bottom: str = BOTTOM) -> Stack:
    """The initial stack ``⊥_2``: one word holding the bottom symbol."""
    return Stack(((Letter(bottom),),))


# ----------------------------------------------------------------------
# the encoding: a configuration as a finite binary tree
# ----------------------------------------------------------------------

def encode_stack(stack: Stack) -> Tree:
    """The tree of a stack: blocks of a blockline hang off each other to the
    right, the blockline a block induces hangs below it to the left."""
    return _encode_blockline(list(stack.words),
                             stack.words[0][0].label())


def _encode_blockline(words: List[Tuple[Letter, ...]], label: str) -> Tree:
    """The encoding of a blockline — a list of words all beginning with the
    same letter, the one `label` names.

    The first maximal block is the longest run of words that agree on their
    first *two* letters; it becomes the left subtree, with its shared first
    letter stripped, and whatever is left of the blockline becomes the right
    subtree, which repeats no label.
    """
    first = words[0]
    if len(first) == 1:                        # a block that is one letter
        if len(words) == 1:
            return Tree(label)
        return Tree(label, None, _encode_blockline(words[1:], SEP))

    second = first[1]
    size = 1
    while size < len(words) and len(words[size]) > 1 and \
            words[size][1] == second:
        size += 1
    induced = _encode_blockline([word[1:] for word in words[:size]],
                                second.label())
    if size == len(words):
        return Tree(label, induced, None)
    return Tree(label, induced, _encode_blockline(words[size:], SEP))


def decode_stack(tree: Tree) -> Stack:
    """The stack a tree encodes.

    Every word ends where the descent to the left ends, and every node that is
    a right child starts the next word. The collapse links are read off the
    positions: a level 2 link points to the substack whose width is the number
    of right children up to and including this node in the traversal.
    """
    words: List[Tuple[Letter, ...]] = []
    separators = 0                             # right children seen so far

    def visit(node: Tree, prefix: Tuple[Letter, ...], is_right: bool) -> None:
        nonlocal separators
        if is_right:
            separators += 1
        here = prefix if node.label == SEP \
            else prefix + (_parse_label(node.label, separators),)
        if node.left is None:
            words.append(here)                 # the word ends here
        else:
            visit(node.left, here, False)
        if node.right is not None:
            visit(node.right, here, True)

    visit(tree, (), False)
    return Stack(tuple(words))


def _parse_label(label: str, separators: int) -> Letter:
    """The letter a node label names, with its link restored from the number
    of right children up to that node."""
    symbol, _, level = label.rpartition(':')
    if level not in ('1', '2') or not symbol:
        raise ValueError(f"{label!r} labels no stack letter")
    return Letter(symbol, 1) if level == '1' \
        else Letter(symbol, 2, separators)


def encode_configuration(configuration: Configuration) -> Tree:
    """The tree of a configuration: the state labels the root and the stack
    hangs below it to the left."""
    return Tree(f'<{configuration.state}>',
                encode_stack(configuration.stack), None)


def decode_configuration(tree: Tree) -> Configuration:
    """The configuration a tree encodes."""
    if tree.left is None or tree.right is not None or \
            not (tree.label.startswith('<') and tree.label.endswith('>')):
        raise ValueError(f"{tree.label!r} roots no configuration")
    return Configuration(tree.label[1:-1], decode_stack(tree.left))


# ----------------------------------------------------------------------
# the automata
# ----------------------------------------------------------------------

class _Encoding:
    """The tree alphabet of one system, and every automaton authored over it.

    The stack operations are tree rewrites, and each is small: an operation
    touches the *topmost* word, which lives on the path that takes every right
    child it can and a left child otherwise — the last path of the traversal.
    So each automaton reads a convolution of two configuration trees, finds a
    bounded rewrite at the end of that path, and checks that the two trees
    agree everywhere else. Carrying a finished rewrite up to the root is what
    pins it to the topmost word: nothing may follow it in the traversal.
    """

    #: the two trees agree on this subtree
    SAME = 'same'
    #: the rewrite is done, and nothing in this subtree follows it
    DONE = 'done'
    #: the whole convolution is a step
    ACCEPT = 'accept'

    def __init__(self, states: Sequence[str], symbols: Sequence[str],
                 bottom: str) -> None:
        self.bottom_label = f'{bottom}:1'
        #: the labels of letter nodes: the bottom symbol carries a level 1
        #: link and never occurs anywhere but at the bottom
        self.letters = [self.bottom_label] + [
            f'{symbol}:{level}' for symbol in symbols if symbol != bottom
            for level in (1, 2)]
        self.states = [f'<{state}>' for state in states]
        #: everything that can label a node of a stack tree
        self.nodes = self.letters + [SEP]
        self.alphabet = set(self.nodes) | set(self.states) | {PAD}

    # -- the domain ----------------------------------------------------
    def universe(self):
        """The encoding trees.

        A configuration tree is a state over a bottom node; below that, left
        children are letters and right children are separators. The one
        further condition is Kartzow's: no node may have ``(σ,1)`` both as its
        left child and as its right child's left child. Two adjacent blocks
        would then begin with the same letter — a level 1 link is fixed by its
        position, so those blocks are really one — and the encoding would stop
        being injective. Level 2 links carry their own value, so the same
        label may legitimately occur twice.
        """
        table = {}
        marks = ['-'] + [label.rpartition(':')[0] for label in self.letters
                         if label.endswith(':1')]
        # the bottom symbol labels the bottom node and nothing else, so it is
        # never anybody's left child
        lefts = [None] + [f'letter {label}' for label in self.letters
                          if label != self.bottom_label]
        rights = [None] + [f'sep {mark}' for mark in marks]

        def mark_of(left):
            """The symbol this node's left child announces, when the left
            child carries a level 1 link."""
            if left is None:
                return '-'
            label = left.partition(' ')[2]
            return label.rpartition(':')[0] if label.endswith(':1') else '-'

        for label in self.nodes:
            for left in lefts:
                for right in rights:
                    if right is not None and \
                            mark_of(left) == right.partition(' ')[2] != '-':
                        continue                 # the two blocks are one block
                    if label == SEP:
                        target = f'sep {mark_of(left)}'
                    elif label == self.bottom_label:
                        target = 'bottom'
                    else:
                        target = f'letter {label}'
                    table[(left, right, (label,))] = target
        for state in self.states:
            table[('bottom', None, (state,))] = self.ACCEPT
        return self._build(1, table)

    def equality(self):
        """``x = y``: the same tree twice."""
        table = self._agreement({})
        for state in self.states:
            table[(self.SAME, None, (state, state))] = self.ACCEPT
        return self._build(2, table)

    # -- the topmost letter --------------------------------------------
    def top_letter(self, accepted):
        """The configurations whose topmost letter has one of the given
        labels.

        The topmost letter sits at the end of the traversal's last path, so it
        is read off bottom-up: a node inherits the letter its own descent ends
        at — through the right child if it has one, else the left — and falls
        back to its own label when that descent finds none.
        """
        table = {}
        options = [None] + [f'top {label}' for label in self.letters + ['-']]
        for label in self.nodes:
            for left in options:
                for right in options:
                    top, _ = self._descent(label, self._top_of(left),
                                           self._top_of(right))
                    table[(left, right, (label,))] = f'top {top}'
        for label in accepted:
            for state in self.states:
                table[(f'top {label}', None, (state,))] = self.ACCEPT
        return self._build(1, table)

    @staticmethod
    def _top_of(state):
        """The letter a `top`/`gone` state names, or None for an absent
        child."""
        return None if state is None else state.partition(' ')[2].split()[0]

    @staticmethod
    def _descent(label, left_top, right_top, left_pure=True):
        """Where the descent from a node ends, and whether it got there
        without ever taking a right child.

        The descent takes the right child when there is one, so a letter found
        below a right child is not reached purely by going left — which is
        exactly the distinction ``collapse`` needs, since a link points at the
        *last* separator the descent passed on its way down.
        """
        own = label if label != SEP else '-'
        if right_top is not None:
            return (right_top, False) if right_top != '-' else (own, True)
        if left_top is not None:
            return (left_top, left_pure) if left_top != '-' else (own, True)
        return own, True

    # -- one stack operation at a time ---------------------------------
    def clone(self):
        """``clone_2``: the topmost word is duplicated. Its copy shares all of
        it, so the copy contributes no letter — just a separator, hung as a
        right child on the very last node of the traversal."""
        table = self._skeleton()
        table[(None, None, (PAD, SEP))] = 'new'
        for label in self.nodes:
            table[(None, 'new', (label, label))] = self.DONE
        return self._build(2, table)

    def push(self, symbol: str, level: int):
        """``push_{symbol,level}``: one letter is written onto the topmost
        word, as a left child of the last node.

        When the letter carries a level 1 link this can make the topmost word
        agree again with its neighbour — the two blocks the last separator
        divides begin with the same letter, so they merge back into one. The
        separator then disappears and the topmost word joins the block to its
        left, at the far end of that block's chain of separators. Both trees
        are produced; the merging case is the one whose plain reading is not
        an encoding tree at all, so the domain sorts them out.
        """
        written = f'{symbol}:{level}'
        table = self._skeleton()
        table[(None, None, (PAD, written))] = 'new'
        for label in self.nodes:
            table[('new', None, (label, label))] = self.DONE
        plain = self._build(2, table)
        if level == 2:
            return plain

        table = self._skeleton()
        table[(None, None, (PAD, SEP))] = 'new'        # the block joins here
        table[(None, None, (SEP, PAD))] = 'gone'       # the separator goes
        for label in self.nodes:
            spine = 'chain =' if label == written else 'chain !'
            for left in (None, self.SAME):
                table[(left, 'new', (label, label))] = spine
                for below in ('chain =', 'chain !'):
                    table[(left, below, (label, label))] = spine
            table[('chain =', 'gone', (label, label))] = self.DONE
        return minimize(plain.union(self._build(2, table)))

    def pop1(self):
        """``pop_1``: the topmost letter goes.

        Either it is a node of its own — the last node of the traversal — and
        simply disappears, or the topmost word ends at a separator, in which
        case the letter is shared with the word below and what moves is the
        separator: it climbs to just above the letter that was dropped.
        """
        table = self._skeleton()
        for label in self.letters:
            if label != self.bottom_label:       # the bottom never pops
                table[(None, None, (label, PAD))] = 'gone leaf'
        for label in self.nodes:
            table[('gone leaf', None, (label, label))] = self.DONE

        table[(None, None, (SEP, PAD))] = 'gone'
        table[(None, None, (PAD, SEP))] = 'new'
        for below in ('gone', 'separators'):
            for left in (None, self.SAME):
                table[(left, below, (SEP, SEP))] = 'separators'
                for label in self.letters:
                    table[(left, below, (label, label))] = 'letter'
        for label in self.nodes:
            table[('letter', 'new', (label, label))] = self.DONE
        return self._build(2, table)

    def pop2(self):
        """``pop_2``: the topmost word goes, and with it every node that no
        other word passes through — the last separator of the tree and the
        letters hanging below it."""
        table = self._skeleton()
        table[(None, None, (SEP, PAD))] = 'gone'
        for label in self.letters:               # the word's own letters
            table[(None, None, (label, PAD))] = 'gone chain'
            table[('gone chain', None, (label, PAD))] = 'gone chain'
        table[('gone chain', None, (SEP, PAD))] = 'gone'
        for label in self.nodes:
            for left in (None, self.SAME):
                table[(left, 'gone', (label, label))] = self.DONE
        return self._build(2, table)

    def collapse2(self):
        """``collapse`` on a level 2 link: the stack is cut back to the width
        the link records.

        That width counts the separators up to the topmost letter, so the cut
        falls exactly at the last separator the descent passed *before*
        reaching that letter — and everything from there on goes. Which
        separator that is has to be read off the deleted part itself, so the
        automaton carries, through the region only the first tree has, both
        the letter its descent ends at and whether it got there without taking
        a right child.
        """
        table = self._skeleton()
        region = [None] + [f'gone {label} {pure}'
                           for label in self.letters + ['-']
                           for pure in ('left', 'right')]
        for label in self.nodes:
            for left in region:
                for right in region:
                    top, pure = self._descent(
                        label, self._top_of(left), self._top_of(right),
                        left is None or left.endswith('left'))
                    table[(left, right, (label, PAD))] = \
                        f'gone {top} {"left" if pure else "right"}'
        for label in self.letters:
            if not label.endswith(':2'):
                continue                         # a level 1 link is a pop_1
            for node in self.nodes:
                for left in (None, self.SAME):
                    table[(left, f'gone {label} left', (node, node))] = \
                        self.DONE
        return self._build(2, table)

    # -- shared pieces --------------------------------------------------
    def _skeleton(self):
        """The transitions every operation shares: the two trees agree
        wherever the rewrite is not, and a finished rewrite climbs to the
        root with nothing after it."""
        return self._climb(self._agreement({}))

    def _agreement(self, table):
        for label in self.nodes:
            for left in (None, self.SAME):
                for right in (None, self.SAME):
                    table[(left, right, (label, label))] = self.SAME
        return table

    def _climb(self, table):
        """Carry a finished rewrite to the root. Coming up from a right child
        anything may sit to the left, but coming up from a left child there
        must be no right child at all: otherwise the rewrite would not have
        been at the end of the traversal, and so not on the topmost word."""
        for label in self.nodes:
            table[(self.DONE, None, (label, label))] = self.DONE
            table[(None, self.DONE, (label, label))] = self.DONE
            table[(self.SAME, self.DONE, (label, label))] = self.DONE
        for source in self.states:
            for target in self.states:
                table[(self.DONE, None, (source, target))] = self.ACCEPT
        return table

    def _build(self, arity, table):
        return partial_tree_automaton(self.alphabet, arity, table,
                                      {self.ACCEPT})

    def lift(self, automaton, position: int, arity: int = 2):
        """A predicate on one configuration, read on one tape of a pair. The
        padding closure comes first: the other tape's tree may reach deeper
        than this one's."""
        return minimize(expand(minimize(attach_padding(automaton, PAD)),
                               arity, [position]))


# ----------------------------------------------------------------------
# the system
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Rule:
    """One transition: in state `state`, with `symbol` on top of the stack,
    read `label`, go to `target` and apply `operation`. A `symbol` of None
    matches any topmost symbol."""
    state: str
    symbol: Optional[str]
    label: str
    target: str
    operation: Operation

    def applies_to(self, configuration: Configuration) -> bool:
        """Whether this rule's state and stack guard match."""
        return self.state == configuration.state and (
            self.symbol is None or
            self.symbol == configuration.stack.top().symbol)


class Level2CPS:
    """A collapsible pushdown system of level 2.

    :param transitions: the rules, as ``(state, symbol, label, target,
        operation)`` tuples. The symbol may be None or ``'-'`` to match any
        topmost symbol; the operation is a `Operation` or its spelling, one of
        ``'clone'``, ``'push <symbol> [1|2]'``, ``'pop 1'``, ``'pop 2'``,
        ``'collapse'``.
    :param bottom: the bottom-of-stack symbol, which no push may write.
    :param initial_state: the control state of the initial configuration;
        the first rule's state when omitted.
    :param states: further control states, if some appear in no rule.
    :param symbols: further stack symbols, likewise.
    """

    #: characters the encoding uses to build tree labels
    RESERVED = ':<>'

    def __init__(self, transitions: Iterable[Sequence],
                 bottom: str = BOTTOM, initial_state: Optional[str] = None,
                 states: Iterable[str] = (),
                 symbols: Iterable[str] = ()) -> None:
        self.bottom = bottom
        self.rules = [self._rule(transition) for transition in transitions]

        found_states = {rule.state for rule in self.rules}
        found_states |= {rule.target for rule in self.rules}
        found_symbols = {bottom}
        for rule in self.rules:
            if rule.symbol is not None:
                found_symbols.add(rule.symbol)
            if rule.operation.kind == 'push':
                found_symbols.add(rule.operation.symbol)

        self.states = sorted(found_states | set(states))
        self.symbols = sorted(found_symbols | set(symbols))
        self.labels = sorted({rule.label for rule in self.rules})
        self.initial_state = initial_state if initial_state is not None else (
            self.rules[0].state if self.rules else
            (self.states[0] if self.states else 'q'))

        if bottom in {rule.operation.symbol for rule in self.rules}:
            raise ValueError(
                f"the bottom symbol {bottom!r} cannot be pushed; it is what "
                f"marks the bottom of a stack")
        if self.initial_state not in self.states:
            self.states = sorted(set(self.states) | {self.initial_state})
        # only states and stack symbols become tree labels; an edge label is
        # never encoded, so anything non-empty will do for one
        for name in self.states + self.symbols:
            self._check(name)
        if not all(self.labels):
            raise ValueError("an edge label may not be empty")

    def _check(self, name: str) -> None:
        if not name:
            raise ValueError("a state or stack symbol may not be empty")
        if any(character in name for character in self.RESERVED):
            raise ValueError(
                f"{name!r} uses one of the reserved characters "
                f"{self.RESERVED!r}, which the tree labels are built from")
        if min(name) <= PAD:
            raise ValueError(
                f"{name!r} would not sort after the padding symbol {PAD!r}, "
                f"which the tree convolutions rely on")

    @staticmethod
    def _rule(transition) -> Rule:
        if isinstance(transition, Rule):
            return transition
        state, symbol, label, target, operation = transition
        return Rule(state, None if symbol in (None, '-') else symbol,
                    label, target, Operation.parse(operation))

    # -- running the system --------------------------------------------
    def initial(self) -> Configuration:
        """The initial configuration: the initial state over ``⊥_2``."""
        return Configuration(self.initial_state, initial_stack(self.bottom))

    def step(self, configuration: Configuration
             ) -> List[Tuple[str, Configuration]]:
        """Every successor of a configuration, as ``(label, configuration)``
        pairs — the Python oracle the automata are checked against, and the
        obvious way to run the system. A collapsible pushdown system is
        nondeterministic, so there may be several, or none."""
        successors = []
        for rule in self.rules:
            if not rule.applies_to(configuration):
                continue
            stack = configuration.stack.apply(rule.operation)
            if stack is not None:
                successors.append((rule.label,
                                   Configuration(rule.target, stack)))
        return successors

    def reachable(self, bound: int = 6) -> List[Configuration]:
        """The configurations reachable from the initial one in at most
        `bound` steps.

        The *unbounded* reachability relation is tree-automatic too (Kartzow,
        Prop. 5.1) but is not built here, so this is a search rather than a
        decision procedure — useful for seeing a system run, and for checking
        the graph against small examples.
        """
        seen, frontier = [self.initial()], [self.initial()]
        for _ in range(bound):
            frontier = [successor for configuration in frontier
                        for _, successor in self.step(configuration)
                        if successor not in seen]
            for configuration in frontier:
                if configuration not in seen:
                    seen.append(configuration)
        return seen

    # -- the configuration graph ---------------------------------------
    def configuration_graph(self, **kwargs) -> 'Level2CPG':
        """The graph of all configurations under one step of the system."""
        return Level2CPG(self, **kwargs)

    # -- element codec --------------------------------------------------
    @staticmethod
    def encode(configuration: Configuration) -> Tree:
        """The tree encoding a configuration."""
        return encode_configuration(configuration)

    @staticmethod
    def decode(tree: Tree) -> Configuration:
        """The configuration a tree encodes."""
        return decode_configuration(tree)

    def __repr__(self):
        return (f"<Level2CPS states={len(self.states)} "
                f"symbols={len(self.symbols)} rules={len(self.rules)}>")


def _names(prefix: str, tokens: Sequence[str]) -> List[str]:
    """Relation names for a family, one per token and in the same order: the
    token itself where it makes a readable symbol, its position otherwise —
    all of them or none, so that the names of one family stay uniform.

    The result is a list rather than a dictionary because two tokens may well
    be equal, and then it is their positions that tell them apart.
    """
    names = [f'{prefix}{token}' for token in tokens]
    if len(set(names)) == len(names) and \
            all(token.isascii() and token.isalnum() for token in tokens):
        return names
    return [f'{prefix}{index}' for index in range(len(tokens))]


class Level2CPG:
    """The configuration graph of a `Level2CPS`, as a tree-automatic
    structure.

    Vertices are *all* configurations — every control state over every level 2
    collapsible stack — and there is a ``γ``-labelled edge wherever one rule
    of the system takes one configuration to another. The domain is regular
    because the encoding is a bijection onto a regular set of trees; the edges
    are regular because every stack operation is a bounded rewrite at the end
    of the tree's last path.

    The presentation carries ``E`` (a step under any label), ``Edge…`` (one
    per label of the system), ``Eq``, and — built on first use — the stack
    operations ``Clone``, ``Push…``, ``Pop1``, ``Pop2``, ``Collapse`` as
    relations of their own, the control-state predicates ``State…``, and the
    predicates ``Top…``, ``Level1``, ``Level2`` for the topmost letter.

    :param system: the `Level2CPS`.
    :param max_states: optional cap on the subset determinizations inside
        projection.
    """

    def __init__(self, system: Level2CPS,
                 max_states: Optional[int] = None) -> None:
        from autstr.symbolic import FunctionCodec
        from autstr.tree_presentations import TreeAutomaticPresentation
        self.system = system
        encoding = _Encoding(system.states, system.symbols, system.bottom)
        self.encoding = encoding

        self.state_names = dict(zip(system.states,
                                    _names('State', system.states)))
        # the bottom symbol is spelled ⊥ by convention, which makes no
        # readable relation name, so it is the one symbol named for its role
        self.symbol_names = dict(zip(system.symbols, _names(
            'Top', ['Bottom' if symbol == system.bottom else symbol
                    for symbol in system.symbols])))
        self.label_names = dict(zip(system.labels,
                                    _names('Edge', system.labels)))
        pushes = sorted({(rule.operation.symbol, rule.operation.level)
                         for rule in system.rules
                         if rule.operation.kind == 'push'})
        self.push_names = dict(zip(pushes, _names(
            'Push', [f'{symbol}{level}' for symbol, level in pushes])))
        self._check_names()

        edges = self._edges()
        presentation = TreeAutomaticPresentation(
            {'U': encoding.universe(), 'Eq': encoding.equality(),
             'E': edges[None],
             **{self.label_names[label]: edges[label]
                for label in system.labels}},
            padding_symbol=PAD, max_states=max_states)
        # the rest is worth having by name, but no query has asked for it yet
        presentation._declare_deferred({
            **{name: (lambda state=state: self._state_automaton(state))
               for state, name in self.state_names.items()},
            **{name: (lambda symbol=symbol: encoding.top_letter(
                [f'{symbol}:{level}' for level in (1, 2)]))
               for symbol, name in self.symbol_names.items()},
            **{f'Level{level}': (lambda level=level: encoding.top_letter(
                [label for label in encoding.letters
                 if label.endswith(f':{level}')])) for level in (1, 2)},
            'Clone': encoding.clone,
            'Pop1': encoding.pop1,
            'Pop2': encoding.pop2,
            'Collapse': self._collapse,
            **{self.push_names[symbol, level]:
               (lambda symbol=symbol, level=level:
                encoding.push(symbol, level))
               for symbol, level in pushes},
        })
        self.presentation = presentation
        self.graph = InfiniteGraph(
            presentation, edge='E', directed=True,
            codec=FunctionCodec(system.encode, system.decode))

    #: the relation names that do not depend on the system
    FIXED = ('U', 'Eq', 'E', 'Clone', 'Pop1', 'Pop2', 'Collapse',
             'Level1', 'Level2')

    def _check_names(self) -> None:
        """No two relations may end up with the same name: the presentation
        holds them in a dictionary, so a clash would quietly drop one instead
        of failing."""
        names = (list(self.FIXED) + list(self.state_names.values())
                 + list(self.symbol_names.values())
                 + list(self.label_names.values())
                 + list(self.push_names.values()))
        clashes = sorted({name for name in names if names.count(name) > 1})
        if clashes:
            raise ValueError(
                f"the states, symbols and labels of this system name the same "
                f"relation twice ({', '.join(clashes)}); rename one of them")

    # -- the automata of this system -----------------------------------
    def _state_automaton(self, state: str):
        """The configurations whose control state is `state`."""
        encoding = self.encoding
        table = {}
        for label in encoding.nodes:
            for left in (None, 'stack'):
                for right in (None, 'stack'):
                    table[(left, right, (label,))] = 'stack'
        table[('stack', None, (f'<{state}>',))] = encoding.ACCEPT
        return partial_tree_automaton(encoding.alphabet, 1, table,
                                      {encoding.ACCEPT})

    def _collapse(self):
        """``collapse``, at either link level: on a level 2 link the stack is
        cut back to the recorded width, and on a level 1 link — which always
        points at the preceding letter — it is exactly ``pop_1``."""
        encoding = self.encoding
        level1 = encoding.top_letter([label for label in encoding.letters
                                      if label.endswith(':1')])
        return minimize(encoding.collapse2().union(minimize(
            encoding.lift(level1, 0).intersection(
                minimize(attach_padding(encoding.pop1(), PAD))))))

    def _operation(self, operation: Operation):
        """The relation of one stack operation."""
        encoding = self.encoding
        if operation.kind == 'clone':
            return encoding.clone()
        if operation.kind == 'push':
            return encoding.push(operation.symbol, operation.level)
        if operation.kind == 'pop':
            return encoding.pop1() if operation.level == 1 else encoding.pop2()
        return self._collapse()

    def _edges(self):
        """One step, per label and in total.

        A rule is the conjunction of four conditions — the control state
        before, the topmost symbol, the control state after, and the stack
        operation — so it is built by intersecting four automata rather than
        authored as one. The label's relation is the union over its rules.
        """
        encoding = self.encoding
        operations, predicates = {}, {}

        def operation(key):
            if key not in operations:
                operations[key] = minimize(attach_padding(
                    self._operation(key), PAD))
            return operations[key]

        def predicate(kind, key, position):
            if (kind, key, position) not in predicates:
                automaton = self._state_automaton(key) if kind == 'state' \
                    else encoding.top_letter([f'{key}:{level}'
                                              for level in (1, 2)])
                predicates[kind, key, position] = encoding.lift(
                    automaton, position)
            return predicates[kind, key, position]

        per_label = {label: [] for label in self.system.labels}
        for rule in self.system.rules:
            step = operation(rule.operation)
            step = minimize(step.intersection(
                predicate('state', rule.state, 0)))
            step = minimize(step.intersection(
                predicate('state', rule.target, 1)))
            if rule.symbol is not None:
                step = minimize(step.intersection(
                    predicate('symbol', rule.symbol, 0)))
            per_label[rule.label].append(step)

        edges = {label: _union(pieces) or _empty(encoding)
                 for label, pieces in per_label.items()}
        edges[None] = _union(list(edges.values())) or _empty(encoding)
        return edges

    # -- interface ------------------------------------------------------
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
        """Whether every configuration has at most one successor — first-order,
        and so decidable, even though reachability is not built here."""
        return self.presentation.check(
            'all x.(all y.(all z.((E(x,y) & E(x,z)) -> Eq(y,z))))')

    def __repr__(self):
        return (f"<Level2CPG of {self.system!r} "
                f"labels={len(self.system.labels)}>")


def _union(automata):
    """The union of a list of automata, or None if there is none."""
    if not automata:
        return None
    result = automata[0]
    for automaton in automata[1:]:
        result = minimize(result.union(automaton))
    return result


def _empty(encoding: _Encoding):
    """The empty binary relation."""
    from autstr.tree_presentations import tree_zero
    return tree_zero(2, encoding.alphabet)
