"""Returns and loops of a level 2 collapsible pushdown system.

Reachability in a collapsible pushdown graph is built out of four kinds of run
(Kartzow 2013, §4), all of them concerned with what a stack can do *before* it
drops below where it started:

* a **return** takes a stack to the one below it,
* a **loop** takes a stack back to itself, and splits into a **high loop**,
  which never drops a letter, and a **low loop**, which drops exactly one and
  writes it back,
* a **1-loop** takes a stack to one with the same topmost word and more words
  underneath.

The pivotal fact is that which of these exist depends only on the stack's
**topmost word** — not on anything below it. So each word `w` has a *summary*:
four relations on the control states, saying between which states a return, a
high loop, a low loop and a 1-loop of `w` exist. Kartzow's Prop. 4.20 then says
the summary of `wσ` is determined by the summary of `w` together with σ, which
makes the summaries the states of a finite automaton reading a word bottom to
top.

**How this computes them.** The paper's own effectiveness argument routes
through µ-calculus model checking on collapsible pushdown graphs, which is a
decision procedure of its own. It is not needed: the decomposition lemmas the
paper proves are already a closed system of rules, each saying how one kind of
run is built from shorter ones —

    return      = high loop, then a pop, or a drop into the word below
                  followed by a return of it, or a pushed level 2 letter
                  collapsed after a 1-loop
    high loop   = (push, loop of the longer word, pop) and (clone, return),
                  closed under composition
    low loop    = drop a level 1 letter, loop of the word below, write it back
    1-loop      = loop, clone, loop, clone, …
    loop        = high loop, or high loop then low loop then high loop

— and the least fixpoint of those rules is the summary. The rules refer to the
summaries of *longer* words, which is why this is one simultaneous fixpoint
over the whole table rather than an induction on word length.

**Why all four together.** A level 2 letter pushed at width d carries a link to
width d−1, and a clone copies the letter *with its link*, so collapsing it from
the copy drops two words at once — overshooting the copy's own return. Such a
run is no composition of two returns, and what covers it is exactly a 1-loop:
push the letter, let the stack grow underneath while the topmost word comes
back, then collapse. Returns therefore need 1-loops, which need loops, which
need returns.

Reference: A. Kartzow, *Collapsible Pushdown Graphs of Level 2 are
Tree-Automatic*, LMCS 9(1), 2013, §4.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Tuple

from autstr.collapsible import SEP, Level2CPS
from autstr.sparse_tree_automata import SparseTreeAutomaton, Tree
from autstr.utils.tree_automata_tools import partial_tree_automaton

#: a relation on control states
Relation = FrozenSet[Tuple[str, str]]

#: a letter of a word, as the summaries see it: a symbol and a link level. The
#: link *value* is always zero here — a summary is asked of a word on its own,
#: where a link into the stack below means nothing.
Letter = Tuple[str, int]


def compose(left: Relation, right: Relation) -> Relation:
    """The relational composition ``left ; right``."""
    if not left or not right:
        return frozenset()
    targets: Dict[str, set] = {}
    for source, middle in right:
        targets.setdefault(source, set()).add(middle)
    return frozenset((source, target)
                     for source, middle in left
                     for target in targets.get(middle, ()))


def closure(relation: Relation, states) -> Relation:
    """The reflexive transitive closure."""
    result = relation | frozenset((state, state) for state in states)
    while True:
        grown = result | compose(result, result)
        if grown == result:
            return result
        result = grown


@dataclass(frozen=True)
class Summary:
    """What the runs of one word are, between which control states.

    :param symbol: the word's topmost symbol.
    :param level: the link level of its topmost letter.
    :param ret: pairs (q, q') with a return — a run to the stack below.
    :param hloop: pairs with a high loop — back to the same stack, never
        dropping the topmost letter.
    :param lloop: pairs with a low loop — dropping the topmost letter and
        writing it back.
    :param oneloop: pairs with a 1-loop — back to the same topmost word, with
        more words underneath.
    """
    symbol: Optional[str] = None
    level: int = 0
    ret: Relation = frozenset()
    hloop: Relation = frozenset()
    lloop: Relation = frozenset()
    oneloop: Relation = frozenset()

    @property
    def loop(self) -> Relation:
        """Every loop: a high loop, or a high loop, a low loop and a high loop
        in sequence (Kartzow, Cor. 4.17)."""
        return self.hloop | compose(compose(self.hloop, self.lloop),
                                    self.hloop)

    def relations(self):
        return self.ret, self.hloop, self.lloop, self.oneloop


#: the summary of the empty word: nothing below the bottom letter, so no run of
#: any kind, and no letter to drop onto
EMPTY = Summary()


class Summaries:
    """The summaries of the words a system can build.

    The rules for a word's runs refer to the runs of *longer* words — pushing a
    letter and dropping it again is how a run stays where it is — so this is
    one simultaneous least fixpoint rather than an induction on length. The
    fixpoint is taken over the words themselves, up to a length bound, with
    everything longer treated as having no runs at all.

    That makes the result an **under-approximation**: every run it reports is
    real, and one it misses would need a word longer than the bound to be
    written down. The bound is raised until the summaries stop changing, which
    is where the fixpoint has been reached — for the automaton of Kartzow's
    Prop. 4.20 the summaries of long words repeat, so this terminates on the
    systems it is meant for, and `converged` says whether it did.

    :param system: the collapsible pushdown system.
    :param depth: how many letters above the bottom to unroll before raising
        the bound; raised up to `limit` while the summaries keep growing.
    :param limit: the longest word the fixpoint will consider.
    """

    def __init__(self, system: Level2CPS, depth: int = 3,
                 limit: int = 7) -> None:
        self.system = system
        self.states = list(system.states)
        self.letters = [(symbol, level) for symbol in system.symbols
                        if symbol != system.bottom for level in (1, 2)]
        self.bottom = (system.bottom, 1)
        self._moves = {symbol: self._transitions(symbol)
                       for symbol in system.symbols}
        self.converged = False
        self.values: Dict[tuple, Summary] = {}
        self.depth = depth
        previous_bound = 0
        #: the words whose summary stopped growing when the bound last rose —
        #: a value can only grow, so agreeing twice means it is the truth
        self.stable: set = set()
        for bound in range(depth, limit + 1):
            previous, self.values = self.values, self._fixpoint(bound)
            self.depth = bound
            self.stable = {path for path in previous
                           if previous[path] == self.values[path]}
            # the longest words of a round are the ones whose own extensions
            # were pinned empty, so they always improve when the bound rises;
            # convergence is about the words below that edge
            if previous and all(
                    previous[path] == self.values[path] for path in previous
                    if len(path) < previous_bound):
                self.converged = True
                break
            previous_bound = bound

    # -- the transitions available on one topmost symbol ----------------
    def _transitions(self, symbol: str) -> Dict[object, Relation]:
        """The moves of the system when `symbol` is on top, by operation."""
        moves: Dict[object, set] = {}
        for rule in self.system.rules:
            if rule.symbol is not None and rule.symbol != symbol:
                continue
            operation = rule.operation
            if operation.kind == 'push':
                key = ('push', operation.symbol, operation.level)
            elif operation.kind == 'pop':
                key = f'pop{operation.level}'
            else:
                key = operation.kind
            moves.setdefault(key, set()).add((rule.state, rule.target))
        return {key: frozenset(pairs) for key, pairs in moves.items()}

    def moves(self, symbol: str, key) -> Relation:
        return self._moves.get(symbol, {}).get(key, frozenset())

    def drops(self, letter: Letter) -> Relation:
        """The moves that remove the topmost letter: a pop of level 1, and a
        collapse when the link is of level 1 — a level 1 link always points at
        the preceding letter, so collapsing on one is popping it."""
        symbol, level = letter
        dropped = self.moves(symbol, 'pop1')
        if level == 1:
            dropped |= self.moves(symbol, 'collapse')
        return dropped

    # -- the fixpoint ---------------------------------------------------
    def _paths(self, bound: int):
        """Every word of at most `bound` letters above the bottom one."""
        paths, frontier = [(self.bottom,)], [(self.bottom,)]
        for _ in range(bound):
            frontier = [path + (letter,) for path in frontier
                        for letter in self.letters]
            paths.extend(frontier)
        return paths

    def _fixpoint(self, bound: int) -> Dict[tuple, Summary]:
        """The least fixpoint of the rules over the words of that length.

        Every rule is monotone in the summaries it reads, so starting from "no
        runs at all" and applying them until nothing changes climbs to the
        least fixpoint — which is the true summary for every word whose runs
        stay within the bound.
        """
        values = {path: Summary(path[-1][0], path[-1][1])
                  for path in self._paths(bound)}
        while True:
            grown = {}
            for path, current in values.items():
                below = values[path[:-1]] if len(path) > 1 else EMPTY
                grown[path] = self._solve(below, path, values, current)
            if grown == values:
                return values
            values = grown

    def transitions(self) -> Tuple[Summary, Dict[Tuple[Summary, Letter],
                                                 Summary]]:
        """The summaries as an automaton: the summary of the bottom letter,
        and a table saying how one letter extends a summary.

        This is the merge Kartzow's Prop. 4.20 licenses — words with the same
        summary behave alike, so they are one state. Words at the edge of the
        fixpoint are left out of the merge: their own extensions were pinned
        empty, so what they say about a letter is not yet the truth.
        """
        if not self.converged:
            raise ValueError(
                f"the summaries were still growing at {self.depth} letters, so "
                f"words that look alike need not behave alike yet; raise the "
                f"limit")
        # a word near the edge has its own extensions pinned empty, so what
        # it says about a letter is short of the truth; only the words whose
        # summary has stopped growing may speak
        table: Dict[Tuple[Summary, Letter], Summary] = {}
        for path in sorted(self.stable, key=len):
            summary = self.values[path]
            for letter in self.letters:
                longer = self.values.get(path + (letter,))
                if longer is None or path + (letter,) not in self.stable:
                    continue
                seen = table.setdefault((summary, letter), longer)
                if seen != longer:
                    raise ValueError(
                        f"two words with the same summary disagree about the "
                        f"letter {letter!r}, so the summaries do not determine "
                        f"the automaton; raise the limit")
        start = self.values[(self.bottom,)]

        # the merge has to reproduce what the fixpoint computed
        for path in self.stable:
            summary = self.values[path]
            walked = start
            for letter in path[1:]:
                walked = table.get((walked, letter))
                if walked is None:
                    break
            if walked is not None and walked != summary:
                raise ValueError(
                    f"the merged automaton disagrees with the summary of "
                    f"{path}; raise the limit")
        return start, table

    def of_word(self, word) -> Summary:
        """The summary of a word, given as its letters from the bottom up."""
        path = tuple((symbol, level) for symbol, level in word)
        if path not in self.values:
            raise ValueError(
                f"the fixpoint was taken over words of up to {self.depth} "
                f"letter(s) above the bottom, which {path} exceeds")
        return self.values[path]

    def _solve(self, below: Summary, path: tuple, values: Dict[tuple, Summary],
               current: Summary) -> Summary:
        """One round of the rules for the word `path`.

        `below` is the summary of the word without its topmost letter, and
        `values` supplies the longer words — a letter pushed on top of this one
        makes a word one longer, and what that word's runs are is what says
        whether the letter can be pushed and dropped again.
        """
        symbol, level = path[-1]
        drop = self.drops(path[-1])             # remove this letter
        clone = self.moves(symbol, 'clone')
        pop2 = self.moves(symbol, 'pop2')
        pushes = [(key, pairs) for key, pairs in self._moves.get(
            symbol, {}).items() if isinstance(key, tuple)]
        longer = {key: values.get(path + ((key[1], key[2]),), EMPTY)
                  for key, _ in pushes}

        # a high loop: push a letter, loop on the longer word, and drop it
        # again; or clone, and return the copy
        high = compose(clone, current.ret)
        for key, pairs in pushes:
            high |= compose(compose(pairs, longer[key].loop),
                            self.drops((key[1], key[2])))
        high = closure(high, self.states)

        # a low loop drops this letter -- only a level 1 link may be dropped
        # and written back as it was -- loops below, and writes it back
        low = frozenset()
        if level == 1 and below is not EMPTY:
            low = compose(compose(drop, below.loop),
                          self.moves(below.symbol, ('push', symbol, 1)))

        # a 1-loop leaves the stack taller than it found it, with the topmost
        # word back as it was. The growth may happen here, by cloning this
        # word; above, by pushing a letter and growing under the longer word
        # before dropping it again; or below, by dropping this letter and
        # growing under the word beneath before writing it back.
        loop = high | compose(compose(high, low), high)
        grow = clone
        for key, pairs in pushes:
            grow |= compose(compose(pairs, longer[key].oneloop),
                            self.drops((key[1], key[2])))
        if level == 1 and below is not EMPTY:
            grow |= compose(compose(drop, below.oneloop),
                            self.moves(below.symbol, ('push', symbol, 1)))
        # at least one of those, with loops in between and around
        once = compose(compose(loop, grow), loop)
        one = compose(once, closure(once, self.states))

        # a return ends by popping the word, by dropping this letter and
        # returning the word below, or by collapsing a level 2 letter pushed
        # on top after the stack has grown underneath it
        end = pop2 | compose(drop, below.ret)
        for key, pairs in pushes:
            inner = longer[key]
            # pushing does not change the width, so a return of the longer
            # word lands exactly where a return of this one has to
            end |= compose(pairs, inner.ret)
            if key[2] != 2:
                continue                        # only a level 2 link collapses
            # the letter points one word down, so collapsing it ends the
            # return from wherever the stack has grown to -- or from right
            # here, if it never grew
            end |= compose(compose(pairs, inner.loop | inner.oneloop),
                           self.moves(key[1], 'collapse'))
        ret = compose(high, end)

        return Summary(symbol, level, ret, high, low, one)

    def __repr__(self):
        return (f"<Summaries of {self.system!r}: {len(set(self.values.values()))}"
                f" distinct over {len(self.values)} words, depth {self.depth}"
                f"{'' if self.converged else ', not converged'}>")


# ----------------------------------------------------------------------
# reading a word down the tree
# ----------------------------------------------------------------------

class Annotation:
    """The summary of the word from the root of an encoding tree to each node,
    written on a tape of its own.

    Kartzow's automata ask, at a node `d`, which returns and loops the stack
    that node stands for has — and that stack's topmost word is the word read
    from the root *down* to `d`. A bottom-up automaton cannot read downwards,
    so the answer is carried on a second tape, where the check becomes local: a
    node's annotation is its parent's extended by the node's own letter, and a
    separator carries its parent's along unchanged.

    The tape is scaffolding — the construction pushes it through a projection
    and then drops it, which is what
    `autstr.utils.tree_automata_tools.restrict_alphabet` is for.

    :param encoding: the tree alphabet of the system, from `autstr.collapsible`.
    :param summaries: the summaries of that system's words.
    """

    #: the annotation of the root, where no letter has been read yet
    START = '~start'

    def __init__(self, encoding, summaries: Summaries) -> None:
        self.encoding = encoding
        self.summaries = summaries
        first, table = summaries.transitions()
        self.names = {}
        for summary in [first] + [value for _, value in sorted(
                table.items(), key=lambda item: repr(item[0]))]:
            self.names.setdefault(summary, f'~{len(self.names)}')
        #: annotation letter -> letter -> annotation letter
        self.table = {(self.names[summary], letter): self.names[value]
                      for (summary, letter), value in table.items()}
        self.first = self.names[first]
        self.letters = [self.START] + sorted(self.names.values())
        self.alphabet = set(encoding.alphabet) | set(self.letters)

    def extend(self, annotation: str, label: str) -> Optional[str]:
        """The annotation a node carries, given its parent's and its own
        label — or None where no encoding tree has that shape."""
        if label == SEP:
            return annotation                   # a separator reads no letter
        if label not in self.encoding.letters:
            return None                         # a state labels only the root
        symbol, _, level = label.rpartition(':')
        if annotation == self.START:
            return self.first if label == self.encoding.bottom_label else None
        return self.table.get((annotation, (symbol, int(level))))

    def of_tree(self, tree, annotation: Optional[str] = None) -> Tree:
        """The annotation of a configuration tree, as a tree of its own — the
        oracle the automaton is checked against."""
        annotation = self.START if annotation is None else annotation
        here = annotation if tree.label in self.encoding.states \
            else self.extend(annotation, tree.label)
        if here is None:
            raise ValueError(f"{tree.label!r} cannot follow {annotation!r}")
        return Tree(here,
                    self.of_tree(tree.left, here) if tree.left else None,
                    self.of_tree(tree.right, here) if tree.right else None)

    def automaton(self) -> SparseTreeAutomaton:
        """Two tapes: a configuration tree, and its annotation.

        A node's state is what a parent has to know about it — the annotation
        it carries and the label it carries — since the parent is where the
        two can be compared.
        """
        table = {}
        pairs = [(label, annotation) for label in self.encoding.nodes
                 for annotation in self.names.values()]
        options = [None] + [f'{annotation}|{label}'
                            for label, annotation in pairs]

        def fits(parent: str, child) -> bool:
            """Whether a child may hang below a node annotated `parent`."""
            if child is None:
                return True
            annotation, _, label = child.partition('|')
            return self.extend(parent, label) == annotation

        for label, annotation in pairs:
            for left in options:
                for right in options:
                    if fits(annotation, left) and fits(annotation, right):
                        table[(left, right, (label, annotation))] = \
                            f'{annotation}|{label}'
        # the root carries the state and reads no letter of its own
        for state in self.encoding.states:
            for left in options:
                if fits(self.START, left):
                    table[(left, None, (state, self.START))] = 'accept'
        return partial_tree_automaton(self.alphabet, 2, table, {'accept'})
