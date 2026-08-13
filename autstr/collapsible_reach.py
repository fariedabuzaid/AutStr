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

from autstr.collapsible import PAD, SEP, Level2CPS
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

    def automaton(self, alphabet=None) -> SparseTreeAutomaton:
        """Two tapes: a configuration tree, and its annotation.

        :param alphabet: the alphabet to build over, when a construction has
            letters of its own beside these — an automaton can only be
            combined with others that read the same one.

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
        return partial_tree_automaton(alphabet or self.alphabet, 2,
                                      table, {'accept'})


# ----------------------------------------------------------------------
# the relations Reach decomposes into
# ----------------------------------------------------------------------

class Relations:
    """The relations whose composition is reachability.

    Kartzow's Remark 4.4 splits every run into four stretches: `A` drops whole
    words, `B` drops letters from the topmost word, `C` pushes letters back on,
    and `D` grows the stack again. All four are reflexive, so reachability is
    their composition and needs no automaton of its own —

        Reach(x,y) ≡ ∃d ∃e ∃f. A(x,d) ∧ B(d,e) ∧ C(e,f) ∧ D(f,y)

    which is a formula the engine evaluates once the four are installed.

    Each is checked on four tapes — the two configurations, the summary
    annotation of the first, and a guessed control state per node — and the
    last two are projected away afterwards.

    :param system: the collapsible pushdown system.
    :param summaries: its summaries; computed if not given.
    """

    #: a node no run visits, and so carries no guessed states
    NONE = '@-'

    def __init__(self, system: Level2CPS, summaries: Optional[Summaries] = None
                 ) -> None:
        from autstr.collapsible import _Encoding
        self.system = system
        self.encoding = _Encoding(system.states, system.symbols, system.bottom)
        self.summaries = summaries or Summaries(system)
        self.annotation = Annotation(self.encoding, self.summaries)
        # a letter the run drops carries the states it is in before and after
        # dropping it, which is what makes every check local
        self.guesses = [self.NONE] + [
            f'@{before},{after}' for before in system.states
            for after in system.states]
        self.alphabet = set(self.annotation.alphabet) | set(self.guesses)

    def summary_of(self, annotation: str) -> Optional[Summary]:
        """The summary an annotation letter stands for."""
        for summary, name in self.annotation.names.items():
            if name == annotation:
                return summary
        return None

    def drop_moves(self, symbol: str, level: int) -> Relation:
        """The transitions that take the topmost letter off: a pop of level 1,
        and a collapse when the link is of level 1."""
        moves = self.summaries.moves(symbol, 'pop1')
        if level == 1:
            moves |= self.summaries.moves(symbol, 'collapse')
        return moves

    def guessed(self, guess: str) -> Optional[Tuple[str, str]]:
        """The pair of states a guess letter carries, or None."""
        if guess == self.NONE:
            return None
        before, _, after = guess[1:].partition(',')
        return before, after

    def dropping(self, annotation: str, label: str, guess: str) -> bool:
        """Whether a letter may be dropped as the guess says: a high loop of
        the word ending at it, then one transition that takes it off.

        The word is the one read from the root down to this node, which is
        what the annotation names — so the check needs nothing but this node's
        own four tapes.
        """
        pair = self.guessed(guess)
        summary = self.summary_of(annotation)
        if pair is None or summary is None or label not in self.encoding.letters:
            return False
        symbol, _, level = label.rpartition(':')
        return pair in compose(summary.hloop,
                               self.drop_moves(symbol, int(level)))

    def b(self) -> SparseTreeAutomaton:
        """``B``: the topmost word loses letters, and nothing goes below what
        is left.

        The two trees agree except along a tail of the first one's last path,
        which is deleted; the second may gain a single separator where its own
        last word now ends. Climbing that tail is the run: at each letter a
        high loop and one drop, the state after one drop being the state before
        the next.
        """
        table = {}
        annotations = list(self.annotation.names.values())
        marks = [(label, annotation)
                 for label in self.encoding.nodes for annotation in annotations]
        pairs = [(before, after) for before in self.system.states
                 for after in self.system.states]

        # the two trees agree here and no run step happens
        for label, annotation in marks:
            for left in (None, 'same'):
                for right in (None, 'same'):
                    table[(left, right, (label, label, annotation, self.NONE))] \
                        = 'same'
        # the separator the second tree gains, where its last word now ends.
        # One is added exactly when one is deleted -- the last word\'s
        # divergence from the word below it moves up -- and never more.
        table[(None, None, (PAD, SEP, PAD, self.NONE))] = 'added'

        # below the deepest letter dropped, the old last separator goes
        for annotation in annotations:
            table[(None, None, (SEP, PAD, annotation, self.NONE))] = 'cut'

        # a letter the run drops: the deepest starts the chain, each one above
        # continues it, and a deleted separator sets the chain\'s mark
        for label, annotation in marks:
            for guess in self.guesses:
                if not self.dropping(annotation, label, guess):
                    continue
                first, last = self.guessed(guess)
                symbols = (label, PAD, annotation, guess)
                table[(None, None, symbols)] = f'chain {first} {last}'
                table[('cut', None, symbols)] = f'cutchain {first} {last}'
                for start_state in self.system.states:
                    for kind in ('chain', 'cutchain'):
                        table[(f'{kind} {start_state} {first}', None,
                               symbols)] = f'{kind} {start_state} {last}'
            if label != SEP:
                continue
            for first, last in pairs:
                # a separator on the way marks the chain; a second one would
                # mean two words ended there, which no chain of pops does
                symbols = (SEP, PAD, annotation, self.NONE)
                table[(f'chain {first} {last}', None, symbols)] = \
                    f'cutchain {first} {last}'
                table[(None, f'chain {first} {last}', symbols)] = \
                    f'cutchain {first} {last}'

        # the chain stops at a node both trees keep. From there up nothing may
        # follow it in either tree: coming from a right child anything already
        # read may sit to the left, but coming from a left child there must be
        # no right child at all.
        for label, annotation in marks:
            plain = (label, label, annotation, self.NONE)
            for first, last in pairs:
                for chain, kind in ((f'chain {first} {last}', 'done'),
                                    (f'cutchain {first} {last}', 'cutdone')):
                    for below in (chain, f'{kind} {first} {last}'):
                        table[(below, None, plain)] = f'{kind} {first} {last}'
                        table[('same', below, plain)] = f'{kind} {first} {last}'
                        table[(None, below, plain)] = f'{kind} {first} {last}'
                # the deleted separator is made good by the added one
                for below in (f'cutchain {first} {last}',
                              f'cutdone {first} {last}'):
                    table[(below, 'added', plain)] = f'grown {first} {last}'
                grown = f'grown {first} {last}'
                table[(grown, None, plain)] = grown
                table[('same', grown, plain)] = grown
                table[(None, grown, plain)] = grown

        # the root: the states the run began and ended in are the two
        # configurations\' own, and a run of no steps leaves the tree alone
        for source in self.system.states:
            for target in self.system.states:
                labels = (f'<{source}>', f'<{target}>', Annotation.START,
                          self.NONE)
                for kind in ('done', 'grown'):
                    table[(f'{kind} {source} {target}', None, labels)] = 'accept'
                if source == target:
                    table[('same', None, labels)] = 'accept'
        return partial_tree_automaton(self.alphabet, 4, table, {'accept'})

    def without_scaffolding(self, checker: SparseTreeAutomaton,
                            annotated: int = 0) -> SparseTreeAutomaton:
        """A four-tape checker as a relation on two configurations.

        The annotation is required to be the real one — that is what the
        annotation automaton says — and then both it and the guess are
        quantified away, leaving the alphabet to be narrowed back to the one
        the configurations are written in.

        :param annotated: which configuration the annotation belongs to. The
            words a run passes through are those of the *longer* of the two,
            which is the first tape where letters come off and the second
            where they go on.
        """
        from autstr.utils.tree_automata_tools import (
            attach_padding, expand, minimize, project, restrict_alphabet,
        )
        annotated = minimize(expand(minimize(attach_padding(
            self.annotation.automaton(self.alphabet), PAD)), 4,
            [annotated, 2]))
        result = minimize(minimize(
            attach_padding(checker, PAD)).intersection(annotated))
        for tape in (3, 2):
            result = minimize(attach_padding(
                project(result, tape, PAD), PAD))
        return restrict_alphabet(result, self.encoding.alphabet)

    def looping(self, annotation: str, guess: str) -> bool:
        """Whether the guess is a high loop of the word this node names."""
        pair = self.guessed(guess)
        summary = self.summary_of(annotation)
        return pair is not None and summary is not None and \
            pair in summary.hloop

    def push_moves(self, annotation: str, label: str) -> Relation:
        """The transitions that write `label` on a stack whose topmost word is
        the one `annotation` names.

        A pop is guarded by the letter it takes off, so a node can check it
        alone; a push is guarded by the letter already on top, which is the one
        *above* the letter written — and which is not this node's label when
        the letter goes below a separator. The summary knows it either way: it
        carries the topmost symbol of the word it names.
        """
        summary = self.summary_of(annotation)
        written, _, level = label.rpartition(':')
        if summary is None or summary.symbol is None or not level.isdigit():
            return frozenset()
        return self.summaries.moves(summary.symbol,
                                    ('push', written, int(level)))

    def c(self) -> SparseTreeAutomaton:
        """``C``: the topmost word gains letters, and the run never dips below
        where it started.

        B read backwards, and the same tree shape with the two configurations
        exchanged: the second one's last path is longer by a tail, and the
        first carries the one separator the second loses. The run is a high
        loop and a push at each letter gained — the loop belonging to the node
        that names its word, the push to the node above, which is where the
        letter it writes is still on top.
        """
        table = {}
        annotations = list(self.annotation.names.values())
        marks = [(label, annotation)
                 for label in self.encoding.nodes for annotation in annotations]
        letters = [label for label in self.encoding.letters]
        states = self.system.states

        def chain(kind, label, first, final):
            return f'{kind}|{label}|{first}|{final}'

        # Where the trees agree, carry up the annotation of the word the top
        # path ends at. With no letter gained at all, C is a single high loop
        # of that word — the run may touch where it started, which is what
        # separates C from A and B — and this is where the root checks it.
        agreed = [f'same {name}' for name in annotations]
        for label, annotation in marks:
            for left in [None] + agreed:
                for right in [None] + agreed:
                    ends = right or left or f'same {annotation}'
                    table[(left, right, (label, label, annotation, self.NONE))] \
                        = ends
        # the separator the first tree carries, where its last word ends, and
        # the one the second loses
        table[(None, None, (SEP, PAD, PAD, self.NONE))] = 'added'
        for annotation in annotations:
            table[(None, None, (PAD, SEP, annotation, self.NONE))] = 'cut'

        # a letter the run pushes. The node checks the high loop of its own
        # word; the push that put the letter there is checked by its parent.
        for label, annotation in marks:
            if label not in letters:
                continue
            for guess in self.guesses:
                if not self.looping(annotation, guess):
                    continue
                first, last = self.guessed(guess)
                symbols = (PAD, label, annotation, guess)
                table[(None, None, symbols)] = chain('chain', label, first, last)
                table[('cut', None, symbols)] = \
                    chain('cutchain', label, first, last)
                for below in letters:
                    moves = self.push_moves(annotation, below)
                    for entered, final in moves:
                        if entered != last:
                            continue
                        for kind in ('chain', 'cutchain'):
                            for deepest in states:
                                table[(chain(kind, below, final, deepest), None,
                                       symbols)] = \
                                    chain(kind, label, first, deepest)

        # a separator the second tree loses, on the way down the tail
        for annotation in annotations:
            symbols = (PAD, SEP, annotation, self.NONE)
            for label in letters:
                for first in states:
                    for final in states:
                        table[(chain('chain', label, first, final), None,
                               symbols)] = chain('cutchain', label, first, final)
                        table[(None, chain('chain', label, first, final),
                               symbols)] = chain('cutchain', label, first, final)

        # where the tail begins: one more high loop, then the push that starts
        # it -- both belonging to this node, which both trees keep
        for label, annotation in marks:
            summary = self.summary_of(annotation)
            plain = (label, label, annotation, self.NONE)
            for below in letters:
                moves = self.push_moves(annotation, below)
                for entered, final in moves:
                    for start in states:
                        if (start, entered) not in summary.hloop:
                            continue
                        for kind, done in (('chain', 'done'),
                                           ('cutchain', 'cutdone')):
                            for deepest in states:
                                source = chain(kind, below, final, deepest)
                                target = f'{done} {start} {deepest}'
                                table[(source, None, plain)] = target
                                for kept in agreed:
                                    table[(kept, source, plain)] = target
                                table[(None, source, plain)] = target
            for first in states:
                for final in states:
                    for kind in ('done', 'cutdone'):
                        below_state = f'{kind} {first} {final}'
                        table[(below_state, None, plain)] = below_state
                        for kept in agreed:
                            table[(kept, below_state, plain)] = below_state
                        table[(None, below_state, plain)] = below_state
                    grown = f'grown {first} {final}'
                    table[(f'cutdone {first} {final}', 'added', plain)] = grown
                    table[(grown, None, plain)] = grown
                    for kept in agreed:
                        table[(kept, grown, plain)] = grown
                    table[(None, grown, plain)] = grown

        for source in states:
            for target in states:
                labels = (f'<{source}>', f'<{target}>', Annotation.START,
                          self.NONE)
                for kind in ('done', 'grown'):
                    table[(f'{kind} {source} {target}', None, labels)] = 'accept'
                # no letter gained: one high loop of the topmost word
                for name in annotations:
                    summary = self.summary_of(name)
                    if summary is not None and \
                            (source, target) in summary.hloop:
                        table[(f'same {name}', None, labels)] = 'accept'
        return partial_tree_automaton(self.alphabet, 4, table, {'accept'})

    def a(self) -> SparseTreeAutomaton:
        """``A``: the stack loses whole words.

        Kartzow's Lemma 4.11 decomposes such a run into pieces that are
        returns (F1), a 1-loop then a level 2 collapse (F2), or a 1-loop then a
        pop that some later F2 closes off (F3). The words dropped are the
        encoding's separators, and the run drops them from the last backwards
        — so within a subtree the chain runs through the right child's
        separators, then the left child's, then the node itself, which is
        reverse traversal order.

        A collapse spans several words at once, which sounds like a link
        reaching across the tree. It is not: a level 2 link records the number
        of separators up to its letter, and the stack it points at is the
        tree's prefix at that separator — which is exactly where the region
        being deleted begins. So an F3-F2 group is a chain climbing the top
        path *inside* that region, a 1-loop and a pop at each letter and a
        1-loop and the collapse at the last, closed off by the region's own
        root. Every check stays local, and the group then contributes a pair of
        states to the outer chain just as a return does.
        """
        table = {}
        annotations = list(self.annotation.names.values())
        marks = [(label, annotation)
                 for label in self.encoding.nodes for annotation in annotations]
        states = self.system.states

        for label, annotation in marks:
            for left in (None, 'same'):
                for right in (None, 'same'):
                    table[(left, right, (label, label, annotation, self.NONE))] \
                        = 'same'

        # inside a word the first tree drops, nothing is guessed
        for label, annotation in marks:
            for left in (None, 'gone'):
                for right in (None, 'gone'):
                    table[(left, right, (label, PAD, annotation, self.NONE))] \
                        = 'gone'

        # a separator the first tree drops: one return of the word it names
        for annotation in annotations:
            summary = self.summary_of(annotation)
            for guess in self.guesses:
                pair = self.guessed(guess)
                if pair is None or summary is None or pair not in summary.ret:
                    continue
                first, last = pair
                symbols = (SEP, PAD, annotation, guess)
                for left in (None, 'gone'):
                    table[(left, None, symbols)] = f'ret {first} {last}'
                    # the words to the right go first, this one after them
                    for entered in states:
                        table[(left, f'ret {entered} {first}', symbols)] = \
                            f'ret {entered} {last}'

        # an F3*F2 group: the run climbs the region's top path, taking a
        # letter off after each 1-loop, and collapses on the last of them
        for label, annotation in marks:
            if label not in self.encoding.letters:
                continue
            summary = self.summary_of(annotation)
            symbol, _, level = label.rpartition(':')
            for guess in self.guesses:
                pair = self.guessed(guess)
                if pair is None or summary is None:
                    continue
                first, last = pair
                symbols = (label, PAD, annotation, guess)
                # The 1-loop before each step may leave the stack as it was:
                # a collapse straight off the top is no return -- it drops
                # more than one word -- so nothing else would cover it.
                idling = summary.oneloop | summary.loop
                # ... a letter the group pops on its way up
                if pair in compose(idling,
                                   self.drop_moves(symbol, int(level))):
                    for below in (None, 'gone'):
                        table[(below, None, symbols)] = f'group {first} {last}'
                    for start in states:
                        # the node the chain came from is this one's left
                        # child where the word goes on, and its right where a
                        # separator ends the word -- the path takes either
                        for other in (None, 'gone'):
                            table[(f'group {start} {first}', other,
                                   symbols)] = f'group {start} {last}'
                            table[(other, f'group {start} {first}',
                                   symbols)] = f'group {start} {last}'
                # ... and the letter it collapses on, which ends the group
                if int(level) == 2 and pair in compose(
                        idling, self.summaries.moves(symbol, 'collapse')):
                    for below in (None, 'gone'):
                        table[(below, None, symbols)] = f'sprung {first} {last}'
                        table[(None, below, symbols)] = f'sprung {first} {last}'
                    for start in states:
                        for other in (None, 'gone'):
                            table[(f'group {start} {first}', other,
                                   symbols)] = f'sprung {start} {last}'
                            table[(other, f'group {start} {first}',
                                   symbols)] = f'sprung {start} {last}'

        # below the collapsed letter the region is dropped without any run
        # step of its own, and the group closes at the region's root
        for label, annotation in marks:
            symbols = (label, PAD, annotation, self.NONE)
            for first in states:
                for last in states:
                    sprung = f'sprung {first} {last}'
                    if label in self.encoding.letters:
                        for other in (None, 'gone'):
                            table[(sprung, other, symbols)] = sprung
                            table[(other, sprung, symbols)] = sprung
                    else:                       # the region's root separator
                        for other in (None, 'gone'):
                            table[(sprung, other, symbols)] = \
                                f'ret {first} {last}'
                            table[(other, sprung, symbols)] = \
                                f'ret {first} {last}'
                        for entered in states:
                            table[(sprung, f'ret {entered} {first}',
                                   symbols)] = f'ret {entered} {last}'

        # a node both trees keep, below which words were dropped
        for label, annotation in marks:
            plain = (label, label, annotation, self.NONE)
            for first in states:
                for last in states:
                    done = f'done {first} {last}'
                    table[(f'ret {first} {last}', None, plain)] = done
                    table[('same', f'ret {first} {last}', plain)] = done
                    table[(None, f'ret {first} {last}', plain)] = done
                    table[(done, None, plain)] = done
                    table[('same', done, plain)] = done
                    table[(None, done, plain)] = done

        for source in states:
            for target in states:
                labels = (f'<{source}>', f'<{target}>', Annotation.START,
                          self.NONE)
                table[(f'done {source} {target}', None, labels)] = 'accept'
                if source == target:
                    table[('same', None, labels)] = 'accept'
        return partial_tree_automaton(self.alphabet, 4, table, {'accept'})

    def collapses_on_links(self) -> bool:
        """Whether the system can collapse on a level 2 link at all — the
        case that makes `a` need its F2 and F3 pieces rather than returns
        alone."""
        pushes = {(rule.operation.symbol, rule.operation.level)
                  for rule in self.system.rules
                  if rule.operation.kind == 'push'}
        if not any(level == 2 for _, level in pushes):
            return False
        return any(rule.operation.kind == 'collapse'
                   for rule in self.system.rules)

    def d(self) -> SparseTreeAutomaton:
        """``D``: the stack grows, and the run never dips below where it
        started.

        Kartzow's Cor. 4.10 says such a run visits the milestones of the stack
        it builds in order, consecutive ones joined by a single operation and a
        loop — and the encoding puts those milestones in traversal order, one
        per node. The run therefore only ever moves *forward* through the tree,
        which is what decides where each operation goes: descending to a left
        child is a push, and a new word is reached by cloning at the deepest
        point of the word before it and then popping back up, one letter per
        level, until the two words part company.

        Cloning at the deepest point rather than where the words part is the
        whole of it. The other way round would let a word be shorter than the
        copy it came from for nothing, which is a run no system need have.
        """
        table = {}
        annotations = list(self.annotation.names.values())
        marks = [(label, annotation)
                 for label in self.encoding.nodes for annotation in annotations]
        letters = list(self.encoding.letters)
        states = self.system.states

        def grew(cloned, label, arrive, leave):
            return f'grew|{int(cloned)}|{label}|{arrive}|{leave}'

        for label, annotation in marks:
            for left in (None, 'same'):
                for right in (None, 'same'):
                    table[(left, right, (label, label, annotation, self.NONE))] \
                        = 'same'

        for label, annotation in marks:
            summary = self.summary_of(annotation)
            if summary is None:
                continue
            loop = summary.loop
            clones = self.summaries.moves(summary.symbol or '', 'clone')
            for guess in self.guesses:
                pair = self.guessed(guess)
                if pair is None:
                    continue
                arrive, leave = pair
                symbols = (PAD, label, annotation, guess)

                # the deepest milestone of a word: the run loops there, and
                # clones if another word is still to come
                if (arrive, leave) in loop:
                    table[(None, None, symbols)] = \
                        grew(False, label, arrive, leave)
                if (arrive, leave) in compose(loop, clones):
                    table[(None, None, symbols)] = \
                        grew(True, label, arrive, leave)

                for below in letters:
                    symbol, _, level = below.rpartition(':')
                    down = compose(loop, self.push_moves(annotation, below))
                    up = compose(self.drop_moves(symbol, int(level)), loop)
                    for entered in states:
                        if (arrive, entered) not in down:
                            continue
                        for came_back in states:
                            # deeper and never back: this word is the last
                            child = grew(False, below, entered, came_back)
                            if came_back == leave:
                                table[(child, None, symbols)] = \
                                    grew(False, label, arrive, leave)
                            # deeper, cloned down there, and popping back up
                            cloned = grew(True, below, entered, came_back)
                            if (came_back, leave) in up:
                                table[(cloned, None, symbols)] = \
                                    grew(True, label, arrive, leave)
                            # ... until the words part, where the new one
                            # hangs as this node's right child
                            for last in states:
                                sibling = grew(False, SEP, leave, last)
                                if (came_back, leave) in up:
                                    table[(cloned, sibling, symbols)] = \
                                        grew(False, label, arrive, last)
                                shared = grew(True, SEP, leave, last)
                                if (came_back, leave) in up:
                                    table[(cloned, shared, symbols)] = \
                                        grew(True, label, arrive, last)

                # a word that parts from this one right here: the clone
                # happened below, and nothing was popped since
                for last in states:
                    for kind in (False, True):
                        sibling = grew(kind, SEP, leave, last)
                        if (arrive, leave) in compose(loop, clones):
                            table[(None, sibling, symbols)] = \
                                grew(kind, label, arrive, last)

        # Where the growth hangs, and the run starts. The first new word is
        # a clone of the one this node names, and this is the only place that
        # clone can be checked -- inside the region every node is one the
        # second tree alone has.
        for label, annotation in marks:
            summary = self.summary_of(annotation)
            plain = (label, label, annotation, self.NONE)
            if summary is None:
                continue
            starts = compose(summary.loop,
                             self.summaries.moves(summary.symbol or '', 'clone'))
            for arrive in states:
                for last in states:
                    child = grew(False, SEP, arrive, last)
                    for first in states:
                        if (first, arrive) not in starts:
                            continue
                        done = f'done {first} {last}'
                        for other in (None, 'same'):
                            table[(other, child, plain)] = done
                        table[(child, None, plain)] = done
            for first in states:
                for last in states:
                    done = f'done {first} {last}'
                    table[(done, None, plain)] = done
                    table[('same', done, plain)] = done
                    table[(None, done, plain)] = done

        for source in states:
            for target in states:
                labels = (f'<{source}>', f'<{target}>', Annotation.START,
                          self.NONE)
                table[(f'done {source} {target}', None, labels)] = 'accept'
                if source == target:
                    table[('same', None, labels)] = 'accept'
        return partial_tree_automaton(self.alphabet, 4, table, {'accept'})

    def reach(self) -> SparseTreeAutomaton:
        """Reachability: a run of any length, between any two configurations.

        Kartzow's Remark 4.4 splits every run into four stretches — words come
        off, then letters, then letters go back on, then words — and each is
        reflexive, so no run is excluded by having to pass through all four.
        The relation is therefore the composition, which is a first-order
        formula over the four and needs no automaton of its own::

            Reach(x,y) ≡ ∃u ∃v ∃w. A(x,u) ∧ B(u,v) ∧ C(v,w) ∧ D(w,y)

        The quantifiers range over configurations, so the domain of the scratch
        structure is the encoding trees themselves.
        """
        from autstr.tree_presentations import TreeAutomaticPresentation
        scratch = TreeAutomaticPresentation(
            {'U': self.encoding.universe(),
             'A': self.without_scaffolding(self.a()),
             'B': self.without_scaffolding(self.b()),
             'C': self.without_scaffolding(self.c(), annotated=1),
             'D': self.without_scaffolding(self.d(), annotated=1)},
            padding_symbol=PAD)
        return scratch.evaluate(
            'exists u.(exists v.(exists w.('
            'A(x,u) & B(u,v) & C(v,w) & D(w,y))))')
