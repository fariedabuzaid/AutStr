"""Returns, loops and 1-loops of a level 2 collapsible pushdown system.

Kartzow's reachability construction rests on a decomposition of every run into
*returns* (a run from a stack to the one below it), *loops* (a run from a stack
back to itself) and *1-loops*, and on the fact that which of these exist
depends only on the stack's **topmost word**. This module's oracle is the
definitions themselves, searched by hand: a bounded breadth-first walk over
configurations that prunes whatever the definition forbids.

Being bounded, the oracle is an under-approximation — what it finds really is
there, and what it misses needs a longer run than the bound allows. That is the
right direction for checking a construction that claims to find *everything*:
every pair the oracle exhibits must appear in the computed set.
"""
import random
from collections import deque

import pytest

from autstr.collapsible import (
    Configuration, Letter, Level2CPS, Operation, PAD, Stack, _Encoding,
    encode_configuration, initial_stack,
)
from autstr.collapsible_reach import Annotation, Relations, Summaries
from autstr.sparse_tree_automata import Tree, convolve_trees


def zeroed(word):
    """The word with every level 2 link set to 0 — Kartzow's ``w↓₀``.

    Returns and loops are asked of a word rather than of a stack, and a link
    into the stack below has no meaning once the stack is forgotten. Zeroing
    them also makes those links unusable, which is exactly what the definition
    of a return demands: it may not use the links stored in the topmost word.
    """
    return tuple(letter if letter.level == 1 else Letter(letter.symbol, 2, 0)
                 for letter in word)


def search(system, start, allowed, goal, bound=2000, size=6,
           empty_run=False):
    """The states in which `goal` is reached from `start`.

    A breadth-first walk over configurations. `allowed` prunes the ones the
    definition forbids as *intermediate* stacks — the goal is still recognized
    when it is reached — and `size` caps the stacks so the walk terminates.

    :param empty_run: whether the run of length zero counts, i.e. whether
        `start` itself is reported when it already satisfies `goal`.
    """
    seen, frontier, reached = {start}, deque([start]), set()
    if empty_run and goal(start):
        reached.add(start.state)
    steps = 0
    while frontier and steps < bound:
        configuration = frontier.popleft()
        steps += 1
        for _, successor in system.step(configuration):
            if goal(successor):
                reached.add(successor.state)
            if successor in seen or not allowed(successor):
                continue
            stack = successor.stack
            if stack.width > size or max(map(len, stack.words)) > size:
                continue
            seen.add(successor)
            frontier.append(successor)
    return reached


def ex_ret(system, word):
    """``ExRet(w)``: the pairs (q, q') with a return from ``(q, w:w)`` to
    ``(q', w)``.

    A return may not visit a substack of its target before the last step, and
    every substack of the one-word stack ``w`` has width 1 — so the run must
    stay at width 2 or more until it arrives.
    """
    word = zeroed(word)
    target = Stack((word,))
    return {(state, reached)
            for state in system.states
            for reached in search(
                system, Configuration(state, Stack((word, word))),
                allowed=lambda c: c.stack.width >= 2,
                goal=lambda c: c.stack == target)}


def ex_loop(system, word, kind='any'):
    """``ExLoop(w)``, ``ExHLoop(w)``, ``ExLLoop(w)``: the pairs (q, q') with a
    loop from ``(q, w)`` back to ``(q', w)``.

    A loop may drop below its own topmost word only through letters carrying
    level 1 links: reaching ``pop_1^k`` of the stack is allowed only when the k
    letters it removed all had level 1 links. A *high* loop never drops at all;
    a *low* loop drops on its first step and returns on its last.

    The run of length zero is a loop, so ``ExLoop`` and ``ExHLoop`` contain
    every ``(q, q)``; a low loop has to take its two steps, so ``ExLLoop`` need
    not. That convention is what makes "one operation followed by some loop"
    cover the bare operation as well.
    """
    word = zeroed(word)
    stack = Stack((word,))

    def staying(base, high):
        """A loop of `base` may push and pop above it freely; what it may not
        do is drop below, except through letters carrying level 1 links — and
        a high loop may not drop at all."""
        def allowed(configuration):
            current = configuration.stack
            if current.width > 1:
                return True                   # above the word: unconstrained
            here = current.words[0]
            if len(here) >= len(base):
                return True                   # the word itself, or longer
            return not high and all(letter.level == 1
                                    for letter in base[len(here):])
        return allowed

    if kind == 'low':
        # a low loop steps down to pop_1(s), loops there, and writes the
        # letter back — so the letter has to be one that may be dropped
        below = stack.pop1()
        if below is None or word[-1].level != 1:
            return set()
        pairs = set()
        for state in system.states:
            for _, first in system.step(Configuration(state, stack)):
                if first.stack != below:
                    continue
                # the part between the two steps is a loop of the word below,
                # and may itself be empty
                for last in search(system, first,
                                   staying(below.words[0], False),
                                   goal=lambda c: c.stack == below,
                                   empty_run=True):
                    pairs |= {(state, target.state) for _, target
                              in system.step(Configuration(last, below))
                              if target.stack == stack}
        return pairs

    allowed = staying(word, kind == 'high')
    return {(state, other)
            for state in system.states
            for other in search(system, Configuration(state, stack), allowed,
                                goal=lambda c: c.stack == stack,
                                empty_run=True)}


def ex_one_loop(system, word):
    """``ExOneLoop(w)``: the pairs (q, q') with a 1-loop from ``(q, w)`` to
    ``(q', t:w)`` for some *stack* t — the stack grows underneath, and the
    topmost word comes back exactly as it was.

    The definition also asks that wherever the run shortens the topmost word
    it later returns; rather than track that, this forbids shortening it at
    all, which is a run of the required shape either way. So this direction
    under-approximates more than the search bound alone does.
    """
    word = zeroed(word)
    return {(state, reached)
            for state in system.states
            for reached in search(
                system, Configuration(state, Stack((word,))),
                allowed=lambda c: len(c.stack.words[-1]) >= len(word),
                goal=lambda c: c.stack.width > 1 and c.stack.words[-1] == word)}


# ----------------------------------------------------------------------
# the oracle against systems whose runs can be counted by hand
# ----------------------------------------------------------------------

BOTTOM = (Letter('⊥'),)


class TestReturns:
    def test_one_rule_that_pops(self):
        """The only return is the single pop_2 step."""
        system = Level2CPS([('p', None, 'o', 'q', 'pop 2')])
        assert ex_ret(system, BOTTOM) == {('p', 'q')}

    def test_cloning_alone_never_returns(self):
        system = Level2CPS([('p', None, 'c', 'p', 'clone')])
        assert ex_ret(system, BOTTOM) == set()

    def test_a_collapse_of_a_freshly_pushed_letter_returns(self):
        """Push a letter linked to the stack below, then collapse it: the
        stack drops back by one word, which is a return."""
        system = Level2CPS([('p', None, 'a', 'r', 'push a 2'),
                            ('r', 'a', 'c', 'q', 'collapse')])
        assert ex_ret(system, BOTTOM) == {('p', 'q')}

    def test_a_return_may_take_a_detour_upwards(self):
        """Clone first, pop twice: still a return, since the run never dips
        below the target before the end. The two single pops are returns of
        their own, from the states they start in."""
        system = Level2CPS([('p', None, 'c', 'r', 'clone'),
                            ('r', None, 'o', 's', 'pop 2'),
                            ('s', None, 'o', 'q', 'pop 2')])
        assert ex_ret(system, BOTTOM) == {('p', 'q'), ('r', 's'), ('s', 'q')}

    def test_a_run_that_dips_below_the_target_is_no_return(self):
        """Popping to width 1 and cloning back is not a return: it visits the
        target's own substack on the way."""
        system = Level2CPS([('p', None, 'o', 'r', 'pop 2'),
                            ('r', None, 'c', 'q', 'clone')])
        assert ('p', 'q') not in ex_ret(system, BOTTOM)
        assert ex_ret(system, BOTTOM) == {('p', 'r')}


class TestLoops:
    def test_a_loop_that_never_leaves_the_word(self):
        """Clone and pop back: a high loop, since it never drops a letter."""
        system = Level2CPS([('p', None, 'c', 'r', 'clone'),
                            ('r', None, 'o', 'q', 'pop 2')])
        assert ('p', 'q') in ex_loop(system, BOTTOM)
        assert ('p', 'q') in ex_loop(system, BOTTOM, kind='high')

    def test_a_loop_that_drops_a_level_1_letter_and_writes_it_back(self):
        word = (Letter('⊥'), Letter('a'))
        system = Level2CPS([('p', 'a', 'o', 'r', 'pop 1'),
                            ('r', None, 'w', 'q', 'push a 1')])
        assert ('p', 'q') in ex_loop(system, word)
        assert ('p', 'q') in ex_loop(system, word, kind='low')
        assert ('p', 'q') not in ex_loop(system, word, kind='high')

    def test_a_level_2_letter_may_not_be_dropped(self):
        """The same run, but the letter carries a level 2 link — then passing
        below it is not a loop at all."""
        word = (Letter('⊥'), Letter('a', 2, 0))
        system = Level2CPS([('p', 'a', 'o', 'r', 'pop 1'),
                            ('r', None, 'w', 'q', 'push a 2')])
        assert ('p', 'q') not in ex_loop(system, word)
        assert ex_loop(system, word) == {(q, q) for q in system.states}


class TestOneLoops:
    def test_the_stack_grows_underneath(self):
        """Clone: the topmost word is back where it started, with a copy of
        itself underneath."""
        system = Level2CPS([('p', None, 'c', 'q', 'clone')])
        assert ('p', 'q') in ex_one_loop(system, BOTTOM)

    def test_popping_is_no_one_loop(self):
        system = Level2CPS([('p', None, 'o', 'q', 'pop 2')])
        assert ex_one_loop(system, BOTTOM) == set()


# ----------------------------------------------------------------------
# the computed summaries, against that oracle
# ----------------------------------------------------------------------

SYMBOLS = ('a', 'b')
OPERATIONS = ['clone', 'pop 1', 'pop 2', 'collapse',
              'push a 1', 'push a 2', 'push b 1', 'push b 2']

#: words to compare on, from the bare bottom letter to one of every kind
SAMPLE_WORDS = [
    (Letter('⊥'),),
    (Letter('⊥'), Letter('a')),
    (Letter('⊥'), Letter('a', 2, 0)),
    (Letter('⊥'), Letter('b'), Letter('a', 2, 0)),
]


def as_path(word):
    return [(letter.symbol, letter.level) for letter in word]


def oracle(system, word):
    """Every set the oracle can find, as the computed summary spells them."""
    return {'ret': ex_ret(system, word),
            'hloop': ex_loop(system, word, kind='high'),
            'lloop': ex_loop(system, word, kind='low'),
            'loop': ex_loop(system, word),
            'oneloop': ex_one_loop(system, word)}


def computed(summaries, word):
    summary = summaries.of_word(as_path(word))
    return {'ret': summary.ret, 'hloop': summary.hloop,
            'lloop': summary.lloop, 'loop': summary.loop,
            'oneloop': summary.oneloop}


def random_system(rng):
    states = [str(index) for index in range(rng.randint(1, 3))]
    return Level2CPS([(rng.choice(states),
                       rng.choice([None] + list(SYMBOLS) + ['⊥']),
                       f'g{index}', rng.choice(states),
                       rng.choice(OPERATIONS))
                      for index in range(rng.randint(1, 6))],
                     symbols=SYMBOLS)


class TestSummaries:
    """The computed summaries against the searched ones.

    The oracle under-approximates — its search is bounded, and it will not
    chase the side condition on 1-loops — so what it finds must always appear
    in the computed summary. On the small systems here the two agree exactly.
    """

    CASES = {
        'a single pop': [('p', None, 'o', 'q', 'pop 2')],
        'cloning only': [('p', None, 'c', 'p', 'clone')],
        'push then collapse': [('p', None, 'a', 'r', 'push a 2'),
                               ('r', 'a', 'c', 'q', 'collapse')],
        'clone then two pops': [('p', None, 'c', 'r', 'clone'),
                                ('r', None, 'o', 's', 'pop 2'),
                                ('s', None, 'o', 'q', 'pop 2')],
        'drop and rewrite': [('p', 'a', 'o', 'r', 'pop 1'),
                             ('r', None, 'w', 'q', 'push a 1')],
        'clone and pop back': [('p', None, 'c', 'r', 'clone'),
                               ('r', None, 'o', 'q', 'pop 2')],
    }

    @pytest.mark.parametrize("case", sorted(CASES))
    def test_against_the_oracle(self, case):
        system = Level2CPS(self.CASES[case], symbols=SYMBOLS)
        summaries = Summaries(system, depth=3, limit=4)
        for word in SAMPLE_WORDS:
            assert computed(summaries, word) == oracle(system, word), word

    def test_against_the_oracle_on_random_systems(self):
        rng = random.Random(2026)
        for _ in range(25):
            system = random_system(rng)
            summaries = Summaries(system, depth=3, limit=4)
            for word in SAMPLE_WORDS:
                found, said = oracle(system, word), computed(summaries, word)
                for name, pairs in found.items():
                    assert pairs <= said[name], (name, word, system.rules)

    def test_raising_the_bound_only_finds_more(self):
        """The fixpoint is taken over words up to a length, so a longer bound
        can only turn up runs the shorter one had no room to write down."""
        rng = random.Random(7)
        for _ in range(10):
            system = random_system(rng)
            shallow = Summaries(system, depth=2, limit=2)
            deeper = Summaries(system, depth=4, limit=4)
            for word in SAMPLE_WORDS[:3]:
                for name, pairs in computed(shallow, word).items():
                    assert pairs <= computed(deeper, word)[name], name

    def test_a_word_beyond_the_bound_is_refused(self):
        system = Level2CPS(self.CASES['cloning only'], symbols=SYMBOLS)
        summaries = Summaries(system, depth=2, limit=2)
        with pytest.raises(ValueError, match="exceeds"):
            summaries.of_word(as_path(
                (Letter('⊥'), Letter('a'), Letter('a'), Letter('a'))))

    def test_the_summary_of_the_paper_s_example(self):
        """Hague et al.'s system: from state 1 a cloned word can be popped
        back, and the pushed letter's link is what lets state 2 collapse all
        the way out."""
        system = Level2CPS([('0', None, 'Cl', '1', 'clone'),
                            ('1', None, 'A', '0', 'push a 2'),
                            ('1', None, 'B', '2', 'push a 2'),
                            ('2', 'a', 'P', '2', 'pop 1'),
                            ('2', 'a', 'Co', '0', 'collapse')])
        summaries = Summaries(system, depth=3, limit=4)
        for word in SAMPLE_WORDS[:3]:
            assert computed(summaries, word) == oracle(system, word), word

    def test_summaries_have_a_readable_repr(self):
        system = Level2CPS(self.CASES['a single pop'], symbols=SYMBOLS)
        assert 'Summaries' in repr(Summaries(system, depth=2, limit=2))


class TestAnnotation:
    """The summary annotation tape: a node carries the summary of the word
    read from the root down to it, and the automaton checks that locally."""

    @pytest.fixture(scope="class")
    def setup(self):
        system = Level2CPS([('0', None, 'Cl', '1', 'clone'),
                            ('1', None, 'A', '0', 'push a 2'),
                            ('1', None, 'B', '2', 'push a 1'),
                            ('2', 'a', 'P', '2', 'pop 1'),
                            ('2', 'a', 'Co', '0', 'collapse')])
        encoding = _Encoding(system.states, system.symbols, system.bottom)
        annotation = Annotation(encoding, Summaries(system, depth=4, limit=6))
        return system, annotation, annotation.automaton()

    def convolution(self, annotation, tree, marks):
        return convolve_trees([tree, marks], frozenset(annotation.alphabet),
                              PAD)

    def test_the_right_annotation_is_accepted(self, setup):
        system, annotation, automaton = setup
        for configuration in system.reachable(bound=6):
            tree = encode_configuration(configuration)
            marks = annotation.of_tree(tree)
            assert automaton.accepts(
                self.convolution(annotation, tree, marks)), configuration

    def test_the_annotation_follows_the_word_down_the_tree(self, setup):
        _, annotation, _ = setup
        # the root reads no letter; below it the annotations follow the
        # summaries of ⊥, ⊥a, ... and a separator repeats its parent's
        tree = encode_configuration(Configuration(
            '0', Stack(((Letter('⊥'), Letter('a', 2, 0)),
                        (Letter('⊥'),)))))
        marks = annotation.of_tree(tree)
        assert marks.label == Annotation.START
        assert marks.left.label == annotation.first
        assert marks.left.right.label == marks.left.label   # a separator

    def test_a_wrong_annotation_is_rejected(self, setup):
        system, annotation, automaton = setup
        others = [letter for letter in annotation.letters
                  if letter != Annotation.START]
        for configuration in system.reachable(bound=5):
            tree = encode_configuration(configuration)
            marks = annotation.of_tree(tree)
            if marks.left is None:
                continue
            for wrong in others:
                if wrong == marks.left.label:
                    continue
                spoiled = Tree(marks.label,
                               Tree(wrong, marks.left.left, marks.left.right),
                               marks.right)
                assert not automaton.accepts(
                    self.convolution(annotation, tree, spoiled))

    def test_an_annotation_of_the_wrong_shape_is_rejected(self, setup):
        system, annotation, automaton = setup
        configuration = system.reachable(bound=4)[-1]
        tree = encode_configuration(configuration)
        marks = annotation.of_tree(tree)
        # one node short, and one node too many
        assert not automaton.accepts(
            self.convolution(annotation, tree, Tree(marks.label)))
        assert not automaton.accepts(self.convolution(
            annotation, tree,
            Tree(marks.label, marks.left, Tree(annotation.first))))

    def test_summaries_merge_into_an_automaton(self):
        """Words that behave alike are one state — and while the fixpoint is
        still moving, the merge is refused rather than guessed at."""
        system = Level2CPS([('0', None, 'c', '0', 'clone'),
                            ('0', None, 'p', '0', 'push a 2')])
        first, table = Summaries(system, depth=4, limit=6).transitions()
        assert first.symbol == '⊥'
        assert all(isinstance(key[1], tuple) for key in table)


# ----------------------------------------------------------------------
# the relations reachability decomposes into
# ----------------------------------------------------------------------

def substacks(stack):
    """Every substack: pop the words, then the letters."""
    out, current = [], stack
    while current is not None:
        inner = current
        while inner is not None:
            out.append(inner)
            inner = inner.pop1()
        current = current.pop2()
    return out


def drops_to(source, target):
    """Is the target the source with letters taken off its topmost word?"""
    current = source
    while current is not None:
        if current == target:
            return True
        current = current.pop1()
    return False


def b_oracle(system, source, target, bound=3000):
    """``B``: a run to a shorter topmost word that never dips below it.

    As with ``A``, dropping no letter at all leaves only the empty run.
    """
    if not drops_to(source.stack, target.stack):
        return False
    if source.stack == target.stack:
        return source == target
    forbidden = set(substacks(target.stack))
    seen, frontier, steps = {source}, deque([source]), 0
    while frontier and steps < bound:
        configuration = frontier.popleft()
        steps += 1
        for _, successor in system.step(configuration):
            if successor == target:
                return True
            if successor in seen or successor.stack in forbidden:
                continue
            if successor.stack.width > 4 or \
                    max(map(len, successor.stack.words)) > 6:
                continue
            seen.add(successor)
            frontier.append(successor)
    return False


def walk_stacks(seed, count, width=3, height=4):
    """Stacks from a random walk, kept small enough to search exhaustively."""
    rng, stack, out = random.Random(seed), initial_stack(), []
    operations = ([Operation('clone')]
                  + [Operation('push', symbol, level)
                     for symbol in SYMBOLS for level in (1, 2)]
                  + [Operation('pop', level=1), Operation('pop', level=2)])
    while len(out) < count:
        stack = stack.apply(rng.choice(operations)) or initial_stack()
        if stack.width <= width and max(map(len, stack.words)) <= height:
            out.append(stack)
    return out


class TestRelationB:
    """``B`` — the topmost word loses letters, and the run never dips below
    what is left. The two trees then differ along a tail of the first one's
    last path, and the second may gain the one separator the first loses."""

    CASES = {
        'pop and push back': [('0', None, 'p', '1', 'pop 1'),
                              ('1', None, 'u', '0', 'push a 1')],
        'a loop before each pop': [('0', None, 'c', '1', 'clone'),
                                   ('1', None, 'o', '0', 'pop 2'),
                                   ('0', 'a', 'p', '2', 'pop 1'),
                                   ('2', None, 'u', '0', 'push b 1')],
    }

    def relation(self, rules):
        system = Level2CPS(rules, symbols=SYMBOLS)
        relations = Relations(system)
        return system, relations, relations.without_scaffolding(relations.b())

    def holds(self, relations, relation, source, target):
        return relation.accepts(convolve_trees(
            [encode_configuration(source), encode_configuration(target)],
            frozenset(relations.encoding.alphabet), PAD))

    @pytest.mark.parametrize("case", sorted(CASES))
    def test_against_the_search(self, case):
        system, relations, relation = self.relation(self.CASES[case])
        configurations = [Configuration(state, stack)
                          for stack in walk_stacks(7, 8)
                          for state in system.states]
        for source in configurations:
            for target in configurations:
                assert self.holds(relations, relation, source, target) == \
                    b_oracle(system, source, target), (source, target)

    def test_it_is_reflexive(self):
        """Every configuration reaches itself by the run of no steps, so the
        two trees are simply equal."""
        system, relations, relation = self.relation(
            self.CASES['pop and push back'])
        for stack in walk_stacks(3, 6):
            for state in system.states:
                configuration = Configuration(state, stack)
                assert self.holds(relations, relation, configuration,
                                  configuration)

    def test_dropping_a_word_is_not_dropping_letters(self):
        """A pair whose widths differ is refused whatever the runs do: B moves
        within the topmost word."""
        system, relations, relation = self.relation(
            self.CASES['pop and push back'])
        tall = Stack(((Letter('⊥'), Letter('a')), (Letter('⊥'), Letter('a'))))
        assert not self.holds(relations, relation,
                              Configuration('0', tall),
                              Configuration('0', Stack((tall.words[0],))))


def c_oracle(system, source, target, bound=3000):
    """``C``: a run to a longer topmost word that never dips below where it
    started."""
    if not drops_to(target.stack, source.stack):
        return False
    if source == target:
        return True
    forbidden = {stack for stack in substacks(source.stack)
                 if stack != source.stack}
    seen, frontier, steps = {source}, deque([source]), 0
    while frontier and steps < bound:
        configuration = frontier.popleft()
        steps += 1
        for _, successor in system.step(configuration):
            if successor.stack in forbidden:
                continue                        # never below where it started
            if successor == target:
                return True
            if successor in seen:
                continue
            if successor.stack.width > 4 or \
                    max(map(len, successor.stack.words)) > 6:
                continue
            seen.add(successor)
            frontier.append(successor)
    return False


class TestRelationC:
    """``C`` — the topmost word gains letters, and the run never dips below
    where it started. B read backwards, with one asymmetry that matters: a pop
    is guarded by the letter it removes, but a push by the letter already on
    top, which is the one *above* the letter written."""

    CASES = {
        'push and pop back': [('0', None, 'u', '1', 'push a 1'),
                              ('1', None, 'p', '0', 'pop 1')],
        'a loop before each push': [('0', None, 'c', '1', 'clone'),
                                    ('1', None, 'o', '0', 'pop 2'),
                                    ('0', None, 'u', '2', 'push a 1'),
                                    ('2', 'a', 'v', '0', 'push b 1')],
    }

    def relation(self, rules):
        system = Level2CPS(rules, symbols=SYMBOLS)
        relations = Relations(system)
        return system, relations, relations.without_scaffolding(
            relations.c(), annotated=1)

    def holds(self, relations, relation, source, target):
        return relation.accepts(convolve_trees(
            [encode_configuration(source), encode_configuration(target)],
            frozenset(relations.encoding.alphabet), PAD))

    @pytest.mark.parametrize("case", sorted(CASES))
    def test_against_the_search(self, case):
        system, relations, relation = self.relation(self.CASES[case])
        configurations = [Configuration(state, stack)
                          for stack in walk_stacks(7, 8)
                          for state in system.states]
        for source in configurations:
            for target in configurations:
                assert self.holds(relations, relation, source, target) == \
                    c_oracle(system, source, target), (source, target)

    def test_it_is_reflexive(self):
        system, relations, relation = self.relation(
            self.CASES['push and pop back'])
        for stack in walk_stacks(3, 6):
            for state in system.states:
                configuration = Configuration(state, stack)
                assert self.holds(relations, relation, configuration,
                                  configuration)

    def test_a_letter_pushed_below_a_separator(self):
        """The letter goes onto the last word, which the tree hangs under a
        separator — so the symbol guarding the push is not the separator's but
        the nearest letter above it, which the annotation knows."""
        system, relations, relation = self.relation(
            self.CASES['push and pop back'])
        word = (Letter('⊥'), Letter('a'))
        twice = Stack((word, word))
        longer = Stack((word, word + (Letter('a'),)))
        assert self.holds(relations, relation,
                          Configuration('0', twice),
                          Configuration('1', longer))


def pops_words_to(source, target):
    """Is the target the source with whole words taken off?"""
    current = source
    while current is not None:
        if current == target:
            return True
        current = current.pop2()
    return False


def a_oracle(system, source, target, bound=3000):
    """``A``: a run that drops whole words and never dips below the target.

    With no word dropped at all the condition already fails at the run's first
    step — the stack it starts on *is* a substack of the target — so nothing
    but the empty run qualifies. A loop that stays where it is belongs to C,
    whose condition excludes only the *proper* substacks.
    """
    if not pops_words_to(source.stack, target.stack):
        return False
    if source.stack == target.stack:
        return source == target
    forbidden = set(substacks(target.stack))
    seen, frontier, steps = {source}, deque([source]), 0
    while frontier and steps < bound:
        configuration = frontier.popleft()
        steps += 1
        for _, successor in system.step(configuration):
            if successor == target:
                return True
            if successor in seen or successor.stack in forbidden:
                continue
            if successor.stack.width > 4 or \
                    max(map(len, successor.stack.words)) > 5:
                continue
            seen.add(successor)
            frontier.append(successor)
    return False


class TestRelationA:
    """``A`` — the stack loses whole words.

    Kartzow's decomposition allows three kinds of piece: a return, a 1-loop
    then a level 2 collapse, and a 1-loop then a pop that a later collapse
    closes off. A collapse spanning several words looks like a link reaching
    across the tree, but a level 2 link records the number of separators up to
    its letter, and the stack it points at is the tree's prefix at that
    separator — exactly where the deleted region begins. So the group is a
    chain climbing that region's own top path.
    """

    EXACT = {
        'clone and pop': [('0', None, 'c', '1', 'clone'),
                          ('1', None, 'o', '0', 'pop 2')],
        'a loop before each pop': [('0', None, 'u', '1', 'push a 1'),
                                   ('1', 'a', 'p', '0', 'pop 1'),
                                   ('0', None, 'o', '0', 'pop 2')],
        'a collapse over one word': [('0', None, 'u', '1', 'push a 2'),
                                     ('1', 'a', 'x', '2', 'collapse')],
    }
    #: a collapse that drops every word the clones added
    SPANNING = [('0', None, 'c', '0', 'clone'),
                ('0', None, 'u', '1', 'push a 2'),
                ('1', None, 'k', '1', 'clone'),
                ('1', 'a', 'x', '2', 'collapse')]

    def relation(self, rules):
        system = Level2CPS(rules, symbols=SYMBOLS)
        relations = Relations(system)
        return system, relations, relations.without_scaffolding(
            relations.a())

    def holds(self, relations, relation, source, target):
        return relation.accepts(convolve_trees(
            [encode_configuration(source), encode_configuration(target)],
            frozenset(relations.encoding.alphabet), PAD))

    @pytest.mark.parametrize("case", sorted(EXACT))
    def test_against_the_search(self, case):
        system, relations, relation = self.relation(self.EXACT[case])
        configurations = [Configuration(state, stack)
                          for stack in walk_stacks(5, 8, width=3, height=3)
                          for state in system.states]
        for source in configurations:
            for target in configurations:
                assert self.holds(relations, relation, source, target) == \
                    a_oracle(system, source, target), (source, target)

    def test_a_collapse_that_spans_several_words(self):
        system, relations, relation = self.relation(self.SPANNING)
        assert relations.collapses_on_links()
        configurations = [Configuration(state, stack)
                          for stack in walk_stacks(5, 8, width=3, height=3)
                          for state in system.states]
        for source in configurations:
            for target in configurations:
                assert self.holds(relations, relation, source, target) == \
                    a_oracle(system, source, target), (source, target)

    def test_a_collapse_no_sequence_of_returns_could_make(self):
        """The only rule is a collapse, so the system has no return at all —
        and the stack still drops two words in one step, which nothing but the
        1-loop-and-collapse piece can account for."""
        system = Level2CPS([('p', 'a', 'x', 'q', 'collapse')], symbols=SYMBOLS)
        relations = Relations(system)
        relation = relations.without_scaffolding(relations.a())
        assert all(not summary.ret
                   for summary in relations.summaries.values.values())

        word = (Letter('⊥'), Letter('a', 2, 1))
        tall = Stack(((Letter('⊥'),), word, word))
        short = Stack(((Letter('⊥'),),))
        assert tall.collapse() == short          # two words in one step
        assert self.holds(relations, relation, Configuration('p', tall),
                          Configuration('q', short))
        # and only from the state the collapse fires in
        assert not self.holds(relations, relation, Configuration('q', tall),
                              Configuration('q', short))

    def test_it_is_reflexive(self):
        system, relations, relation = self.relation(self.EXACT['clone and pop'])
        for stack in walk_stacks(3, 6):
            for state in system.states:
                configuration = Configuration(state, stack)
                assert self.holds(relations, relation, configuration,
                                  configuration)


def d_oracle(system, source, target, bound=3000):
    """``D``: the stack grows, and after the first step nothing is a substack
    of where it started."""
    current = target.stack
    grows = False
    while current is not None:
        if current == source.stack:
            grows = True
            break
        current = current.pop2()
    if not grows:
        return False
    if source.stack == target.stack:
        return source == target
    forbidden = set(substacks(source.stack))
    seen, frontier, steps = {source}, deque([source]), 0
    while frontier and steps < bound:
        configuration = frontier.popleft()
        steps += 1
        for _, successor in system.step(configuration):
            if successor.stack in forbidden:
                continue
            if successor == target:
                return True
            if successor in seen:
                continue
            if successor.stack.width > 4 or \
                    max(map(len, successor.stack.words)) > 5:
                continue
            seen.add(successor)
            frontier.append(successor)
    return False


class TestRelationD:
    """``D`` — the stack grows. **Not correct yet.**

    Kartzow's Cor. 4.10 walks the milestones of the stack being built, which
    the encoding puts in traversal order. What this does not yet handle are
    the *generalised* milestones: between one word and the next the run passes
    through the previous word cloned whole, and only then pops it down to
    where the two words part. Getting that wrong shows up in both directions,
    and the two cases below are kept as a record of it.
    """

    CLONE_ONLY = [('0', None, 'c', '1', 'clone')]
    NO_POPS = [('0', None, 'c', '1', 'clone'),
               ('1', None, 'o', '0', 'pop 2')]

    def relation(self, rules):
        system = Level2CPS(rules, symbols=SYMBOLS)
        relations = Relations(system)
        return system, relations, relations.without_scaffolding(
            relations.d(), annotated=1)

    def holds(self, relations, relation, source, target):
        return relation.accepts(convolve_trees(
            [encode_configuration(source), encode_configuration(target)],
            frozenset(relations.encoding.alphabet), PAD))

    def test_one_clone(self):
        system, relations, relation = self.relation(self.CLONE_ONLY)
        for word in ((Letter('⊥'),), (Letter('⊥'), Letter('a'))):
            assert self.holds(relations, relation,
                              Configuration('0', Stack((word,))),
                              Configuration('1', Stack((word, word))))

    def test_it_is_reflexive(self):
        system, relations, relation = self.relation(self.CLONE_ONLY)
        for stack in walk_stacks(3, 6):
            for state in system.states:
                configuration = Configuration(state, stack)
                assert self.holds(relations, relation, configuration,
                                  configuration)

    @pytest.mark.xfail(reason="the generalised milestones are not handled: "
                              "the run clones the whole word before popping "
                              "it down to where the two words part")
    def test_a_word_shorter_than_its_clone_needs_pops(self):
        """The system can only clone and pop words, never letters — so the
        second word cannot be shorter than the first."""
        system, relations, relation = self.relation(self.NO_POPS)
        long_word = (Letter('⊥'), Letter('a'), Letter('a'))
        short_word = (Letter('⊥'), Letter('a'))
        assert not self.holds(
            relations, relation,
            Configuration('0', Stack((long_word,))),
            Configuration('1', Stack((long_word, short_word))))
