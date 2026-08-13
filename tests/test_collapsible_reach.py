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

from autstr.collapsible import Configuration, Letter, Level2CPS, Stack
from autstr.collapsible_reach import Summaries


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
