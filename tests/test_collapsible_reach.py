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
from collections import deque

from autstr.collapsible import Configuration, Letter, Level2CPS, Stack


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
    droppable = {word[:len(word) - k]
                 for k in range(len(word) + 1)
                 if all(letter.level == 1 for letter in word[len(word) - k:])}

    def allowed(configuration):
        current = configuration.stack
        if current.width > 1:
            return True                       # above the word: unconstrained
        if kind == 'high' and current.words[0] != word:
            return False                      # a high loop never drops
        return current.words[0] in droppable or len(current.words[0]) > len(word)

    pairs = set()
    for state in system.states:
        start = Configuration(state, stack)
        if kind == 'low':
            # a low loop steps down to pop_1(s) first and comes back last
            below = stack.pop1()
            if below is None:
                continue
            starts = [target for _, target in system.step(start)
                      if target.stack == below]
            reached = set()
            for first in starts:
                # the part between the two steps may itself be empty
                for last in search(system, first, allowed,
                                   goal=lambda c: c.stack == below,
                                   empty_run=True):
                    # ... and the final step climbs back onto the word
                    reached |= {target.state for _, target
                                in system.step(Configuration(last, below))
                                if target.stack == stack}
            pairs |= {(state, other) for other in reached}
        else:
            pairs |= {(state, other) for other in search(
                system, start, allowed, goal=lambda c: c.stack == stack,
                empty_run=True)}
    return pairs


def ex_one_loop(system, word):
    """``ExOneLoop(w)``: the pairs (q, q') with a 1-loop from ``(q, w)`` to
    ``(q', t:w)`` for some 2-word t — the stack grows underneath, and the
    topmost word comes back exactly as it was."""
    word = zeroed(word)
    return {(state, reached)
            for state in system.states
            for reached in search(
                system, Configuration(state, Stack((word,))),
                allowed=lambda c: True,
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
