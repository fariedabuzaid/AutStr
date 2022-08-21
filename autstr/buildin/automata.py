from typing import Set
import itertools as it

from automata.fa.dfa import DFA

def length_automaton(n, sigma) -> DFA:
    """
    Creates an automaton that recognizes all words over sigma with length n

    :param n: The length
    :param sigma: The input symbols
    :return:
    """
    states = {f'{i}' for i in range(n + 1)}
    transitions = {q: {a: str(int(q) + 1) if int(q) < n else q for a in sigma} for q in states}

    dfa = DFA(
        states=states,
        input_symbols=sigma,
        initial_state='0',
        transitions=transitions,
        final_states={f'{n - 1}'}
    )
    return dfa

def k_longer_automaton(k: int, r: int, sigma: Set[str], padding_symbol='*') -> DFA:
    """
    Creates an automaton that recognizes exactly the :math:`(r+1)`-tuple where the :math:`(r+1)`-th word is at least
    :math:`k` letters longer than all other words in the tuple.
    :param k: Minimal length distance to the other words.
    :param r: Number of other words
    :param sigma: The base alphabet
    :return: The k-longer-automaton
    """
    assert padding_symbol in sigma

    states = set(range(k+1))
    states.add(-1)
    input_symbols = set(it.product(sigma, repeat=r + 1))
    initial_state = 0

    def get_nex_state(i, t):
        if i >= 0:
            if all([a == padding_symbol for a in t[:-1]]) and t[-1] != padding_symbol:
                return min([i + 1, k])  # measure distance
            elif t[-1] == padding_symbol:
                return -1  # reject
            elif i == 0:
                return 0  # wait for all other words to end
            else:
                return -1  # reject
        else:
            return -1

    transitions = {
        i: {
            t: get_nex_state(i, t) for t in input_symbols
        } for i in range(-1, k+1)
    }

    final_states = {k}

    dfa = DFA(
        states=states,
        input_symbols=input_symbols,
        initial_state=initial_state,
        transitions=transitions,
        final_states=final_states
    )

    return dfa


def zero() -> DFA:
    """
    Creates an automaton over the empty symbol set that recognizes the empty language

    :return:
    """
    result = DFA(
        states={'0'},
        input_symbols=set(),
        initial_state='0',
        transitions={'0': {}},
        final_states=set()
    )

    return result


def one() -> DFA:
    """
        Creates an automaton over the empty symbol set that recognizes language that contains only the empty word.

        :return:
        """
    result = DFA(
        states={'0'},
        input_symbols=set(),
        initial_state='0',
        transitions={
            '0': {},
        },
        final_states={'0'}
    )

    return result
