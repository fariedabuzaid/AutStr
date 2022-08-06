from typing import Set

from automata.fa.dfa import DFA
from visual_automata.fa.dfa import VisualDFA


def generate_new_elements(M: Set, n: int):
    """
    Generate a list of n elements which are not contained in x
    :param M: The set
    :param n: The number of new elements to generate
    :return: List of string elements that guaranteed to be not in M
    """
    i = 0
    result = []
    for _ in range(n):
        while True:
            element = str(i)
            if element not in M:
                result.append(element)
                i += 1
                break
            else:
                i += 1

    if n == 1:
        return result[0]
    else:
        return result


def draw_automaton(dfa: DFA):
    """
    Convenience method for better drawings of an automaton.
    :param dfa: the automaton
    :return:
    """
    return VisualDFA(dfa).show_diagram()
