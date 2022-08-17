from functools import cmp_to_key
from heapq import heapify, heappop, heappush
from typing import Set, List

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


def cmp_llex(v: str, w: str, padding_symbol: str = '*'):
    """
    Length-lexicographic ordering of a tuple of strings ignoring the padding symbol
    :param v:
    :param w:
    :return:
    """
    v = v[0]
    w = w[0]

    if max([len(vc) for vc in v]) < max([len(wc) for wc in w]):
        return -1
    elif max([len(vc) for vc in v]) > max([len(wc) for wc in w]):
        return 1
    else:
        for vc, wc in zip(v, w):
            if vc < wc:
                return -1
            elif vc > wc:
                return 1

        return 0


def heapify_llex(arr: list):
    """
    heaplyfy a list under the llex order
    :param arr:
    :return:
    """
    s = list(map(cmp_to_key(cmp_llex), arr))
    heapify(s)
    return s


def heappop_llex(heap):
    """
    heapop wrapper for llex order
    :param heap: The heap list
    :return:
    """
    return heappop(heap).obj


def heappush_llex(heap, x):
    """
    Heappop wrapper for lllex order
    :param heap: The heap list
    :param x: The element to push
    :return:
    """
    x = cmp_to_key(cmp_llex)(x)
    heappush(heap, x)


def get_unique_id(current_id: List[str], n: int = 1):
    max_element = max(current_id)
    if n == 1:
        return f'{max_element}0'
    else:
        n_symbols = len(str(n))
        return [f'{max_element}{i:{n_symbols}d}' for i in range(n)]
