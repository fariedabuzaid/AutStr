from functools import cmp_to_key
from heapq import heapify, heappop, heappush
from typing import Set, List, Union

def cmp_llex(v: str, w: str) -> int:
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


def heapify_llex(arr: List[object]) -> List[object]:
    """
    heaplyfy a list under the llex order
    :param arr:
    :return:
    """
    s = list(map(cmp_to_key(cmp_llex), arr))
    heapify(s)
    return s


def heappop_llex(heap: List[object]) -> object:
    """
    heapop wrapper for llex order
    :param heap: The heap list
    :return:
    """
    return heappop(heap).obj


def heappush_llex(heap: List[object], x: object) -> None:
    """
    Heappop wrapper for lllex order
    :param heap: The heap list
    :param x: The element to push
    :return:
    """
    x = cmp_to_key(cmp_llex)(x)
    heappush(heap, x)


def get_unique_id(current_id: List[str], n: int = 1) -> Union[str, List[str]]:
    max_element = max(current_id)
    if n == 1:
        return f'{max_element}0'
    else:
        n_symbols = len(str(n))
        return [f'{max_element}{i:{n_symbols}d}' for i in range(n)]
