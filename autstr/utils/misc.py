from functools import cmp_to_key
from heapq import heapify, heappop, heappush
from typing import FrozenSet, Set, List, Tuple, Union

import numpy as np

def cmp_llex(v: str, w: str) -> int:
    """
    Length-lexicographic ordering of a tuple of strings ignoring the padding symbol.

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
    heaplyfy a list under the llex order.

    :param arr:
    :return:
    """
    s = list(map(cmp_to_key(cmp_llex), arr))
    heapify(s)
    return s


def heappop_llex(heap: List[object]) -> object:
    """
    heapop wrapper for llex order.

    :param heap: The heap list
    :return:
    """
    return heappop(heap).obj


def heappush_llex(heap: List[object], x: object) -> None:
    """
    Heappop wrapper for lllex order.

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


# ====== Symbol Encoding/Decoding ======
def encode_symbol(tuple_symbol: Tuple[int], base_alphabet: FrozenSet[int]) -> int:
    """Encode a symbol tuple into a single integer."""
    if not tuple_symbol:
        return 0
    m = len(base_alphabet)
    alphabet_sorted = sorted(base_alphabet)
    mapping = {sym: idx for idx, sym in enumerate(alphabet_sorted)}
    enc = 0
    for sym in tuple_symbol:
        enc = enc * m + mapping[sym]
    return enc


def decode_symbol(enc: int, arity: int, base_alphabet: FrozenSet[int]) -> Tuple[int]:
    """Decode an integer into a symbol tuple."""
    if arity == 0:
        return ()
    m = len(base_alphabet)
    alphabet_sorted = sorted(base_alphabet)
    symbols = []
    num = enc
    for _ in range(arity):
        num, r = divmod(num, m)
        symbols.append(alphabet_sorted[r])
    return tuple(reversed(symbols))

def complement(values, min_val: int, max_val: int) -> np.ndarray:
    """Find the complement of a set of values within a specified range.

    Args:
        values (np.ndarray): The input array of values.
        min_val (int): The minimum value of the range.
        max_val (int): The maximum value of the range.

    Returns:
        np.ndarray: The sorted values of the range missing from the input.
    """
    values = np.asarray(values)
    if values.size == 0:
        return np.array([], dtype=values.dtype)
    return np.setdiff1d(np.arange(min_val, max_val + 1), values)