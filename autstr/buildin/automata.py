import itertools as it
from typing import Optional, Set
from autstr.sparse_automata import SparseDFA
import numpy as np

from autstr.utils.misc import encode_symbol

def length_automaton(n: int, base_alphabet: Set[int]) -> SparseDFA:
    """
    Creates an automaton that recognizes all words over base_alphabet with length exactly n.

    :param n: The exact word length
    :param base_alphabet: Set of integer symbols
    :return: SparseDFA recognizing words of length n
    """
    # States: 0 (start), 1, 2, ..., n (accepting), n+1 (dead)
    num_states = n + 2
    # Default transitions: move to next state or dead state
    default_states = np.array([i + 1 for i in range(n)] + [n + 1, n + 1], dtype=np.int32)
    # No exceptions needed (same behavior for all symbols)
    exception_symbols = np.full((num_states, 0), -1, dtype=np.int32)
    exception_states = np.full((num_states, 0), -1, dtype=np.int32)
    # Only state n is accepting
    is_accepting = np.array([False] * n + [True, False])
    start_state = 0

    return SparseDFA(
        num_states=num_states,
        default_states=default_states,
        exception_symbols=exception_symbols,
        exception_states=exception_states,
        is_accepting=is_accepting,
        start_state=start_state,
        symbol_arity=1,
        base_alphabet=base_alphabet
    )

def k_longer_automaton(k: int, r: int, base_alphabet: Set[int], padding_symbol: int) -> SparseDFA:
    """
    Creates an automaton recognizing (r+1)-tuples where the last word is at least k letters 
    longer than the other r words.

    :param k: Minimal length difference
    :param r: Number of reference words
    :param base_alphabet: Set of integer symbols
    :param padding_symbol: Padding symbol integer
    :return: SparseDFA for the k-longer condition
    """
    # State mapping: [-1, 0, 1, ..., k] -> [0, 1, 2, ..., k+1]
    state_mapping = {s: i for i, s in enumerate(range(-1, k+1))}
    num_states = len(state_mapping)
    sorted_alphabet = sorted(base_alphabet)
    arity = r + 1
    
    # Precompute all symbol tuples and their encodings
    symbol_tuples = list(it.product(sorted_alphabet, repeat=arity))
    symbol_encodings = [encode_symbol(t, base_alphabet) for t in symbol_tuples]
    
    # Initialize DFA components
    default_states = np.full(num_states, state_mapping[-1], dtype=np.int32)  # Default to dead state
    exception_list = [[] for _ in range(num_states)]
    
    # Build transitions
    for state in range(-1, k+1):
        state_idx = state_mapping[state]
        for t, enc in zip(symbol_tuples, symbol_encodings):
            # Compute next state
            if state == -1:
                next_state = -1  # Stay in dead state
            else:
                if all(x == padding_symbol for x in t[:-1]) and t[-1] != padding_symbol:
                    next_state = min(state + 1, k)  # Count extra length
                elif t[-1] == padding_symbol:
                    next_state = -1  # Reject if padding the last word
                elif state == 0:
                    next_state = 0  # Wait for other words to end
                else:
                    next_state = -1  # Reject otherwise
            
            next_state_idx = state_mapping[next_state]
            if next_state_idx != state_mapping[-1]:
                exception_list[state_idx].append((enc, next_state_idx))
    
    # Find max exceptions needed
    max_exceptions = max(len(ex_list) for ex_list in exception_list) if exception_list else 0
    
    # Build exception arrays
    exception_symbols = np.full((num_states, max_exceptions), -1, dtype=np.int32)
    exception_states = np.full((num_states, max_exceptions), -1, dtype=np.int32)

    for i, ex_list in enumerate(exception_list):
        if ex_list:
            syms, states = zip(*ex_list)
            exception_symbols[i, :len(syms)] = syms
            exception_states[i, :len(states)] = states
    
    # Final states: state k (meaning we've counted k extra symbols)
    is_accepting = np.array([i == state_mapping[k] for i in range(num_states)])
    
    return SparseDFA(
        num_states=num_states,
        default_states=default_states,
        exception_symbols=exception_symbols,
        exception_states=exception_states,
        is_accepting=is_accepting,
        start_state=state_mapping[0],
        symbol_arity=arity,
        base_alphabet=base_alphabet
    )


def zero(symbol_arity: int = 1, base_alphabet: Optional[Set[int]] = None) -> SparseDFA:
    """Automaton that rejects all inputs."""
    base_alphabet = base_alphabet or {0}
    return SparseDFA(
        num_states=1,
        default_states=np.array([0], dtype=np.int32),
        exception_symbols=np.full((1, 0), -1, dtype=np.int32),
        exception_states=np.full((1, 0), -1, dtype=np.int32),
        is_accepting=np.array([False]),
        start_state=0,
        symbol_arity=symbol_arity,
        base_alphabet=base_alphabet
    )

def one(symbol_arity: int = 1, base_alphabet: Optional[Set[int]] = None) -> SparseDFA:
    """Automaton that accepts all inputs."""
    base_alphabet = base_alphabet or {0}
    return SparseDFA(
        num_states=1,
        default_states=np.array([0], dtype=np.int32),
        exception_symbols=np.full((1, 0), -1, dtype=np.int32),
        exception_states=np.full((1, 0), -1, dtype=np.int32),
        is_accepting=np.array([True]),
        start_state=0,
        symbol_arity=symbol_arity,
        base_alphabet=base_alphabet
    )