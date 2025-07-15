import itertools as it
from typing import Optional, Set
from autstr.sparse_automata import SparseDFA
from jax import numpy as jnp

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
    default_states = jnp.array([i + 1 for i in range(n)] + [n + 1, n + 1])
    # No exceptions needed (same behavior for all symbols)
    exception_symbols = jnp.full((num_states, 0), -1)
    exception_states = jnp.full((num_states, 0), -1)
    # Only state n is accepting
    is_accepting = jnp.array([False] * n + [True, False])
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
    default_states = jnp.full(num_states, state_mapping[-1])  # Default to dead state
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
    exception_symbols = jnp.full((num_states, max_exceptions), -1)
    exception_states = jnp.full((num_states, max_exceptions), -1)
    
    for i, ex_list in enumerate(exception_list):
        if ex_list:
            syms, states = zip(*ex_list)
            exception_symbols = exception_symbols.at[i, :len(syms)].set(jnp.array(syms))
            exception_states = exception_states.at[i, :len(states)].set(jnp.array(states))
    
    # Final states: state k (meaning we've counted k extra symbols)
    is_accepting = jnp.array([i == state_mapping[k] for i in range(num_states)])
    
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
        default_states=jnp.array([0]),
        exception_symbols=jnp.full((1, 0), -1),
        exception_states=jnp.full((1, 0), -1),
        is_accepting=jnp.array([False]),
        start_state=0,
        symbol_arity=symbol_arity,
        base_alphabet=base_alphabet
    )

def one(symbol_arity: int = 1, base_alphabet: Optional[Set[int]] = None) -> SparseDFA:
    """Automaton that accepts all inputs."""
    base_alphabet = base_alphabet or {0}
    return SparseDFA(
        num_states=1,
        default_states=jnp.array([0]),
        exception_symbols=jnp.full((1, 0), -1),
        exception_states=jnp.full((1, 0), -1),
        is_accepting=jnp.array([True]),
        start_state=0,
        symbol_arity=symbol_arity,
        base_alphabet=base_alphabet
    )