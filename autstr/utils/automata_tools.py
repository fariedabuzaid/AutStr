import heapq
import jax
import jax.numpy as jnp
import numpy as np
from collections import deque, defaultdict
from functools import partial
from typing import Generator, Optional, Callable, Dict, List, Set, Union
import itertools as it

from autstr.utils.logic import get_free_elementary_vars
from autstr.sparse_automata import SparseDFA, SparseNFA
from autstr.buildin.automata import one
from autstr.utils.misc import decode_symbol, encode_symbol, complement




# ====== Helper Functions ======
def pad(dfa: SparseDFA, padding_symbol: int = -1) -> SparseDFA:
    """Pad the automaton to accept trailing padding symbols by:
    1. Creating a new sub-automaton for (pad_tuple)*
    2. Connecting original accepting states to the new sub-automaton
    3. Determinizing and minimizing the result
    """
    arity = dfa.symbol_arity
    base_alphabet = dfa.base_alphabet
    if padding_symbol == -1:
        padding_symbol = sorted(base_alphabet)[0]  # Default to first symbol
    pad_tuple = (padding_symbol,) * arity
    pad_enc = encode_symbol(pad_tuple, base_alphabet)
    
    # If automaton is empty, return it immediately
    if dfa.is_empty():
        return dfa

    # Convert JAX arrays to native Python types
    default_states_np = np.array(dfa.default_states)
    exception_symbols_np = np.array(dfa.exception_symbols)
    exception_states_np = np.array(dfa.exception_states)
    is_accepting_np = np.array(dfa.is_accepting)
    
    # Step 1: Build NFA components
    n_orig = dfa.num_states
    n_pad = n_orig  # State for padding loop
    n_dead = n_orig + 1  # Dead state
    num_states = n_orig + 2

    # Base state array (using native Python types)
    base_state_arr = default_states_np.tolist() + [n_dead, n_dead]

    # Acceptance array (original acceptors + pad state)
    is_accepting_arr = is_accepting_np.tolist() + [True, False]

    # Calculate needed exception slots
    extra_slots = 1  # For new pad transitions
    new_max_exceptions = dfa.max_exceptions + extra_slots

    # Initialize exception arrays
    exception_symbols_arr = np.full((num_states, new_max_exceptions), -1, dtype=np.int32)
    exception_states_arr = np.full((num_states, new_max_exceptions), -1, dtype=np.int32)

    # Copy original exceptions
    for i in range(n_orig):
        for j in range(dfa.max_exceptions):
            sym = exception_symbols_np[i, j]
            if sym != -1:
                exception_symbols_arr[i, j] = sym
                exception_states_arr[i, j] = exception_states_np[i, j]

    # Add new transitions:
    # 1. From original accepting states to pad state on padding symbol
    for i in range(n_orig):
        if is_accepting_np[i]:
            # Find first available slot
            slot = np.where(exception_symbols_arr[i] == -1)[0]
            if slot.size > 0:
                slot_idx = slot[0]
                exception_symbols_arr[i, slot_idx] = pad_enc
                exception_states_arr[i, slot_idx] = n_pad

    # 2. From pad state to itself on padding symbol
    exception_symbols_arr[n_pad, 0] = pad_enc
    exception_states_arr[n_pad, 0] = n_pad

    # Build NFA using native Python types
    nfa = SparseNFA(
        num_states=num_states,
        base_state=base_state_arr,
        exception_symbols=exception_symbols_arr,
        exception_states=exception_states_arr,
        is_accepting=is_accepting_arr,
        start_state=dfa.start_state,
        symbol_arity=arity,
        base_alphabet=base_alphabet
    )

    # Convert to DFA and minimize
    return nfa.determinize()#.minimize()

def unpad(dfa: SparseDFA, padding_symbol: int = -1, remove_blank: bool = False) -> SparseDFA:
    """Remove trailing padding symbols from accepted words."""
    arity = dfa.symbol_arity
    base_alphabet = dfa.base_alphabet
    if padding_symbol == -1:
        padding_symbol = sorted(base_alphabet)[0]  # Default to first symbol
    pad_tuple = (padding_symbol,) * arity
    pad_tuple_enc = encode_symbol(pad_tuple, base_alphabet)
    
    # Compute closure under padding symbols
    closure = {}
    for q in range(dfa.num_states):
        closure[q] = set()
        stack = [q]
        while stack:
            state = stack.pop()
            # Convert state to native int for set operations
            state_int = int(state) if hasattr(state, '__int__') else state
            
            if state_int not in closure[q]:
                closure[q].add(state_int)
                # Follow padding transitions
                if pad_tuple_enc in dfa.exception_symbols[state]:
                    idx = jnp.where(dfa.exception_symbols[state] == pad_tuple_enc)[0]
                    if idx.size > 0:
                        next_state = dfa.exception_states[state, idx[0]]
                        # Convert to native int before adding to stack
                        stack.append(int(next_state))  
                # Default transition for padding
                else:
                    next_state = dfa.default_states[state]
                    # Avoid infinite loops from self-transitions
                    if int(next_state) != state_int:  
                        # Convert to native int before adding to stack
                        stack.append(int(next_state))  
    
    # New acceptance: states that can reach a final state via padding
    new_accepting = jnp.array([
        any(dfa.is_accepting[p] for p in closure[q]) for q in range(dfa.num_states)
    ])
    
    # Create new automaton
    if remove_blank:
        # Remove padding symbol from input symbols
        new_base_alphabet = base_alphabet - {padding_symbol}
        # Filter out padding transitions
        new_exception_symbols = jnp.full_like(dfa.exception_symbols, -1)
        new_exception_states = jnp.full_like(dfa.exception_states, -1)
        
        for q in range(dfa.num_states):
            valid_mask = (dfa.exception_symbols[q] != pad_tuple_enc) & (dfa.exception_symbols[q] != -1)
            valid_indices = jnp.where(valid_mask)[0]
            num_valid = len(valid_indices)
            
            if num_valid > 0:
                new_exception_symbols = new_exception_symbols.at[q, :num_valid].set(
                    dfa.exception_symbols[q, valid_indices]
                )
                new_exception_states = new_exception_states.at[q, :num_valid].set(
                    dfa.exception_states[q, valid_indices]
                )
        
        return SparseDFA(
            num_states=dfa.num_states,
            default_states=dfa.default_states,
            exception_symbols=new_exception_symbols,
            exception_states=new_exception_states,
            is_accepting=new_accepting,
            start_state=dfa.start_state,
            symbol_arity=arity,
            base_alphabet=new_base_alphabet
        ).minimize()
    else:
        return SparseDFA(
            num_states=dfa.num_states,
            default_states=dfa.default_states,
            exception_symbols=dfa.exception_symbols,
            exception_states=dfa.exception_states,
            is_accepting=new_accepting,
            start_state=dfa.start_state,
            symbol_arity=arity,
            base_alphabet=base_alphabet
        ).minimize()

def product(dfa: SparseDFA, n: int) -> SparseDFA:
    """Create the n-fold Cartesian product of the automaton's language."""
    if n == 0:
        return one()
    if n == 1:
        return dfa
    else:
        result = dfa
        for _ in range(n-1):
            result = stack(result, dfa)
        return result

def stack(dfa1: SparseDFA, dfa2: SparseDFA) -> SparseDFA:
    """
    Creates a stacked automaton that recognizes the concatenation of two relations
    without explicitly generating all possible symbols.
    
    The new automaton accepts tuples (x1,...,xk,y1,...,yl) where:
        (x1,...,xk) is accepted by dfa1 and 
        (y1,...,yl) is accepted by dfa2
        
    Args:
        dfa1: First automaton of arity k
        dfa2: Second automaton of arity l
        
    Returns:
        SparseDFA of arity k+l recognizing the stacked relation
    """
    # Validate common base alphabet
    if dfa1.base_alphabet != dfa2.base_alphabet:
        raise ValueError("Automata must have the same base alphabet")
    
    dfa1 = pad(dfa1)
    dfa2 = pad(dfa2)
    
    # Get arities
    k = dfa1.symbol_arity
    l = dfa2.symbol_arity
    arity = k + l
    base_alphabet = dfa1.base_alphabet
    
    # Create product states
    n1 = dfa1.num_states
    n2 = dfa2.num_states
    num_states = n1 * n2
    
    # On-the-fly construction: only generate reachable states
    start_pair = (dfa1.start_state, dfa2.start_state)
    queue = deque([start_pair])
    state_map = {start_pair: 0}
    
    new_default_states_list = []
    new_exception_symbols_list = []
    new_exception_states_list = []
    new_is_accepting_list = []
    
    # Helper function to split symbol
    def split_symbol(full_symbol_enc):
        """Split encoded symbol into two components"""
        full_tuple = decode_symbol(full_symbol_enc, arity, base_alphabet)
        s1_tuple = full_tuple[:k]
        s2_tuple = full_tuple[k:]
        s1_enc = encode_symbol(s1_tuple, base_alphabet)
        s2_enc = encode_symbol(s2_tuple, base_alphabet)
        return s1_enc, s2_enc
    
    # Build product automaton
    idx_counter = 0
    while queue:
        current_pair = queue.popleft()
        i, j = current_pair
        current_idx = state_map[current_pair]
        
        # Add acceptance status
        new_is_accepting_list.append(bool(dfa1.is_accepting[i]) and bool(dfa2.is_accepting[j]))
        
        # Collect all unique symbols that cause an exception in either DFA
        # or are part of the full alphabet
        all_relevant_symbols = set()
        
        # Add symbols that are exceptions in dfa1
        for pos1 in range(dfa1.max_exceptions):
            s1_enc = int(dfa1.exception_symbols[i, pos1])
            if s1_enc == -1:
                continue
            # Generate corresponding symbols for the full arity
            for symbol_char in base_alphabet:
                base_tuple = decode_symbol(s1_enc, k, base_alphabet)
                full_tuple = base_tuple + (symbol_char,) * l
                full_enc = encode_symbol(full_tuple, base_alphabet)
                all_relevant_symbols.add(full_enc)
        
        # Add symbols that are exceptions in dfa2
        for pos2 in range(dfa2.max_exceptions):
            s2_enc = int(dfa2.exception_symbols[j, pos2])
            if s2_enc == -1:
                continue
            # Generate corresponding symbols for the full arity
            for symbol_char in base_alphabet:
                base_tuple = decode_symbol(s2_enc, l, base_alphabet)
                full_tuple = (symbol_char,) * k + base_tuple
                full_enc = encode_symbol(full_tuple, base_alphabet)
                all_relevant_symbols.add(full_enc)
        
        # Add all symbols from the combined alphabet to ensure all transitions are considered
        for symbol_tuple_chars in it.product(sorted(base_alphabet), repeat=arity):
            all_relevant_symbols.add(encode_symbol(symbol_tuple_chars, base_alphabet))

        # Determine default transition for the product state
        # The default transition for the product automaton is formed by the default transitions
        # of the individual automata.
        def_i = int(dfa1.default_states[i])
        def_j = int(dfa2.default_states[j])
        default_target_pair = (def_i, def_j)
        
        if default_target_pair not in state_map:
            state_map[default_target_pair] = len(state_map)
            queue.append(default_target_pair)
        new_default_states_list.append(state_map[default_target_pair])
        
        # Process exceptions for the current product state
        current_exceptions_symbols = []
        current_exceptions_states = []
        
        for full_enc in sorted(list(all_relevant_symbols)): # Sort for deterministic output
            s1_enc, s2_enc = split_symbol(full_enc)
            
            next_i = int(dfa1.transition(i, s1_enc))
            next_j = int(dfa2.transition(j, s2_enc))
            next_pair = (next_i, next_j)
            
            # Only add as an exception if it deviates from the default transition
            if next_pair != default_target_pair:
                if next_pair not in state_map:
                    state_map[next_pair] = len(state_map)
                    queue.append(next_pair)
                current_exceptions_symbols.append(full_enc)
                current_exceptions_states.append(state_map[next_pair])
        
        new_exception_symbols_list.append(current_exceptions_symbols)
        new_exception_states_list.append(current_exceptions_states)
        
        idx_counter += 1

    # Pad exceptions to uniform length
    num_new_states = len(state_map)
    max_exceptions = max(len(ex) for ex in new_exception_symbols_list) if new_exception_symbols_list else 0
    padded_ex_syms = jnp.full((num_new_states, max_exceptions), -1, dtype=jnp.int32)
    padded_ex_states = jnp.full((num_new_states, max_exceptions), -1, dtype=jnp.int32)
    
    for i in range(num_new_states):
        syms = new_exception_symbols_list[i]
        states = new_exception_states_list[i]
        if syms:
            padded_ex_syms = padded_ex_syms.at[i, :len(syms)].set(jnp.array(syms, dtype=jnp.int32))
            padded_ex_states = padded_ex_states.at[i, :len(states)].set(jnp.array(states, dtype=jnp.int32))
    
    # Create and return the stacked automaton
    return SparseDFA(
        num_states=num_new_states,
        default_states=jnp.array(new_default_states_list),
        exception_symbols=jnp.array(padded_ex_syms),
        exception_states=jnp.array(padded_ex_states),
        is_accepting=jnp.array(new_is_accepting_list),
        start_state=0, # Start state is always 0 in the new mapping
        symbol_arity=arity,
        base_alphabet=base_alphabet
    )

def projection(dfa: SparseDFA, i: int) -> SparseDFA:
    """Project the automaton by existentially quantifying the i-th position."""
    arity = dfa.symbol_arity
    base_alphabet = dfa.base_alphabet
    new_arity = arity - 1
    
    # Create new automaton via subset construction
    start_state = frozenset([int(dfa.start_state)])
    state_queue = deque([start_state])
    state_to_id = {start_state: 0}
    id_to_state = [start_state]
    
    # New DFA components
    new_default_states = []
    new_exception_symbols = []
    new_exception_states = []
    new_accepting = []
    
    # Generate all possible symbols for the new alphabet
    new_alphabet = set(it.product(sorted(base_alphabet), repeat=new_arity))
    
    while state_queue:
        state_set = state_queue.popleft()
        state_id = state_to_id[state_set]
        
        # For each possible symbol in new alphabet
        trans_map = {}
        for symbol_tuple in new_alphabet:
            next_set = set()
            for q in state_set:
                # Try all possible values at position i
                for a in base_alphabet:
                    full_symbol = symbol_tuple[:i] + (a,) + symbol_tuple[i:]
                    full_symbol_enc = encode_symbol(full_symbol, base_alphabet)
                    next_state = dfa.transition(q, full_symbol_enc)
                    next_set.add(int(next_state))  # Convert to native int
            trans_map[symbol_tuple] = frozenset(next_set)
        
        # Handle case with no transitions
        if not trans_map:
            # Create a dead state if no transitions exist
            dead_state = frozenset()
            if dead_state not in state_to_id:
                new_id = len(id_to_state)
                state_to_id[dead_state] = new_id
                id_to_state.append(dead_state)
                # Dead state never accepts and transitions to itself
                new_accepting.append(False)
                new_default_states.append(new_id)
                new_exception_symbols.append([])
                new_exception_states.append([])
            default_target = state_to_id[dead_state]
            state_id_for_default = default_target
        else:
            # Find the most common transition
            targets = list(trans_map.values())
            default_target_set = max(set(targets), key=targets.count)
            
            # Get or create state ID for default target
            if default_target_set not in state_to_id:
                new_id = len(id_to_state)
                state_to_id[default_target_set] = new_id
                id_to_state.append(default_target_set)
                state_queue.append(default_target_set)
                state_id_for_default = new_id
            else:
                state_id_for_default = state_to_id[default_target_set]
            
            # Create exceptions for deviations
            exceptions = []
            for symbol_tuple, target_set in trans_map.items():
                if target_set != default_target_set:
                    # Get or create state ID for this target
                    if target_set not in state_to_id:
                        new_id = len(id_to_state)
                        state_to_id[target_set] = new_id
                        id_to_state.append(target_set)
                        state_queue.append(target_set)
                        target_id = new_id
                    else:
                        target_id = state_to_id[target_set]
                    
                    symbol_enc = encode_symbol(symbol_tuple, base_alphabet)
                    exceptions.append((symbol_enc, target_id))
            
            new_exception_symbols.append([e[0] for e in exceptions])
            new_exception_states.append([e[1] for e in exceptions])
        
        new_default_states.append(state_id_for_default)
        new_accepting.append(any(dfa.is_accepting[q] for q in state_set))
    
    # Pad exceptions
    max_ex = max(len(ex) for ex in new_exception_symbols) if new_exception_symbols else 0
    padded_ex_syms = jnp.full((len(id_to_state), max_ex), -1)
    padded_ex_states = jnp.full((len(id_to_state), max_ex), -1)
    
    for idx, (syms, states) in enumerate(zip(new_exception_symbols, new_exception_states)):
        if syms:
            padded_ex_syms = padded_ex_syms.at[idx, :len(syms)].set(jnp.array(syms, dtype=jnp.int32))
            padded_ex_states = padded_ex_states.at[idx, :len(states)].set(jnp.array(states, dtype=jnp.int32))
    
    return SparseDFA(
        num_states=len(id_to_state),
        default_states=jnp.array(new_default_states, dtype=jnp.int32),
        exception_symbols=padded_ex_syms,
        exception_states=padded_ex_states,
        is_accepting=jnp.array(new_accepting),
        start_state=0,
        symbol_arity=new_arity,
        base_alphabet=base_alphabet
    )

def expand(dfa, new_arity: int, pos: List[int]):
    # Input validation (unchanged)
    original_arity = dfa.symbol_arity
    base_alphabet = dfa.base_alphabet
    sorted_alphabet = sorted(base_alphabet)
    m = len(base_alphabet)
    new_num_states = dfa.num_states
    K = m ** (new_arity - len(set(pos)))  # Account for duplicate positions
    new_max_exceptions = dfa.max_exceptions * K
    
    # Precompute fixed positions and free indices
    fixed_mask = jnp.zeros(new_arity, dtype=bool)
    for idx in pos:
        fixed_mask = fixed_mask.at[idx].set(True)
    free_indices = jnp.where(~fixed_mask, size=new_arity, fill_value=-1)[0]
    free_count = jnp.sum(~fixed_mask).item()
    
    # Precompute encoding powers
    powers = m ** jnp.arange(new_arity-1, -1, -1, dtype=jnp.int64)
    sorted_alphabet_arr = jnp.arange(len(sorted_alphabet))
    
    # Prepare new DFA arrays
    new_default_states = jnp.zeros(new_num_states, dtype=jnp.int32)
    new_exception_symbols = jnp.full((new_num_states, new_max_exceptions), -1, dtype=jnp.int32)
    new_exception_states = jnp.full((new_num_states, new_max_exceptions), -1, dtype=jnp.int32)
    
    # Process each state
    for state in range(dfa.num_states):
        # Calculate new default state via frequency counts
        # The default state for the expanded DFA is simply the default state of the original DFA
        # as the 'default' transition applies when no exception matches, regardless of arity.
        new_default_states = new_default_states.at[state].set(int(dfa.default_states[state]))

        # Collect original exceptions
        orig_exceptions = dfa.exception_symbols[state][dfa.exception_symbols[state] != -1]
        orig_targets = dfa.exception_states[state][dfa.exception_symbols[state] != -1]     
        
        # Process exception blocks
        all_exception_symbols = []
        all_exception_states = []
        for sym_enc, target in zip(orig_exceptions, orig_targets):
            # Decode and convert to indices
            orig_tuple = decode_symbol(int(sym_enc), original_arity, base_alphabet)

            # Check consistency for duplicate positions
            expected = {}
            consistent = True
            for orig_idx, new_idx in enumerate(pos):
                val = orig_tuple[orig_idx]
                if new_idx in expected and expected[new_idx] != val:
                    consistent = False
                    break
                expected[new_idx] = val
            if not consistent:
                continue  # Skip inconsistent symbol

            # Generate expanded symbols as a block
            expanded_symbols = generate_expanded_block(
                sym_enc, 
                sorted_alphabet_arr,
                fixed_mask,
                free_indices,
                free_count,
                powers,
                original_arity,
                m,
                pos
            )
            all_exception_symbols.append(expanded_symbols)
            all_exception_states.append(jnp.full_like(expanded_symbols, target))
        
        if all_exception_symbols:
            # Flatten the lists and pad to max_exceptions
            all_exception_symbols = jnp.concatenate(all_exception_symbols)
            all_exception_states = jnp.concatenate(all_exception_states)
            new_exception_symbols = new_exception_symbols.at[state, :len(all_exception_symbols)].set(all_exception_symbols)
            new_exception_states = new_exception_states.at[state, :len(all_exception_states)].set(all_exception_states)
        
    return SparseDFA(
        num_states=new_num_states,
        default_states=new_default_states,
        exception_symbols=new_exception_symbols,
        exception_states=new_exception_states,
        is_accepting=dfa.is_accepting,
        start_state=dfa.start_state,
        symbol_arity=new_arity,
        base_alphabet=base_alphabet
    )


def generate_expanded_block(sym_enc, sorted_alphabet_arr, fixed_mask, free_indices, 
                           free_count, powers, original_arity, m, pos):
    # Decode original symbol
    powers_orig = m ** jnp.arange(original_arity-1, -1, -1, dtype=jnp.int64)
    digits = (sym_enc // powers_orig) % m
    orig_tuple = sorted_alphabet_arr[digits]
    
    # Create fixed values template
    fixed_values = jnp.full(len(fixed_mask), -1, dtype=sorted_alphabet_arr.dtype)
    for orig_idx, new_idx in enumerate(pos):
        fixed_values = fixed_values.at[new_idx].set(orig_tuple[orig_idx])
    
    # Generate free combinations
    if free_count > 0:
        n_comb = m ** free_count
        grid = jnp.indices((m,)*free_count).reshape(free_count, -1).T
        free_vals = sorted_alphabet_arr[grid]
        symbol_tensors = jnp.tile(fixed_values, (n_comb, 1))
        symbol_tensors = symbol_tensors.at[:, free_indices[:free_count]].set(free_vals)
    else:
        symbol_tensors = fixed_values[None, :]
    
    # Encode symbols
    indices = jnp.searchsorted(sorted_alphabet_arr, symbol_tensors)
    return jnp.sum(indices * powers, axis=1, dtype=jnp.int32)

# We'll define a custom heap structure for length-lexicographic ordering
class LengthLexHeap:
    def __init__(self):
        self.heap = []
        
    def push(self, item):
        # item: (word_tuple, state)
        # word_tuple is tuple of strings
        # Priority: 1. Total length (sum of lengths), 2. Lex order
        total_length = max(len(comp) for comp in item[0])
        heapq.heappush(self.heap, (total_length, item[0], item[1]))
        
    def pop(self):
        _, word, state = heapq.heappop(self.heap)
        return (word, state)
        
    def __len__(self):
        return len(self.heap)

def iterate_language(dfa: SparseDFA, decoder: Callable = None, 
                    backward: bool = False, padding_symbol: int = -1) -> Generator:
    """
    Generator over the language of a SparseDFA. Yields words in length-lexicographic order.
    Note: The algorithm assumes minimality and optimal sparsity of the automaton.

    :param dfa: Sparse automaton
    :param decoder: Function to decode words to Python objects
    :param backward: If True, generate words in reverse order
    :param padding_symbol: Integer representing padding symbol
    :return: Generator of words (or decoded objects)
    """
    successors = {q: dfa.successors(q) for q in range(dfa.num_states)}
    nonempty = {q for q in range(dfa.num_states) if len(successors[q]) > 0 or q not in successors[q]}

    arity = dfa.symbol_arity
    
    # Build reversed transitions: state -> symbol -> set of previous states
    rev_transitions = {}
    for state in range(dfa.num_states):
        rev_transitions[state] = {}


    start_set = {dfa.start_state}
    final_set = set(jnp.where(dfa.is_accepting)[0].tolist())

    # Initialize heap with starting states
    heap = LengthLexHeap()
    for state in start_set:
        if state in nonempty:
            # Represent words as tuple of empty strings
            heap.push((tuple(["" for _ in range(arity)]), state))
    
    def cat(word, symbol):
        """Concatenate symbol to word based on direction."""
        if backward:
            return str(symbol) + word
        else:
            return word + str(symbol)
        
    def push(heap, word_tuple, sym_enc, next_state):
        """Push a new word onto the heap with the given extension symbol and next state."""
        if encode_symbol((padding_symbol,) * arity, dfa.base_alphabet) == sym_enc:
            # Skip padding symbols
            return
        # Decode symbol
        symbol_tuple = decode_symbol(sym_enc, arity, dfa.base_alphabet)

        # Create new word components
        new_components = []
        for comp, sym in zip(word_tuple, symbol_tuple):
            if sym == padding_symbol:
                # Keep component unchanged
                new_components.append(comp)
            else:
                # Prepend symbol to component
                new_components.append(cat(comp, sym))

        new_word_tuple = tuple(new_components)

        # Add to heap
        heap.push((new_word_tuple, next_state))

    # Main loop
    visited_words = set()
    while heap:
        word_tuple, state = heap.pop()
        
        # Skip duplicates
        word_key = (state, word_tuple)
        if word_key in visited_words:
            continue
        visited_words.add(word_key)
        
        # Check if we've reached a final state
        if state in final_set:
            if decoder:
                yield decoder(word_tuple)
            else:
                yield word_tuple
        
        # process transitions
        ex_mask = dfa.exception_symbols[state] != -1
        ex_symbols = dfa.exception_symbols[state, ex_mask]
        ex_states = dfa.exception_states[state, ex_mask]
        for sym_enc, next_state in zip(ex_symbols, ex_states):
            sym_enc, next_state = int(sym_enc), int(next_state)
            if next_state not in nonempty:
                continue
            
            push(heap, word_tuple, sym_enc, next_state)
        
        default = int(dfa.default_states[state])
        if default in nonempty:
            # get all non-exception symbols
            default_symbols = complement(ex_symbols, 0, len(dfa.base_alphabet)**dfa.symbol_arity - 1)
            for sym_enc in default_symbols:
                push(heap, word_tuple, sym_enc, default)





def lsbf_Z_automaton(z: int) -> SparseDFA:
    """
    Creates a SparseDFA for LSB-first representation of integer z with sign bit and padding.
    Alphabet encoding:
        "*" = 0
        "0" = 1
        "1" = 2
    """
    # Handle special case for zero
    if z == 0:
        return SparseDFA(
            num_states=4,
            default_states=jnp.array([3, 3, 3, 3], dtype=jnp.int32),
            exception_symbols=jnp.array([[1], [1], [0], [-1]], dtype=jnp.int32),  # "0"=1, "*"=0
            exception_states=jnp.array([[1], [2], [2], [-1]], dtype=jnp.int32),
            is_accepting=jnp.array([False, False, True, False]),
            start_state=0,
            symbol_arity=1,
            base_alphabet={"*", "0", "1"}  # "*"=0, "0"=1, "1"=2
        )
    
    # Determine sign and magnitude
    sign_symbol = 1 if z >= 0 else 2  # "0"=1 for positive, "1"=2 for negative
    magnitude = abs(z)
    
    # Convert to LSB-first bits (without trailing zeros)
    bits = []
    while magnitude:
        bits.append(2 if magnitude & 1 else 1)  # 1→"0"=1, 2→"1"=2
        magnitude >>= 1
    
    # Create representation: [sign_symbol] + bits (LSB first)
    rep = [sign_symbol] + bits
    n = len(rep)
    
    # States: 
    # 0 to n-1: processing representation
    # n: accepting state (after full representation)
    # n+1: dead state
    num_states = n + 2
    
    # Create arrays with vectorized operations
    default_states = jnp.full(num_states, n+1, dtype=jnp.int32)  # Default to dead state
    
    # Exception symbols: rep for states 0..n-1, 0 ('*') for state n
    exception_symbols = jnp.full((num_states, 1), -1, dtype=jnp.int32)
    exception_symbols = exception_symbols.at[:n, 0].set(jnp.array(rep, dtype=jnp.int32))
    exception_symbols = exception_symbols.at[n, 0].set(0)  # '*' for accepting state
    
    # Exception states: next state for representation, self for padding
    exception_states = jnp.full((num_states, 1), -1, dtype=jnp.int32)
    exception_states = exception_states.at[:n, 0].set(jnp.arange(1, n+1))
    exception_states = exception_states.at[n, 0].set(n)  # loop in accepting state
    
    # Accepting state is state n
    is_accepting = jnp.zeros(num_states, dtype=bool)
    is_accepting = is_accepting.at[n].set(True)
    
    return SparseDFA(
        num_states=num_states,
        default_states=default_states,
        exception_symbols=exception_symbols,
        exception_states=exception_states,
        is_accepting=is_accepting,
        start_state=0,
        symbol_arity=1,
        base_alphabet={"*", "0", "1"}  # "*"=0, "0"=1, "1"=2
    )