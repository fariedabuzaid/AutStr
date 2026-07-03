import heapq
import numpy as np
from collections import deque, defaultdict, Counter
from functools import partial
from typing import Generator, Optional, Callable, Dict, List, Set, Union
import itertools as it

from autstr.utils.logic import get_free_elementary_vars
from autstr.sparse_automata import SparseDFA, SparseNFA, _sort_exception_rows, _sorted_row_lookup
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
    valid = exception_symbols_np != -1
    exception_symbols_arr[:n_orig, :dfa.max_exceptions] = np.where(valid, exception_symbols_np, -1)
    exception_states_arr[:n_orig, :dfa.max_exceptions] = np.where(valid, exception_states_np, -1)

    # Add new transitions:
    # 1. From original accepting states to pad state on padding symbol,
    #    in the first free slot (the extra slot guarantees one exists)
    accepting_rows = np.flatnonzero(is_accepting_np[:n_orig])
    free_slot = (exception_symbols_arr[accepting_rows] == -1).argmax(axis=1)
    exception_symbols_arr[accepting_rows, free_slot] = pad_enc
    exception_states_arr[accepting_rows, free_slot] = n_pad

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

    defaults = np.asarray(dfa.default_states, dtype=np.int64)
    ex_symbols = np.asarray(dfa.exception_symbols, dtype=np.int64)
    ex_states = np.asarray(dfa.exception_states, dtype=np.int64)
    accepting = np.asarray(dfa.is_accepting, dtype=bool)

    # The padding successor of each state (a deterministic function)
    if ex_symbols.shape[1] > 0:
        is_pad = ex_symbols == pad_tuple_enc
        hit = is_pad.any(axis=1)
        first = is_pad.argmax(axis=1)
        pad_next = np.where(
            hit,
            np.take_along_axis(ex_states, first[:, None], axis=1)[:, 0],
            defaults
        )
    else:
        pad_next = defaults

    # New acceptance: states that can reach a final state via padding.
    # OR of acceptance over each state's forward orbit by iterative doubling.
    new_accepting = accepting.copy()
    steps = 1
    while steps < dfa.num_states:
        new_accepting |= new_accepting[pad_next]
        pad_next = pad_next[pad_next]
        steps *= 2

    # Create new automaton
    if remove_blank:
        # Remove padding symbol from input symbols
        new_base_alphabet = base_alphabet - {padding_symbol}
        # Filter out padding transitions and left-compact the rows
        keep = (ex_symbols != pad_tuple_enc) & (ex_symbols != -1)
        order = np.argsort(~keep, axis=1, kind='stable')
        keep_sorted = np.take_along_axis(keep, order, axis=1)
        new_exception_symbols = np.where(keep_sorted, np.take_along_axis(ex_symbols, order, axis=1), -1)
        new_exception_states = np.where(keep_sorted, np.take_along_axis(ex_states, order, axis=1), -1)

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
    padded_ex_syms = np.full((num_new_states, max_exceptions), -1, dtype=np.int32)
    padded_ex_states = np.full((num_new_states, max_exceptions), -1, dtype=np.int32)

    for i in range(num_new_states):
        syms = new_exception_symbols_list[i]
        states = new_exception_states_list[i]
        if syms:
            padded_ex_syms[i, :len(syms)] = syms
            padded_ex_states[i, :len(states)] = states

    # Create and return the stacked automaton
    return SparseDFA(
        num_states=num_new_states,
        default_states=np.array(new_default_states_list, dtype=np.int32),
        exception_symbols=padded_ex_syms,
        exception_states=padded_ex_states,
        is_accepting=np.array(new_is_accepting_list, dtype=bool),
        start_state=0, # Start state is always 0 in the new mapping
        symbol_arity=arity,
        base_alphabet=base_alphabet
    )

def projection(dfa: SparseDFA, i: int) -> SparseDFA:
    """Project the automaton by existentially quantifying the i-th position.

    Sparse subset construction: for a subset S, only projected symbols whose
    insertions hit an exception of some member of S can lead anywhere other
    than the subset of member defaults. Candidates are therefore derived from
    the members' exception tables instead of enumerating the full projected
    alphabet, and all transitions of a subset are resolved in one batched
    numpy lookup.
    """
    arity = dfa.symbol_arity
    base_alphabet = dfa.base_alphabet
    m = len(base_alphabet)
    new_arity = arity - 1
    n_proj_symbols = m ** new_arity

    defaults = np.asarray(dfa.default_states, dtype=np.int64)
    ex_syms = np.asarray(dfa.exception_symbols, dtype=np.int64)
    ex_states = np.asarray(dfa.exception_states, dtype=np.int64)
    sorted_syms, sorted_targets = _sort_exception_rows(ex_syms, ex_states)
    acc = np.asarray(dfa.is_accepting, dtype=bool)
    max_ex = ex_syms.shape[1]

    # Weight of the projected-out digit: enc = (high*m + digit_i)*p + low
    p = m ** (arity - 1 - i)
    insert_offsets = np.arange(m, dtype=np.int64) * p

    state_to_id = {}
    id_to_set = []
    new_accepting = []

    def get_id(key):
        idx = state_to_id.get(key)
        if idx is None:
            idx = state_to_id[key] = len(id_to_set)
            members = np.array(key, dtype=np.int64)
            id_to_set.append(members)
            new_accepting.append(bool(acc[members].any()))
        return idx

    get_id((int(dfa.start_state),))

    new_default_states = []
    new_exception_symbols = []
    new_exception_states = []

    next_unprocessed = 0
    while next_unprocessed < len(id_to_set):
        members = id_to_set[next_unprocessed]
        next_unprocessed += 1

        # Candidate projected symbols: projections of members' exceptions
        member_syms = ex_syms[members]
        valid_syms = member_syms[member_syms != -1]
        cand = np.unique(valid_syms // (p * m) * p + valid_syms % p)

        default_key = tuple(np.unique(defaults[members]).tolist())

        exceptions = []
        if cand.size > 0:
            # All insertions of the candidates: (n_cand, m)
            full = (cand[:, None] // p) * (p * m) + insert_offsets + cand[:, None] % p

            # Batched transition lookup via binary search: (n_members, n_cand, m)
            n_members, n_cand = members.shape[0], cand.shape[0]
            flat_cands = np.broadcast_to(full.reshape(-1), (n_members, n_cand * m))
            next_states = _sorted_row_lookup(
                sorted_syms[members], sorted_targets[members],
                defaults[members], flat_cands
            ).reshape(n_members, n_cand, m)

            for j in range(n_cand):
                key = tuple(np.unique(next_states[:, j, :]).tolist())
                if key != default_key:
                    exceptions.append((int(cand[j]), key))

            if len(exceptions) == n_proj_symbols:
                # Every projected symbol is an exception, so the all-defaults
                # target can never be reached; make the most common target the
                # default instead of introducing a spurious state.
                counts = Counter(key for _, key in exceptions)
                default_key = counts.most_common(1)[0][0]
                exceptions = [(sym, key) for sym, key in exceptions if key != default_key]

        new_default_states.append(get_id(default_key))
        new_exception_symbols.append([sym for sym, _ in exceptions])
        new_exception_states.append([get_id(key) for _, key in exceptions])

    # Pad exceptions
    num_new_states = len(id_to_set)
    max_new_ex = max(len(ex) for ex in new_exception_symbols) if new_exception_symbols else 0
    padded_ex_syms = np.full((num_new_states, max_new_ex), -1, dtype=np.int32)
    padded_ex_states = np.full((num_new_states, max_new_ex), -1, dtype=np.int32)

    for idx, (syms, states) in enumerate(zip(new_exception_symbols, new_exception_states)):
        if syms:
            padded_ex_syms[idx, :len(syms)] = syms
            padded_ex_states[idx, :len(states)] = states

    return SparseDFA(
        num_states=num_new_states,
        default_states=np.array(new_default_states, dtype=np.int32),
        exception_symbols=padded_ex_syms,
        exception_states=padded_ex_states,
        is_accepting=np.array(new_accepting, dtype=bool),
        start_state=0,
        symbol_arity=new_arity,
        base_alphabet=base_alphabet
    )

def expand(dfa, new_arity: int, pos: List[int]):
    """Expand a DFA of arity k to new_arity by placing original tape t at new
    position pos[t]; the remaining positions accept any symbol. Every expanded
    exception is a closed-form function of an original exception, so the whole
    construction is computed as one batched numpy operation over all exceptions.
    """
    original_arity = dfa.symbol_arity
    base_alphabet = dfa.base_alphabet
    m = len(base_alphabet)
    num_states = dfa.num_states

    ex_syms = np.asarray(dfa.exception_symbols, dtype=np.int64)
    ex_states = np.asarray(dfa.exception_states, dtype=np.int64)
    max_ex = ex_syms.shape[1]

    # Digit representation of all exception symbols: (num_states, max_ex, k)
    powers_orig = m ** np.arange(original_arity - 1, -1, -1, dtype=np.int64)
    digits = (ex_syms[:, :, None] // powers_orig) % m

    powers_new = m ** np.arange(new_arity - 1, -1, -1, dtype=np.int64)

    # Fixed part of the expanded encoding, plus consistency check for
    # duplicate positions (all original tapes mapped to the same new position
    # must carry the same value; otherwise the exception has no expansion).
    valid = ex_syms != -1
    fixed_enc = np.zeros((num_states, max_ex), dtype=np.int64)
    first_at = {}
    for orig_idx, new_idx in enumerate(pos):
        if new_idx in first_at:
            valid &= digits[:, :, orig_idx] == digits[:, :, first_at[new_idx]]
        else:
            first_at[new_idx] = orig_idx
            fixed_enc += digits[:, :, orig_idx] * powers_new[new_idx]

    # Encodings of all combinations at the free positions: (K,)
    free_pos = [p for p in range(new_arity) if p not in first_at]
    free_count = len(free_pos)
    if free_count > 0:
        grid = np.indices((m,) * free_count).reshape(free_count, -1).T  # (K, free_count)
        free_enc = grid @ powers_new[free_pos]
    else:
        free_enc = np.zeros(1, dtype=np.int64)
    K = free_enc.shape[0]

    # Expanded exceptions: each original exception becomes a block of K
    # symbols with the same target. Flattened per state, blocks keep the
    # original exception order.
    exp_syms = (fixed_enc[:, :, None] + free_enc).reshape(num_states, max_ex * K)
    exp_states = np.broadcast_to(ex_states[:, :, None], (num_states, max_ex, K)).reshape(num_states, max_ex * K)
    exp_valid = np.broadcast_to(valid[:, :, None], (num_states, max_ex, K)).reshape(num_states, max_ex * K)

    # Left-compact valid entries within each row, pad with -1
    order = np.argsort(~exp_valid, axis=1, kind='stable')
    exp_valid_sorted = np.take_along_axis(exp_valid, order, axis=1)
    new_exception_symbols = np.where(exp_valid_sorted, np.take_along_axis(exp_syms, order, axis=1), -1)
    new_exception_states = np.where(exp_valid_sorted, np.take_along_axis(exp_states, order, axis=1), -1)

    return SparseDFA(
        num_states=num_states,
        default_states=dfa.default_states,
        exception_symbols=new_exception_symbols,
        exception_states=new_exception_states,
        is_accepting=dfa.is_accepting,
        start_state=dfa.start_state,
        symbol_arity=new_arity,
        base_alphabet=base_alphabet
    )

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
    final_set = set(np.flatnonzero(dfa.is_accepting).tolist())

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





def permute_tapes(dfa: SparseDFA, perm: List[int]) -> SparseDFA:
    """Reorder the tapes of a multi-tape automaton: tape t of the result is
    tape perm[t] of the input. Only the symbol encodings change."""
    if sorted(perm) != list(range(dfa.symbol_arity)):
        raise ValueError(f"perm must be a permutation of range({dfa.symbol_arity})")
    m = len(dfa.base_alphabet)
    powers = m ** np.arange(dfa.symbol_arity - 1, -1, -1, dtype=np.int64)
    symbols = dfa.exception_symbols.astype(np.int64)
    digits = (symbols[:, :, None] // powers) % m
    new_symbols = (digits[:, :, perm] * powers).sum(axis=2)
    new_symbols = np.where(dfa.exception_symbols == -1, -1, new_symbols).astype(np.int32)
    return SparseDFA(
        num_states=dfa.num_states,
        default_states=dfa.default_states,
        exception_symbols=new_symbols,
        exception_states=dfa.exception_states,
        is_accepting=dfa.is_accepting,
        start_state=dfa.start_state,
        symbol_arity=dfa.symbol_arity,
        base_alphabet=dfa.base_alphabet
    )


def word_automaton(word: List, base_alphabet: Set, padding_symbol=None) -> SparseDFA:
    """Automaton accepting exactly the given word, optionally followed by
    trailing padding symbols.

    :param word: sequence of symbols from base_alphabet
    :param base_alphabet: the base alphabet
    :param padding_symbol: if given, accept word followed by any number of
        padding symbols
    :return: SparseDFA of arity 1 recognizing {word}·{pad}*
    """
    n = len(word)
    # States 0..n-1 read the word, n accepts (with optional pad loop), n+1 dead
    num_states = n + 2
    dead = n + 1

    max_exc = 1
    default_states = np.full(num_states, dead, dtype=np.int32)
    exception_symbols = np.full((num_states, max_exc), -1, dtype=np.int32)
    exception_states = np.full((num_states, max_exc), -1, dtype=np.int32)

    for i, symbol in enumerate(word):
        exception_symbols[i, 0] = encode_symbol((symbol,), base_alphabet)
        exception_states[i, 0] = i + 1

    if padding_symbol is not None:
        exception_symbols[n, 0] = encode_symbol((padding_symbol,), base_alphabet)
        exception_states[n, 0] = n

    is_accepting = np.zeros(num_states, dtype=bool)
    is_accepting[n] = True

    return SparseDFA(
        num_states=num_states,
        default_states=default_states,
        exception_symbols=exception_symbols,
        exception_states=exception_states,
        is_accepting=is_accepting,
        start_state=0,
        symbol_arity=1,
        base_alphabet=base_alphabet
    )


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
            default_states=np.array([3, 3, 3, 3], dtype=np.int32),
            exception_symbols=np.array([[1], [1], [0], [-1]], dtype=np.int32),  # "0"=1, "*"=0
            exception_states=np.array([[1], [2], [2], [-1]], dtype=np.int32),
            is_accepting=np.array([False, False, True, False]),
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
    default_states = np.full(num_states, n+1, dtype=np.int32)  # Default to dead state

    # Exception symbols: rep for states 0..n-1, 0 ('*') for state n
    exception_symbols = np.full((num_states, 1), -1, dtype=np.int32)
    exception_symbols[:n, 0] = rep
    exception_symbols[n, 0] = 0  # '*' for accepting state

    # Exception states: next state for representation, self for padding
    exception_states = np.full((num_states, 1), -1, dtype=np.int32)
    exception_states[:n, 0] = np.arange(1, n+1)
    exception_states[n, 0] = n  # loop in accepting state

    # Accepting state is state n
    is_accepting = np.zeros(num_states, dtype=bool)
    is_accepting[n] = True
    
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