import heapq
import numpy as np
from collections import deque
from typing import Callable, Dict, Generator, List, Set
import itertools as it

from autstr.mtbdd import NONE, bits_of, var_tables
from autstr.sparse_automata import SparseDFA, SparseNFA
from autstr.buildin.automata import one
from autstr.utils.misc import decode_symbol, encode_symbol, complement




# ====== Helper Functions ======
def _symbol_assignment(symbol: int, arity: int, m: int, bits: int) -> List[int]:
    """The binary variable assignment of a convolution symbol."""
    div, shift = var_tables(arity, m, bits)
    return [int((symbol // div[v]) % m) >> int(shift[v]) & 1
            for v in range(arity * bits)]


def pad(dfa: SparseDFA, padding_symbol: int = -1) -> SparseDFA:
    """Accept the language followed by any number of padding symbols.

    An accepting state may already have a transition on the padding symbol, so
    adding the padding loop makes the automaton nondeterministic: the padding
    symbol now leads both to the original target and into the padding loop.
    Only that one symbol changes, which on the diagrams is a single path
    rewrite; the subset construction then restores determinism.
    """
    arity = dfa.symbol_arity
    base_alphabet = dfa.base_alphabet
    if padding_symbol == -1:
        padding_symbol = sorted(base_alphabet)[0]
    pad_enc = encode_symbol((padding_symbol,) * arity, base_alphabet)

    if dfa.is_empty():
        return dfa

    store = dfa.store
    m, bits = dfa.m, dfa.bits
    n = dfa.num_states
    PAD, DEAD = n, n + 1
    pad_assignment = _symbol_assignment(pad_enc, arity, m, bits)

    subsets: List[int] = []                        # sets of states, as bitsets
    subset_ids: Dict[int, int] = {}

    def subset_id(mask: int) -> int:
        idx = subset_ids.get(mask)
        if idx is None:
            idx = subset_ids[mask] = len(subsets)
            subsets.append(mask)
        return idx

    def singleton(target: int) -> int:
        return subset_id(1 << target)

    singleton_cache: Dict[int, int] = {}
    nodes = []
    for q in range(n):
        node = store.apply1(int(dfa.nodes[q]), singleton, singleton_cache)
        if dfa.is_accepting[q]:
            current = int(store.eval_batch(np.array([node]),
                                           np.array([pad_enc], dtype=np.int64),
                                           arity, m, bits)[0])
            node = store.set_path(node, pad_assignment,
                                  subset_id(subsets[current] | (1 << PAD)))
        nodes.append(node)

    dead = store.const(singleton(DEAD), arity, m, bits)
    nodes.append(store.set_path(dead, pad_assignment, singleton(PAD)))
    nodes.append(dead)

    nfa = SparseNFA(
        n + 2, is_accepting=np.r_[dfa.is_accepting, True, False],
        start_state=dfa.start_state, symbol_arity=arity,
        base_alphabet=base_alphabet,
        nodes=np.array(nodes, dtype=np.int64), subsets=subsets)
    return nfa.determinize()

def unpad(dfa: SparseDFA, padding_symbol: int = -1) -> SparseDFA:
    """Remove trailing padding symbols from accepted words: a state becomes
    accepting iff reading padding from it can reach an accepting state. The
    padding successor is a function, so its orbit is closed by iterative
    doubling; the transition diagrams are untouched."""
    arity = dfa.symbol_arity
    base_alphabet = dfa.base_alphabet
    if padding_symbol == -1:
        padding_symbol = sorted(base_alphabet)[0]
    pad_enc = encode_symbol((padding_symbol,) * arity, base_alphabet)

    pad_next = dfa.store.eval_batch(
        dfa.nodes, np.full(dfa.num_states, pad_enc, dtype=np.int64),
        arity, dfa.m, dfa.bits)

    new_accepting = np.asarray(dfa.is_accepting, dtype=bool).copy()
    steps = 1
    while steps < dfa.num_states:
        new_accepting |= new_accepting[pad_next]
        pad_next = pad_next[pad_next]
        steps *= 2

    return SparseDFA(
        dfa.num_states, is_accepting=new_accepting,
        start_state=dfa.start_state, symbol_arity=arity,
        base_alphabet=base_alphabet, nodes=dfa.nodes).minimize()

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
    """Existentially quantify tape i.

    The projected transition of a state is the union of the m cofactors of its
    diagram on tape i's variable block — a diagram over *sets* of states — and
    the subset construction then folds those over the members of each subset.
    Neither the source nor the projected alphabet is ever enumerated.
    """
    arity = dfa.symbol_arity
    if arity < 2:
        raise ValueError("cannot project the only tape")
    store = dfa.store
    m, bits = dfa.m, dfa.bits
    new_arity = arity - 1

    subsets: List[int] = []                        # sets of states, as bitsets
    subset_ids: Dict[int, int] = {}

    def subset_id(mask: int) -> int:
        idx = subset_ids.get(mask)
        if idx is None:
            idx = subset_ids[mask] = len(subsets)
            subsets.append(mask)
        return idx

    def singleton(target: int) -> int:
        return subset_id(1 << target)

    def union(a: int, b: int) -> int:
        if a == NONE or b == NONE:
            return NONE
        return subset_id(subsets[a] | subsets[b])

    # drop tape i's variable block; the tapes below it move up one block
    varmap = [(v // bits - (1 if v // bits > i else 0)) * bits + v % bits
              for v in range(arity * bits)]
    singleton_cache: Dict[int, int] = {}
    union_cache: Dict[int, int] = {}
    rename_cache: Dict[int, int] = {}

    set_nodes = np.empty(dfa.num_states, dtype=np.int64)
    for q in range(dfa.num_states):
        node = store.apply1(int(dfa.nodes[q]), singleton, singleton_cache)
        node = store.quantify_letter(node, i, m, bits, union, union_cache)
        set_nodes[q] = store.rename(node, varmap, rename_cache)

    new_subsets: List[int] = []
    new_ids: Dict[int, int] = {}

    def state_of(sid: int) -> int:
        mask = subsets[sid]
        idx = new_ids.get(mask)
        if idx is None:
            idx = new_ids[mask] = len(new_subsets)
            new_subsets.append(mask)
        return idx

    state_cache: Dict[int, int] = {}
    state_of(singleton(int(dfa.start_state)))
    nodes: List[int] = []
    index = 0
    while index < len(new_subsets):                # grows inside apply1
        members = bits_of(new_subsets[index])
        index += 1
        node = int(set_nodes[members[0]])
        for q in members[1:]:
            node = store.apply2(node, int(set_nodes[q]), union, union_cache)
        nodes.append(store.apply1(node, state_of, state_cache))

    accepting_mask = 0
    for q in np.flatnonzero(dfa.is_accepting).tolist():
        accepting_mask |= 1 << q
    accepting = np.array([bool(mask & accepting_mask) for mask in new_subsets],
                         dtype=bool)
    return SparseDFA(len(new_subsets), is_accepting=accepting, start_state=0,
                     symbol_arity=new_arity, base_alphabet=dfa.base_alphabet,
                     nodes=np.array(nodes, dtype=np.int64))


def expand(dfa, new_arity: int, pos: List[int]) -> SparseDFA:
    """Expand a DFA of arity k to new_arity by placing original tape t at new
    position pos[t]; the remaining positions accept any symbol.

    This is a variable renaming on the transition diagrams: the new tapes'
    variables simply do not occur. Repeated entries in `pos` identify tapes,
    which restricts the relation to their diagonal.
    """
    store = dfa.store
    m, bits = dfa.m, dfa.bits
    varmap = [pos[v // bits] * bits + v % bits
              for v in range(dfa.symbol_arity * bits)]
    valid = store.const(0, new_arity, m, bits)

    def keep_valid(target: int, ok: int) -> int:
        # the new tapes are unconstrained by the source, so their invalid
        # binary codes must be excluded explicitly
        return NONE if (target == NONE or ok == NONE) else target

    rename_cache: Dict[int, int] = {}
    mask_cache: Dict[int, int] = {}
    nodes = [store.apply2(store.rename(int(node), varmap, rename_cache),
                          valid, keep_valid, mask_cache)
             for node in dfa.nodes.tolist()]

    return SparseDFA(
        dfa.num_states, is_accepting=dfa.is_accepting,
        start_state=dfa.start_state, symbol_arity=new_arity,
        base_alphabet=dfa.base_alphabet,
        nodes=np.array(nodes, dtype=np.int64))


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
    tape perm[t] of the input — a permutation of the variable blocks."""
    k = dfa.symbol_arity
    if sorted(perm) != list(range(k)):
        raise ValueError(f"perm must be a permutation of range({k})")
    inverse = [0] * k
    for new_tape, old_tape in enumerate(perm):
        inverse[old_tape] = new_tape
    bits = dfa.bits
    varmap = [inverse[v // bits] * bits + v % bits for v in range(k * bits)]
    cache: Dict[int, int] = {}
    nodes = [dfa.store.rename(int(node), varmap, cache)
             for node in dfa.nodes.tolist()]
    return SparseDFA(
        dfa.num_states, is_accepting=dfa.is_accepting,
        start_state=dfa.start_state, symbol_arity=k,
        base_alphabet=dfa.base_alphabet,
        nodes=np.array(nodes, dtype=np.int64))


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