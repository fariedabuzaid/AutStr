import jax
import jax.numpy as jnp
import numpy as np
from collections import deque, defaultdict
from functools import partial
from typing import Tuple, Optional, Callable, Dict, List, Set, Union
import graphviz

from autstr.utils.misc import decode_symbol
from autstr.utils.misc import encode_symbol, complement


class SparseDFA:
    def __init__(self, num_states: int, default_states: jnp.ndarray,
                 exception_symbols: jnp.ndarray, exception_states: jnp.ndarray,
                 is_accepting: jnp.ndarray, start_state: int,
                 symbol_arity: int = 1, base_alphabet: Optional[Set[int]] = None):
        self.num_states = num_states
        self.default_states = default_states
        self.exception_symbols = exception_symbols
        self.exception_states = exception_states
        self.is_accepting = is_accepting
        self.start_state = start_state
        self.max_exceptions = exception_symbols.shape[1]
        self.symbol_arity = symbol_arity
        self.base_alphabet = base_alphabet or self._infer_base_alphabet()
        self.base_alphabet_frozen = frozenset(self.base_alphabet)
        
    def _infer_base_alphabet(self) -> Set[int]:
        """Infer base alphabet from exception symbols."""
        symbols = set(np.unique(self.exception_symbols))
        symbols.discard(-1)
        return symbols or {0}
    
    def encode_symbol(self, symbol_tuple: Tuple[int]) -> int:
        return encode_symbol(symbol_tuple, self.base_alphabet_frozen)
    
    def decode_symbol(self, symbol_enc: int) -> Tuple[int]:
        return decode_symbol(symbol_enc, self.symbol_arity, self.base_alphabet_frozen)
    
    def transition(self, state: int, symbol: int) -> int:
        """Get the next state for a given symbol."""
        # Check for exception transitions
        for i in range(self.max_exceptions):
            ex_symbol = int(self.exception_symbols[state, i])
            if ex_symbol == -1:
                continue  # Skip padding
            if ex_symbol == symbol:
                return int(self.exception_states[state, i])
        
        # Return default state if no exception matches
        return int(self.default_states[state])

    def compute(self, word: jnp.ndarray) -> int:
        def step(state, symbol):
            return self.transition(state, symbol), None
        final_state, _ = jax.lax.scan(step, self.start_state, word)
        return final_state

    def accepts(self, word: jnp.ndarray) -> jnp.ndarray:
        final_state = self.compute(word)
        return self.is_accepting[final_state]
    
    def is_empty(self) -> bool:
        """Check if the language is empty"""
        # BFS to find any accepting state
        visited = np.zeros(self.num_states, dtype=bool)
        queue = deque([self.start_state])
        
        while queue:
            state = queue.popleft()
            if self.is_accepting[state]:
                return False
            if visited[state]:
                continue
            visited[state] = True
            
            # Check all transitions
            next_states = set()
            next_states.add(int(self.default_states[state]))
            for i in range(self.max_exceptions):
                sym = self.exception_symbols[state, i]
                if sym != -1:
                    next_states.add(int(self.exception_states[state, i]))
            
            for ns in next_states:
                if not visited[ns]:
                    queue.append(ns)
                    
        return True

    def complement(self) -> 'SparseDFA':
        new_accepting = ~self.is_accepting
        return SparseDFA(
            self.num_states,
            self.default_states,
            self.exception_symbols,
            self.exception_states,
            new_accepting,
            self.start_state,
            self.symbol_arity,
            self.base_alphabet
        )

    def intersection(self, other: 'SparseDFA') -> 'SparseDFA':
        return self._product(other, combine_accept=lambda a, b: a & b)
    

    def union(self, other: 'SparseDFA') -> 'SparseDFA':
        return self._product(other, combine_accept=lambda a, b: a | b)

    def _vectorized_transition(self, state: int, symbols: jnp.ndarray) -> jnp.ndarray:
        """Vectorized transition lookup for a batch of symbols."""
        if symbols.size == 0:
            return jnp.array([], dtype=jnp.int32)
        
        default_next = self.default_states[state]
        result = jnp.full(symbols.shape, default_next, dtype=jnp.int32)
        
        # Process exceptions
        ex_syms = self.exception_symbols[state]
        ex_states = self.exception_states[state]
        valid_mask = ex_syms != -1
        ex_syms_valid = ex_syms[valid_mask]
        ex_states_valid = ex_states[valid_mask]
        
        for i in range(len(ex_syms_valid)):
            mask = symbols == ex_syms_valid[i]
            result = jnp.where(mask, ex_states_valid[i], result)
        
        return result

    def _product(self, other: 'SparseDFA', combine_accept: Callable[[bool, bool], bool]) -> 'SparseDFA':
        if self.symbol_arity != other.symbol_arity:
            raise ValueError("Product requires same symbol arity")
        
        start_pair = (self.start_state, other.start_state)
        queue = deque([start_pair])
        state_map = {start_pair: 0}
        
        new_default_states_list = []
        new_exception_symbols_list = []
        new_exception_states_list = []
        new_is_accepting_list = []
        
        while queue:
            current_pair = queue.popleft()
            i, j = current_pair
            current_idx = state_map[current_pair]
            
            new_is_accepting_list.append(combine_accept(
                bool(self.is_accepting[i]),
                bool(other.is_accepting[j]))
            )
            
            # Get relevant symbols
            ex_i_mask = self.exception_symbols[i] != -1
            ex_j_mask = other.exception_symbols[j] != -1
            ex_i_symbols = self.exception_symbols[i][ex_i_mask]
            ex_j_symbols = other.exception_symbols[j][ex_j_mask]
            all_relevant_symbols = jnp.unique(jnp.concatenate([ex_i_symbols, ex_j_symbols]))
            
            # Default transition
            def_i = int(self.default_states[i])
            def_j = int(other.default_states[j])
            default_target_pair = (def_i, def_j)
            
            if default_target_pair not in state_map:
                state_map[default_target_pair] = len(state_map)
                queue.append(default_target_pair)
            new_default_states_list.append(state_map[default_target_pair])
            
            # Process exceptions in batch
            current_exceptions_symbols = []
            current_exceptions_states = []
            
            if all_relevant_symbols.size > 0:
                # Vectorized transition lookups
                next_i_vec = self._vectorized_transition(i, all_relevant_symbols)
                next_j_vec = other._vectorized_transition(j, all_relevant_symbols)
                
                # Identify non-default transitions
                non_default_mask = (next_i_vec != def_i) | (next_j_vec != def_j)
                non_default_symbols = all_relevant_symbols[non_default_mask]
                non_default_i = next_i_vec[non_default_mask]
                non_default_j = next_j_vec[non_default_mask]
                
                # Process non-default transitions
                for idx in range(non_default_symbols.shape[0]):
                    sym = int(non_default_symbols[idx])
                    next_pair = (int(non_default_i[idx]), int(non_default_j[idx]))
                    
                    if next_pair not in state_map:
                        state_map[next_pair] = len(state_map)
                        queue.append(next_pair)
                    current_exceptions_symbols.append(sym)
                    current_exceptions_states.append(state_map[next_pair])
            
            new_exception_symbols_list.append(current_exceptions_symbols)
            new_exception_states_list.append(current_exceptions_states)
        
        # Convert results to JAX arrays
        num_new_states = len(state_map)
        max_exceptions = max(len(ex) for ex in new_exception_symbols_list) if new_exception_symbols_list else 0
        
        padded_ex_syms = jnp.full((num_new_states, max_exceptions), -1, dtype=jnp.int32)
        padded_ex_states = jnp.full((num_new_states, max_exceptions), -1, dtype=jnp.int32)
        
        for i, (syms, states) in enumerate(zip(new_exception_symbols_list, new_exception_states_list)):
            if syms:
                padded_ex_syms = padded_ex_syms.at[i, :len(syms)].set(jnp.array(syms, dtype=jnp.int32))
                padded_ex_states = padded_ex_states.at[i, :len(states)].set(jnp.array(states, dtype=jnp.int32))
        
        return SparseDFA(
            num_new_states,
            jnp.array(new_default_states_list, dtype=jnp.int32),
            padded_ex_syms,
            padded_ex_states,
            jnp.array(new_is_accepting_list, dtype=bool),
            0,
            self.symbol_arity,
            self.base_alphabet.union(other.base_alphabet)
        )

    def alphabet_projection(self, projection_map: jnp.ndarray) -> 'SparseNFA':
        new_ex_symbols = jnp.where(
            self.exception_symbols != -1,
            projection_map[self.exception_symbols],
            -1
        )
        return SparseNFA(
            num_states=self.num_states,
            base_state=self.default_states,
            exception_symbols=new_ex_symbols,
            exception_states=self.exception_states,
            is_accepting=self.is_accepting,
            start_state=self.start_state,
            symbol_arity=self.symbol_arity,
            base_alphabet=self.base_alphabet
        )
    
    def intersect_subtapes(self, other: 'SparseDFA', tapes: List[int]) -> 'SparseDFA':
        """
        Intersects two automata on specified tapes of the first automaton.
        
        Args:
            other: Second automaton with arity = len(tapes)
            tapes: List of tape indices from self to project to other
            
        Returns:
            SparseDFA recognizing {x in L(self) | (x[tapes]) in L(other)}
        """
        # TODO: Buggy (not all exception symbols induced by other are considered)
        # Validate inputs
        k = self.symbol_arity
        l = other.symbol_arity
        if len(tapes) != l:
            raise ValueError(f"Tapes length ({len(tapes)}) must match other.arity ({l})")
        if not all(0 <= t < k for t in tapes):
            raise ValueError("All tape indices must be in [0, self.arity-1]")
        if self.base_alphabet != other.base_alphabet:
            raise ValueError("Automata must have the same base alphabet")
        
        base_alphabet = self.base_alphabet
        n1 = self.num_states
        n2 = other.num_states
        num_states = n1 * n2
        
        # Initialize arrays
        default_states = np.zeros(num_states, dtype=np.int32)
        exception_symbols = [[] for _ in range(num_states)]
        exception_states = [[] for _ in range(num_states)]
        is_accepting = np.zeros(num_states, dtype=bool)
        
        # Helper to project a full symbol to subtapes
        def project_symbol(full_enc: int) -> int:
            """Project full symbol to specified tapes"""
            full_tuple = decode_symbol(full_enc, k, base_alphabet)
            proj_tuple = tuple(full_tuple[t] for t in tapes)
            return encode_symbol(proj_tuple, base_alphabet)
        
        # Build product automaton
        for i in range(n1):
            for j in range(n2):
                idx = i * n2 + j
                
                # Default transitions for both automata
                default_i = int(self.default_states[i])
                default_j = int(other.default_states[j])
                default_states[idx] = default_i * n2 + default_j
                
                # Set acceptance
                is_accepting[idx] = self.is_accepting[i] and other.is_accepting[j]
                
                # Collect symbols that would cause exceptions
                exception_symbols_set = set()
                
                # Add exception symbols from self
                for pos1 in range(self.max_exceptions):
                    s_enc = int(self.exception_symbols[i, pos1])
                    if s_enc == -1:
                        continue
                    exception_symbols_set.add(s_enc)
                
                # Add exception symbols from other (via projection)
                for pos2 in range(other.max_exceptions):
                    p_enc = int(other.exception_symbols[j, pos2])
                    if p_enc == -1:
                        continue
                    # Create a representative symbol by:
                    # 1. Decoding the projected symbol
                    # 2. Creating a full symbol with default values
                    # 3. Setting the specified tapes
                    base0 = sorted(base_alphabet)[0]  # default symbol
                    full_tuple = [base0] * k
                    proj_tuple = decode_symbol(p_enc, l, base_alphabet)
                    for idx, t in enumerate(tapes):
                        full_tuple[t] = proj_tuple[idx]
                    full_enc = encode_symbol(tuple(full_tuple), base_alphabet)
                    exception_symbols_set.add(full_enc)
                
                # Process exception symbols
                for full_enc in exception_symbols_set:
                    # Get transition in self
                    next_i = self.transition(i, full_enc)
                    
                    # Get projection and transition in other
                    proj_enc = project_symbol(full_enc)
                    next_j = other.transition(j, proj_enc)
                    
                    next_state = next_i * n2 + next_j
                    
                    # Only store if different from default
                    if next_state != default_states[idx]:
                        exception_symbols[idx].append(full_enc)
                        exception_states[idx].append(next_state)
        
        # Pad exceptions to uniform length
        max_exceptions = max(len(ex) for ex in exception_symbols) if exception_symbols else 0
        padded_ex_syms = np.full((num_states, max_exceptions), -1, dtype=np.int32)
        padded_ex_states = np.full((num_states, max_exceptions), -1, dtype=np.int32)
        
        for i, syms in enumerate(exception_symbols):
            if syms:
                padded_ex_syms[i, :len(syms)] = syms
                padded_ex_states[i, :len(syms)] = exception_states[i]
        
        # Start state
        start_state = self.start_state * n2 + other.start_state
        
        return SparseDFA(
            num_states=num_states,
            default_states=jnp.array(default_states),
            exception_symbols=jnp.array(padded_ex_syms),
            exception_states=jnp.array(padded_ex_states),
            is_accepting=jnp.array(is_accepting),
            start_state=start_state,
            symbol_arity=k,
            base_alphabet=base_alphabet
        )

    def regular_right_quotient(self, other: 'SparseDFA') -> 'SparseDFA':
        nA, nB = self.num_states, other.num_states
        
        # Initialize reachability matrix
        reachable = jnp.zeros((nA, nB), dtype=bool)
        for i in range(nA):
            for j in range(nB):
                reachable = reachable.at[i, j].set(
                    bool(self.is_accepting[i]) & bool(other.is_accepting[j])
                )
        
        # Backward propagation
        changed = True
        while changed:
            new_reachable = reachable.copy()
            changed = False
            
            for i in range(nA):
                for j in range(nB):
                    if reachable[i, j]:
                        continue
                    
                    # Get unique symbols from both states
                    ex_i = self.exception_symbols[i][self.exception_symbols[i] != -1]
                    ex_j = other.exception_symbols[j][other.exception_symbols[j] != -1]
                    all_symbols = jnp.unique(jnp.concatenate([ex_i, ex_j]))
                    
                    # Check if any symbol leads to a reachable state
                    for sym in all_symbols:
                        next_i = self.transition(i, sym)
                        next_j = other.transition(j, sym)
                        if reachable[next_i, next_j]:
                            new_reachable = new_reachable.at[i, j].set(True)
                            changed = True
                            break
            
            reachable = new_reachable
        
        # New acceptance: state i is accepting if (i, other.start_state) is reachable
        new_accept = jnp.array([
            reachable[i, other.start_state] for i in range(nA)
        ])
        
        return SparseDFA(
            nA,
            self.default_states,
            self.exception_symbols,
            self.exception_states,
            new_accept,
            self.start_state,
            self.symbol_arity,
            self.base_alphabet
        )
    
    def fill_defaults(self) -> 'SparseDFA':
        """Fills in default transitions for all states. If default state is currently -1, it will be set to the most common exception state."""
        # check for unused default states (default_state == -1)
        ex_states = jnp.where(
            self.exception_states == -1,
            self.num_states,
            self.exception_states
        )

        n_symbols = len(self.base_alphabet_frozen)**self.symbol_arity

        counts = jnp.apply_along_axis(
            jnp.bincount, axis=-1, arr=ex_states, length=self.num_states + 1
        )[:, :-1]

        if (self.default_states.min() < 0) or jnp.any(counts.sum(axis=-1) == n_symbols):

            change_mask = (self.default_states == -1) | (counts.sum(axis=-1) == n_symbols)
            self.default_states = jnp.where(
                change_mask, jnp.argmax(counts, axis=-1), self.default_states
            )

            # delete exceptions to new default states
            new_ex_symbols = jnp.full((self.num_states, self.max_exceptions), -1, dtype=jnp.int32)
            change_mask_expanded = jnp.expand_dims(change_mask, axis=1)
            change_mask_expanded = jnp.repeat(change_mask_expanded, self.max_exceptions, axis=-1)
            new_defaults_expanded = jnp.expand_dims(self.default_states, axis=1)
            new_defaults_expanded = jnp.repeat(new_defaults_expanded, self.max_exceptions, axis=-1)
            self.exception_symbols = jnp.where(
                (self.exception_states == new_defaults_expanded) & change_mask_expanded, new_ex_symbols, self.exception_symbols
            )
            self.exception_states = jnp.where(
                (self.exception_states == new_defaults_expanded) & change_mask_expanded, jnp.full_like(self.exception_states, -1), self.exception_states
            )
        
        return self
        

    def minimize(self) -> 'SparseDFA':
        """Minimizes the DFA using Hopcroft's algorithm with sparse optimizations"""
        self.fill_defaults()

        # Convert to numpy for processing
        default_states_np = np.array(self.default_states)
        exception_symbols_np = np.array(self.exception_symbols)
        exception_states_np = np.array(self.exception_states)
        is_accepting_np = np.array(self.is_accepting)

        n_symbols = len(self.base_alphabet_frozen)**self.symbol_arity
        
        # Step 1: Precompute all symbols used
        all_symbols = set(exception_symbols_np.flatten())
        all_symbols.discard(-1)
        all_symbols = sorted(all_symbols)
        
        # Step 2: Compute reachable states
        reachable = set()
        queue = deque([self.start_state])
        while queue:
            state = queue.popleft()
            if state in reachable:
                continue
            reachable.add(state)
            
            # Default transition exists?
            if (exception_symbols_np[state] != -1).sum() < n_symbols:
                # Add default state
                default_target = default_states_np[state]
                if default_target not in reachable:
                    queue.append(default_target)
                
            # Add exception states
            for i in range(self.max_exceptions):
                sym = exception_symbols_np[state, i]
                if sym == -1:
                    continue
                target = exception_states_np[state, i]
                if target not in reachable:
                    queue.append(target)
        
        # Step 3: Initial partition - accepting and non-accepting states
        accepting_states = set()
        non_accepting_states = set()
        for state in reachable:
            if is_accepting_np[state]:
                accepting_states.add(state)
            else:
                non_accepting_states.add(state)
                
        if not accepting_states:
            partitions = [non_accepting_states]
        elif not non_accepting_states:
            partitions = [accepting_states]
        else:
            partitions = [accepting_states, non_accepting_states]
        
        # Precompute state to partition index mapping
        state_to_part_idx = {}
        for part_idx, part in enumerate(partitions):
            for state in part:
                state_to_part_idx[state] = part_idx
        
        # Step 4: Partition refinement
        changed = True
        while changed:
            changed = False
            new_partitions = []
            # Precompute state to partition index for current partitions
            state_to_part_idx = {}
            for part_idx, part in enumerate(partitions):
                for state in part:
                    state_to_part_idx[state] = part_idx
            
            for part in partitions:
                # Split partition based on transitions
                split_dict = defaultdict(list)
                for state in part:
                    # Create key based on transitions
                    key = []
                    
                    # Add default transition partition
                    default_next = default_states_np[state]
                    if default_next in reachable:
                        key.append(state_to_part_idx[default_next])
                    
                    # Add exception transitions
                    for symbol in all_symbols:
                        next_state = self._get_transition(
                            state, symbol, 
                            default_states_np, 
                            exception_symbols_np, 
                            exception_states_np
                        )
                        key.append(state_to_part_idx[next_state])
                    
                    split_dict[tuple(key)].append(state)
                
                # Add new partitions from splits
                for subgroup in split_dict.values():
                    new_partitions.append(set(subgroup))
                    if len(subgroup) < len(part):
                        changed = True
            
            if changed:
                partitions = new_partitions
        
        # Create mapping from old states to new state indices
        state_to_part = {}
        for part_idx, part in enumerate(partitions):
            for state in part:
                state_to_part[state] = part_idx
        
        # Build new DFA
        num_new_states = len(partitions)
        new_accepting = np.zeros(num_new_states, dtype=bool)
        new_start = state_to_part[self.start_state]
        
        # Initialize new transition components
        new_defaults = np.zeros(num_new_states, dtype=np.int32)
        new_ex_syms = []
        new_ex_states = []
        
        # Precompute state to partition index for transition processing
        state_to_part_idx = {}
        for part_idx, part in enumerate(partitions):
            for state in part:
                state_to_part_idx[state] = part_idx
        
        # Build transitions for new states
        for part_idx, part in enumerate(partitions):
            # Use representative state for transition information
            rep_state = next(iter(part))
            
            # Default transition
            default_target = default_states_np[rep_state]
            if default_target in reachable:
                new_defaults[part_idx] = state_to_part_idx[default_target]
            else:
                new_defaults[part_idx] = -1
            
            # Collect exceptions
            exceptions = []
            for i in range(self.max_exceptions):
                sym = exception_symbols_np[rep_state, i]
                if sym == -1:
                    continue
                target = exception_states_np[rep_state, i]
                new_target = state_to_part_idx[target]
                # Only add exception if different from default
                if new_target != new_defaults[part_idx]:
                    exceptions.append((sym, new_target))
            
            # Store exceptions
            new_ex_syms.append([sym for sym, _ in exceptions])
            new_ex_states.append([state for _, state in exceptions])
            
            # Set acceptance
            new_accepting[part_idx] = is_accepting_np[rep_state]
        
        # Pad exception arrays
        max_exceptions = max(len(ex) for ex in new_ex_syms) if new_ex_syms else 0
        padded_ex_syms = np.full((num_new_states, max_exceptions), -1)
        padded_ex_states = np.full((num_new_states, max_exceptions), -1)
        
        for i in range(num_new_states):
            syms = new_ex_syms[i] + [-1] * (max_exceptions - len(new_ex_syms[i]))
            states = new_ex_states[i] + [-1] * (max_exceptions - len(new_ex_states[i]))
            padded_ex_syms[i] = syms
            padded_ex_states[i] = states
        
        # Convert back to JAX arrays
        new_defaults_jax = jnp.array(new_defaults)
        padded_ex_syms_jax = jnp.array(padded_ex_syms)
        padded_ex_states_jax = jnp.array(padded_ex_states)
        new_accepting_jax = jnp.array(new_accepting)
        
        return SparseDFA(
            num_new_states,
            new_defaults_jax,
            padded_ex_syms_jax,
            padded_ex_states_jax,
            new_accepting_jax,
            new_start,
            self.symbol_arity,
            self.base_alphabet
        )
    
    def sparsify(self) -> 'SparseDFA':
        """Computes the sparsest possible equivalent DFA by choosing optimal default states.
        

        Returns:
            SparseDFA with optimized default states and exceptions. A default state is changed if the most common exception state is more frequent than the default state.
        """
        ex_states = jnp.where(
            self.exception_states == -1,
            self.num_states,
            self.exception_states
        )
        
        counts = jnp.apply_along_axis(
            jnp.bincount, axis=-1, arr=ex_states, length=self.num_states + 1
        )

        counts = counts[:, :-1]  # Ignore padding (num_states)
        n_symbols = len(self.base_alphabet_frozen)
        change_mask = counts.max(axis=-1) > (n_symbols ** self.symbol_arity) - counts.sum(axis=-1)
        
        # get new default states
        maxfreq_states = jnp.argmax(counts, axis=-1)
        new_defaults = jnp.where(change_mask, maxfreq_states, self.default_states)
        

        # delete exceptions to new default states
        new_ex_symbols = jnp.full((self.num_states, self.max_exceptions), -1, dtype=jnp.int32)
        change_mask_expanded = jnp.expand_dims(change_mask, axis=1)
        change_mask_expanded = jnp.repeat(change_mask_expanded, self.max_exceptions, axis=-1)
        new_defaults_expanded = jnp.expand_dims(new_defaults, axis=1)
        new_defaults_expanded = jnp.repeat(new_defaults_expanded, self.max_exceptions, axis=-1)
        new_ex_symbols = jnp.where(
            (self.exception_states == new_defaults_expanded) & change_mask_expanded, new_ex_symbols, self.exception_symbols
        )
        new_ex_states = jnp.full((self.num_states, self.max_exceptions), -1, dtype=jnp.int32)
        new_ex_states = jnp.where(
            (self.exception_states == new_defaults_expanded) & change_mask_expanded, new_ex_states, self.exception_states
        )

        # For each state that changed, fill in the exceptions with the old default trasitions
        for state in jnp.arange(self.num_states)[change_mask & (self.default_states != -1)]:
            old_def_ex_syms = complement(
                self.exception_symbols[state][self.exception_symbols[state] != -1], 0, n_symbols**self.symbol_arity - 1
            )

            free_places = jnp.arange(new_ex_symbols.shape[1])[new_ex_symbols[state] == -1][:len(old_def_ex_syms)]
            new_ex_symbols = new_ex_symbols.at[state, free_places].set(old_def_ex_syms)
            new_ex_states = new_ex_states.at[state, free_places].set(self.default_states[state])
        
        # Reorder exception symbols and states to minimize max_exceptions
        # Find the minimum number of -1 entries in any row
        num_minus_ones = jnp.sum(new_ex_symbols == -1, axis=1)
        min_minus_ones_per_row = jnp.min(num_minus_ones)

        if min_minus_ones_per_row > 0:
            # Create new padded arrays with reduced max_exceptions
            new_max_exceptions = self.max_exceptions - min_minus_ones_per_row
            reordered_ex_symbols = jnp.full((self.num_states, new_max_exceptions), -1, dtype=jnp.int32)
            reordered_ex_states = jnp.full((self.num_states, new_max_exceptions), -1, dtype=jnp.int32)

            for i in range(self.num_states):
                # Get non -1 entries
                valid_symbols = new_ex_symbols[i][new_ex_symbols[i] != -1]
                valid_states = new_ex_states[i][new_ex_states[i] != -1]
                
                # Fill into the reordered arrays
                reordered_ex_symbols = reordered_ex_symbols.at[i, :len(valid_symbols)].set(valid_symbols)
                reordered_ex_states = reordered_ex_states.at[i, :len(valid_states)].set(valid_states)
            
            new_ex_symbols = reordered_ex_symbols
            new_ex_states = reordered_ex_states

        return SparseDFA(
            self.num_states,
            new_defaults,
            new_ex_symbols,
            new_ex_states,
            self.is_accepting,
            self.start_state,
            self.symbol_arity,
            self.base_alphabet
        )
    
    def _get_transition(self, state, symbol, default_states, exception_symbols, exception_states):
        """Helper to get transition using numpy arrays"""
        for i in range(self.max_exceptions):
            if exception_symbols[state, i] == symbol:
                return exception_states[state, i]
        return default_states[state]
    
    def __str__(self) -> str:
        lines = []
        lines.append(f"SparseDFA with {self.num_states} states (arity={self.symbol_arity})")
        lines.append(f"Start state: {self.start_state}")
        
        # List accepting states
        accepting_states = [i for i in range(self.num_states) if self.is_accepting[i]]
        lines.append(f"Accepting states: {accepting_states}")
        
        # Add transitions header
        lines.append("\nTransitions:")
        lines.append("State | Default | Exceptions")
        lines.append("------|---------|-----------")
        
        # Process each state
        for state in range(self.num_states):
            default = self.default_states[state]
            
            # Collect exception transitions
            exceptions = []
            for i in range(self.max_exceptions):
                sym = self.exception_symbols[state, i]
                target = self.exception_states[state, i]
                if sym != -1 and target != -1:
                    sym = decode_symbol(sym, self.symbol_arity, self.base_alphabet_frozen)
                    exceptions.append(f"{sym}→{target}")
            
            # Format exceptions or show none
            exceptions_str = ", ".join(exceptions) if exceptions else "None"
            
            # Format state row
            state_str = f"{state}{'*' if self.is_accepting[state] else ''}"
            lines.append(f"{state_str:<5} | {default:<7} | {exceptions_str}")
        
        return "\n".join(lines)
    
    def show_diagram(self, filename: str = "automaton", format: str = "png", view: bool = True) -> graphviz.Digraph:
        """
        Visualize the automaton using Graphviz, showing both default and exception transitions.
        This version ensures all transitions are properly displayed.
        """
        dot = graphviz.Digraph(engine='dot')
        dot.attr(rankdir='LR')
        
        # Add nodes
        for state in range(self.num_states):
            if self.is_accepting[state]:
                dot.node(str(state), shape='doublecircle')
            else:
                dot.node(str(state), shape='circle')
        
        # Add start arrow
        dot.node('__start__', '', shape='none', width='0', height='0')
        dot.edge('__start__', str(self.start_state))
        
        # Collect all transitions
        all_transitions = {}
        
        # First, add exception transitions
        for state in range(self.num_states):
            for i in range(self.max_exceptions):
                symbol = int(self.exception_symbols[state, i])
                if symbol == -1:  # Skip padding
                    continue
                target = int(self.exception_states[state, i])
                key = (state, target)
                
                # Decode symbol
                symbol_tuple = decode_symbol(symbol, self.symbol_arity, self.base_alphabet)
                symbol_str = str(symbol_tuple) if self.symbol_arity > 1 else str(symbol_tuple[0])
                
                if key not in all_transitions:
                    all_transitions[key] = []
                all_transitions[key].append(symbol_str)
        
        # Then add default transitions
        for state in range(self.num_states):
            default_target = int(self.default_states[state])
            key = (state, default_target)
            
            # Only add default if not already covered by exceptions
            if key not in all_transitions:
                all_transitions[key] = []
            
            # Add "default" label to the list
            all_transitions[key].append("default")
        
        # Create edges with all labels
        for (from_state, to_state), symbols in all_transitions.items():
            # Combine all symbols for this edge
            label = ", ".join(sorted(set(symbols)))
            dot.edge(str(from_state), str(to_state), label=label)
        
        # Render and view
        dot.render(filename=filename, format=format, view=view)
        return dot
    

class SparseNFA:
    def __init__(self, num_states: int, base_state: jnp.ndarray,
                 exception_symbols: jnp.ndarray, exception_states: jnp.ndarray,
                 is_accepting: jnp.ndarray, start_state: int,
                 symbol_arity: int = 1, base_alphabet: Optional[Set[int]] = None):
        self.num_states = num_states
        self.base_state = base_state
        self.exception_symbols = exception_symbols
        self.exception_states = exception_states
        self.is_accepting = is_accepting
        self.start_state = start_state
        self.max_exceptions = exception_symbols.shape[1]
        self.symbol_arity = symbol_arity
        self.base_alphabet = base_alphabet if base_alphabet is not None else self._infer_base_alphabet()
        
    def _infer_base_alphabet(self) -> Set[int]:
        """Infer base alphabet from exception symbols"""
        symbols = set(np.unique(self.exception_symbols))
        symbols.discard(-1)
        return symbols
    
    def _step(self, current_set: jnp.ndarray, symbol: int) -> Tuple[jnp.ndarray, None]:
        """JAX-compatible single step transition"""
        # Base transitions
        base_next = jnp.zeros(self.num_states, dtype=bool)
        base_next = base_next.at[self.base_state].set(
            base_next[self.base_state] | current_set
        )
        
        # Exception transitions
        exception_next = jnp.zeros(self.num_states, dtype=bool)
        symbol_mask = (self.exception_symbols == symbol) & (self.exception_symbols != -1)
        
        # Process each exception slot
        for i in range(self.max_exceptions):
            # Get mask for this slot
            slot_mask = symbol_mask[:, i]
            slot_targets = self.exception_states[:, i]
            
            # Find states that are active and have matching exception
            valid_transitions = current_set & slot_mask
            
            # Update exception_next for this slot
            exception_next = exception_next.at[slot_targets].set(
                exception_next[slot_targets] | valid_transitions
            )
        
        # Combine base and exception transitions
        next_set = base_next | exception_next
        return next_set, None

    def compute(self, word: jnp.ndarray) -> jnp.ndarray:
        """Returns the set of states after processing the word"""
        init = jnp.zeros(self.num_states, dtype=bool)
        init = init.at[self.start_state].set(True)
        final_set, _ = jax.lax.scan(self._step, init, word)
        return final_set

    def accepts(self, word: jnp.ndarray) -> jnp.ndarray:
        """Checks if the NFA accepts the word"""
        final_set = self.compute(word)
        return jnp.any(jnp.logical_and(final_set, self.is_accepting))
    
    def determinize(self) -> SparseDFA:
        """Converts NFA to DFA using subset construction"""
        # Start state is epsilon closure (but no epsilons)
        start_set = frozenset([self.start_state])
        state_queue = deque([start_set])
        state_to_id = {start_set: 0}
        id_to_state = [start_set]
        
        # DFA components
        dfa_defaults = []
        dfa_ex_syms = []  # List of exception symbols per state
        dfa_ex_states = []  # List of exception states per state
        dfa_accepting = [any(self.is_accepting[q] for q in start_set)]
        
        while state_queue:
            current = state_queue.popleft()
            current_id = state_to_id[current]
            
            # Compute base states and exception map
            base_states = set()
            exception_map = {}
            has_exception = {}
            
            for q in current:
                base_states.add(int(self.base_state[q]))

                for i in range(self.max_exceptions):
                    sym = self.exception_symbols[q, i]
                    if sym == -1:
                        continue
                    target = self.exception_states[q, i]
                    if sym not in exception_map:
                        exception_map[sym] = set()
                        has_exception[sym] = set()

                    exception_map[sym].add(int(target))
                    has_exception[sym].add(int(q))
            
            # Default transition: T0 = base_states
            T0 = frozenset(base_states)
            if T0 not in state_to_id:
                new_id = len(id_to_state)
                state_to_id[T0] = new_id
                id_to_state.append(T0)
                dfa_accepting.append(any(self.is_accepting[q] for q in T0))
                state_queue.append(T0)
                default_target = new_id
            else:
                default_target = state_to_id[T0]
            dfa_defaults.append(default_target)
            
            # Process exception symbols
            ex_symbols = []
            ex_states = []
            for sym, targets in exception_map.items():
                # Next state = base_states ∪ exception targets
                targets_from_base = set([self.base_state[q] for q in current if q not in has_exception[sym]])
                T_sym = frozenset(targets_from_base | targets)
                if T_sym not in state_to_id:
                    new_id = len(id_to_state)
                    state_to_id[T_sym] = new_id
                    id_to_state.append(T_sym)
                    dfa_accepting.append(any(self.is_accepting[q] for q in T_sym))
                    state_queue.append(T_sym)
                    ex_states.append(new_id)
                else:
                    ex_states.append(state_to_id[T_sym])
                ex_symbols.append(sym)
            
            dfa_ex_syms.append(ex_symbols)
            dfa_ex_states.append(ex_states)
        
        # Pad exception arrays
        max_ex = max(len(syms) for syms in dfa_ex_syms) if dfa_ex_syms else 0
        padded_ex_syms = jnp.full((len(id_to_state), max_ex), -1)
        padded_ex_states = jnp.full((len(id_to_state), max_ex), -1)
        
        for i in range(len(id_to_state)):
            if i < len(dfa_ex_syms):
                syms = dfa_ex_syms[i] + [-1] * (max_ex - len(dfa_ex_syms[i]))
                states = dfa_ex_states[i] + [-1] * (max_ex - len(dfa_ex_states[i]))
                padded_ex_syms = padded_ex_syms.at[i].set(jnp.array(syms))
                padded_ex_states = padded_ex_states.at[i].set(jnp.array(states))
        
        return SparseDFA(
            num_states=len(id_to_state),
            default_states=jnp.array(dfa_defaults, dtype=jnp.int32),
            exception_symbols=padded_ex_syms,
            exception_states=padded_ex_states,
            is_accepting=jnp.array(dfa_accepting, dtype=bool),
            start_state=0,
            symbol_arity=self.symbol_arity,
            base_alphabet=self.base_alphabet
        )
    
    def __str__(self) -> str:
        """Returns a string representation of the NFA"""
        lines = []
        lines.append(f"SparseNFA with {self.num_states} states (arity={self.symbol_arity})")
        lines.append(f"Start state: {self.start_state}")
        
        # List accepting states
        accepting_states = [i for i in range(self.num_states) if self.is_accepting[i]]
        lines.append(f"Accepting states: {accepting_states}")
        
        # Add transitions header
        lines.append("\nTransitions:")
        lines.append("State | Base | Exceptions")
        lines.append("------|------|-----------")
        
        # Helper to decode symbols
        def symbol_to_str(sym: int) -> str:
            if sym == -1:
                return "ε"
            tup = decode_symbol(sym, self.symbol_arity, self.base_alphabet)
            if self.symbol_arity == 1:
                return str(tup[0])
            return str(tup)
        
        # Process each state
        for state in range(self.num_states):
            base = self.base_state[state]
            
            # Collect exception transitions
            exceptions = []
            for i in range(self.max_exceptions):
                sym = int(self.exception_symbols[state, i])
                if sym == -1:
                    continue
                target = int(self.exception_states[state, i])
                symbol_str = symbol_to_str(sym)
                exceptions.append(f"{symbol_str}→{target}")
            
            # Format exceptions or show none
            exceptions_str = ", ".join(exceptions) if exceptions else "None"
            
            # Format state row
            state_str = f"{state}{'*' if self.is_accepting[state] else ''}"
            lines.append(f"{state_str:<5} | {base:<4} | {exceptions_str}")
        
        return "\n".join(lines)

    def show_diagram(self, filename: str = "nfa", format: str = "png", view: bool = True) -> graphviz.Digraph:
        """Visualizes the NFA using Graphviz"""
        try:
            import graphviz
        except ImportError:
            raise ImportError("Graphviz is required for visualization. Install with 'pip install graphviz'")
        
        dot = graphviz.Digraph(engine='dot')
        dot.attr(rankdir='LR')
        
        # Helper to decode symbols
        def symbol_to_str(sym: int) -> str:
            if sym == -1:
                return "ε"
            tup = decode_symbol(sym, self.symbol_arity, self.base_alphabet)
            if self.symbol_arity == 1:
                return str(tup[0])
            return str(tup)
        
        # Add nodes
        for state in range(self.num_states):
            if self.is_accepting[state]:
                dot.node(str(state), shape='doublecircle')
            else:
                dot.node(str(state), shape='circle')
        
        # Add start arrow
        dot.node('__start__', '', shape='none', width='0', height='0')
        dot.edge('__start__', str(self.start_state))
        
        # Collect transitions
        base_transitions = {}
        exception_transitions = defaultdict(lambda: defaultdict(set))
        
        # Base transitions
        for state in range(self.num_states):
            base_target = int(self.base_state[state])
            base_transitions[(state, base_target)] = "default"
        
        # Exception transitions
        for state in range(self.num_states):
            for i in range(self.max_exceptions):
                sym = int(self.exception_symbols[state, i])
                if sym == -1:
                    continue
                target = int(self.exception_states[state, i])
                symbol_str = symbol_to_str(sym)
                exception_transitions[(state, target)][symbol_str] = True
        
        # Add base transitions to graph
        for (from_state, to_state), label in base_transitions.items():
            dot.edge(str(from_state), str(to_state), label=label, style='dashed', color='blue')
        
        # Add exception transitions to graph
        for (from_state, to_state), symbols in exception_transitions.items():
            label = ", ".join(sorted(symbols.keys()))
            dot.edge(str(from_state), str(to_state), label=label)
        
        # Render and view
        dot.render(filename=filename, format=format, view=view)
        return dot