import json
import numpy as np
from collections import deque, defaultdict, Counter

# JAX is optional: it is only used to accelerate batched word processing on
# fixed automata (accepts_batch), where jit/scan/GPU pay off. All construction
# algorithms run on numpy.
try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True

    @jax.jit
    def _jax_run_batch(defaults, ex_symbols, ex_states, start, words):
        """Run a (B, L) batch of encoded words; returns the (B,) final states."""
        def step(states, symbols):
            eq = ex_symbols[states] == symbols[:, None]
            hit = eq.any(axis=1)
            first = jnp.argmax(eq, axis=1)
            targets = jnp.take_along_axis(ex_states[states], first[:, None], axis=1)[:, 0]
            return jnp.where(hit, targets, defaults[states]), None

        init = jnp.full(words.shape[0], start, dtype=jnp.int32)
        final, _ = jax.lax.scan(step, init, words.T)
        return final
except ImportError:
    _HAS_JAX = False
from typing import Tuple, Optional, Callable, List, Set
import graphviz

import struct
import zlib


from autstr.utils.misc import decode_symbol
from autstr.utils.misc import encode_symbol, complement



# File format structure:
# [Header (16 bytes)]
#   - Magic number: 4 bytes ('SDFA')
#   - Version: 1 byte
#   - Reserved: 3 bytes (0)
#   - Checksum: 4 bytes (CRC32 of payload)
#   - Payload size: 4 bytes
# [Payload]
#   - Metadata (20 bytes)
#   - Base alphabet
#   - Default states
#   - Exception symbols
#   - Exception states
#   - Acceptance array

class SparseDFASerializer:
    VERSION = 2  # Bump version for new format
    HEADER_FORMAT = "4sB3sII"
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
    METADATA_FORMAT = "IIIII"
    METADATA_SIZE = struct.calcsize(METADATA_FORMAT)
    
    @classmethod
    def serialize(cls, dfa: 'SparseDFA', filename: str) -> None:
        """Serialize SparseDFA to binary file"""
        # Prepare payload components
        payload = cls._create_payload(dfa)
        
        # Create header
        checksum = zlib.crc32(payload)
        header = struct.pack(
            cls.HEADER_FORMAT,
            b'SDFA',           # Magic number
            cls.VERSION,       # Format version
            b'\0\0\0',         # Reserved bytes
            checksum,          # CRC32 checksum
            len(payload)       # Payload size
        )
        
        # Write to file
        with open(filename, 'wb') as f:
            f.write(header)
            f.write(payload)
    
    @classmethod
    def deserialize(cls, filename: str) -> 'SparseDFA':
        """Deserialize SparseDFA from binary file"""
        with open(filename, 'rb') as f:
            # Read and validate header
            header = f.read(cls.HEADER_SIZE)
            magic, version, _, checksum, payload_size = struct.unpack(cls.HEADER_FORMAT, header)
            
            if magic != b'SDFA':
                raise ValueError("Invalid file format (bad magic number)")
            if version != cls.VERSION:
                raise ValueError(f"Unsupported version: {version}")
            
            # Read and validate payload
            payload = f.read(payload_size)
            if zlib.crc32(payload) != checksum:
                raise ValueError("Data corruption detected (checksum mismatch)")
            
            return cls._parse_payload(payload)
    
    @classmethod
    def _create_payload(cls, dfa: 'SparseDFA') -> bytes:
        """Create binary payload from SparseDFA"""
        # Convert arrays to numpy for efficient serialization
        default_states = np.array(dfa.default_states, dtype=np.uint32)
        exception_symbols = np.array(dfa.exception_symbols, dtype=np.int32)
        exception_states = np.array(dfa.exception_states, dtype=np.int32)
        is_accepting = np.array(dfa.is_accepting, dtype=np.uint8)
        
        # Serialize base alphabet as JSON
        base_alphabet_json = json.dumps(sorted(dfa.base_alphabet)).encode('utf-8')
        base_alphabet_len = len(base_alphabet_json)
        
        # Pack metadata
        metadata = struct.pack(
            cls.METADATA_FORMAT,
            dfa.num_states,
            dfa.max_exceptions,
            dfa.start_state,
            dfa.symbol_arity,
            base_alphabet_len
        )
        
        # Pack components
        components = [
            metadata,
            base_alphabet_json,
            default_states.tobytes(),
            exception_symbols.tobytes(),
            exception_states.tobytes(),
            is_accepting.tobytes()
        ]
        
        return b''.join(components)
    
    @classmethod
    def _parse_payload(cls, payload: bytes) -> 'SparseDFA':
        """Parse binary payload into SparseDFA"""
        # Unpack metadata
        meta = struct.unpack(
            cls.METADATA_FORMAT,
            payload[:cls.METADATA_SIZE]
        )
        num_states, max_exceptions, start_state, symbol_arity, alpha_len = meta
        
        # Calculate offsets
        offset = cls.METADATA_SIZE
        base_alphabet_json = payload[offset:offset+alpha_len]
        base_alphabet = set(json.loads(base_alphabet_json.decode('utf-8')))
        offset += alpha_len
        
        # Calculate array sizes
        states_bytes = num_states * 4
        exceptions_bytes = num_states * max_exceptions * 4
        accepting_bytes = num_states
        
        # Extract arrays (frombuffer views are read-only, so copy via astype)
        default_states = np.frombuffer(
            payload[offset:offset+states_bytes],
            dtype=np.uint32
        ).astype(np.int32)
        offset += states_bytes

        exception_symbols = np.frombuffer(
            payload[offset:offset+exceptions_bytes],
            dtype=np.int32
        ).reshape(num_states, max_exceptions).copy()
        offset += exceptions_bytes

        exception_states = np.frombuffer(
            payload[offset:offset+exceptions_bytes],
            dtype=np.int32
        ).reshape(num_states, max_exceptions).copy()
        offset += exceptions_bytes

        is_accepting = np.frombuffer(
            payload[offset:offset+accepting_bytes],
            dtype=np.uint8
        ).astype(bool)
        
        return SparseDFA(
            num_states=num_states,
            default_states=default_states,
            exception_symbols=exception_symbols,
            exception_states=exception_states,
            is_accepting=is_accepting,
            start_state=start_state,
            symbol_arity=symbol_arity,
            base_alphabet=base_alphabet
        )
    
    @classmethod
    def to_bytes(cls, dfa: 'SparseDFA') -> bytes:
        """Serialize SparseDFA to bytes object"""
        payload = cls._create_payload(dfa)
        header = struct.pack(
            cls.HEADER_FORMAT,
            b'SDFA',
            cls.VERSION,
            b'\0\0\0',
            zlib.crc32(payload),
            len(payload)
        )
        return header + payload
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'SparseDFA':
        """Deserialize SparseDFA from bytes object"""
        if len(data) < cls.HEADER_SIZE:
            raise ValueError("Data too short for header")
        
        header = data[:cls.HEADER_SIZE]
        magic, version, _, checksum, payload_size = struct.unpack(cls.HEADER_FORMAT, header)
        
        if magic != b'SDFA':
            raise ValueError("Invalid SparseDFA format")
        if version != cls.VERSION:
            raise ValueError(f"Unsupported SparseDFA version: {version}")
        
        payload = data[cls.HEADER_SIZE:cls.HEADER_SIZE+payload_size]
        if len(payload) != payload_size:
            raise ValueError("Payload size mismatch")
        if zlib.crc32(payload) != checksum:
            raise ValueError("SparseDFA data corruption detected")
        
        return cls._parse_payload(payload)


def _sort_exception_rows(ex_symbols: np.ndarray, ex_states: np.ndarray):
    """Sort each state's exception row by symbol (with -1 padding first),
    keeping targets aligned. Precondition for _sorted_row_lookup."""
    order = np.argsort(ex_symbols, axis=1, kind='stable')
    return (np.take_along_axis(ex_symbols, order, axis=1),
            np.take_along_axis(ex_states, order, axis=1))


def _sorted_row_lookup(sorted_syms: np.ndarray, sorted_targets: np.ndarray,
                       defaults: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Batched per-row transition lookup via binary search.

    For each row f and candidate symbol candidates[f, j], returns the
    matching exception target or defaults[f]. Rows must be sorted by symbol
    (see _sort_exception_rows). Memory and time are linear in the number of
    rows/candidates — never materializes a (rows, exceptions, candidates)
    tensor. Entries for -1 (padding) candidates are meaningless.
    """
    num_rows, max_ex = sorted_syms.shape
    num_cands = candidates.shape[1]
    if max_ex == 0 or num_cands == 0:
        return np.broadcast_to(defaults[:, None], (num_rows, num_cands)).copy()

    # Make rows disjoint key ranges so one flat searchsorted resolves all rows
    offset = np.int64(max(int(sorted_syms.max(initial=0)), int(candidates.max(initial=0))) + 2)
    row_ids = np.arange(num_rows, dtype=np.int64)[:, None]
    keys = (row_ids * offset + sorted_syms).ravel()
    queries = (row_ids * offset + candidates).ravel()

    pos = np.searchsorted(keys, queries)
    pos_clipped = np.minimum(pos, keys.size - 1)
    hit = keys[pos_clipped] == queries
    result = np.where(hit, sorted_targets.ravel()[pos_clipped],
                      np.repeat(defaults, num_cands))
    return result.reshape(num_rows, num_cands)


class SparseDFA:
    def __init__(self, num_states: int, default_states: np.ndarray,
                 exception_symbols: np.ndarray, exception_states: np.ndarray,
                 is_accepting: np.ndarray, start_state: int,
                 symbol_arity: int = 1, base_alphabet: Optional[Set[int]] = None):
        self.num_states = num_states
        self.default_states = np.asarray(default_states, dtype=np.int32)
        self.exception_symbols = np.asarray(exception_symbols, dtype=np.int32)
        self.exception_states = np.asarray(exception_states, dtype=np.int32)
        self.is_accepting = np.asarray(is_accepting, dtype=bool)
        self.start_state = int(start_state)
        self.max_exceptions = self.exception_symbols.shape[1]
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
        matches = np.flatnonzero(self.exception_symbols[state] == symbol)
        if matches.size > 0:
            return int(self.exception_states[state, matches[0]])
        return int(self.default_states[state])

    def compute(self, word: np.ndarray) -> int:
        """Final state after reading the word (encoded symbols).

        A word is a sequential dependency chain, so this cannot be vectorized;
        instead the loop avoids per-element numpy dispatch entirely: exceptions
        are flattened once into a sorted list of packed (state, symbol) keys
        and each step is plain integer arithmetic plus one C bisect (~10x
        faster than per-symbol numpy operations)."""
        from bisect import bisect_left
        num_symbols = len(self.base_alphabet_frozen) ** self.symbol_arity
        rows, cols = np.nonzero(self.exception_symbols >= 0)
        key_arr = rows.astype(np.int64) * num_symbols + \
            self.exception_symbols[rows, cols]
        order = np.argsort(key_arr, kind="stable")
        keys = key_arr[order].tolist()
        targets = self.exception_states[rows[order], cols[order]].tolist()
        n_keys = len(keys)
        defaults = self.default_states.tolist()

        state = self.start_state
        for symbol in np.asarray(word, dtype=np.int64).tolist():
            key = state * num_symbols + symbol
            pos = bisect_left(keys, key)
            state = targets[pos] if pos < n_keys and keys[pos] == key \
                else defaults[state]
        return state

    def accepts(self, word) -> bool:
        # encode word for internal representation
        word = np.array([encode_symbol(s, self.base_alphabet_frozen) for s in word], dtype=np.int64)
        final_state = self.compute(word)
        return bool(self.is_accepting[final_state])

    def _encode_words(self, words) -> np.ndarray:
        """Normalize a batch of equal-length words to an encoded (B, L) array."""
        if isinstance(words, np.ndarray) and np.issubdtype(words.dtype, np.integer):
            return words
        return np.array([
            [encode_symbol(s, self.base_alphabet_frozen) for s in word]
            for word in words
        ], dtype=np.int64)

    def accepts_batch(self, words) -> np.ndarray:
        """Batched acceptance check for many equal-length words at once.

        :param words: either an already-encoded integer array of shape
            (batch, length), or a sequence of equal-length words of symbol
            tuples (encoded like accepts()).
        :returns: boolean array of shape (batch,).

        Uses JAX (jit + scan, GPU if available) when installed, otherwise a
        vectorized numpy fallback.
        """
        words = self._encode_words(words)
        if words.ndim != 2:
            raise ValueError("accepts_batch expects a (batch, length) input")
        if words.shape[1] == 0:
            return np.full(words.shape[0], self.is_accepting[self.start_state])

        if _HAS_JAX and self.max_exceptions > 0:
            final = _jax_run_batch(
                jnp.asarray(self.default_states),
                jnp.asarray(self.exception_symbols),
                jnp.asarray(self.exception_states),
                self.start_state,
                jnp.asarray(words, dtype=jnp.int32)
            )
            return np.asarray(self.is_accepting[np.asarray(final)])

        states = np.full(words.shape[0], self.start_state, dtype=np.int64)
        for t in range(words.shape[1]):
            if self.max_exceptions > 0:
                eq = self.exception_symbols[states] == words[:, t, None]
                hit = eq.any(axis=1)
                first = eq.argmax(axis=1)
                targets = self.exception_states[states, first]
                states = np.where(hit, targets, self.default_states[states])
            else:
                states = self.default_states[states].astype(np.int64)
        return self.is_accepting[states]
    
    def is_empty(self) -> bool:
        """Check if the language is empty"""
        # Frontier-batched BFS to find any accepting state. The default
        # transition only exists if the exceptions don't cover the alphabet.
        defaults = np.asarray(self.default_states, dtype=np.int64)
        ex_symbols = np.asarray(self.exception_symbols, dtype=np.int64)
        ex_states = np.asarray(self.exception_states, dtype=np.int64)
        accepting = np.asarray(self.is_accepting, dtype=bool)
        n_symbols = len(self.base_alphabet_frozen)**self.symbol_arity

        visited = np.zeros(self.num_states, dtype=bool)
        frontier = np.array([int(self.start_state)], dtype=np.int64)
        visited[frontier] = True
        while frontier.size > 0:
            if accepting[frontier].any():
                return False
            valid = ex_symbols[frontier] != -1
            has_default = valid.sum(axis=1) < n_symbols
            successors = np.unique(np.concatenate([
                ex_states[frontier][valid],
                defaults[frontier][has_default]
            ]))
            successors = successors[~visited[successors]]
            visited[successors] = True
            frontier = successors

        return True
    
    def is_finite(self) -> bool:
        # Step 1: Find reachable states (forward BFS)
        reachable = set()
        queue = deque([self.start_state])
        while queue:
            state = queue.popleft()
            if state in reachable:
                continue
            reachable.add(state)
            # Default transition
            default_target = int(self.default_states[state])
            if default_target not in reachable:
                queue.append(default_target)
            # Exception transitions
            for i in range(self.max_exceptions):
                sym = int(self.exception_symbols[state, i])
                if sym == -1:
                    continue
                target = int(self.exception_states[state, i])
                if target not in reachable:
                    queue.append(target)
        
        # Step 2: Find co-reachable states (backward BFS from accepting states)
        # Precompute reverse transition map
        rev_map = defaultdict(set)
        for u in range(self.num_states):
            # Default transitions
            v_def = int(self.default_states[u])
            rev_map[v_def].add(u)
            # Exception transitions
            for i in range(self.max_exceptions):
                sym = int(self.exception_symbols[u, i])
                if sym != -1:
                    v_ex = int(self.exception_states[u, i])
                    rev_map[v_ex].add(u)
        
        co_reachable = set()
        # Initialize queue with all accepting states
        queue = deque([state for state in range(self.num_states) if self.is_accepting[state]])
        while queue:
            state = queue.popleft()
            if state in co_reachable:
                continue
            co_reachable.add(state)
            for pred in rev_map[state]:
                if pred not in co_reachable:
                    queue.append(pred)
        
        # Step 3: Useful states (reachable and co-reachable)
        useful_states = reachable & co_reachable
        
        # If no useful states, language is empty -> finite
        if not useful_states:
            return True
        
        # Step 4: Build graph for useful states
        graph = {}
        for u in useful_states:
            neighbors = set()
            # Default transition
            v_def = int(self.default_states[u])
            if v_def in useful_states:
                neighbors.add(v_def)
            # Exception transitions
            for i in range(self.max_exceptions):
                sym = int(self.exception_symbols[u, i])
                if sym != -1:
                    v_ex = int(self.exception_states[u, i])
                    if v_ex in useful_states:
                        neighbors.add(v_ex)
            graph[u] = neighbors
        
        # Step 5: Cycle detection with iterative DFS
        color = {state: 0 for state in useful_states}  # 0: white, 1: gray, 2: black
        for state in useful_states:
            if color[state] == 0:
                stack = [state]
                while stack:
                    u = stack.pop()
                    if color[u] == 0:
                        color[u] = 1  # Mark as gray
                        stack.append(u)  # Push back for backtracking
                        for v in graph[u]:
                            if color[v] == 0:
                                stack.append(v)
                            elif color[v] == 1:
                                return False  # Cycle found -> infinite language
                    else:
                        color[u] = 2  # Mark as black
        return True  # No cycles -> finite language

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

    def _vectorized_transition(self, state: int, symbols: np.ndarray) -> np.ndarray:
        """Vectorized transition lookup for a batch of symbols."""
        symbols = np.asarray(symbols)
        if symbols.size == 0:
            return np.array([], dtype=np.int32)

        if self.max_exceptions == 0:
            return np.full(symbols.shape, self.default_states[state], dtype=np.int32)

        eq = self.exception_symbols[state][:, None] == symbols
        hit = eq.any(axis=0)
        first = eq.argmax(axis=0)
        return np.where(hit, self.exception_states[state][first], self.default_states[state]).astype(np.int32)

    def reverse_transition(self, state: int, symbol: int) -> np.ndarray:
        """Get the states that transition to the given state on the given symbol."""
        default_mask = (self.default_states == state) & np.all(self.exception_symbols != symbol, axis=1)
        ex_mask = ((self.exception_states == state) & (self.exception_symbols == symbol)).any(axis=1)

        return np.flatnonzero(default_mask | ex_mask)

    def successors(self, state: int) -> np.ndarray:
        """Get all successor states from a given state."""
        default_succ = self.default_states[state]
        ex_succ = self.exception_states[state, self.exception_symbols[state] != -1]
        return np.unique(np.concatenate([np.array([default_succ]), ex_succ]))

    def _product(self, other: 'SparseDFA', combine_accept: Callable[[bool, bool], bool]) -> 'SparseDFA':
        if self.symbol_arity != other.symbol_arity:
            raise ValueError("Product requires same symbol arity")

        # Host-side arrays with pre-sorted exception rows: the BFS below
        # resolves each frontier's transitions with one batched binary search.
        n2 = other.num_states
        d1 = np.asarray(self.default_states, dtype=np.int64)
        acc1 = np.asarray(self.is_accepting, dtype=bool)
        e1_syms, e1_states = _sort_exception_rows(
            np.asarray(self.exception_symbols, dtype=np.int64),
            np.asarray(self.exception_states, dtype=np.int64))
        d2 = np.asarray(other.default_states, dtype=np.int64)
        acc2 = np.asarray(other.is_accepting, dtype=bool)
        e2_syms, e2_states = _sort_exception_rows(
            np.asarray(other.exception_symbols, dtype=np.int64),
            np.asarray(other.exception_states, dtype=np.int64))

        # Pairs (i, j) are encoded as i * n2 + j
        start_key = int(self.start_state) * n2 + int(other.start_state)
        state_map = {start_key: 0}
        frontier = np.array([start_key], dtype=np.int64)

        new_default_states_list = []
        new_exception_symbols_list = []
        new_exception_states_list = []
        new_is_accepting_list = []

        # Level-synchronous BFS: process the whole frontier as one batch.
        while frontier.size > 0:
            i = frontier // n2
            j = frontier % n2

            new_is_accepting_list.extend(combine_accept(acc1[i], acc2[j]).tolist())

            def_i = d1[i]
            def_j = d2[j]
            def_key = def_i * n2 + def_j

            # Candidate symbols: exceptions of either component, sorted per row
            # so duplicates are adjacent and the output is in symbol order.
            cand = np.sort(np.concatenate([e1_syms[i], e2_syms[j]], axis=1), axis=1)
            if cand.shape[1] > 0:
                keep = cand != -1
                keep[:, 1:] &= cand[:, 1:] != cand[:, :-1]  # drop duplicates

                next_i = _sorted_row_lookup(e1_syms[i], e1_states[i], def_i, cand)
                next_j = _sorted_row_lookup(e2_syms[j], e2_states[j], def_j, cand)
                next_key = next_i * n2 + next_j
                keep &= next_key != def_key[:, None]  # keep non-default only
            else:
                next_key = cand
                keep = np.zeros_like(cand, dtype=bool)

            # Assign ids in BFS discovery order (default target first, then
            # exception targets in symbol order); only dict ops left in Python.
            new_keys = []
            def_key_list = def_key.tolist()
            for row in range(frontier.size):
                key = def_key_list[row]
                idx = state_map.get(key)
                if idx is None:
                    idx = state_map[key] = len(state_map)
                    new_keys.append(key)
                new_default_states_list.append(idx)

                row_ids = []
                for key in next_key[row][keep[row]].tolist():
                    idx = state_map.get(key)
                    if idx is None:
                        idx = state_map[key] = len(state_map)
                        new_keys.append(key)
                    row_ids.append(idx)
                new_exception_symbols_list.append(cand[row][keep[row]].tolist())
                new_exception_states_list.append(row_ids)

            frontier = np.array(new_keys, dtype=np.int64)

        # Pad exception rows to uniform width
        num_new_states = len(state_map)
        max_exceptions = max(len(ex) for ex in new_exception_symbols_list) if new_exception_symbols_list else 0

        padded_ex_syms = np.full((num_new_states, max_exceptions), -1, dtype=np.int32)
        padded_ex_states = np.full((num_new_states, max_exceptions), -1, dtype=np.int32)

        for i, (syms, states) in enumerate(zip(new_exception_symbols_list, new_exception_states_list)):
            if syms:
                padded_ex_syms[i, :len(syms)] = syms
                padded_ex_states[i, :len(states)] = states

        return SparseDFA(
            num_new_states,
            np.array(new_default_states_list, dtype=np.int32),
            padded_ex_syms,
            padded_ex_states,
            np.array(new_is_accepting_list, dtype=bool),
            0,
            self.symbol_arity,
            self.base_alphabet.union(other.base_alphabet)
        )

    def alphabet_projection(self, projection_map: np.ndarray) -> 'SparseNFA':
        projection_map = np.asarray(projection_map)
        new_ex_symbols = np.where(
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
            default_states=default_states,
            exception_symbols=padded_ex_syms,
            exception_states=padded_ex_states,
            is_accepting=is_accepting,
            start_state=start_state,
            symbol_arity=k,
            base_alphabet=base_alphabet
        )

    def regular_right_quotient(self, other: 'SparseDFA') -> 'SparseDFA':
        nA, nB = self.num_states, other.num_states

        # Initialize reachability matrix
        reachable = self.is_accepting[:, None] & other.is_accepting[None, :]

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
                    all_symbols = np.unique(np.concatenate([ex_i, ex_j]))

                    # Check if any symbol leads to a reachable state
                    for sym in all_symbols.tolist():
                        next_i = self.transition(i, sym)
                        next_j = other.transition(j, sym)
                        if reachable[next_i, next_j]:
                            new_reachable[i, j] = True
                            changed = True
                            break

            reachable = new_reachable

        # New acceptance: state i is accepting if (i, other.start_state) is reachable
        new_accept = reachable[:, other.start_state]
        
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
    
    def _exception_target_stats(self, ex_states: np.ndarray):
        """Per-state statistics over exception targets, computed sparsely
        (never materializes a (num_states x num_states) counts matrix).

        :returns: (n_exceptions, max_count, most_common_target) arrays.
            States without exceptions get max_count 0 and target 0, matching
            the argmax of an all-zero counts row.
        """
        n = self.num_states
        n_exceptions = np.zeros(n, dtype=np.int64)
        max_count = np.zeros(n, dtype=np.int64)
        most_common = np.zeros(n, dtype=np.int64)
        if ex_states.shape[1] == 0:
            return n_exceptions, max_count, most_common

        rows = np.repeat(np.arange(n, dtype=np.int64), ex_states.shape[1])
        targets = ex_states.astype(np.int64).ravel()
        valid = targets != -1
        rows, targets = rows[valid], targets[valid]
        if rows.size == 0:
            return n_exceptions, max_count, most_common

        n_exceptions = np.bincount(rows, minlength=n)
        keys, counts = np.unique(rows * np.int64(n) + targets, return_counts=True)
        key_rows = keys // n
        key_targets = keys % n
        np.maximum.at(max_count, key_rows, counts)
        # First (smallest) target attaining the max, matching np.argmax ties
        at_max = counts == max_count[key_rows]
        rows_at_max, first = np.unique(key_rows[at_max], return_index=True)
        most_common[rows_at_max] = key_targets[np.flatnonzero(at_max)[first]]
        return n_exceptions, max_count, most_common

    def fill_defaults(self) -> 'SparseDFA':
        """Fills in default transitions for all states. If default state is currently -1, it will be set to the most common exception state."""
        defaults = np.asarray(self.default_states, dtype=np.int64)
        ex_symbols = np.asarray(self.exception_symbols, dtype=np.int64)
        ex_states = np.asarray(self.exception_states, dtype=np.int64)

        n_symbols = len(self.base_alphabet_frozen)**self.symbol_arity
        n_exceptions, _, most_common = self._exception_target_stats(ex_states)
        covers_alphabet = n_exceptions == n_symbols

        if (defaults.min(initial=0) < 0) or covers_alphabet.any():
            change_mask = (defaults == -1) | covers_alphabet
            new_defaults = np.where(change_mask, most_common, defaults)

            # delete exceptions to new default states
            delete = change_mask[:, None] & (ex_states == new_defaults[:, None])
            self.default_states = new_defaults.astype(np.int32)
            self.exception_symbols = np.where(delete, -1, ex_symbols).astype(np.int32)
            self.exception_states = np.where(delete, -1, ex_states).astype(np.int32)

        return self
        

    def minimize(self) -> 'SparseDFA':
        """Minimizes the DFA using vectorized Moore partition refinement.

        Partitions are refined by grouping states on signature rows
        (current partition, default-target partition, exception-target
        partitions) with np.unique until the partition count is stable.
        """
        self.fill_defaults()

        defaults = np.asarray(self.default_states, dtype=np.int64)
        ex_symbols = np.asarray(self.exception_symbols, dtype=np.int64)
        ex_states = np.asarray(self.exception_states, dtype=np.int64)
        accepting = np.asarray(self.is_accepting, dtype=bool)

        n_symbols = len(self.base_alphabet_frozen)**self.symbol_arity
        start = int(self.start_state)

        # Step 1: reachable states (frontier-batched BFS). The default
        # transition only exists if the exceptions don't cover the alphabet.
        reachable_mask = np.zeros(self.num_states, dtype=bool)
        reachable_mask[start] = True
        frontier = np.array([start], dtype=np.int64)
        while frontier.size > 0:
            valid = ex_symbols[frontier] != -1
            has_default = valid.sum(axis=1) < n_symbols
            successors = np.unique(np.concatenate([
                ex_states[frontier][valid],
                defaults[frontier][has_default]
            ]))
            successors = successors[~reachable_mask[successors]]
            reachable_mask[successors] = True
            frontier = successors

        reachable = np.flatnonzero(reachable_mask)
        row_of = np.full(self.num_states, -1, dtype=np.int64)
        row_of[reachable] = np.arange(reachable.size)

        # Step 2: transition table of the reachable states over all used
        # symbols, resolved with batched binary search in row chunks (memory
        # stays linear; the table itself is the only quadratic-ish object).
        # A state whose default target is unreachable has exceptions for
        # every used symbol, so the default never leaks into the table and
        # all targets are reachable.
        used_symbols = np.unique(ex_symbols[reachable][ex_symbols[reachable] != -1])
        num_syms = used_symbols.size
        if num_syms > 0:
            sorted_syms, sorted_targets = _sort_exception_rows(
                ex_symbols[reachable], ex_states[reachable])
            reach_defaults = defaults[reachable]
            table = np.empty((reachable.size, num_syms), dtype=np.int32)
            chunk = max(1, (1 << 23) // num_syms)
            for begin in range(0, reachable.size, chunk):
                rows = slice(begin, begin + chunk)
                cands = np.broadcast_to(used_symbols, (sorted_syms[rows].shape[0], num_syms))
                targets = _sorted_row_lookup(
                    sorted_syms[rows], sorted_targets[rows], reach_defaults[rows], cands)
                table[rows] = row_of[targets]
        else:
            table = np.zeros((reachable.size, 0), dtype=np.int32)

        # Default column: partition of the default target, or -1 sentinel if
        # the default target is unreachable (matching the old behavior of
        # keying only on reachable default targets).
        default_reach = reachable_mask[defaults[reachable]]
        default_rows = np.where(default_reach, row_of[np.where(default_reach, defaults[reachable], 0)], 0)

        # Step 3+4: refinement from the accepting/non-accepting partition.
        # Signature rows (current label, default label, transition labels)
        # are compressed into two independent 64-bit hashes, accumulated over
        # column chunks so the (states x symbols) label gather never fully
        # materializes. A collision would need two distinct signatures to
        # agree on both hashes (~ states^2 / 2^127 per round) — negligible.
        labels = accepting[reachable].astype(np.int64)
        num_parts = int(labels.max(initial=0)) + 1
        rng = np.random.default_rng(0xA757)
        weights = rng.integers(-2**62, 2**62, size=(2, num_syms + 2), dtype=np.int64) | 1
        chunk = max(1, (1 << 23) // max(reachable.size, 1))
        with np.errstate(over='ignore'):
            while True:
                default_col = np.where(default_reach, labels[default_rows], -1)
                hashes = labels[:, None] * weights[:, 0] + default_col[:, None] * weights[:, 1]
                for begin in range(0, num_syms, chunk):
                    cols = labels[table[:, begin:begin + chunk]]
                    hashes += cols @ weights[:, 2 + begin:2 + begin + cols.shape[1]].T
                _, labels = np.unique(hashes, axis=0, return_inverse=True)
                labels = labels.reshape(-1).astype(np.int64)
                new_parts = int(labels.max(initial=0)) + 1
                if new_parts == num_parts:
                    break
                num_parts = new_parts

        # Renumber partitions by first occurrence and pick that first state
        # as the representative
        _, first_occurrence = np.unique(labels, return_index=True)
        perm = np.empty(num_parts, dtype=np.int64)
        perm[np.argsort(first_occurrence, kind='stable')] = np.arange(num_parts)
        labels = perm[labels]
        reps = reachable[np.sort(first_occurrence)]

        part_of_state = np.full(self.num_states, -1, dtype=np.int64)
        part_of_state[reachable] = labels

        # Build new DFA from the representatives
        rep_defaults = defaults[reps]
        rep_default_reach = reachable_mask[rep_defaults]
        new_defaults = np.where(
            rep_default_reach,
            part_of_state[np.where(rep_default_reach, rep_defaults, 0)],
            -1
        )

        # Exceptions of the representatives, dropping those equal to the default
        rep_syms = ex_symbols[reps]
        valid = rep_syms != -1
        rep_targets = part_of_state[np.where(valid, ex_states[reps], 0)]
        keep = valid & (rep_targets != new_defaults[:, None])

        order = np.argsort(~keep, axis=1, kind='stable')
        keep_sorted = np.take_along_axis(keep, order, axis=1)
        padded_ex_syms = np.where(keep_sorted, np.take_along_axis(rep_syms, order, axis=1), -1)
        padded_ex_states = np.where(keep_sorted, np.take_along_axis(rep_targets, order, axis=1), -1)
        width = int(keep.sum(axis=1).max(initial=0))

        return SparseDFA(
            reps.size,
            new_defaults,
            padded_ex_syms[:, :width],
            padded_ex_states[:, :width],
            accepting[reps],
            int(part_of_state[start]),
            self.symbol_arity,
            self.base_alphabet
        ).sparsify()
    
    def sparsify(self) -> 'SparseDFA':
        """Computes the sparsest possible equivalent DFA by choosing optimal default states.
        

        Returns:
            SparseDFA with optimized default states and exceptions. A default state is changed if the most common exception state is more frequent than the default state.
        """
        num_states = self.num_states
        defaults = np.asarray(self.default_states, dtype=np.int64)
        ex_symbols = np.asarray(self.exception_symbols, dtype=np.int64)
        ex_states = np.asarray(self.exception_states, dtype=np.int64)
        n_total_symbols = len(self.base_alphabet_frozen) ** self.symbol_arity

        n_exceptions, max_count, most_common = self._exception_target_stats(ex_states)
        # Change the default if some exception target is more frequent than the
        # number of symbols served by the current default
        change_mask = max_count > n_total_symbols - n_exceptions
        new_defaults = np.where(change_mask, most_common, defaults)

        # delete exceptions to new default states
        delete = change_mask[:, None] & (ex_states == new_defaults[:, None])
        new_ex_symbols = np.where(delete, -1, ex_symbols)
        new_ex_states = np.where(delete, -1, ex_states)

        # For each state that changed, fill in the exceptions with the old default transitions
        for state in np.flatnonzero(change_mask & (defaults != -1)):
            row = ex_symbols[state]
            old_def_ex_syms = np.setdiff1d(np.arange(n_total_symbols), row[row != -1])

            free_places = np.flatnonzero(new_ex_symbols[state] == -1)[:len(old_def_ex_syms)]
            new_ex_symbols[state, free_places] = old_def_ex_syms[:len(free_places)]
            new_ex_states[state, free_places] = defaults[state]

        # Left-compact rows and shrink max_exceptions to the widest row
        valid = new_ex_symbols != -1
        order = np.argsort(~valid, axis=1, kind='stable')
        valid_sorted = np.take_along_axis(valid, order, axis=1)
        new_ex_symbols = np.where(valid_sorted, np.take_along_axis(new_ex_symbols, order, axis=1), -1)
        new_ex_states = np.where(valid_sorted, np.take_along_axis(new_ex_states, order, axis=1), -1)
        width = int(valid.sum(axis=1).max(initial=0))
        new_ex_symbols = new_ex_symbols[:, :width]
        new_ex_states = new_ex_states[:, :width]

        return SparseDFA(
            num_states,
            new_defaults,
            new_ex_symbols,
            new_ex_states,
            self.is_accepting,
            self.start_state,
            self.symbol_arity,
            self.base_alphabet
        )
    
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
    
    def sparse_dfa_to_file(self, filename: str) -> None:
        SparseDFASerializer.serialize(self, filename)

    @classmethod
    def sparse_dfa_from_file(cls, filename: str) -> 'SparseDFA':
        return SparseDFASerializer.deserialize(filename)
    

class SparseNFA:
    def __init__(self, num_states: int, base_state: np.ndarray,
                 exception_symbols: np.ndarray, exception_states: np.ndarray,
                 is_accepting: np.ndarray, start_state: int,
                 symbol_arity: int = 1, base_alphabet: Optional[Set[int]] = None):
        self.num_states = num_states
        self.base_state = np.asarray(base_state, dtype=np.int32)
        self.exception_symbols = np.asarray(exception_symbols, dtype=np.int32)
        self.exception_states = np.asarray(exception_states, dtype=np.int32)
        self.is_accepting = np.asarray(is_accepting, dtype=bool)
        self.start_state = int(start_state)
        self.max_exceptions = self.exception_symbols.shape[1]
        self.symbol_arity = symbol_arity
        self.base_alphabet = base_alphabet if base_alphabet is not None else self._infer_base_alphabet()

    def _infer_base_alphabet(self) -> Set[int]:
        """Infer base alphabet from exception symbols"""
        symbols = set(np.unique(self.exception_symbols))
        symbols.discard(-1)
        return symbols

    def _step(self, current_set: np.ndarray, symbol: int) -> np.ndarray:
        """Single step transition of the active state set"""
        next_set = np.zeros(self.num_states, dtype=bool)
        # Base transitions of all active states
        next_set[self.base_state[current_set]] = True
        # Exception transitions of active states on this symbol
        matches = current_set[:, None] & (self.exception_symbols == symbol)
        next_set[self.exception_states[matches]] = True
        return next_set

    def compute(self, word: np.ndarray) -> np.ndarray:
        """Returns the set of states after processing the word"""
        current = np.zeros(self.num_states, dtype=bool)
        current[self.start_state] = True
        for symbol in np.asarray(word, dtype=np.int64).tolist():
            current = self._step(current, symbol)
        return current

    def accepts(self, word: np.ndarray) -> bool:
        """Checks if the NFA accepts the word"""
        final_set = self.compute(word)
        return bool(np.any(final_set & self.is_accepting))
    
    def determinize(self) -> SparseDFA:
        """Converts NFA to DFA using sparse subset construction.

        Exceptions override the base transition (a member with an exception
        on a symbol does not contribute its base target for that symbol).
        Only symbols occurring as exceptions of subset members can lead
        anywhere other than the set of member base targets, so candidates are
        derived from the exception tables and resolved in batched numpy ops.
        """
        bases = np.asarray(self.base_state, dtype=np.int64)
        sorted_syms, sorted_targets = _sort_exception_rows(
            np.asarray(self.exception_symbols, dtype=np.int64),
            np.asarray(self.exception_states, dtype=np.int64))
        accepting = np.asarray(self.is_accepting, dtype=bool)
        n_total_symbols = len(self.base_alphabet) ** self.symbol_arity

        state_to_id = {}
        id_to_set = []
        dfa_accepting = []

        def get_id(key):
            idx = state_to_id.get(key)
            if idx is None:
                idx = state_to_id[key] = len(id_to_set)
                members = np.array(key, dtype=np.int64)
                id_to_set.append(members)
                dfa_accepting.append(bool(accepting[members].any()))
            return idx

        get_id((int(self.start_state),))

        dfa_defaults = []
        dfa_ex_syms = []
        dfa_ex_states = []

        next_unprocessed = 0
        while next_unprocessed < len(id_to_set):
            members = id_to_set[next_unprocessed]
            next_unprocessed += 1

            member_syms = sorted_syms[members]
            cand = np.unique(member_syms[member_syms != -1])

            default_key = tuple(np.unique(bases[members]).tolist())

            exceptions = []
            if cand.size > 0:
                # Spans of matching exception slots per (member, candidate)
                # via binary search on the sorted rows: NFA rows may hold the
                # same symbol several times, so all slots in the span count.
                n_members, n_cand = members.shape[0], cand.size
                offset = np.int64(int(member_syms.max(initial=0)) + 2)
                row_ids = np.arange(n_members, dtype=np.int64)[:, None]
                keys = (row_ids * offset + member_syms).ravel()
                queries = (row_ids * offset + cand).ravel()
                left = np.searchsorted(keys, queries, side='left')
                lengths = np.searchsorted(keys, queries, side='right') - left

                has_exception = (lengths > 0).reshape(n_members, n_cand)
                total = int(lengths.sum())
                positions = np.repeat(left, lengths) + (
                    np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths))
                match_targets = sorted_targets[members].ravel()[positions]
                match_cand = np.repeat(np.arange(n_members * n_cand) % n_cand, lengths)
                member_bases = bases[members]

                for j in range(n_cand):
                    # Exception targets plus base targets of members without
                    # an exception on this symbol
                    key = tuple(np.unique(np.concatenate([
                        match_targets[match_cand == j],
                        member_bases[~has_exception[:, j]]
                    ])).tolist())
                    if key != default_key:
                        exceptions.append((int(cand[j]), key))

                if len(exceptions) == n_total_symbols:
                    # Every symbol is an exception, so the all-bases target can
                    # never be reached; make the most common target the default
                    # instead of introducing a spurious state.
                    counts = Counter(key for _, key in exceptions)
                    default_key = counts.most_common(1)[0][0]
                    exceptions = [(sym, key) for sym, key in exceptions if key != default_key]

            dfa_defaults.append(get_id(default_key))
            dfa_ex_syms.append([sym for sym, _ in exceptions])
            dfa_ex_states.append([get_id(key) for _, key in exceptions])

        # Pad exception arrays
        num_states = len(id_to_set)
        max_ex = max(len(syms) for syms in dfa_ex_syms) if dfa_ex_syms else 0
        padded_ex_syms = np.full((num_states, max_ex), -1, dtype=np.int32)
        padded_ex_states = np.full((num_states, max_ex), -1, dtype=np.int32)

        for i, (syms, states) in enumerate(zip(dfa_ex_syms, dfa_ex_states)):
            if syms:
                padded_ex_syms[i, :len(syms)] = syms
                padded_ex_states[i, :len(states)] = states

        return SparseDFA(
            num_states=num_states,
            default_states=np.array(dfa_defaults, dtype=np.int32),
            exception_symbols=padded_ex_syms,
            exception_states=padded_ex_states,
            is_accepting=np.array(dfa_accepting, dtype=bool),
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