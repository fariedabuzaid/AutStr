import json
import numpy as np
from collections import deque, defaultdict

# JAX is optional: it is only used to accelerate batched word processing on
# fixed automata (accepts_batch), where jit/scan/GPU pay off. All construction
# algorithms run on numpy.
try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True

    @jax.jit
    def _jax_run_batch(table, start, words):
        """Run a (B, L) batch of encoded words; returns the (B,) final states."""
        def step(states, symbols):
            return table[states, symbols], None

        init = jnp.full(words.shape[0], start, dtype=jnp.int32)
        final, _ = jax.lax.scan(step, init, words.T)
        return final

except ImportError:
    _HAS_JAX = False
from typing import Dict, Tuple, Optional, Callable, List, Set
import graphviz

import struct
import zlib


from autstr.mtbdd import (
    NONE, STORE, TOP, ComputedTable, NodeStore, bits_of, num_bits, var_tables,
)
from autstr.utils.misc import decode_symbol, encode_symbol



# File format structure:
# [Header (16 bytes)]
#   - Magic number: 4 bytes ('SDFA')
#   - Version: 1 byte
#   - Reserved: 3 bytes (0)
#   - Checksum: 4 bytes (CRC32 of payload)
#   - Payload size: 4 bytes
# [Payload]
#   - Metadata (20 bytes: num_states, num_nodes, start, arity, alphabet size)
#   - Base alphabet (JSON)
#   - Acceptance array
#   - Transition diagrams: var / lo / hi / term arrays, then one root per state

class SparseDFASerializer:
    VERSION = 3  # diagram payload (v2 stored flat exception rows)
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
            
            # Read and validate payload
            payload = f.read(payload_size)
            if zlib.crc32(payload) != checksum:
                raise ValueError("Data corruption detected (checksum mismatch)")
            
            return cls._parse_payload(payload, version)
    
    @classmethod
    def _create_payload(cls, dfa: 'SparseDFA') -> bytes:
        """Binary payload: the shared sub-DAG of the state diagrams, so an
        automaton over a convolution alphabet too wide to enumerate still
        serializes in the size of its diagrams."""
        var, lo, hi, term, roots = STORE.export(dfa.nodes.tolist())
        base_alphabet_json = json.dumps(sorted(dfa.base_alphabet)).encode('utf-8')

        metadata = struct.pack(
            cls.METADATA_FORMAT,
            dfa.num_states,
            len(var),
            dfa.start_state,
            dfa.symbol_arity,
            len(base_alphabet_json)
        )
        return b''.join([
            metadata,
            base_alphabet_json,
            np.asarray(dfa.is_accepting, dtype=np.uint8).tobytes(),
            var.astype(np.int64).tobytes(),
            lo.astype(np.int64).tobytes(),
            hi.astype(np.int64).tobytes(),
            term.astype(np.int64).tobytes(),
            roots.astype(np.int64).tobytes(),
        ])

    @classmethod
    def _parse_payload(cls, payload: bytes, version: int = None) -> 'SparseDFA':
        if version is not None and version != cls.VERSION:
            if version == 2:
                return cls._parse_payload_v2(payload)
            raise ValueError(f"Unsupported SparseDFA version: {version}")
        num_states, num_nodes, start_state, symbol_arity, alphabet_len = \
            struct.unpack(cls.METADATA_FORMAT, payload[:cls.METADATA_SIZE])
        offset = cls.METADATA_SIZE
        base_alphabet = set(json.loads(payload[offset:offset + alphabet_len]))
        offset += alphabet_len

        is_accepting = np.frombuffer(payload, dtype=np.uint8, count=num_states,
                                     offset=offset).astype(bool)
        offset += num_states

        arrays = []
        for count in (num_nodes, num_nodes, num_nodes, num_nodes, num_states):
            arrays.append(np.frombuffer(payload, dtype=np.int64, count=count,
                                        offset=offset))
            offset += count * 8
        var, lo, hi, term, roots = arrays
        nodes = STORE.import_nodes(var, lo, hi, term, roots)

        return SparseDFA(num_states, is_accepting=is_accepting,
                         start_state=start_state, symbol_arity=symbol_arity,
                         base_alphabet=base_alphabet, nodes=nodes)

    @classmethod
    def _parse_payload_v2(cls, payload: bytes) -> 'SparseDFA':
        """Read the pre-diagram format: default target plus exception rows."""
        num_states, max_exceptions, start_state, symbol_arity, alphabet_len = \
            struct.unpack(cls.METADATA_FORMAT, payload[:cls.METADATA_SIZE])
        offset = cls.METADATA_SIZE
        base_alphabet = set(json.loads(payload[offset:offset + alphabet_len]))
        offset += alphabet_len

        defaults = np.frombuffer(payload, dtype=np.uint32, count=num_states,
                                 offset=offset).astype(np.int64)
        offset += num_states * 4
        count = num_states * max_exceptions
        ex_symbols = np.frombuffer(payload, dtype=np.int32, count=count,
                                   offset=offset).reshape(num_states, max_exceptions)
        offset += count * 4
        ex_states = np.frombuffer(payload, dtype=np.int32, count=count,
                                  offset=offset).reshape(num_states, max_exceptions)
        offset += count * 4
        is_accepting = np.frombuffer(payload, dtype=np.uint8, count=num_states,
                                     offset=offset).astype(bool)

        return SparseDFA(num_states, defaults, ex_symbols, ex_states,
                         is_accepting, start_state, symbol_arity,
                         base_alphabet)

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
        
        payload = data[cls.HEADER_SIZE:cls.HEADER_SIZE+payload_size]
        if len(payload) != payload_size:
            raise ValueError("Payload size mismatch")
        if zlib.crc32(payload) != checksum:
            raise ValueError("SparseDFA data corruption detected")
        
        return cls._parse_payload(payload, version)


class SparseDFA:
    """Deterministic automaton over a convolution alphabet.

    Each state carries one *shared multi-terminal BDD* over the binary digits
    of the symbol (see `autstr.mtbdd`) instead of a default target plus a row
    of ``symbol -> target`` exceptions. A transition that ignores a tape never
    tests that tape's variables, so cylindrification is a variable renaming
    rather than a duplication of every row once per letter of every new tape —
    which is what the pipeline used to spend its memory on.

    The constructor still accepts the flat form, and `default_states`,
    `exception_symbols` and `exception_states` remain available as decoded
    views for inspection, rendering and serialization of narrow automata: the
    default of a decoded state is its most common target, so the view is the
    sparsest one (what `sparsify` used to compute).
    """

    def __init__(self, num_states: int, default_states=(),
                 exception_symbols=(), exception_states=(),
                 is_accepting=(), start_state: int = 0,
                 symbol_arity: int = 1,
                 base_alphabet: Optional[Set[int]] = None, nodes=None):
        self.num_states = int(num_states)
        self.is_accepting = np.asarray(is_accepting, dtype=bool)
        self.start_state = int(start_state)
        self.symbol_arity = int(symbol_arity)
        self.base_alphabet = base_alphabet if base_alphabet is not None \
            else self._infer_base_alphabet(exception_symbols)
        self.base_alphabet_frozen = frozenset(self.base_alphabet)

        self.store = STORE
        self.m = len(self.base_alphabet_frozen)
        self.bits = num_bits(self.m)
        self.nvars = self.symbol_arity * self.bits
        self._dense: Optional[np.ndarray] = None
        self._decoded = None

        if nodes is not None:
            self.nodes = np.asarray(nodes, dtype=np.int64)
        else:
            self.nodes = self._compile(default_states, exception_symbols,
                                       exception_states)

    @staticmethod
    def _infer_base_alphabet(exception_symbols) -> Set[int]:
        symbols = set(np.unique(np.asarray(exception_symbols)).tolist())
        symbols.discard(-1)
        return symbols or {0}

    def _compile(self, default_states, exception_symbols,
                 exception_states) -> np.ndarray:
        n = self.num_states
        defaults = np.asarray(default_states, dtype=np.int64).reshape(-1)
        ex_syms = np.asarray(exception_symbols, dtype=np.int64)
        ex_targets = np.asarray(exception_states, dtype=np.int64)
        if ex_syms.size == 0:
            ex_syms = np.zeros((n, 0), dtype=np.int64)
            ex_targets = np.zeros((n, 0), dtype=np.int64)
        else:
            ex_syms = ex_syms.reshape(n, -1)
            ex_targets = ex_targets.reshape(n, -1)

        nodes = np.empty(n, dtype=np.int64)
        for q in range(n):
            keep = (ex_syms[q] >= 0) & (ex_targets[q] >= 0)
            symbols, targets = ex_syms[q][keep], ex_targets[q][keep]
            order = np.argsort(symbols, kind="stable")
            symbols, targets = symbols[order], targets[order]
            if len(symbols):
                symbols, first = np.unique(symbols, return_index=True)
                targets = targets[first]
            base = int(defaults[q]) if q < len(defaults) else 0
            if base < 0:
                # "no default": the exceptions were meant to cover the
                # alphabet, so any target serves — take the most common one
                base = int(np.bincount(targets).argmax()) if len(targets) else 0
            nodes[q] = self.store.build_rows(symbols, targets, base,
                                             self.symbol_arity, self.m,
                                             self.bits)
        return nodes

    # ---------------- symbols ----------------

    @property
    def num_symbols(self) -> int:
        return self.m ** self.symbol_arity

    @property
    def num_nodes(self) -> int:
        """Distinct diagram nodes carrying this automaton's transitions."""
        return self.store.size(self.nodes.tolist())

    def encode_symbol(self, symbol_tuple: Tuple[int]) -> int:
        return encode_symbol(symbol_tuple, self.base_alphabet_frozen)

    def decode_symbol(self, symbol_enc: int) -> Tuple[int]:
        return decode_symbol(symbol_enc, self.symbol_arity,
                             self.base_alphabet_frozen)

    # ---------------- decoded (flat) view ----------------

    def dense_next(self, max_entries: int = 1 << 24) -> np.ndarray:
        """The full ``(num_states, num_symbols)`` next-state table. Only for
        automata narrow enough to enumerate; the pipeline never calls it."""
        if self._dense is None:
            size = self.num_states * self.num_symbols
            if size > max_entries:
                raise ValueError(
                    f"transition table of {size} entries is too large to "
                    f"materialize; use the diagrams instead")
            symbols = np.tile(np.arange(self.num_symbols, dtype=np.int64),
                              self.num_states)
            nodes = np.repeat(self.nodes, self.num_symbols)
            self._dense = self.store.eval_batch(
                nodes, symbols, self.symbol_arity, self.m, self.bits
            ).reshape(self.num_states, self.num_symbols)
        return self._dense

    def _decode(self):
        """Flat view: each state's most common target becomes its default and
        the remaining symbols become exceptions (the sparsest flat form)."""
        if self._decoded is None:
            table = self.dense_next()
            n, S = table.shape
            counts = np.zeros((n, self.num_states), dtype=np.int64)
            np.add.at(counts, (np.repeat(np.arange(n), S), table.ravel()), 1)
            defaults = counts.argmax(axis=1)
            deviates = table != defaults[:, None]
            width = int(deviates.sum(axis=1).max(initial=0))
            symbols = np.full((n, width), -1, dtype=np.int32)
            targets = np.full((n, width), -1, dtype=np.int32)
            for q in range(n):
                where = np.flatnonzero(deviates[q])
                symbols[q, :len(where)] = where
                targets[q, :len(where)] = table[q, where]
            self._decoded = (defaults.astype(np.int32), symbols, targets)
        return self._decoded

    @property
    def default_states(self) -> np.ndarray:
        return self._decode()[0]

    @property
    def exception_symbols(self) -> np.ndarray:
        return self._decode()[1]

    @property
    def exception_states(self) -> np.ndarray:
        return self._decode()[2]

    @property
    def max_exceptions(self) -> int:
        return self._decode()[1].shape[1]

    # ---------------- running words ----------------

    def transition(self, state: int, symbol: int) -> int:
        return int(self.store.eval_batch(
            np.array([self.nodes[state]]), np.array([symbol], dtype=np.int64),
            self.symbol_arity, self.m, self.bits)[0])

    def _vectorized_transition(self, state: int, symbols: np.ndarray) -> np.ndarray:
        symbols = np.asarray(symbols, dtype=np.int64)
        if symbols.size == 0:
            return np.array([], dtype=np.int32)
        return self.store.eval_batch(
            np.full(symbols.shape, self.nodes[state]), symbols,
            self.symbol_arity, self.m, self.bits).astype(np.int32)

    def compute(self, word: np.ndarray) -> int:
        """Final state after reading the word (encoded symbols).

        A word is a sequential dependency chain, so this cannot be vectorized.
        Narrow automata run off the dense table (one list index per symbol);
        wide ones descend the diagram, memoizing each (state, symbol) step."""
        word = np.asarray(word, dtype=np.int64).tolist()
        state = self.start_state
        try:
            table = self.dense_next(1 << 20).tolist()
        except ValueError:
            table = None
        if table is not None:
            for symbol in word:
                state = table[state][symbol]
            return state

        store = self.store
        var, lo, hi, term = store.var, store.lo, store.hi, store.term
        div, shift = var_tables(self.symbol_arity, self.m, self.bits)
        div, shift = div.tolist(), shift.tolist()
        nodes, m, cache = self.nodes.tolist(), self.m, {}
        for symbol in word:
            key = state * self.num_symbols + symbol
            target = cache.get(key)
            if target is None:
                node = nodes[state]
                while var[node] != TOP:
                    v = var[node]
                    node = hi[node] if (symbol // div[v]) % m >> shift[v] & 1 \
                        else lo[node]
                target = cache[key] = term[node]
            state = target
        return state

    def accepts(self, word) -> bool:
        word = np.array([encode_symbol(s, self.base_alphabet_frozen)
                         for s in word], dtype=np.int64)
        return bool(self.is_accepting[self.compute(word)])

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

        Uses JAX (jit + scan, GPU if available) when installed and the dense
        next-state table fits; otherwise a vectorized numpy fallback (over the
        table, or over the diagrams when the alphabet is too wide to enumerate).
        """
        words = self._encode_words(words)
        if words.ndim != 2:
            raise ValueError("accepts_batch expects a (batch, length) input")
        if words.shape[1] == 0:
            return np.full(words.shape[0], self.is_accepting[self.start_state])

        try:
            table = self.dense_next()
        except ValueError:
            table = None

        if table is None:
            states = np.full(words.shape[0], self.start_state, dtype=np.int64)
            for t in range(words.shape[1]):
                states = self.store.eval_batch(
                    self.nodes[states], words[:, t], self.symbol_arity,
                    self.m, self.bits)
            return self.is_accepting[states]

        if _HAS_JAX:
            final = _jax_run_batch(jnp.asarray(table, dtype=jnp.int32),
                                   self.start_state,
                                   jnp.asarray(words, dtype=jnp.int32))
            return np.asarray(self.is_accepting[np.asarray(final)])

        states = np.full(words.shape[0], self.start_state, dtype=np.int64)
        for t in range(words.shape[1]):
            states = table[states, words[:, t]]
        return self.is_accepting[states]

    # ---------------- structure ----------------

    def successors(self, state: int) -> np.ndarray:
        """All successor states of a state — the terminals of its diagram."""
        return np.array(self.store.terminals(int(self.nodes[state])),
                        dtype=np.int64)

    def reverse_transition(self, state: int, symbol: int) -> np.ndarray:
        """The states that transition to `state` on `symbol`."""
        targets = self.store.eval_batch(
            self.nodes, np.full(self.num_states, symbol, dtype=np.int64),
            self.symbol_arity, self.m, self.bits)
        return np.flatnonzero(targets == state)

    def _reachable(self) -> np.ndarray:
        seen = np.zeros(self.num_states, dtype=bool)
        seen[self.start_state] = True
        frontier = [self.start_state]
        while frontier:
            targets = np.unique(np.concatenate(
                [self.successors(q) for q in frontier]))
            targets = targets[~seen[targets]]
            seen[targets] = True
            frontier = targets.tolist()
        return seen

    def is_empty(self) -> bool:
        return not bool((self._reachable() & self.is_accepting).any())

    def is_finite(self) -> bool:
        reachable = set(np.flatnonzero(self._reachable()).tolist())

        rev_map = defaultdict(set)
        for u in range(self.num_states):
            for v in self.successors(u).tolist():
                rev_map[v].add(u)

        co_reachable = set()
        queue = deque(np.flatnonzero(self.is_accepting).tolist())
        while queue:
            state = queue.popleft()
            if state in co_reachable:
                continue
            co_reachable.add(state)
            queue.extend(pred for pred in rev_map[state]
                         if pred not in co_reachable)

        useful = reachable & co_reachable
        if not useful:
            return True                       # empty language: finite

        graph = {u: {v for v in self.successors(u).tolist() if v in useful}
                 for u in useful}

        # iterative DFS cycle detection: a cycle among useful states means
        # arbitrarily long accepted words
        color = {state: 0 for state in useful}      # 0 white, 1 gray, 2 black
        for state in useful:
            if color[state] != 0:
                continue
            stack = [state]
            while stack:
                u = stack.pop()
                if color[u] == 0:
                    color[u] = 1
                    stack.append(u)
                    for v in graph[u]:
                        if color[v] == 0:
                            stack.append(v)
                        elif color[v] == 1:
                            return False
                else:
                    color[u] = 2
        return True

    # ---------------- boolean operations ----------------

    def complement(self) -> 'SparseDFA':
        """Flip acceptance — the transition diagrams are untouched."""
        return SparseDFA(self.num_states, is_accepting=~self.is_accepting,
                         start_state=self.start_state,
                         symbol_arity=self.symbol_arity,
                         base_alphabet=self.base_alphabet, nodes=self.nodes)

    def intersection(self, other: 'SparseDFA') -> 'SparseDFA':
        return self._product(other, combine_accept=lambda a, b: a & b)

    def union(self, other: 'SparseDFA') -> 'SparseDFA':
        return self._product(other, combine_accept=lambda a, b: a | b)

    def _product(self, other: 'SparseDFA',
                 combine_accept: Callable[[bool, bool], bool]) -> 'SparseDFA':
        if self.symbol_arity != other.symbol_arity:
            raise ValueError("Product requires same symbol arity")
        if self.base_alphabet_frozen != other.base_alphabet_frozen:
            raise ValueError("Product requires same base alphabet")

        # The diagram of a product state is the pairwise `apply` of the
        # factors' diagrams; its terminal operation allocates product state
        # ids, so the reachable pairs discover themselves.
        store = self.store
        ids: Dict[Tuple[int, int], int] = {}
        pairs: List[Tuple[int, int]] = []

        def get_id(a: int, b: int) -> int:
            key = (a, b)
            idx = ids.get(key)
            if idx is None:
                idx = ids[key] = len(pairs)
                pairs.append(key)
            return idx

        def op(ta: int, tb: int) -> int:
            if ta == NONE or tb == NONE:
                return NONE
            return get_id(ta, tb)

        cache: Dict[int, int] = {}
        get_id(self.start_state, other.start_state)
        nodes: List[int] = []
        index = 0
        while index < len(pairs):               # pairs grows inside apply2
            a, b = pairs[index]
            nodes.append(store.apply2(int(self.nodes[a]), int(other.nodes[b]),
                                      op, cache))
            index += 1

        first = np.array([p[0] for p in pairs], dtype=np.int64)
        second = np.array([p[1] for p in pairs], dtype=np.int64)
        return SparseDFA(
            len(pairs),
            is_accepting=combine_accept(self.is_accepting[first],
                                        other.is_accepting[second]),
            start_state=0, symbol_arity=self.symbol_arity,
            base_alphabet=self.base_alphabet.union(other.base_alphabet),
            nodes=np.array(nodes, dtype=np.int64))

    def alphabet_projection(self, projection_map: np.ndarray) -> 'SparseNFA':
        """Relabel symbols by `projection_map`, which may merge them and thus
        make the automaton nondeterministic (inspection-scale: it decodes)."""
        projection_map = np.asarray(projection_map)
        new_ex_symbols = np.where(self.exception_symbols != -1,
                                  projection_map[self.exception_symbols], -1)
        return SparseNFA(
            num_states=self.num_states, base_state=self.default_states,
            exception_symbols=new_ex_symbols,
            exception_states=self.exception_states,
            is_accepting=self.is_accepting, start_state=self.start_state,
            symbol_arity=self.symbol_arity, base_alphabet=self.base_alphabet)

    def intersect_subtapes(self, other: 'SparseDFA', tapes: List[int]) -> 'SparseDFA':
        """{x in L(self) | x[tapes] in L(other)}.

        Expressed as a product with `other` cylindrified onto `tapes`, so no
        symbol is enumerated."""
        from autstr.utils.automata_tools import expand
        k, l = self.symbol_arity, other.symbol_arity
        if len(tapes) != l:
            raise ValueError(f"Tapes length ({len(tapes)}) must match "
                             f"other.arity ({l})")
        if not all(0 <= t < k for t in tapes):
            raise ValueError("All tape indices must be in [0, self.arity-1]")
        if self.base_alphabet != other.base_alphabet:
            raise ValueError("Automata must have the same base alphabet")
        return self.intersection(expand(other, k, list(tapes)))

    def regular_right_quotient(self, other: 'SparseDFA') -> 'SparseDFA':
        """{u | uv in L(self) for some v in L(other)}: a state of self is
        accepting iff, paired with other's start state, it can synchronously
        reach a pair of accepting states."""
        table_a, table_b = self.dense_next(), other.dense_next()
        reachable = self.is_accepting[:, None] & other.is_accepting[None, :]
        while True:
            # (nA, nB, S): does symbol s lead to an already-reachable pair?
            step = reachable[table_a[:, None, :], table_b[None, :, :]].any(axis=2)
            grown = reachable | step
            if (grown == reachable).all():
                break
            reachable = grown

        return SparseDFA(self.num_states,
                         is_accepting=reachable[:, other.start_state],
                         start_state=self.start_state,
                         symbol_arity=self.symbol_arity,
                         base_alphabet=self.base_alphabet, nodes=self.nodes)

    # ---------------- normalization ----------------

    def fill_defaults(self) -> 'SparseDFA':
        """No-op: a diagram has no default slot to fill."""
        return self

    def sparsify(self) -> 'SparseDFA':
        """No-op: the diagram representation is already the sparse one (its
        decoded view picks each state's most common target as the default)."""
        return self

    def minimize(self) -> 'SparseDFA':
        """Moore partition refinement over the transition diagrams.

        Relabelling a state's diagram by the current partition yields the
        function ``symbol -> class of target``; hash-consing means two states
        induce the same function exactly when the relabelled diagrams are the
        same node, so a refinement round is one `apply1` per state."""
        store = self.store
        reach = self._reachable()
        keep = np.flatnonzero(reach)
        new_of_old = np.full(self.num_states, -1, dtype=np.int64)
        new_of_old[keep] = np.arange(len(keep))

        relabel: Dict[int, int] = {}
        nodes = np.array([store.apply1(int(self.nodes[q]),
                                       lambda t: int(new_of_old[t]), relabel)
                          for q in keep.tolist()], dtype=np.int64)
        accepting = self.is_accepting[keep]

        labels = accepting.astype(np.int64)
        num_parts = len(np.unique(labels))
        while True:
            round_cache: Dict[int, int] = {}
            relabelled = np.array(
                [store.apply1(int(node), lambda t: int(labels[t]), round_cache)
                 for node in nodes.tolist()], dtype=np.int64)
            _, refined = np.unique(np.stack([labels, relabelled], axis=1),
                                   axis=0, return_inverse=True)
            refined = refined.reshape(-1).astype(np.int64)
            parts = len(np.unique(refined))
            labels = refined
            if parts == num_parts:
                break
            num_parts = parts

        # renumber partitions by first occurrence, keep that state as the
        # representative
        _, first = np.unique(labels, return_index=True)
        perm = np.empty(num_parts, dtype=np.int64)
        perm[np.argsort(first, kind='stable')] = np.arange(num_parts)
        labels = perm[labels]
        reps = np.sort(first)

        final: Dict[int, int] = {}
        new_nodes = np.array([store.apply1(int(nodes[r]),
                                           lambda t: int(labels[t]), final)
                              for r in reps.tolist()], dtype=np.int64)
        return SparseDFA(
            num_parts, is_accepting=accepting[reps],
            start_state=int(labels[new_of_old[self.start_state]]),
            symbol_arity=self.symbol_arity, base_alphabet=self.base_alphabet,
            nodes=new_nodes)

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
    
    def show_diagram(self, filename: str = "automaton", format: str = "png", view: bool = False) -> graphviz.Digraph:
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
    

def recode(dfa: 'SparseDFA', new_alphabet, letter_map=None) -> 'SparseDFA':
    """Re-express an automaton over a different base alphabet.

    `letter_map` sends each of the automaton's letters to a letter of
    `new_alphabet` (injectively; the identity by default). Letters of the new
    alphabet outside the image are rejected: a fresh dead state absorbs them,
    which is what the closure constructions want -- a factor of a disjoint
    union must reject the other side's letters, and a factor of a direct
    product must reject the tag letters.

    The rewrite happens on the diagrams, one pass over the nodes, so widening
    the alphabet does not rebuild any transition table.
    """
    old_letters = sorted(dfa.base_alphabet_frozen)
    new_letters = sorted(frozenset(new_alphabet))
    letter_map = letter_map or {letter: letter for letter in old_letters}
    missing = set(old_letters) - set(letter_map)
    if missing:
        raise ValueError(f"letter_map does not cover {sorted(missing)}")
    index = {letter: i for i, letter in enumerate(new_letters)}
    try:
        digit_map = [index[letter_map[letter]] for letter in old_letters]
    except KeyError as exc:
        raise ValueError(f"{exc.args[0]!r} is not in the new alphabet") from exc

    store = dfa.store
    dead = dfa.num_states
    new_m = len(new_letters)
    new_bits = num_bits(new_m)
    nodes = [store.recode_letters(int(node), dfa.symbol_arity, dfa.m, dfa.bits,
                                  new_m, new_bits, digit_map, dead)
             for node in dfa.nodes.tolist()]
    nodes.append(store.const(dead, dfa.symbol_arity, new_m, new_bits))
    return SparseDFA(dead + 1, is_accepting=np.r_[dfa.is_accepting, False],
                     start_state=dfa.start_state,
                     symbol_arity=dfa.symbol_arity,
                     base_alphabet=set(new_letters),
                     nodes=np.array(nodes, dtype=np.int64))


def _mask(states) -> int:
    """Pack an iterable of state indices into an integer bitset."""
    mask = 0
    for q in states:
        mask |= 1 << int(q)
    return mask


def reduce_set_nfa(store, nodes, subsets, is_accepting, start: int,
                   arity: int, m: int, bits: int):
    """Shrink a set-valued NFA before determinizing it.

    `nodes[q]` is a diagram from symbols to sets of successors (terminals are
    indices into `subsets`, whose entries are bitsets of states). Two
    reductions apply, both language-preserving and both cheap next to the
    subset construction that follows:

    - states that reach no accepting state, and states unreachable from
      `start`, are dropped: they enlarge every subset that contains them
      without ever affecting acceptance;
    - the remainder is quotiented by *forward bisimulation* — q and q' merge
      when they agree on acceptance and, on every symbol, their successor sets
      have the same classes. Relabelling a state's diagram so each terminal
      becomes the *class* bitset of its target set turns the state's signature
      into a hash-consed node id, so a refinement round is one `apply1` per
      state.

    Determinizing the quotient explores subsets of classes, which are images
    of the subsets of states, so the state count can only shrink.

    :returns: ``(nodes, subsets, subset_ids, is_accepting, start)`` over the
        classes, or None when the language is empty.
    """
    n = len(nodes)
    is_accepting = np.asarray(is_accepting, dtype=bool)

    successors = []
    for q in range(n):
        mask = 0
        for sid in store.terminals(int(nodes[q])):
            mask |= subsets[sid]
        successors.append(mask)

    forward = np.zeros(n, dtype=bool)
    forward[start] = True
    stack = [start]
    while stack:
        for t in bits_of(successors[stack.pop()]):
            if not forward[t]:
                forward[t] = True
                stack.append(t)

    predecessors = [[] for _ in range(n)]
    for q in range(n):
        for t in bits_of(successors[q]):
            predecessors[t].append(q)
    backward = np.zeros(n, dtype=bool)
    stack = np.flatnonzero(is_accepting).tolist()
    for q in stack:
        backward[q] = True
    while stack:
        for p in predecessors[stack.pop()]:
            if not backward[p]:
                backward[p] = True
                stack.append(p)

    keep = forward & backward
    if not keep[start]:
        return None                                # no accepting run at all
    live = np.flatnonzero(keep)
    live_mask = _mask(live.tolist())

    # ---- forward bisimulation over the live states ----
    cls_of = np.full(n, -1, dtype=np.int64)
    cls_of[live] = is_accepting[live].astype(np.int64)
    parts = len(np.unique(cls_of[live]))
    while True:
        class_ids: Dict[int, int] = {}

        def class_mask(sid: int) -> int:
            targets = 0
            for q in bits_of(subsets[sid] & live_mask):
                targets |= 1 << int(cls_of[q])
            idx = class_ids.get(targets)
            if idx is None:
                idx = class_ids[targets] = len(class_ids)
            return idx

        cache: Dict[int, int] = {}
        signature = np.array(
            [store.apply1(int(nodes[q]), class_mask, cache)
             for q in live.tolist()], dtype=np.int64)
        _, refined = np.unique(np.stack([cls_of[live], signature], axis=1),
                               axis=0, return_inverse=True)
        refined = refined.reshape(-1).astype(np.int64)
        new_parts = len(np.unique(refined))
        cls_of[live] = refined
        if new_parts == parts:
            break
        parts = new_parts

    # ---- rebuild over the classes ----
    representative = np.zeros(parts, dtype=np.int64)
    representative[cls_of[live]] = live            # last wins; any will do
    new_subsets: List[int] = []
    new_ids: Dict[int, int] = {}

    def new_subset_id(mask: int) -> int:
        idx = new_ids.get(mask)
        if idx is None:
            idx = new_ids[mask] = len(new_subsets)
            new_subsets.append(mask)
        return idx

    def to_classes(sid: int) -> int:
        # dropped states vanish here, so a transition into them alone yields
        # the empty set of classes (a rejecting sink)
        targets = 0
        for q in bits_of(subsets[sid] & live_mask):
            targets |= 1 << int(cls_of[q])
        return new_subset_id(targets)

    cache = {}
    new_nodes = np.array([store.apply1(int(nodes[representative[c]]),
                                       to_classes, cache)
                          for c in range(parts)], dtype=np.int64)
    new_accepting = is_accepting[representative]
    return (new_nodes, new_subsets, new_ids, new_accepting,
            int(cls_of[start]))


def _determinize_set_nfa(store, nodes, subsets, subset_ids, is_accepting,
                         start: int, arity: int, m: int, bits: int,
                         base_alphabet, gc_threshold: int = 1 << 21
                         ) -> 'SparseDFA':
    """Subset construction over a (reduced) set-valued NFA.

    The construction runs in its own `NodeStore`. A subset's set-valued
    diagram is dead the moment `apply1` has relabelled it into the new
    automaton's diagram, but hash-consing keeps it alive for good — on a
    determinization of any size that garbage is the bulk of the store. A
    scratch store can be mark-swept: when it outgrows `gc_threshold`, keep
    only the NFA's diagrams and the finished states' diagrams, and drop the
    memos (which map node ids to node ids, so they cannot survive a
    renumbering). The threshold then doubles past the live set, which makes
    the sweeps amortized-linear.

    Only the finished automaton is re-interned into the shared store, so a
    projection no longer leaks its intermediates there either.
    """
    scratch = NodeStore()
    nfa_nodes = scratch.import_nodes(*store.export([int(n) for n in nodes]))
    floor = gc_threshold

    def subset_id(mask: int) -> int:
        idx = subset_ids.get(mask)
        if idx is None:
            idx = subset_ids[mask] = len(subsets)
            subsets.append(mask)
        return idx

    def union(a: int, b: int) -> int:
        if a == NONE or b == NONE:
            return NONE
        return subset_id(subsets[a] | subsets[b])

    dfa_subsets: List[int] = []
    dfa_ids: Dict[int, int] = {}

    def state_of(sid: int) -> int:
        mask = subsets[sid]
        idx = dfa_ids.get(mask)
        if idx is None:
            idx = dfa_ids[mask] = len(dfa_subsets)
            dfa_subsets.append(mask)
        return idx

    union_cache = ComputedTable()
    state_cache: Dict[int, int] = {}
    state_of(subset_id(1 << start))

    dfa_nodes: List[int] = []
    index = 0
    while index < len(dfa_subsets):                # grows inside apply1
        members = bits_of(dfa_subsets[index])
        self_id = index
        index += 1
        if not members:
            # pruning dead states lets a transition land on the empty set of
            # states, which is the rejecting sink
            dfa_nodes.append(scratch.const(self_id, arity, m, bits))
            continue
        node = int(nfa_nodes[members[0]])
        for q in members[1:]:
            node = scratch.apply2(node, int(nfa_nodes[q]), union, union_cache)
        dfa_nodes.append(scratch.apply1(node, state_of, state_cache))

        if len(scratch.var) > gc_threshold:
            split = len(nfa_nodes)
            live, renumber = scratch.collect(list(nfa_nodes) + dfa_nodes)
            nfa_nodes, dfa_nodes = live[:split], live[split:].tolist()
            # the memos map node ids to node ids, and a sweep renumbers them;
            # translating is far cheaper than re-deriving every fold
            union_cache = union_cache.remap(renumber)
            # apply1 memoizes node -> *node*, so both sides are renumbered
            state_cache = {renumber[source]: renumber[target]
                           for source, target in state_cache.items()
                           if source in renumber and target in renumber}
            # the live set grows with the automaton being built, so schedule
            # the next sweep relative to it instead of monotonically: peak
            # memory then stays a small factor above what is actually live
            gc_threshold = max(floor, (3 * len(scratch.var)) // 2)

    accepting_mask = _mask(np.flatnonzero(is_accepting).tolist())
    accepting = np.array([bool(mask & accepting_mask) for mask in dfa_subsets],
                         dtype=bool)
    # drop the scratch store before re-interning, or both copies of the live
    # diagrams are resident at once
    exported = scratch.export(dfa_nodes)
    del scratch, nfa_nodes, dfa_nodes, union_cache, state_cache
    return SparseDFA(len(dfa_subsets), is_accepting=accepting, start_state=0,
                     symbol_arity=arity, base_alphabet=base_alphabet,
                     nodes=store.import_nodes(*exported))


class SparseNFA:
    """Nondeterministic automaton whose states carry a diagram from symbols to
    *sets* of targets (terminals index `self.subsets`).

    The flat constructor keeps the historical shape — a base target per state
    plus ``symbol -> target`` exception rows, where the exceptions of a symbol
    *override* the base rather than adding to it. That is the semantics
    `determinize` always implemented; running the NFA directly now follows the
    same reading.
    """

    def __init__(self, num_states: int, base_state=(), exception_symbols=(),
                 exception_states=(), is_accepting=(), start_state: int = 0,
                 symbol_arity: int = 1,
                 base_alphabet: Optional[Set[int]] = None,
                 nodes=None, subsets=None):
        self.num_states = int(num_states)
        self.is_accepting = np.asarray(is_accepting, dtype=bool)
        self.start_state = int(start_state)
        self.symbol_arity = int(symbol_arity)
        self.base_alphabet = base_alphabet if base_alphabet is not None \
            else SparseDFA._infer_base_alphabet(exception_symbols)
        self.base_alphabet_frozen = frozenset(self.base_alphabet)

        self.store = STORE
        self.m = len(self.base_alphabet_frozen)
        self.bits = num_bits(self.m)

        if nodes is not None:
            self.nodes = np.asarray(nodes, dtype=np.int64)
            self.subsets = list(subsets)
            self._subset_ids = {s: i for i, s in enumerate(self.subsets)}
        else:
            self.subsets = []
            self._subset_ids: Dict[int, int] = {}
            self.nodes = self._compile(base_state, exception_symbols,
                                       exception_states)

    def subset_id(self, mask: int) -> int:
        """Intern a set of states, held as an integer bitset."""
        idx = self._subset_ids.get(mask)
        if idx is None:
            idx = self._subset_ids[mask] = len(self.subsets)
            self.subsets.append(mask)
        return idx

    def _compile(self, base_state, exception_symbols, exception_states):
        n = self.num_states
        bases = np.asarray(base_state, dtype=np.int64).reshape(-1)
        ex_syms = np.asarray(exception_symbols, dtype=np.int64)
        ex_targets = np.asarray(exception_states, dtype=np.int64)
        if ex_syms.size == 0:
            ex_syms = np.zeros((n, 0), dtype=np.int64)
            ex_targets = np.zeros((n, 0), dtype=np.int64)
        else:
            ex_syms = ex_syms.reshape(n, -1)
            ex_targets = ex_targets.reshape(n, -1)

        nodes = np.empty(n, dtype=np.int64)
        for q in range(n):
            keep = (ex_syms[q] >= 0) & (ex_targets[q] >= 0)
            by_symbol: Dict[int, set] = {}
            for s, t in zip(ex_syms[q][keep].tolist(),
                            ex_targets[q][keep].tolist()):
                by_symbol.setdefault(s, set()).add(t)
            symbols = np.array(sorted(by_symbol), dtype=np.int64)
            targets = np.array([self.subset_id(_mask(by_symbol[s]))
                                for s in symbols.tolist()], dtype=np.int64)
            base = self.subset_id(1 << int(bases[q]))
            nodes[q] = self.store.build_rows(symbols, targets, base,
                                             self.symbol_arity, self.m,
                                             self.bits)
        return nodes

    def _step(self, current_set: np.ndarray, symbol: int) -> np.ndarray:
        active = np.flatnonzero(current_set)
        reached = self.store.eval_batch(
            self.nodes[active], np.full(len(active), symbol, dtype=np.int64),
            self.symbol_arity, self.m, self.bits)
        next_set = np.zeros(self.num_states, dtype=bool)
        for sid in reached.tolist():
            next_set[bits_of(self.subsets[sid])] = True
        return next_set

    def compute(self, word: np.ndarray) -> np.ndarray:
        """Returns the set of states after processing the word"""
        current = np.zeros(self.num_states, dtype=bool)
        current[self.start_state] = True
        for symbol in np.asarray(word, dtype=np.int64).tolist():
            current = self._step(current, symbol)
        return current

    def accepts(self, word: np.ndarray) -> bool:
        return bool(np.any(self.compute(word) & self.is_accepting))

    def determinize(self) -> SparseDFA:
        """Subset construction on the diagrams: a subset's transition is the
        union of its members' diagrams, and the union of two set-valued
        diagrams is one `apply`. No symbol is enumerated.

        The NFA is first pruned and quotiented by forward bisimulation (see
        `reduce_set_nfa`), which is cheap and shrinks the subset space."""
        store = self.store
        arity, m, bits = self.symbol_arity, self.m, self.bits

        reduced = reduce_set_nfa(store, self.nodes, self.subsets,
                                 self.is_accepting, self.start_state,
                                 arity, m, bits)
        if reduced is None:
            return SparseDFA(1, is_accepting=[False], start_state=0,
                             symbol_arity=arity,
                             base_alphabet=self.base_alphabet,
                             nodes=np.array([store.const(0, arity, m, bits)]))
        nodes, subsets, subset_ids, accepting, start = reduced
        return _determinize_set_nfa(store, nodes, subsets, subset_ids,
                                    accepting, start, arity, m, bits,
                                    self.base_alphabet)

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

    def show_diagram(self, filename: str = "nfa", format: str = "png", view: bool = False) -> graphviz.Digraph:
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