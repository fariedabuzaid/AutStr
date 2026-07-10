"""A shared multi-terminal BDD store for transition symbols.

Sparse automata over convolution alphabets used to store transitions as flat
``(left, right, symbol) -> target`` rows. That representation forces every
operation to *enumerate symbols*: cylindrification duplicates each row once
per letter of every new tape, padding blankets multiply out, and a set
quantifier over a wide convolution produces exception tables with millions of
rows that all say the same thing about the same digits. This module replaces
the symbol column by a decision diagram over the *digits* of the symbol, which
is the representation MONA uses for the same reason.

**Encoding.** A symbol of a k-tape convolution over a base alphabet of size m
is the integer ``sum_t digit_t * m**(k-1-t)``. Each letter is written in
``bits = ceil(log2 m)`` binary variables, most significant first, and the
global variable order is *tape-major*: variable ``t*bits + j`` is bit j of
tape t. A tape therefore occupies a contiguous block of variables, so
existential projection quantifies a block, and cylindrification renames blocks.

**Nodes.** A node is either a terminal carrying an integer value (a state, or
a subset id, or a class id — the callers decide) or an internal node
``(var, lo, hi)`` whose children test strictly larger variables. Nodes are
reduced (``lo != hi``) and hash-consed in one process-wide store, so
structurally equal transition functions *are* the same node: equality of
behavior is an integer comparison, and the memo tables of `apply` are shared
across pairs of states and across automata.

**Invalid codes.** When m is not a power of two some binary codes denote no
letter. Every node built here maps those codes to the reserved terminal
``NONE``, and every operation propagates it, so the reachable-target set of a
node never contains a state that only an invalid code would reach — the
counting arguments the flat representation needed ("does this pair except all
m preimages of the symbol?") disappear.
"""
from array import array
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


TOP = 1 << 40          # variable index of a terminal node
NONE = -1              # terminal value of an invalid binary code
_INTERNAL = -2         # `term` entry of an internal node


def num_bits(m: int) -> int:
    """Binary variables per letter of an m-letter alphabet."""
    return max(1, (max(m, 1) - 1).bit_length())


class ComputedTable:
    """A memo for `apply` that is exact while small and *lossy* once large.

    A dict memo for `apply2` grows without bound: a set quantifier fills it
    with tens of millions of entries at roughly 180 bytes each, outweighing
    the nodes it caches. But most `apply` calls in a normal query are tiny,
    and a dict beats any hand-rolled table at that size.

    So: a plain dict until it exceeds `dict_limit` entries, then a
    direct-mapped array of (key, value) pairs at 16 bytes per slot that simply
    overwrites on collision. Memory is bounded by the table; a miss only
    recomputes a pure function.

    Correctness rests on `apply`'s terminal operations being deterministic
    functions of their arguments: recomputing an entry allocates no new state
    id or subset, it re-derives the same one. Only the *computed* tables may
    be lossy — the unique table `NodeStore._node_ids` must stay exact, or
    hash-consing breaks and node equality stops meaning function equality.
    """

    __slots__ = ("_dict", "_limit", "mask", "shift", "keys", "vals")

    _MIX = 0x9E3779B97F4A7C15
    _WORD = (1 << 64) - 1
    _EMPTY = -1                       # keys are non-negative packed node pairs

    def __init__(self, cap_log2: int = 23, dict_limit: int = 1 << 19) -> None:
        self._dict = {}
        self._limit = dict_limit
        size = 1 << cap_log2
        self.mask = size - 1
        self.shift = 64 - cap_log2
        self.keys = None
        self.vals = None

    def _slot(self, key: int) -> int:
        return (((key * self._MIX) & self._WORD) >> self.shift) & self.mask

    def _migrate(self) -> None:
        self.keys = array('q', b'\xff' * (8 * (self.mask + 1)))
        self.vals = array('q', bytes(8 * (self.mask + 1)))
        for key, value in self._dict.items():
            slot = self._slot(key)
            self.keys[slot] = key
            self.vals[slot] = value
        self._dict = None

    def get(self, key: int, default=None):
        if self._dict is not None:
            return self._dict.get(key, default)
        slot = self._slot(key)
        if self.keys[slot] == key:
            return self.vals[slot]
        return default

    def __setitem__(self, key: int, value: int) -> None:
        if self._dict is not None:
            self._dict[key] = value
            if len(self._dict) > self._limit:
                self._migrate()
            return
        slot = self._slot(key)
        self.keys[slot] = key
        self.vals[slot] = value

    def _entries(self):
        if self._dict is not None:
            return self._dict.items()
        return ((key, self.vals[slot])
                for slot, key in enumerate(self.keys) if key != self._EMPTY)

    def remap(self, mapping: Dict[int, int]) -> "ComputedTable":
        """A copy of this memo with every node id renumbered, dropping the
        entries whose operands or result did not survive a sweep."""
        fresh = ComputedTable()
        fresh._limit = self._limit
        fresh.mask, fresh.shift = self.mask, self.shift
        for key, value in self._entries():
            left = mapping.get(key >> 32)
            right = mapping.get(key & 0xFFFFFFFF)
            result = mapping.get(value)
            if left is None or right is None or result is None:
                continue
            fresh[(left << 32) | right] = result
        return fresh


def bits_of(mask: int) -> List[int]:
    """The set bit positions of an integer bitset, ascending.

    Subset constructions intern one set of states per union result — including
    every intermediate inside `apply2` — so the sets are held as python ints:
    a bitset costs ``num_states/8`` bytes and hashes in one pass, where a
    frozenset costs tens of bytes *per member*."""
    members = []
    while mask:
        low = mask & -mask
        members.append(low.bit_length() - 1)
        mask ^= low
    return members


class NodeStore:
    """Hash-consed multi-terminal BDD nodes with memoized operations.

    A set quantifier can create millions of nodes, so the per-node cost is
    part of the algorithm: `var`/`lo`/`hi`/`term` are `array('q')` (8 bytes
    each, against 36 for a python list slot holding a boxed int) and the
    unique table is keyed by a single packed integer rather than a 3-tuple.
    """

    _SHIFT = 1 << 34                       # node-id range of the packed key

    def __init__(self) -> None:
        self.var = array('q')
        self.lo = array('q')
        self.hi = array('q')
        self.term = array('q')
        self._terminal_ids: Dict[int, int] = {}
        self._node_ids: Dict[int, int] = {}
        self._cofactor: Dict[int, int] = {}
        self._mux: Dict[int, int] = {}
        self._terminals: Dict[int, Tuple[int, ...]] = {}
        self._const: Dict[Tuple[int, int, int], int] = {}
        self._arrays: Optional[Tuple[np.ndarray, ...]] = None

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        # node ids are indices into this store; copying it would strand every
        # automaton that holds them (callers deepcopy presentations)
        return self

    # ---------------- construction ----------------

    def terminal(self, value: int) -> int:
        node = self._terminal_ids.get(value)
        if node is None:
            node = self._terminal_ids[value] = len(self.var)
            self.var.append(TOP)
            self.lo.append(-1)
            self.hi.append(-1)
            self.term.append(int(value))
        return node

    def make(self, var: int, lo: int, hi: int) -> int:
        if lo == hi:
            return lo
        key = (var * self._SHIFT + lo) * self._SHIFT + hi
        node = self._node_ids.get(key)
        if node is None:
            node = self._node_ids[key] = len(self.var)
            self.var.append(var)
            self.lo.append(lo)
            self.hi.append(hi)
            self.term.append(_INTERNAL)
        return node

    def is_terminal(self, node: int) -> bool:
        return self.var[node] == TOP

    # ---------------- letters ----------------

    def letter(self, tape: int, children: Sequence[int], m: int,
               bits: int) -> int:
        """The binary decision tree over tape `tape`'s variable block that
        selects ``children[d]`` on digit d; codes d >= m lead to NONE."""
        invalid = self.terminal(NONE)

        def build(j: int, prefix: int) -> int:
            if j == bits:
                return children[prefix] if prefix < m else invalid
            return self.make(tape * bits + j,
                             build(j + 1, prefix << 1),
                             build(j + 1, (prefix << 1) | 1))

        return build(0, 0)

    def const(self, value: int, arity: int, m: int, bits: int) -> int:
        """The node mapping every valid symbol of an `arity`-tape convolution
        to `value` (and every invalid code to NONE)."""
        key = (value, arity, m)
        node = self._const.get(key)
        if node is None:
            node = self.terminal(value)
            for tape in range(arity - 1, -1, -1):
                node = self.letter(tape, [node] * m, m, bits)
            self._const[key] = node
        return node

    def build_rows(self, symbols: np.ndarray, targets: np.ndarray,
                   base: int, arity: int, m: int, bits: int) -> int:
        """The node of a transition function given as a base value plus a
        sorted list of ``symbol -> target`` deviations."""
        # the all-base node over tapes tape..arity-1 (variables tape*bits..)
        suffix: Dict[int, int] = {}

        def all_base(tape: int) -> int:
            node = suffix.get(tape)
            if node is None:
                node = self.terminal(base)
                for t in range(arity - 1, tape - 1, -1):
                    node = self.letter(t, [node] * m, m, bits)
                suffix[tape] = node
            return node

        def build(tape: int, lo: int, hi: int) -> int:
            if lo >= hi:
                return all_base(tape)
            if tape == arity:
                return self.terminal(int(targets[lo]))
            div = m ** (arity - 1 - tape)
            digits = (symbols[lo:hi] // div) % m
            cuts = np.searchsorted(digits, np.arange(m + 1))
            children = [build(tape + 1, lo + int(cuts[d]), lo + int(cuts[d + 1]))
                        for d in range(m)]
            return self.letter(tape, children, m, bits)

        return build(0, 0, len(symbols))

    def map_letters(self, node: int, arity: int, old_m: int, old_bits: int,
                    new_m: int, new_bits: int, source: Sequence[int],
                    fill: int) -> int:
        """Re-express a diagram over a different base alphabet.

        `source[d]` names the *old* letter that the new letter d should behave
        like, or -1 to send it to `fill`. The map runs from the new alphabet to
        the old one, so it need not be injective: several new letters may share
        an old one. That is what embeds a factor into a product's pair
        alphabet, where every pair (a, b) behaves like its own component.

        Per tape, take the old block's `old_m` cofactors and reassemble them at
        the new digits. One memoized pass over the nodes -- widening an
        alphabet never rebuilds a transition table, which is why a direct
        product is affordable: the pair alphabet has |A|*|B| letters but its
        diagrams have bits_A + bits_B variables. Letters multiply; bits add.
        """
        if len(source) != new_m:
            raise ValueError("source must give one old letter per new letter")
        if any(d >= old_m for d in source):
            raise ValueError("source names a letter outside the old alphabet")

        suffix: Dict[int, int] = {}

        def all_fill(tape: int) -> int:
            node = suffix.get(tape)
            if node is None:
                node = self.terminal(fill)
                for t in range(arity - 1, tape - 1, -1):
                    node = self.letter(t, [node] * new_m, new_m, new_bits)
                suffix[tape] = node
            return node

        cache: Dict[Tuple[int, int], int] = {}

        def rebuild(current: int, tape: int) -> int:
            if tape == arity:
                return current                     # a terminal: nothing to map
            key = (current, tape)
            result = cache.get(key)
            if result is not None:
                return result
            cofactors: Dict[int, int] = {}

            def old_branch(digit: int) -> int:
                branch = cofactors.get(digit)
                if branch is None:
                    branch = current
                    for j in range(old_bits):
                        bit = (digit >> (old_bits - 1 - j)) & 1
                        branch = self.cofactor(branch, tape * old_bits + j, bit)
                    cofactors[digit] = branch
                return branch

            children = []
            for digit in range(new_m):
                old = source[digit]
                children.append(all_fill(tape + 1) if old < 0
                                else rebuild(old_branch(old), tape + 1))
            result = cache[key] = self.letter(tape, children, new_m, new_bits)
            return result

        return rebuild(node, 0)

    def recode_letters(self, node: int, arity: int, old_m: int, old_bits: int,
                       new_m: int, new_bits: int, digit_map: Sequence[int],
                       fill: int) -> int:
        """Injective relabelling: old letter d becomes new letter
        ``digit_map[d]``; new letters outside the image go to `fill`."""
        if len(set(digit_map)) != len(digit_map):
            raise ValueError("digit_map must be injective")
        if any(not 0 <= d < new_m for d in digit_map):
            raise ValueError("digit_map must land in the new alphabet")
        source = [-1] * new_m
        for old, new in enumerate(digit_map):
            source[new] = old
        return self.map_letters(node, arity, old_m, old_bits, new_m, new_bits,
                                source, fill)

    def set_path(self, node: int, assignment: Sequence[int], value: int) -> int:
        """The node that agrees with `node` everywhere except on the single
        full variable assignment `assignment`, where it takes `value`."""
        nvars = len(assignment)

        def walk(x: int, i: int) -> int:
            if i == nvars:
                return self.terminal(value)
            if self.var[x] == i:
                lo, hi = self.lo[x], self.hi[x]
            else:                                   # variable i is skipped
                lo = hi = x
            if assignment[i]:
                hi = walk(hi, i + 1)
            else:
                lo = walk(lo, i + 1)
            return self.make(i, lo, hi)

        return walk(node, 0)

    # ---------------- operations ----------------

    def cofactor(self, node: int, var: int, bit: int) -> int:
        v = self.var[node]
        if v > var:
            return node
        if v == var:
            return self.hi[node] if bit else self.lo[node]
        key = (node * 1024 + var) * 2 + bit
        result = self._cofactor.get(key)
        if result is None:
            result = self._cofactor[key] = self.make(
                v, self.cofactor(self.lo[node], var, bit),
                self.cofactor(self.hi[node], var, bit))
        return result

    def mux(self, var: int, on_high: int, on_low: int) -> int:
        """The node that behaves like `on_high` where `var` is 1 and like
        `on_low` where it is 0 — `var` may sit below the arguments' roots, in
        which case they are pushed down."""
        key = (var * self._SHIFT + on_high) * self._SHIFT + on_low
        result = self._mux.get(key)
        if result is not None:
            return result
        vh, vl = self.var[on_high], self.var[on_low]
        w = vh if vh < vl else vl
        if w < var:
            h0, h1 = (self.lo[on_high], self.hi[on_high]) if vh == w \
                else (on_high, on_high)
            l0, l1 = (self.lo[on_low], self.hi[on_low]) if vl == w \
                else (on_low, on_low)
            result = self.make(w, self.mux(var, h0, l0), self.mux(var, h1, l1))
        else:
            result = self.make(var, self.cofactor(on_low, var, 0),
                               self.cofactor(on_high, var, 1))
        self._mux[key] = result
        return result

    def rename(self, node: int, varmap: Sequence[int],
               cache: Dict[int, int]) -> int:
        """Substitute variable v by variable ``varmap[v]``. The map need not
        be monotone or injective: identifying two variables restricts the
        function to their diagonal (which is how a relation R(x, x) is built).
        Monotone maps cost one node per node."""
        if self.var[node] == TOP:
            return node
        result = cache.get(node)
        if result is None:
            result = cache[node] = self.mux(
                varmap[self.var[node]],
                self.rename(self.hi[node], varmap, cache),
                self.rename(self.lo[node], varmap, cache))
        return result

    def apply2(self, f: int, g: int, op, cache) -> int:
        """Pointwise combination of two nodes; `op` acts on terminal values.

        `cache` is a memo keyed by the packed node pair: a dict, or a bounded
        `ComputedTable` when the pair space is large enough that an exact memo
        would outweigh the nodes it caches."""
        key = (f << 32) | g
        result = cache.get(key)
        if result is not None:
            return result
        vf, vg = self.var[f], self.var[g]
        if vf == TOP and vg == TOP:
            result = self.terminal(op(self.term[f], self.term[g]))
        else:
            v = vf if vf < vg else vg
            f0, f1 = (self.lo[f], self.hi[f]) if vf == v else (f, f)
            g0, g1 = (self.lo[g], self.hi[g]) if vg == v else (g, g)
            result = self.make(v, self.apply2(f0, g0, op, cache),
                               self.apply2(f1, g1, op, cache))
        cache[key] = result
        return result

    def apply1(self, f: int, fn, cache: Dict[int, int]) -> int:
        """Relabel the terminals of a node."""
        result = cache.get(f)
        if result is not None:
            return result
        if self.var[f] == TOP:
            value = self.term[f]
            result = f if value == NONE else self.terminal(fn(value))
        else:
            result = self.make(self.var[f],
                               self.apply1(self.lo[f], fn, cache),
                               self.apply1(self.hi[f], fn, cache))
        cache[f] = result
        return result

    def quantify_letter(self, node: int, tape: int, m: int, bits: int, op,
                        cache) -> int:
        """Combine the m cofactors of `node` on tape `tape`'s letter with
        `op` — the tape's variables no longer occur in the result."""
        result = None
        for digit in range(m):
            branch = node
            for j in range(bits):
                bit = (digit >> (bits - 1 - j)) & 1
                branch = self.cofactor(branch, tape * bits + j, bit)
            result = branch if result is None \
                else self.apply2(result, branch, op, cache)
        return result

    # ---------------- inspection ----------------

    def terminals(self, node: int) -> Tuple[int, ...]:
        """The terminal values reachable in `node`, sorted, without NONE."""
        memo = self._terminals
        cached = memo.get(node)
        if cached is not None:
            return cached
        stack = [node]
        while stack:
            x = stack[-1]
            if x in memo:
                stack.pop()
                continue
            if self.var[x] == TOP:
                value = self.term[x]
                memo[x] = () if value == NONE else (value,)
                stack.pop()
                continue
            lo, hi = self.lo[x], self.hi[x]
            missing = [c for c in (lo, hi) if c not in memo]
            if missing:
                stack.extend(missing)
                continue
            memo[x] = tuple(sorted(set(memo[lo]) | set(memo[hi])))
            stack.pop()
        return memo[node]

    def reset(self) -> None:
        """Drop every node and memo. Only valid on a scratch store: node ids
        are indices, so any surviving holder of one is left dangling."""
        self.var = array('q')
        self.lo = array('q')
        self.hi = array('q')
        self.term = array('q')
        self._terminal_ids = {}
        self._node_ids = {}
        self._cofactor = {}
        self._mux = {}
        self._terminals = {}
        self._const = {}
        self._arrays = None

    def collect(self, roots):
        """Mark-sweep this store down to the sub-DAG below `roots`. A subset
        construction abandons the set-valued diagram of every subset as soon as
        it has been relabelled, but hash-consing keeps it forever; on a scratch
        store those nodes can be reclaimed.

        Node ids are indices, so a sweep renumbers everything. Returns the
        roots' new ids together with the old -> new map of every surviving
        node, which the caller needs to translate its `apply` memos (they map
        ids to ids, and dropping them instead makes the sweep cost more in
        recomputation than it saves in memory).
        """
        var, lo, hi, term, local = self.export(roots)
        old_ids = self._exported_order
        self.reset()
        renumbered = self._intern_all(var, lo, hi, term)
        return renumbered[local], dict(zip(old_ids, renumbered.tolist()))

    def export(self, roots) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                     np.ndarray, np.ndarray]:
        """Extract the sub-DAG below `roots` as standalone arrays, renumbered
        so that children precede parents. Returns (var, lo, hi, term, roots)."""
        local: Dict[int, int] = {}
        var, lo, hi, term = [], [], [], []
        stack = [(int(r), False) for r in roots]
        while stack:
            node, expanded = stack.pop()
            if node in local:
                continue
            if self.var[node] == TOP:
                local[node] = len(var)
                var.append(TOP)
                lo.append(-1)
                hi.append(-1)
                term.append(self.term[node])
                continue
            if not expanded:
                stack.append((node, True))
                stack.append((self.lo[node], False))
                stack.append((self.hi[node], False))
            else:
                local[node] = len(var)
                var.append(self.var[node])
                lo.append(local[self.lo[node]])
                hi.append(local[self.hi[node]])
                term.append(_INTERNAL)
        order = [0] * len(var)
        for original, index in local.items():
            order[index] = original
        self._exported_order = order
        return (np.array(var, dtype=np.int64), np.array(lo, dtype=np.int64),
                np.array(hi, dtype=np.int64), np.array(term, dtype=np.int64),
                np.array([local[int(r)] for r in roots], dtype=np.int64))

    def _intern_all(self, var, lo, hi, term) -> np.ndarray:
        """Intern an exported sub-DAG (children first); returns local -> id."""
        mapping = np.empty(len(var), dtype=np.int64)
        for i in range(len(var)):
            if var[i] == TOP:
                mapping[i] = self.terminal(int(term[i]))
            else:
                mapping[i] = self.make(int(var[i]), int(mapping[lo[i]]),
                                       int(mapping[hi[i]]))
        return mapping

    def import_nodes(self, var, lo, hi, term, roots) -> np.ndarray:
        """Re-intern an exported sub-DAG (children first) into this store."""
        return self._intern_all(var, lo, hi, term)[
            np.asarray(roots, dtype=np.int64)]

    def size(self, roots) -> int:
        """Number of distinct nodes below the given roots."""
        seen = set()
        stack = list(roots)
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            if self.var[x] != TOP:
                stack.append(self.lo[x])
                stack.append(self.hi[x])
        return len(seen)

    # ---------------- evaluation ----------------

    def _numpy(self):
        if self._arrays is None or len(self._arrays[0]) != len(self.var):
            self._arrays = (np.array(self.var, dtype=np.int64),
                            np.array(self.lo, dtype=np.int64),
                            np.array(self.hi, dtype=np.int64),
                            np.array(self.term, dtype=np.int64))
        return self._arrays

    def eval_batch(self, nodes: np.ndarray, symbols: np.ndarray,
                   arity: int, m: int, bits: int) -> np.ndarray:
        """Terminal value of each node at its symbol (batched descent)."""
        var, lo, hi, term = self._numpy()
        div, shift = var_tables(arity, m, bits)
        nodes = np.array(nodes, dtype=np.int64, copy=True)
        while True:
            active = np.flatnonzero(var[nodes] != TOP)
            if len(active) == 0:
                break
            cur = nodes[active]
            v = var[cur]
            bit = ((symbols[active] // div[v]) % m) >> shift[v] & 1
            nodes[active] = np.where(bit == 1, hi[cur], lo[cur])
        return term[nodes]


def var_tables(arity: int, m: int, bits: int):
    """Per-variable digit divisor and bit shift for symbol decoding."""
    key = (arity, m, bits)
    tables = _VAR_TABLES.get(key)
    if tables is None:
        variables = np.arange(arity * bits, dtype=np.int64)
        div = m ** (arity - 1 - variables // bits)
        shift = bits - 1 - variables % bits
        tables = _VAR_TABLES[key] = (div, shift)
    return tables


_VAR_TABLES: Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]] = {}

STORE = NodeStore()
