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
        self._cofactor: Dict[Tuple[int, int, int], int] = {}
        self._mux: Dict[Tuple[int, int, int], int] = {}
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
        key = (node, var, bit)
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
        key = (var, on_high, on_low)
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

    def apply2(self, f: int, g: int, op, cache: Dict[Tuple[int, int], int]
               ) -> int:
        """Pointwise combination of two nodes; `op` acts on terminal values."""
        key = (f, g)
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
                        cache: Dict[Tuple[int, int], int]) -> int:
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
        return (np.array(var, dtype=np.int64), np.array(lo, dtype=np.int64),
                np.array(hi, dtype=np.int64), np.array(term, dtype=np.int64),
                np.array([local[int(r)] for r in roots], dtype=np.int64))

    def import_nodes(self, var, lo, hi, term, roots) -> np.ndarray:
        """Re-intern an exported sub-DAG (children first) into this store."""
        mapping = np.empty(len(var), dtype=np.int64)
        for i in range(len(var)):
            if var[i] == TOP:
                mapping[i] = self.terminal(int(term[i]))
            else:
                mapping[i] = self.make(int(var[i]), int(mapping[lo[i]]),
                                       int(mapping[hi[i]]))
        return mapping[np.asarray(roots, dtype=np.int64)]

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
