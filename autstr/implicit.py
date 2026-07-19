"""Implicit first-order evaluation, without compiling a query automaton.

The explicit path (`presentations.AutomaticPresentation._build_automaton`) turns
a formula into a product automaton -- intersection for AND, union for OR,
complement for NOT, projection for EXISTS -- and materialises it. That product
blows up, and for the heavy group classes even the *base* multiplication
automaton is infeasible to build (an O(sigma^4) product over the astronomical
advice alphabet).

This module evaluates a formula *implicitly* instead: it keeps composite states
and steps the base automata on the fly. Boolean connectives take the product of
states; EXISTS is handled by an on-the-fly powerset (subset construction); NOT
flips the acceptance test (exact, because every composite stays deterministic
and total). The base automata are given by a small functional interface, so a
class whose presentation cannot even be built is still checkable -- this is the
`simulate` methods of the group classes generalised to arbitrary first-order
formulas.

Everything here works over a *fixed* input: an advice word/tree plus concrete
assignments for some free variables; the remaining free variables are
existentially closed and eliminated by the powerset construction. Because the
advice is fixed input and quantifiers range over element tapes whose per-symbol
alphabet is tiny, the subset construction never touches the huge advice
alphabet; the cost is set by quantifier alternation, not alphabet size.

Two shapes, one combinator pattern:
  * `ImplicitDFA`   -- string automata, run left-to-right.
  * `ImplicitTA`    -- bottom-up tree automata, run post-order.

Beyond the boolean `check_*` entry points, the module offers the
*satisfying-set primitive*: `StringSolutionSet`/`TreeSolutionSet` compute, for
a formula with open free variables over a fixed advice, the exact number of
satisfying assignments (a count DP over the reachable composite states — no
enumeration) and lazily enumerate them. `ImplicitClass`/`ImplicitTreeClass`
package atoms + element alphabet as a first-class *fully implicit
presentation*: a uniformly automatic class given purely functionally, offering
model checking and satisfying-set evaluation and never compiling anything.
"""
import itertools as it
from typing import Callable, Dict, FrozenSet, Iterable, Sequence

from nltk.sem import logic

from autstr.sparse_tree_automata import Tree
from autstr.utils.logic import get_free_elementary_vars
from autstr.utils.misc import encode_symbol, get_unique_id


# ======================================================================
# String automata
# ======================================================================

class ImplicitDFA:
    """A deterministic, total automaton over a set of named tapes, given
    functionally. A symbol is a dict {tape: value}; the automaton reads only its
    own `tapes` from it."""

    def __init__(self, tapes: Iterable[str],
                 initial: Callable[[], object],
                 step: Callable[[object, Dict[str, object]], object],
                 accepting: Callable[[object], bool]):
        self.tapes: FrozenSet[str] = frozenset(tapes)
        self._initial = initial
        self._step = step
        self._accepting = accepting

    def initial(self):
        return self._initial()

    def step(self, state, symbol: Dict[str, object]):
        return self._step(state, symbol)

    def accepting(self, state) -> bool:
        return self._accepting(state)


def dfa_atom(dfa, tapes: Sequence[str]) -> ImplicitDFA:
    """Wrap an explicit `SparseDFA` as an implicit atom. `tapes` names the DFA's
    tapes in the DFA's own symbol order."""
    tapes = list(tapes)

    def step(state, symbol):
        enc = dfa.encode_symbol(tuple(symbol[t] for t in tapes))
        return dfa.transition(state, enc)

    return ImplicitDFA(tapes, lambda: dfa.start_state, step,
                       lambda s: bool(dfa.is_accepting[s]))


def dfa_product(a: ImplicitDFA, b: ImplicitDFA,
                accept: Callable[[bool, bool], bool]) -> ImplicitDFA:
    def step(state, symbol):
        return (a.step(state[0], symbol), b.step(state[1], symbol))

    return ImplicitDFA(a.tapes | b.tapes,
                       lambda: (a.initial(), b.initial()), step,
                       lambda s: accept(a.accepting(s[0]), b.accepting(s[1])))


def dfa_complement(a: ImplicitDFA) -> ImplicitDFA:
    return ImplicitDFA(a.tapes, a.initial, a.step, lambda s: not a.accepting(s))


def dfa_project(a: ImplicitDFA, var: str, alphabet: Sequence) -> ImplicitDFA:
    """EXISTS var: subset construction. State is a frozenset of a-states; each
    step guesses `var`'s symbol over `alphabet` from every state in the set."""
    alphabet = list(alphabet)

    def step(states, symbol):
        out = set()
        for s in states:
            for x in alphabet:
                out.add(a.step(s, {**symbol, var: x}))
        return frozenset(out)

    return ImplicitDFA(a.tapes - {var},
                       lambda: frozenset({a.initial()}), step,
                       lambda states: any(a.accepting(s) for s in states))


def run_dfa(a: ImplicitDFA, inputs: Dict[str, Sequence], length: int) -> bool:
    """Run over `length` synchronised positions; `inputs` gives a word for each
    of the automaton's remaining tapes."""
    state = a.initial()
    for i in range(length):
        state = a.step(state, {t: inputs[t][i] for t in a.tapes})
    return a.accepting(state)


# ======================================================================
# Tree automata (bottom-up)
# ======================================================================

class ImplicitTA:
    """A deterministic bottom-up tree automaton over named tapes. `step` takes
    the node's symbol dict and the child states (None for a missing child)."""

    def __init__(self, tapes: Iterable[str],
                 step: Callable[[Dict[str, object], object, object], object],
                 accepting: Callable[[object], bool]):
        self.tapes: FrozenSet[str] = frozenset(tapes)
        self._step = step
        self._accepting = accepting

    def step(self, symbol, left, right):
        return self._step(symbol, left, right)

    def accepting(self, state) -> bool:
        return self._accepting(state)


def ta_atom(sta, tapes: Sequence[str]) -> ImplicitTA:
    """Wrap an explicit `SparseTreeAutomaton` as an implicit atom."""
    tapes = list(tapes)
    bot = sta.BOT

    def step(symbol, left, right):
        enc = encode_symbol(tuple(symbol[t] for t in tapes),
                            sta.base_alphabet_frozen)
        l = bot if left is None else left
        r = bot if right is None else right
        return int(sta.transitions([l], [r], [enc])[0])

    return ImplicitTA(tapes, step, lambda s: bool(sta.is_accepting[s]))


def ta_product(a: ImplicitTA, b: ImplicitTA,
               accept: Callable[[bool, bool], bool]) -> ImplicitTA:
    def split(state):
        return (None, None) if state is None else state

    def step(symbol, left, right):
        la, lb = split(left)
        ra, rb = split(right)
        return (a.step(symbol, la, ra), b.step(symbol, lb, rb))

    return ImplicitTA(a.tapes | b.tapes, step,
                      lambda s: accept(a.accepting(s[0]), b.accepting(s[1])))


def ta_complement(a: ImplicitTA) -> ImplicitTA:
    return ImplicitTA(a.tapes, a.step, lambda s: not a.accepting(s))


def ta_project(a: ImplicitTA, var: str, alphabet: Sequence) -> ImplicitTA:
    """EXISTS var: bottom-up subset construction. State is a frozenset of
    a-states; a missing child contributes the singleton {None}."""
    alphabet = list(alphabet)

    def step(symbol, left, right):
        lefts = [None] if left is None else list(left)
        rights = [None] if right is None else list(right)
        out = set()
        for x in alphabet:
            sym = {**symbol, var: x}
            for la in lefts:
                for ra in rights:
                    out.add(a.step(sym, la, ra))
        return frozenset(out)

    return ImplicitTA(a.tapes - {var}, step,
                      lambda states: any(a.accepting(s) for s in states))


def run_ta(a: ImplicitTA, inputs: Dict[str, object]) -> bool:
    """Run over labelled trees (one per tape, all of the same shape)."""
    tapes = list(a.tapes)

    def rec(node_map):
        anynode = next(n for n in node_map.values() if n is not None)
        symbol = {t: node_map[t].label for t in tapes}
        left = (rec({t: node_map[t].left for t in tapes})
                if anynode.left is not None else None)
        right = (rec({t: node_map[t].right for t in tapes})
                 if anynode.right is not None else None)
        return a.step(symbol, left, right)

    return a.accepting(rec({t: inputs[t] for t in tapes}))


# ======================================================================
# Satisfying sets (the evaluate_implicit primitive)
# ======================================================================

class StringSolutionSet:
    """The satisfying assignments for the open variables of an implicit
    string automaton over a fixed input.

    One forward pass stores, per position, the reachable composite states
    with their outgoing edges (one per guessed symbol tuple); one backward
    pass counts the accepted suffixes per state. `len` is then the exact
    number of satisfying assignments without any enumeration, truthiness is
    non-emptiness, and iteration lazily yields `{var: word}` dicts (each
    word a list of element symbols, one per position). Deterministic
    composite states make the count exact: every assignment has exactly one
    run."""

    def __init__(self, a: ImplicitDFA, inputs: Dict[str, Sequence],
                 length: int, solve_vars: Sequence[str], alphabet_of):
        self.variables = sorted(solve_vars)
        self._length = length
        self._tuples = list(it.product(
            *[list(alphabet_of(v)) for v in self.variables]))
        fixed = [t for t in a.tapes if t not in set(self.variables)]
        init = a.initial()
        layer = {init}
        edges = []              # edges[i]: {state: [(tuple, next_state)]}
        for i in range(length):
            base = {t: inputs[t][i] for t in fixed}
            out = {}
            nxt = set()
            for s in layer:
                row = []
                for tup in self._tuples:
                    s2 = a.step(s, {**base,
                                    **dict(zip(self.variables, tup))})
                    row.append((tup, s2))
                    nxt.add(s2)
                out[s] = row
            edges.append(out)
            layer = nxt
        cnt = {s: 1 for s in layer if a.accepting(s)}
        counts = [cnt]
        for i in range(length - 1, -1, -1):
            prev = {}
            for s, row in edges[i].items():
                total = sum(cnt.get(s2, 0) for _, s2 in row)
                if total:
                    prev[s] = total
            counts.append(prev)
            cnt = prev
        counts.reverse()
        self._edges = edges
        self._counts = counts
        self._init = init

    def __len__(self):
        return self._counts[0].get(self._init, 0)

    def __bool__(self):
        return len(self) > 0

    def __iter__(self):
        if not len(self):
            return
        words = {v: [] for v in self.variables}

        def rec(i, s):
            if i == self._length:
                yield {v: list(w) for v, w in words.items()}
                return
            nxt = self._counts[i + 1]
            for tup, s2 in self._edges[i][s]:
                if nxt.get(s2, 0):
                    for v, x in zip(self.variables, tup):
                        words[v].append(x)
                    yield from rec(i + 1, s2)
                    for v in self.variables:
                        words[v].pop()

        yield from rec(0, self._init)


class TreeSolutionSet:
    """The satisfying assignments for the open variables of an implicit
    bottom-up tree automaton over a fixed input; assignments are labelled
    trees of the input's exact shape. A bottom-up pass counts, per node and
    reachable state, the subtree labelings that reach it; `len` sums the
    accepting root states, iteration re-derives the labelings top-down."""

    def __init__(self, a: ImplicitTA, inputs: Dict[str, object],
                 solve_vars: Sequence[str], alphabet_of):
        self.variables = sorted(solve_vars)
        self._a = a
        self._tuples = list(it.product(
            *[list(alphabet_of(v)) for v in self.variables]))
        self._fixed = [t for t in a.tapes if t not in set(self.variables)]
        self._tables = {}       # id(shape node) -> {state: count}
        self._root = {t: inputs[t] for t in self._fixed}
        root_tab = self._table(self._root)
        self._total = sum(cn for s, cn in root_tab.items()
                          if a.accepting(s))
        self._root_tab = root_tab

    def _maps(self, node_map):
        anynode = node_map[self._fixed[0]]
        lmap = ({t: node_map[t].left for t in self._fixed}
                if anynode.left is not None else None)
        rmap = ({t: node_map[t].right for t in self._fixed}
                if anynode.right is not None else None)
        return anynode, lmap, rmap

    def _table(self, node_map):
        anynode, lmap, rmap = self._maps(node_map)
        ltab = self._table(lmap) if lmap else {None: 1}
        rtab = self._table(rmap) if rmap else {None: 1}
        base = {t: node_map[t].label for t in self._fixed}
        out = {}
        for tup in self._tuples:
            sym = {**base, **dict(zip(self.variables, tup))}
            for l, cl in ltab.items():
                for r, cr in rtab.items():
                    s = self._a.step(sym, l, r)
                    out[s] = out.get(s, 0) + cl * cr
        self._tables[id(anynode)] = out
        return out

    def __len__(self):
        return self._total

    def __bool__(self):
        return self._total > 0

    def _enum(self, node_map, target):
        anynode, lmap, rmap = self._maps(node_map)
        ltab = (self._tables[id(lmap[self._fixed[0]])] if lmap
                else {None: 1})
        rtab = (self._tables[id(rmap[self._fixed[0]])] if rmap
                else {None: 1})
        base = {t: node_map[t].label for t in self._fixed}
        for tup in self._tuples:
            sym = {**base, **dict(zip(self.variables, tup))}
            for l in ltab:
                for r in rtab:
                    if self._a.step(sym, l, r) != target:
                        continue
                    for lsol in (self._enum(lmap, l) if lmap else ({},)):
                        for rsol in (self._enum(rmap, r) if rmap else ({},)):
                            yield {v: Tree(x, lsol.get(v), rsol.get(v))
                                   for v, x in zip(self.variables, tup)}

    def __iter__(self):
        for s, cn in self._root_tab.items():
            if cn and self._a.accepting(s):
                yield from self._enum(self._root, s)


class MappedSolutions:
    """A solution set with a mapper applied to every assignment value
    (e.g. decoding element words/trees back to group elements)."""

    def __init__(self, base, mapper):
        self._base = base
        self._mapper = mapper
        self.variables = base.variables

    def __len__(self):
        return len(self._base)

    def __bool__(self):
        return bool(self._base)

    def __iter__(self):
        for sol in self._base:
            yield {v: self._mapper(w) for v, w in sol.items()}


# ======================================================================
# The AST evaluator (shared by strings and trees)
# ======================================================================

class _Combinators:
    """Bundles the shape-specific combinators so the AST walk is written once."""

    def __init__(self, atom, product, complement, project):
        self.atom = atom              # (relation_name, arg_tapes) -> automaton
        self.product = product        # (a, b, accept_fn) -> automaton
        self.complement = complement  # a -> automaton
        self.project = project        # (a, var, alphabet) -> automaton


def _build(phi: logic.Expression, comb: _Combinators, alphabet_of):
    """Build the composite implicit automaton for a (relativized) formula."""
    if isinstance(phi, logic.ApplicationExpression):
        name = str(phi.pred)
        args = [str(a) for a in phi.args]
        return comb.atom(name, args)
    if isinstance(phi, logic.AndExpression):
        return comb.product(_build(phi.first, comb, alphabet_of),
                            _build(phi.second, comb, alphabet_of),
                            lambda x, y: x and y)
    if isinstance(phi, logic.OrExpression):
        return comb.product(_build(phi.first, comb, alphabet_of),
                            _build(phi.second, comb, alphabet_of),
                            lambda x, y: x or y)
    if isinstance(phi, logic.ImpExpression):
        return comb.product(comb.complement(_build(phi.first, comb, alphabet_of)),
                            _build(phi.second, comb, alphabet_of),
                            lambda x, y: x or y)
    if isinstance(phi, logic.IffExpression):
        a = _build(phi.first, comb, alphabet_of)
        b = _build(phi.second, comb, alphabet_of)
        return comb.product(a, b, lambda x, y: x == y)
    if isinstance(phi, logic.NegatedExpression):
        return comb.complement(_build(phi.term, comb, alphabet_of))
    if isinstance(phi, logic.ExistsExpression):
        var = str(phi.variable)
        inner = _build(phi.term, comb, alphabet_of)
        return comb.project(inner, var, alphabet_of(var))
    if isinstance(phi, logic.AllExpression):
        var = str(phi.variable)
        inner = _build(phi.term, comb, alphabet_of)
        # forall x. phi  ==  not exists x. not phi
        return comb.complement(
            comb.project(comb.complement(inner), var, alphabet_of(var)))
    raise ValueError(f"unsupported expression: {type(phi).__name__}")


def check_string(phi, atoms: Dict, inputs: Dict[str, Sequence], length: int,
                 alphabet_of) -> bool:
    """Implicitly decide a (relativized) formula over string automata.

    :param atoms: relation name -> an atom builder ``args -> ImplicitDFA`` (or a
        `SparseDFA`, wrapped via `dfa_atom` with the formula's argument order).
    :param inputs: a word for every tape that survives to the top (advice and
        assigned variables).
    :param alphabet_of: var name -> iterable of element symbols to guess for it.
    """
    comb = _Combinators(
        atom=lambda name, args: _mk_atom(atoms[name], args, dfa_atom),
        product=dfa_product, complement=dfa_complement, project=dfa_project)
    return run_dfa(_build(phi, comb, alphabet_of), inputs, length)


def check_tree(phi, atoms: Dict, inputs: Dict[str, object], alphabet_of) -> bool:
    """Implicitly decide a (relativized) formula over tree automata.

    :param atoms: relation name -> an atom builder ``args -> ImplicitTA`` (or a
        `SparseTreeAutomaton`, wrapped via `ta_atom`).
    :param inputs: a labelled tree for every surviving tape (all same shape).
    """
    comb = _Combinators(
        atom=lambda name, args: _mk_atom(atoms[name], args, ta_atom),
        product=ta_product, complement=ta_complement, project=ta_project)
    return run_ta(_build(phi, comb, alphabet_of), inputs)


def _mk_atom(entry, args: Sequence[str], wrap):
    """An atom entry is either a functional builder ``args -> automaton`` or an
    explicit automaton to wrap over the formula's argument order."""
    return entry(args) if callable(entry) else wrap(entry, args)


# ======================================================================
# Class-level entry points (advice-relativized model checking)
# ======================================================================

def relativized_query(phi, assignments, relativize, variable_names,
                      close_free=True):
    """Relativize to a fresh advice variable and add the Adv/Dom guards --
    the same query the explicit `evaluate`/`check` build, but returned as an
    expression for implicit evaluation. With `close_free` (the model-checking
    contract) unassigned free variables are existentially closed; without it
    (the satisfying-set contract) they stay open and are returned as the
    solve variables.

    :param relativize: `UniformlyAutomaticClass._relativize` (static).
    :param variable_names: `UniformlyAutomaticClass._variable_names` (static).
    :returns: (query expression, advice variable name, assigned variable
        names, solve variable names).
    """
    if isinstance(phi, str):
        phi = logic.Expression.fromstring(phi)
    free = get_free_elementary_vars(phi)
    unknown = set(assignments) - set(free)
    if unknown:
        raise ValueError(f"assignments for non-free variables: {unknown}")
    if close_free:
        for x in free:
            if x not in assignments:
                phi = logic.ExistsExpression(logic.Variable(x), phi)
    solve = [] if close_free else sorted(set(free) - set(assignments))
    open_vars = get_free_elementary_vars(phi)
    assigned = sorted(set(open_vars) - set(solve))
    all_vars = sorted(variable_names(phi) | set(open_vars)) or ['p']
    advice_var = get_unique_id(all_vars, 1)
    conjuncts = [relativize(phi, advice_var), f"Adv({advice_var})"]
    conjuncts += [f"Dom({advice_var},{x})" for x in open_vars]
    query = logic.Expression.fromstring(" and ".join(f"({c})" for c in conjuncts))
    return query, advice_var, assigned, solve


def check_class_string(phi, advice, assignments, atoms, element_alphabet,
                       relativize, variable_names) -> bool:
    """Implicit model check over string automata: relativize then run. `atoms`
    maps each wrapped relation (Dom, Adv, and the class relations) to a
    `SparseDFA` or a functional builder `args -> ImplicitDFA`."""
    query, advice_var, assigned, _ = relativized_query(
        phi, assignments, relativize, variable_names)
    advice = list(advice)
    inputs = {advice_var: advice}
    for x in assigned:
        word = list(assignments[x])
        if len(word) != len(advice):
            raise ValueError(f"assignment {x!r} has length {len(word)}, "
                             f"advice has length {len(advice)}")
        inputs[x] = word
    return check_string(query, atoms, inputs, len(advice),
                        lambda v: element_alphabet)


def check_class_tree(phi, advice, assignments, atoms, element_alphabet,
                     relativize, variable_names) -> bool:
    """Implicit model check over tree automata: relativize then run bottom-up.
    `atoms` maps each wrapped relation to a `SparseTreeAutomaton` or a functional
    builder `args -> ImplicitTA`."""
    query, advice_var, assigned, _ = relativized_query(
        phi, assignments, relativize, variable_names)
    inputs = {advice_var: advice}
    for x in assigned:
        inputs[x] = assignments[x]
    return check_tree(query, atoms, inputs, lambda v: element_alphabet)


def evaluate_class_string(phi, advice, assignments, atoms, element_alphabet,
                          relativize, variable_names) -> StringSolutionSet:
    """The satisfying set of a formula over the member presented by the
    advice, computed implicitly: unassigned free variables stay *open* and
    are solved for over the fixed advice. Returns a `StringSolutionSet` of
    `{var: word}` assignments (exact `len` without enumeration)."""
    query, advice_var, assigned, solve = relativized_query(
        phi, assignments, relativize, variable_names, close_free=False)
    advice = list(advice)
    inputs = {advice_var: advice}
    for x in assigned:
        word = list(assignments[x])
        if len(word) != len(advice):
            raise ValueError(f"assignment {x!r} has length {len(word)}, "
                             f"advice has length {len(advice)}")
        inputs[x] = word
    comb = _Combinators(
        atom=lambda name, args: _mk_atom(atoms[name], args, dfa_atom),
        product=dfa_product, complement=dfa_complement, project=dfa_project)
    a = _build(query, comb, lambda v: element_alphabet)
    return StringSolutionSet(a, inputs, len(advice), solve,
                             lambda v: element_alphabet)


def evaluate_class_tree(phi, advice, assignments, atoms, element_alphabet,
                        relativize, variable_names) -> TreeSolutionSet:
    """Tree analog of `evaluate_class_string`: the satisfying assignments
    are labelled trees of the advice's shape."""
    query, advice_var, assigned, solve = relativized_query(
        phi, assignments, relativize, variable_names, close_free=False)
    inputs = {advice_var: advice}
    for x in assigned:
        inputs[x] = assignments[x]
    comb = _Combinators(
        atom=lambda name, args: _mk_atom(atoms[name], args, ta_atom),
        product=ta_product, complement=ta_complement, project=ta_project)
    a = _build(query, comb, lambda v: element_alphabet)
    return TreeSolutionSet(a, inputs, solve, lambda v: element_alphabet)


# ======================================================================
# Fully implicit presentations (the successors-based API)
# ======================================================================

class ImplicitClass:
    """A uniformly automatic class given purely *functionally* -- the fully
    implicit presentation. `atoms` maps every relation name (including the
    wrapped 'Dom' and 'Adv') to a builder ``args -> ImplicitDFA`` (or an
    explicit `SparseDFA` to wrap); `element_alphabet` lists the per-position
    element symbols quantifiers guess from. Nothing is ever compiled: the
    class offers implicit model checking and satisfying-set evaluation only,
    so it reaches members whose presentation automata cannot be built."""

    def __init__(self, atoms: Dict, element_alphabet: Sequence):
        self.atoms = dict(atoms)
        self.element_alphabet = list(element_alphabet)

    @staticmethod
    def _statics():
        from autstr.uniform import UniformlyAutomaticClass as U
        return U._relativize, U._variable_names

    def check(self, phi, advice, **assignments) -> bool:
        """Model check against the member presented by the advice word;
        unassigned free variables are existentially closed."""
        rel, names = self._statics()
        return check_class_string(phi, advice, assignments, self.atoms,
                                  self.element_alphabet, rel, names)

    def evaluate(self, phi, advice, **assignments) -> StringSolutionSet:
        """The satisfying set for the open free variables over the member
        presented by the advice word."""
        rel, names = self._statics()
        return evaluate_class_string(phi, advice, assignments, self.atoms,
                                     self.element_alphabet, rel, names)


class ImplicitTreeClass:
    """Tree analog of `ImplicitClass`: atoms are ``args -> ImplicitTA``
    builders (or explicit `SparseTreeAutomaton` objects), members are
    advice trees."""

    def __init__(self, atoms: Dict, element_alphabet: Sequence):
        self.atoms = dict(atoms)
        self.element_alphabet = list(element_alphabet)

    def check(self, phi, advice, **assignments) -> bool:
        """Model check against the member presented by the advice tree;
        unassigned free variables are existentially closed."""
        rel, names = ImplicitClass._statics()
        return check_class_tree(phi, advice, assignments, self.atoms,
                                self.element_alphabet, rel, names)

    def evaluate(self, phi, advice, **assignments) -> TreeSolutionSet:
        """The satisfying set for the open free variables over the member
        presented by the advice tree."""
        rel, names = ImplicitClass._statics()
        return evaluate_class_tree(phi, advice, assignments, self.atoms,
                                   self.element_alphabet, rel, names)
