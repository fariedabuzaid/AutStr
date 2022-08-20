from automata.fa.dfa import DFA
from automata.fa.nfa import NFA
import itertools as it
from typing import List, Tuple, Callable, Union

from autstr.buildin.automata import one
from autstr.utils.misc import generate_new_elements
from autstr.utils.misc import heapify_llex as heapify
from autstr.utils.misc import heappop_llex as heappop
from autstr.utils.misc import heappush_llex as heappush


def stringlify_states(dfa: Union[DFA, NFA], convert_orgnames: bool = True) -> DFA:
    """
    Auxiliary function to turn the states of a finite automation into strings
    :param dfa: the automaton A
    :param convert_orgnames: If True, converts the original state names into strings. Otherwise, an arbitrary numbering
        is chosen.
    :return: dfa B such that B is identical to A but type(B.states) = Set[str]
    """
    mapping = {s: str(s) if convert_orgnames else str(i) for i, s in enumerate(dfa.states)}
    states = {mapping[q] for q in dfa.states}
    if isinstance(dfa, DFA):
        transitions = {mapping[q]: {a: mapping[dfa.transitions[q][a]] for a in dfa.input_symbols} for q in dfa.states}
    else:
        transitions = {
            mapping[q]: {a: {mapping[p] for p in dfa.transitions[q][a]} for a in dfa.transitions[q]} for q in dfa.states
        }
    initial_state = mapping[dfa.initial_state]
    final_states = {mapping[q] for q in dfa.final_states}

    if isinstance(dfa, DFA):
        dfa_str = DFA(
            initial_state=initial_state,
            states=states,
            transitions=transitions,
            input_symbols=dfa.input_symbols,
            final_states=final_states
        )
    else:
        dfa_str = NFA(
            initial_state=initial_state,
            states=states,
            transitions=transitions,
            input_symbols=dfa.input_symbols,
            final_states=final_states
        )
    return dfa_str


def stringlify_input_symbols(dfa: DFA, convert_orgnames=True) -> DFA:
    """
        Auxiliary function to turn the input_symbols of a finite automation into strings. Note that the function
        :param dfa: the automaton A
        :param convert_orgnames: If True, converts the original input symbol names into strings. Otherwise,
            an arbitrary numbering is chosen.
        :return: dfa B such that B is identical to A but type(B.input_symbols) = Set[str]
    """
    mapping = {s: str(s) if convert_orgnames else str(i) for i, s in enumerate(dfa.input_symbols)}
    input_symbols = {mapping[s] for s in dfa.input_symbols}
    transitions = {q: {mapping[a]: dfa.transitions[q][a] for a in dfa.input_symbols} for q in dfa.states}

    dfa_str = DFA(
        initial_state=dfa.initial_state,
        states=dfa.states,
        transitions=transitions,
        input_symbols=input_symbols,
        final_states=dfa.final_states
    )
    return dfa_str


def projection(dfa: DFA, i: int) -> DFA:
    """Takes an automaton that recognizes a k-ary relation R and a position i<k and returns and automaton that
    recognizes the relation R_{-i} = {(x_1,..., x_{i-1}, x_{i+1},..., x_k) | exists x_i: (x_1,..., x_n) in R}
    :param dfa: The automaton
    :param i: the position
    :return: Automaton presentation of R_{-i}
    """
    assert all([isinstance(s, tuple) for s in dfa.input_symbols])
    assert all([len(s) > i for s in dfa.input_symbols])
    done = set()
    input_symbols = {s[:i] + s[i + 1:] for s in dfa.input_symbols}
    input_symbols_i = {s[i] for s in dfa.input_symbols}
    todo = [frozenset({dfa.initial_state})]
    transitions = {}
    while len(todo) > 0:
        state = todo.pop(0)
        done.add(state)
        state_trans = {s: frozenset() for s in input_symbols}
        for q in state:
            for s in input_symbols:
                state_trans[s] = frozenset(
                    state_trans[s].union({dfa.transitions[q][s[:i] + (a,) + s[i:]] for a in input_symbols_i})
                )
        transitions[state] = state_trans
        for s in state_trans.values():
            if s not in done:
                todo.append(s)

    final = {s for s in done if any([q in dfa.final_states for q in s])}
    result = DFA(
        states=frozenset(done),
        input_symbols=input_symbols,
        initial_state=frozenset({dfa.initial_state}),
        transitions=transitions,
        final_states=final
    ).minify(retain_names=False)

    return stringlify_states(result)


def expand(dfa: DFA, n: int, pos: List) -> DFA:
    """
    Takes an automaton that recognizes a k-ary presentation R and expands it to a n-ary presentation S, n>=k, with
    S = {(x_1,..., x_n) | (x_{pos[0]},..., x_{pos[k-1]}) in R}
    :param dfa: The automaton
    :param n: The arity of the result
    :param pos: where to encode the original relation
    :return: Automaton recognizing {(x_1,..., x_n) | (x_{pos[0]},..., x_{pos[k-1]}) in L(dfa)}
    """
    assert max(pos) < n
    assert len(list(dfa.input_symbols)[0]) == len(pos)

    base_symbols = {symbol[0] for symbol in dfa.input_symbols}

    return DFA(
        initial_state=dfa.initial_state,
        input_symbols=set(it.product(base_symbols, repeat=n)),
        states=dfa.states,
        transitions={
            x: {
                symbol: dfa.transitions[x][tuple(symbol[i] for i in pos)] for symbol in
                it.product(base_symbols, repeat=n)
            } for x in dfa.states
        },
        final_states=dfa.final_states
    ).minify()


def pad(dfa: DFA, padding_symbol: Tuple[str] = ('*',)) -> DFA:
    """
    Create an automaton that recognises L(dfa){padding_symbol}^*. Note that pad does currently only support plain
    languages and not relations
    :param dfa: The automaton
    :param padding_symbol: (*,)
    :return: Automaton that recognizes L(dfa){paddingsymbol}^*
    """
    base_symbols = {a[0] for a in dfa.input_symbols}.union({padding_symbol[0]})
    arity = len(list(dfa.input_symbols)[0])
    input_symbols = set(it.product(base_symbols, repeat=arity))
    good, bad = generate_new_elements(dfa.states, 2)
    states = dfa.states.union({good, bad})
    padding_transitions = {
        q: {padding_symbol * arity: good if q in dfa.final_states else bad} for q in dfa.states
    }
    transitions = {q: dfa.transitions[q] | padding_transitions[q] for q in dfa.states}
    transitions[good] = {a: good if a == padding_symbol * arity else bad for a in input_symbols}
    transitions[bad] = {a: bad for a in input_symbols}
    final_states = dfa.final_states.union({good})

    padded_dfa = DFA(
        states=states,
        input_symbols=input_symbols,
        initial_state=dfa.initial_state,
        transitions=transitions,
        final_states=final_states
    ).minify()

    return padded_dfa


def unpad(dfa: DFA, padding_symbol: Tuple[str] = ('*',), remove_blank: bool = False) -> DFA:
    """
    Creates an automaton that recognizes {w | exists x in (padding_symbol^k)^*: wx in L(dfa)} where k is the arity of
    the relation that is recognized by dfa.
    :param remove_blank: If True, removes the blank symbol from the input symbols
    :param dfa: The automaton
    :param padding_symbol: The padding symbol. Needs to be a length one tuple
    :return: Automaton that recognizes {w | exists x in padding_symbol^*: wx in L(dfa)}
    """
    arity = len(list(dfa.input_symbols)[0])
    if not remove_blank:
        input_symbols = dfa.input_symbols
        sink = generate_new_elements(dfa.states, 1)
        states = dfa.states.union({sink})
        transitions = {
            q: {
                a: dfa.transitions[q][a] if padding_symbol * arity != a else sink for a in dfa.input_symbols
            } for q in dfa.states
        }
        transitions[sink] = {a: sink for a in input_symbols}
    else:
        input_symbols = {a for a in dfa.input_symbols if padding_symbol * arity != a}
        states = dfa.states
        transitions = {
            q: {
                a: dfa.transitions[q][a] for a in dfa.input_symbols if padding_symbol * arity != a
            } for q in dfa.states
        }

    final_states = dfa.final_states
    last_fstates = None
    while final_states != last_fstates:
        last_fstates = final_states
        final_states = final_states.union(
            {q for q in dfa.transitions if dfa.transitions[q][padding_symbol * arity] in final_states}
        )

    unpadded_dfa = DFA(
        states=states,
        input_symbols=input_symbols,
        initial_state=dfa.initial_state,
        transitions=transitions,
        final_states=final_states
    ).minify()

    return unpadded_dfa


def product(dfa: DFA, n: int) -> DFA:
    """
    Create an automaton that recognizes the n-fold cartesian product of L(dfa)
    :param dfa: The automaton
    :param n: number of convolutions
    :return: Automaton that recognizes the n-fold cartesian product of L(dfa)
    """
    if n == 0:
        result = one()
        return result
    base_automata = [stringlify_states(expand(dfa, n, [i])) for i in range(n)]
    result = base_automata[0]
    for base in base_automata[1:]:
        result = result.intersection(base).minify()

    return result


def iterate_language(dfa: DFA, decoder: Callable = None, backward: bool = False, padding_symbol: str = '*'):
    """
    Generator over the language that is represented by a DFA. Provides functionality for decoding of words by user
    defined decoder functions. Iterates through the words in length-lexicographic order.
    :param dfa: The automaton
    :param decoder: Decoder function that maps (tuples of) words to python object
    :param backward: If True, iterate over the elements of the reversed language
    :param padding_symbol: The padding symbol
    :return:
    """
    arity = len(list(dfa.input_symbols)[0])
    nfa = dfa

    nfa = stringlify_states(nfa)

    if backward:
        nonempty = set()
        for q in nfa.states:
            nfa_q = DFA(
                states=nfa.states,
                input_symbols=nfa.input_symbols,
                initial_state=nfa.initial_state,
                final_states={q},
                transitions=nfa.transitions
            )
            if not nfa_q.isempty():
                nonempty.add(q)

        transitions = {q: {a: set() for a in nfa.input_symbols} for q in nfa.states}
        for p in nfa.states:
            for q in nfa.states:
                for a in nfa.transitions[q]:
                    transitions[nfa.transitions[q][a]][a].add(q)

        final = {nfa.initial_state}
    else:
        nonempty = set()
        for q in nfa.states:
            nfa_q = DFA(
                states=nfa.states,
                input_symbols=nfa.input_symbols,
                initial_state=q,
                final_states=nfa.final_states,
                transitions=nfa.transitions
            )
            if not nfa_q.isempty():
                nonempty.add(q)

        transitions = dfa.transitions
        final = nfa.final_states

    queue = [(('',) * arity, q) for q in nfa.final_states]
    queue = heapify(queue)
    while len(queue) > 0:
        word, state = heappop(queue)

        if state in final:
            if decoder is None:
                yield word
            else:
                yield decoder(word)

        for a in transitions[state]:
            if a != (padding_symbol,) * arity:
                for q in transitions[state][a]:
                    if q in nonempty:
                        heappush(
                            queue,
                            (tuple(
                                [f'{wordcomp}{b}' if b != padding_symbol else wordcomp for wordcomp, b in
                                 zip(word, a)]
                            ), q)
                        )


def lsbf_automaton(n: int) -> DFA:
    """
    generates an automation that recognizes exactly the least-significant-bit-first binary encoding of n
    """
    bits = format(n, 'b')[::-1]
    n_bits = len(bits)

    states = set(range(n_bits + 2))
    initial_state = 0
    final_states = {n_bits}
    input_symbols = {('0',), ('1',), ('*',)}

    transitions = {i: {a: i + 1 if a == (bits[i],) else n_bits + 1 for a in input_symbols} for i in range(n_bits)}
    transitions[n_bits] = {a: n_bits + 1 for a in input_symbols}
    transitions[n_bits + 1] = {a: n_bits + 1 for a in input_symbols}

    result = DFA(
        states=states,
        input_symbols=input_symbols,
        initial_state=initial_state,
        transitions=transitions,
        final_states=final_states
    )

    return stringlify_states(result)
