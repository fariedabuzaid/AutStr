from automata.fa.dfa import DFA

def length_automaton(n, sigma) -> DFA:
    """
    Creates an automaton that recognizes all words over sigma with length n
    :param n: The length
    :param sigma: The input symbols
    :return:
    """
    states = {f'{i}' for i in range(n + 1)}
    transitions = {q: {a: str(int(q) + 1) if int(q) < n else q for a in sigma} for q in states}

    dfa = DFA(
        states=states,
        input_symbols=sigma,
        initial_state='0',
        transitions=transitions,
        final_states={f'{n - 1}'}
    )
    return dfa

def zero() -> DFA:
    """
    Creates an automaton over the empty symbol set that recognizes the empty language
    :return:
    """
    result = DFA(
        states={'0'},
        input_symbols=set(),
        initial_state='0',
        transitions={'0': {}},
        final_states=set()
    )

    return result


def one() -> DFA:
    """
        Creates an automaton over the empty symbol set that recognizes language that contains only the empty word.
        :return:
        """
    result = DFA(
        states={'0'},
        input_symbols=set(),
        initial_state='0',
        transitions={
            '0': {},
        },
        final_states={'0'}
    )

    return result
