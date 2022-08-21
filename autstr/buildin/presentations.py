from automata.fa.dfa import DFA
import itertools as it
from autstr.presentations import AutomaticPresentation


def buechi_arithmetic() -> AutomaticPresentation:
    """
    create a presentation of Büchi arithmetic over the natural numbers :math:`\\mathbb{N}`.
    """
    universe = DFA(
        states={'i', '0', '0+', '1', '*'},
        input_symbols={('0',), ('1',), ('*',)},
        transitions={
            'i': {('0',): '0', ('1',): '1', ('*',): '*'},
            '0': {('0',): '0+', ('1',): '1', ('*',): '*'},
            '0+': {('0',): '0+', ('1',): '1', ('*',): '*'},
            '1': {('0',): '0+', ('1',): '1', ('*',): '*'},
            '*': {('0',): '*', ('1',): '*', ('*',): '*'},
        },
        initial_state='i',
        final_states={'0', '1'}
    )

    addition = DFA(
        states={0, 1, 2},
        input_symbols={('0', '0', '0'), ('0', '0', '1'), ('0', '0', '*'), ('0', '1', '0'), ('0', '1', '1'),
                       ('0', '1', '*'), ('0', '*', '0'), ('0', '*', '1'), ('0', '*', '*'),
                       ('1', '0', '0'), ('1', '0', '1'), ('1', '0', '*'), ('1', '1', '0'), ('1', '1', '1'),
                       ('1', '1', '*'), ('1', '*', '0'), ('1', '*', '1'), ('1', '*', '*'),
                       ('*', '0', '0'), ('*', '0', '1'), ('*', '0', '*'), ('*', '1', '0'), ('*', '1', '1'),
                       ('*', '1', '*'), ('*', '*', '0'), ('*', '*', '1'), ('*', '*', '*'),
                       },
        transitions={
            0: {
                ('0', '0', '0'): 0, ('0', '0', '1'): 2, ('0', '0', '*'): 2, ('0', '1', '0'): 2, ('0', '1', '1'): 0,
                ('0', '1', '*'): 2, ('0', '*', '0'): 0, ('0', '*', '1'): 2, ('0', '*', '*'): 2,
                ('1', '0', '0'): 2, ('1', '0', '1'): 0, ('1', '0', '*'): 2, ('1', '1', '0'): 1, ('1', '1', '1'): 2,
                ('1', '1', '*'): 2, ('1', '*', '0'): 2, ('1', '*', '1'): 0, ('1', '*', '*'): 2,
                ('*', '0', '0'): 0, ('*', '0', '1'): 2, ('*', '0', '*'): 2, ('*', '1', '0'): 2, ('*', '1', '1'): 0,
                ('*', '1', '*'): 2, ('*', '*', '0'): 2, ('*', '*', '1'): 2, ('*', '*', '*'): 2,
            },
            1: {
                ('0', '0', '0'): 2, ('0', '0', '1'): 0, ('0', '0', '*'): 2, ('0', '1', '0'): 1, ('0', '1', '1'): 2,
                ('0', '1', '*'): 2, ('0', '*', '0'): 2, ('0', '*', '1'): 0, ('0', '*', '*'): 2,
                ('1', '0', '0'): 1, ('1', '0', '1'): 2, ('1', '0', '*'): 2, ('1', '1', '0'): 2, ('1', '1', '1'): 1,
                ('1', '1', '*'): 2, ('1', '*', '0'): 1, ('1', '*', '1'): 2, ('1', '*', '*'): 2,
                ('*', '0', '0'): 2, ('*', '0', '1'): 0, ('*', '0', '*'): 2, ('*', '1', '0'): 1, ('*', '1', '1'): 2,
                ('*', '1', '*'): 2, ('*', '*', '0'): 2, ('*', '*', '1'): 0, ('*', '*', '*'): 2,
            },
            2: {
                ('0', '0', '0'): 2, ('0', '0', '1'): 2, ('0', '0', '*'): 2, ('0', '1', '0'): 2, ('0', '1', '1'): 2,
                ('0', '1', '*'): 2, ('0', '*', '0'): 2, ('0', '*', '1'): 2, ('0', '*', '*'): 2,
                ('1', '0', '0'): 2, ('1', '0', '1'): 2, ('1', '0', '*'): 2, ('1', '1', '0'): 2, ('1', '1', '1'): 2,
                ('1', '1', '*'): 2, ('1', '*', '0'): 2, ('1', '*', '1'): 2, ('1', '*', '*'): 2,
                ('*', '0', '0'): 2, ('*', '0', '1'): 2, ('*', '0', '*'): 2, ('*', '1', '0'): 2, ('*', '1', '1'): 2,
                ('*', '1', '*'): 2, ('*', '*', '0'): 2, ('*', '*', '1'): 2, ('*', '*', '*'): 2,
            },
        },
        initial_state=0,
        final_states={0}
    )

    input_symbols = {a for a in it.product(['0', '1', '*'], repeat=2)}
    weak_div = DFA(
        states={'0', '1', 'e'},
        input_symbols=input_symbols,
        transitions={
            '0': {
                a: '0' if a == ('0', '0') else '1' if a == ('0', '1') or a == ('1', '1') else 'e' for a in input_symbols
            },
            '1': {
                a: 'e' if a[1] != '*' else '1' for a in input_symbols
            },
            'e': {a: 'e' for a in input_symbols}
        },
        initial_state='0',
        final_states={'1'}
    )

    presentation = AutomaticPresentation({'U': universe, 'A': addition, 'B': weak_div})
    presentation.update(Z='A(x,x,x)')
    presentation.update(Eq='exists z.(Z(z) and A(x,z,y))')
    presentation.update(Pt='B(x,x)')
    presentation.update(Lt='exists z.(not Z(z) and A(x, z, y))')

    return presentation


def buechi_arithmetic_Z() -> AutomaticPresentation:
    """
    Creates a presentation of Büchi arithmetic over :math:`\\mathbb{Z}`.
    """

    base_input_symbols = {'0', '1', '*'}

    universe = DFA(
        states={'-1', 'i+', 'i', '0', '0+', '1', '*'},
        input_symbols={(a,) for a in base_input_symbols},
        transitions={
            '-1': {('0',): 'i', ('1',): 'i+', ('*',): '*'},
            'i+': {('0',): '0+', ('1',): '1', ('*',): '*'},
            'i': {('0',): '0', ('1',): '1', ('*',): '*'},
            '0': {('0',): '0+', ('1',): '1', ('*',): '*'},
            '0+': {('0',): '0+', ('1',): '1', ('*',): '*'},
            '1': {('0',): '0+', ('1',): '1', ('*',): '*'},
            '*': {('0',): '*', ('1',): '*', ('*',): '*'},
        },
        initial_state='-1',
        final_states={'0', '1'}
    )

    addition_input_symbols = set(it.product(base_input_symbols, repeat=3))

    addition_intermediate = DFA(
        states={-1, 0, 1, 2},
        input_symbols=addition_input_symbols,
        transitions={
            -1: {a: 0 if '*' not in a else 2 for a in addition_input_symbols},
            0: {
                ('0', '0', '0'): 0, ('0', '0', '1'): 2, ('0', '0', '*'): 2, ('0', '1', '0'): 2, ('0', '1', '1'): 0,
                ('0', '1', '*'): 2, ('0', '*', '0'): 0, ('0', '*', '1'): 2, ('0', '*', '*'): 2,
                ('1', '0', '0'): 2, ('1', '0', '1'): 0, ('1', '0', '*'): 2, ('1', '1', '0'): 1, ('1', '1', '1'): 2,
                ('1', '1', '*'): 2, ('1', '*', '0'): 2, ('1', '*', '1'): 0, ('1', '*', '*'): 2,
                ('*', '0', '0'): 0, ('*', '0', '1'): 2, ('*', '0', '*'): 2, ('*', '1', '0'): 2, ('*', '1', '1'): 0,
                ('*', '1', '*'): 2, ('*', '*', '0'): 2, ('*', '*', '1'): 2, ('*', '*', '*'): 2,
            },
            1: {
                ('0', '0', '0'): 2, ('0', '0', '1'): 0, ('0', '0', '*'): 2, ('0', '1', '0'): 1, ('0', '1', '1'): 2,
                ('0', '1', '*'): 2, ('0', '*', '0'): 2, ('0', '*', '1'): 0, ('0', '*', '*'): 2,
                ('1', '0', '0'): 1, ('1', '0', '1'): 2, ('1', '0', '*'): 2, ('1', '1', '0'): 2, ('1', '1', '1'): 1,
                ('1', '1', '*'): 2, ('1', '*', '0'): 1, ('1', '*', '1'): 2, ('1', '*', '*'): 2,
                ('*', '0', '0'): 2, ('*', '0', '1'): 0, ('*', '0', '*'): 2, ('*', '1', '0'): 1, ('*', '1', '1'): 2,
                ('*', '1', '*'): 2, ('*', '*', '0'): 2, ('*', '*', '1'): 0, ('*', '*', '*'): 2,
            },
            2: {
                ('0', '0', '0'): 2, ('0', '0', '1'): 2, ('0', '0', '*'): 2, ('0', '1', '0'): 2, ('0', '1', '1'): 2,
                ('0', '1', '*'): 2, ('0', '*', '0'): 2, ('0', '*', '1'): 2, ('0', '*', '*'): 2,
                ('1', '0', '0'): 2, ('1', '0', '1'): 2, ('1', '0', '*'): 2, ('1', '1', '0'): 2, ('1', '1', '1'): 2,
                ('1', '1', '*'): 2, ('1', '*', '0'): 2, ('1', '*', '1'): 2, ('1', '*', '*'): 2,
                ('*', '0', '0'): 2, ('*', '0', '1'): 2, ('*', '0', '*'): 2, ('*', '1', '0'): 2, ('*', '1', '1'): 2,
                ('*', '1', '*'): 2, ('*', '*', '0'): 2, ('*', '*', '1'): 2, ('*', '*', '*'): 2,
            },
        },
        initial_state=-1,
        final_states={0}
    )

    weak_div_input_symbols = {a for a in it.product(['0', '1', '*'], repeat=2)}
    weak_div = DFA(
        states={'-1', '0', '1', 'e'},
        input_symbols=weak_div_input_symbols,
        transitions={
            '-1': {a: '0' if a[1] == '0' else 'e' for a in weak_div_input_symbols},
            '0': {
                a: '0' if a == ('0', '0') else '1' if a == ('0', '1') or a == ('1', '1') else 'e' for a in weak_div_input_symbols
            },
            '1': {
                a: 'e' if a[1] != '*' else '1' for a in weak_div_input_symbols
            },
            'e': {a: 'e' for a in weak_div_input_symbols}
        },
        initial_state='-1',
        final_states={'1'}
    )

    N0_input_symbols = {(a,) for a in base_input_symbols}

    N0 = DFA(
        states={-1, 0, 1},
        input_symbols=N0_input_symbols,
        initial_state=-1,
        transitions={
            -1: {a: 1 if a == ("0",) else 0 for a in N0_input_symbols},
            0: {a: 0 for a in N0_input_symbols},
            1: {a: 1 for a in N0_input_symbols}
        },
        final_states={1}
    )

    presentation = AutomaticPresentation({'U': universe, 'A0': addition_intermediate, 'B': weak_div, 'N0': N0})
    presentation.update(Z='A0(x,x,x)')

    c000 = '(N0(x) and N0(y) and N0(z) and A0(x, y, z))'
    c001 = '(N0(x) and N0(y) and not N0(z) and exists a z0.(Z(z0) and A0(x,y,a) and A0(a,z,z0)))'
    c010 = '(N0(x) and not N0(y) and N0(z) and A0(z, y, x))'
    c011 = '(N0(x) and not N0(y) and not N0(z) and A0(z, x, y))'
    c100 = '(not N0(x) and N0(y) and N0(z) and A0(x, z, y))'
    c101 = '(not N0(x) and N0(y) and not N0(z) and A0(z, y, x))'
    c110 = '(not N0(x) and not N0(y) and N0(z) and exists a z0.(Z(z0) and A0(x,y,a) and A0(a,z,z0)))'
    c111 = '(not N0(x) and not N0(y) and not N0(z) and A0(x,y,z))'
    phi_A = ' or '.join([c000, c001, c010, c011, c100, c101, c110, c111])
    presentation.update(A=phi_A)

    presentation.update(Eq='exists z.(Z(z) and A(x,z,y))')
    presentation.update(Pt='B(x,x) and N0(x)')
    presentation.update(Lt='exists z.(N0(z) and not Z(z) and A(x, z, y))')
    presentation.update(Neg='exists z.(Z(z) and A(x,y,z))')

    # Delete auxiliary relations
    del presentation.automata['A0']

    return presentation
