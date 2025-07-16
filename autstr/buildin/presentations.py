from pathlib import Path
import itertools as it
from typing import Dict, List, Set, Tuple
import jax.numpy as jnp 

from autstr.sparse_automata import SparseDFA
from autstr.presentations import AutomaticPresentation

# Helper function to convert a symbol tuple to an integer encoding
def encode_symbol(tup: Tuple[str], base_alphabet: Set[str]) -> int:
    """Encode a symbol tuple into an integer using base conversion."""
    base = len(base_alphabet)
    mapping = {sym: idx for idx, sym in enumerate(sorted(base_alphabet))}
    enc = 0
    for char in tup:
        enc = enc * base + mapping[char]
    return enc

# Helper function to create a SparseDFA from a traditional DFA description
def create_sparse_dfa(states: List[str], input_symbols: Set[Tuple[str]], 
                     transitions: Dict[str, Dict[Tuple[str], str]], 
                     initial_state: str, final_states: Set[str]) -> SparseDFA:
    """Convert a traditional DFA description to a SparseDFA."""
    # Map states to integers
    state_to_index = {s: i for i, s in enumerate(states)}
    num_states = len(states)
    
    # Determine base alphabet and arity
    base_alphabet = set()
    for sym in input_symbols:
        for char in sym:
            base_alphabet.add(char)
    arity = len(next(iter(input_symbols))) if input_symbols else 0
    
    # Create state arrays
    default_states = []
    exception_symbols = []
    exception_states = []
    is_accepting = jnp.array([state in final_states for state in states], dtype=bool)
    
    # Build symbol mapping
    symbol_map = {}
    for symbol in input_symbols:
        symbol_map[symbol] = encode_symbol(symbol, base_alphabet)
    
    # Process each state
    for state in states:
        # Find most common transition
        next_states = [transitions[state][sym] for sym in input_symbols]
        default_target = max(set(next_states), key=next_states.count)
        default_states.append(state_to_index[default_target])
        
        # Collect exceptions
        exceptions = []
        for symbol, next_state in transitions[state].items():
            if next_state != default_target:
                sym_enc = symbol_map[symbol]
                next_idx = state_to_index[next_state]
                exceptions.append((sym_enc, next_idx))
        
        # Sort exceptions by symbol for consistency
        exceptions.sort(key=lambda x: x[0])
        exception_symbols.append([e[0] for e in exceptions])
        exception_states.append([e[1] for e in exceptions])
    
    # Pad exceptions
    max_exceptions = max(len(e) for e in exception_symbols) if exception_symbols else 0
    padded_ex_syms = jnp.full((num_states, max_exceptions), -1, dtype=jnp.int32)
    padded_ex_states = jnp.full((num_states, max_exceptions), -1, dtype=jnp.int32)
    
    for i in range(num_states):
        if exception_symbols[i]:
            padded_ex_syms = padded_ex_syms.at[i, :len(exception_symbols[i])].set(jnp.array(exception_symbols[i], dtype=jnp.int32))
            padded_ex_states = padded_ex_states.at[i, :len(exception_states[i])].set(jnp.array(exception_states[i], dtype=jnp.int32))
    
    return SparseDFA(
        num_states=num_states,
        default_states=jnp.array(default_states, dtype=jnp.int32),
        exception_symbols=padded_ex_syms,
        exception_states=padded_ex_states,
        is_accepting=is_accepting,
        start_state=state_to_index[initial_state],
        symbol_arity=arity,
        base_alphabet=base_alphabet
    ).minimize()

def BuechiArithmetic() -> AutomaticPresentation:
    """Load the serialized Büchi arithmetic presentation."""
    current_dir = Path(__file__).parent
    bin_path = current_dir / 'bin' / 'buechi.autstr'
    return AutomaticPresentation.automatic_presentation_from_file(str(bin_path))

def buechi_arithmetic() -> AutomaticPresentation:
    """Sparse version of Büchi arithmetic over natural numbers. Compiles definable relation from scratch."""
    # Universe automaton - only accepts valid binary numbers (no '*')
    universe = create_sparse_dfa(
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

    addition = create_sparse_dfa(
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
    weak_div = create_sparse_dfa(
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
    

    # Create presentation
    presentation = AutomaticPresentation({'U': universe, 'A': addition, 'B': weak_div})

    # Add bootstrap remaining relations
    presentation.update(Z='A(x,x,x)')
    presentation.update(Eq='exists z.(Z(z) and A(x,z,y))')
    presentation.update(Pt='B(x,x)')
    presentation.update(Lt='exists z.(not Z(z) and A(x, z, y))')
    
    return presentation

def BuechiArithmeticZ() -> AutomaticPresentation:
    """Load the serialized Büchi arithmetic presentation."""
    current_dir = Path(__file__).parent
    bin_path = current_dir / 'bin' / 'buechiZ.autstr'
    return AutomaticPresentation.automatic_presentation_from_file(str(bin_path))

def buechi_arithmetic_Z() -> AutomaticPresentation:
    """Sparse version of Büchi arithmetic over integers."""
    # Universe automaton
    universe_states = ['-1', 'i+', 'i', '0', '0+', '1', '*']
    universe_symbols = {('0',), ('1',), ('*',)}
    universe_trans = {
        '-1': {('0',): 'i', ('1',): 'i+', ('*',): '*'},
        'i+': {('0',): '0+', ('1',): '1', ('*',): '*'},
        'i': {('0',): '0', ('1',): '1', ('*',): '*'},
        '0': {('0',): '0+', ('1',): '1', ('*',): '*'},
        '0+': {('0',): '0+', ('1',): '1', ('*',): '*'},
        '1': {('0',): '0+', ('1',): '1', ('*',): '*'},
        '*': {('0',): '*', ('1',): '*', ('*',): '*'},
    }
    universe = create_sparse_dfa(
        universe_states, universe_symbols, universe_trans, '-1', {'0', '1'}
    )
    
    # Addition automaton (intermediate)
    add_states = [-1, 0, 1, 2]
    add_symbols = set(it.product(['0', '1', '*'], repeat=3))
    add_trans = {
        -1: {a: 0 if '*' not in a else 2 for a in add_symbols},
        0: {
            ('0','0','0'): 0, ('0','0','1'): 2, ('0','0','*'): 2,
            ('0','1','0'): 2, ('0','1','1'): 0, ('0','1','*'): 2,
            ('0','*','0'): 0, ('0','*','1'): 2, ('0','*','*'): 2,
            ('1','0','0'): 2, ('1','0','1'): 0, ('1','0','*'): 2,
            ('1','1','0'): 1, ('1','1','1'): 2, ('1','1','*'): 2,
            ('1','*','0'): 2, ('1','*','1'): 0, ('1','*','*'): 2,
            ('*','0','0'): 0, ('*','0','1'): 2, ('*','0','*'): 2,
            ('*','1','0'): 2, ('*','1','1'): 0, ('*','1','*'): 2,
            ('*','*','0'): 2, ('*','*','1'): 2, ('*','*','*'): 2,
        },
        1: {
            ('0','0','0'): 2, ('0','0','1'): 0, ('0','0','*'): 2,
            ('0','1','0'): 1, ('0','1','1'): 2, ('0','1','*'): 2,
            ('0','*','0'): 2, ('0','*','1'): 0, ('0','*','*'): 2,
            ('1','0','0'): 1, ('1','0','1'): 2, ('1','0','*'): 2,
            ('1','1','0'): 2, ('1','1','1'): 1, ('1','1','*'): 2,
            ('1','*','0'): 1, ('1','*','1'): 2, ('1','*','*'): 2,
            ('*','0','0'): 2, ('*','0','1'): 0, ('*','0','*'): 2,
            ('*','1','0'): 1, ('*','1','1'): 2, ('*','1','*'): 2,
            ('*','*','0'): 2, ('*','*','1'): 0, ('*','*','*'): 2,
        },
        2: {s: 2 for s in add_symbols}
    }
    addition_intermediate = create_sparse_dfa(add_states, add_symbols, add_trans, -1, {0})
    
    # Weak division automaton
    div_states = ['-1', '0', '1', 'e']
    div_symbols = set(it.product(['0', '1', '*'], repeat=2))
    div_trans = {
        '-1': {a: '0' if a[1] == '0' else 'e' for a in div_symbols},
        '0': {
            a: '0' if a == ('0','0') 
            else '1' if a in {('0','1'), ('1','1')} 
            else 'e' for a in div_symbols
        },
        '1': {
            a: 'e' if a[1] != '*' else '1' for a in div_symbols
        },
        'e': {a: 'e' for a in div_symbols}
    }
    weak_div = create_sparse_dfa(div_states, div_symbols, div_trans, '-1', {'1'})
    
    # N0 automaton
    n0_states = [-1, 0, 1]
    n0_symbols = {('0',), ('1',), ('*',)}
    n0_trans = {
        -1: {('0',): 1, ('1',): 0, ('*',): 0},
        0: {s: 0 for s in n0_symbols},
        1: {s: 1 for s in n0_symbols}
    }
    N0 = create_sparse_dfa(n0_states, n0_symbols, n0_trans, -1, {1})
    
    # Create presentation
    presentation = AutomaticPresentation({
        'U': universe, 
        'A0': addition_intermediate, 
        'B': weak_div, 
        'N0': N0
    })
    presentation.update(Z='A0(x,x,x)')
    
    # Define addition formula
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
    
    # Delete auxiliary relation
    del presentation.automata['A0']
    
    return presentation