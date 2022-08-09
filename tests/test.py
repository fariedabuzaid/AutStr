from automata.fa.nfa import NFA

from austr.buildin.presentations import buechi_arithmetic
from austr.utils.automata_tools import iterate_language

if __name__ == '__main__':
    ba = buechi_arithmetic()
    fnc = lambda x: tuple(int(n.replace('*', ''), base=2) for n in x)
    samples = []
    query = ba.evaluate('exists x y.(B(x, x) and B(y, y) and A(x, z, y))')
    nfa = NFA.from_dfa(query)
    nfa = nfa.reverse()

    for i in iterate_language(query, backward=True): #, decoder=fnc):
        print(i)
        samples.append(i)
        if len(samples) > 10:
            break
