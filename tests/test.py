from autstr.buildin.presentations import buechi_arithmetic
from autstr.utils.automata_tools import iterate_language

if __name__ == '__main__':
    ba = buechi_arithmetic()
    fnc = lambda x: tuple(int(n.replace('*', ''), base=2) for n in x)
    samples = []
    query = ba.evaluate('A(y,y,z) and A(x, x, z) and B(y,y)')

    for i in iterate_language(query, backward=True, decoder=fnc):
        print(i)
        samples.append(i)
        if len(samples) > 10:
            break
    print(query.transitions)
