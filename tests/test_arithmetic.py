from copy import deepcopy

from austr.arithmetic import VariableTerm as Var

from datetime import datetime

if __name__ == '__main__':
    x = Var('x')
    y = Var('y')
    z = Var('z')
    a = Var('a')

    fnc = lambda x: tuple(int(n.replace('*', ''), base=2) for n in x)

    P = x.gt(1)

    res = []
    time = datetime.now()

    for _ in range(4):
        for n, in P:
            print(n)

            P = P & ~(n * y).eq(x).drop(['y'])
            res = []
            time = datetime.now()
            break

    Bxy = ((z | z) & ((x + z).eq(y) | (y + z).eq(x))).drop('z')
    Bxz = ((y | y) & ((x + y).eq(z) | (z + y).eq(x))).drop('y')
    Byz = ((x | x) & ((y + x).eq(z) | (z + x).eq(y))).drop('x')
    triangles = Bxy & deepcopy(Bxy).substitute(y='z') & deepcopy(Bxy).substitute(x='z')

    time = datetime.now()
    triangles.update_presentation()
    print(f'Evaluation time: {datetime.now() - time}')

    for n in triangles:
        print(n)
        if max(n) > 10:
            break

