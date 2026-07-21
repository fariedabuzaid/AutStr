"""Symbolic first-order expressions over automatic structures and classes.

Instead of writing formula strings, build them from variables and symbols the
structure hands out::

    A = BuechiArithmeticZ()
    S = A.symbolic()
    x, y, z = S.vars("x y z")
    phi = ((x + y).eq(z) & z.lt(10)).drop(y)

    phi.check()            # satisfiable?
    phi.evaluate()         # presentation of the satisfying assignments
    (1, 5) in phi          # membership, by the sorted tape order
    list(phi)              # enumerate solutions

The same expressions compile against a `UniformlyAutomaticClass`, where they
define relations uniformly across every member structure.

Relation and function arities come from the automata; which relations are
function graphs, which Python operators they are bound to, and how Python
values encode as elements are declared in a `Signature`.
"""
from autstr.symbolic.backends import Backend, ClassBackend, StructureBackend
from autstr.symbolic.context import (
    FunctionSymbol, Relation, RelationSymbol, SymbolicContext,
    SymbolicSymbolError,
)
from autstr.symbolic.compiler import CompileError
from autstr.symbolic.expr import Formula, Term, Var
from autstr.symbolic.signature import (
    ElementCodec, Function, FunctionCodec, Signature,
)

__all__ = [
    'Backend', 'ClassBackend', 'CompileError', 'ElementCodec', 'Formula',
    'Function', 'FunctionCodec', 'FunctionSymbol', 'Relation',
    'RelationSymbol', 'Signature', 'StructureBackend', 'SymbolicContext',
    'SymbolicSymbolError', 'Term', 'Var',
]
