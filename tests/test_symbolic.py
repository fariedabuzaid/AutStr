"""Tests for the generic symbolic interface."""
from itertools import islice

import pytest

from autstr.buildin.presentations import BuechiArithmeticZ
from autstr.symbolic import (
    CompileError, FunctionCodec, Signature, SymbolicSymbolError,
)


def _encode(n: int):
    """LSB-first encoding of an integer: sign symbol, then magnitude bits."""
    return [('0' if n >= 0 else '1')] + list(format(abs(n), 'b')[::-1])


def _decode(word):
    word = ''.join(word).replace('*', '')
    magnitude = int(word[1:][::-1] or '0', 2)
    return magnitude if word[0] == '0' else -magnitude


def arithmetic_signature() -> Signature:
    signature = Signature(codec=FunctionCodec(_encode, _decode))
    signature.function('+', graph='A', out=2)
    signature.function('neg', graph='Neg', out=1)
    signature.operator('+', '+').operator('-', 'neg')
    signature.operator('eq', 'Eq').operator('lt', 'Lt').operator('gt', 'Gt')
    return signature


@pytest.fixture(scope='module')
def S():
    return BuechiArithmeticZ().symbolic(arithmetic_signature())


# ----------------------------------------------------------------------
# terms and operators
# ----------------------------------------------------------------------

def test_addition_term(S):
    x, y, z = S.vars('x y z')
    relation = (x + y).eq(z).evaluate()
    assert relation.variables == ['x', 'y', 'z']
    assert relation.contains(x=3, y=4, z=7)
    assert not relation.contains(x=3, y=4, z=8)


def test_positional_membership_follows_tape_order(S):
    x, y, z = S.vars('x y z')
    relation = (x + y).eq(z).evaluate()
    assert (3, 4, 7) in relation
    assert (3, 4, 8) not in relation


def test_nested_terms(S):
    x, y, z = S.vars('x y z')
    relation = ((x + y) + z).eq(S.const(10)).evaluate()
    assert relation.contains(x=2, y=3, z=5)
    assert not relation.contains(x=2, y=3, z=6)


def test_constants_from_python_values(S):
    x, y = S.vars('x y')
    relation = (x + 5).eq(y).evaluate()
    assert relation.contains(x=10, y=15)
    assert not relation.contains(x=10, y=16)


def test_negation_operator(S):
    x, y = S.vars('x y')
    relation = (-x).eq(y).evaluate()
    assert relation.contains(x=5, y=-5)
    assert not relation.contains(x=5, y=5)


def test_subtraction_via_declared_inverse(S):
    x, y, z = S.vars('x y z')
    relation = (x - y).eq(z).evaluate()
    assert relation.contains(x=9, y=4, z=5)
    assert not relation.contains(x=9, y=4, z=6)


def test_times_uses_base_two_decomposition(S):
    x, y = S.vars('x y')
    relation = x.times(5).eq(y).evaluate()
    assert relation.contains(x=3, y=15)
    assert not relation.contains(x=3, y=14)


def test_method_operators_from_signature(S):
    x, y = S.vars('x y')
    relation = x.lt(y).evaluate()
    assert relation.contains(x=2, y=7)
    assert not relation.contains(x=7, y=2)


def test_undeclared_operator_is_an_error(S):
    x, y = S.vars('x y')
    with pytest.raises(AttributeError, match='no operator'):
        x.divides(y)


def test_undeclared_multiplication_is_an_error(S):
    x, y = S.vars('x y')
    with pytest.raises(TypeError, match="no operator '\\*'"):
        x * y


# ----------------------------------------------------------------------
# symbols
# ----------------------------------------------------------------------

def test_relation_symbol_arity_is_read_from_the_automaton(S):
    R = S.get_symbolic_rel('A')
    assert R.arity == 3


def test_function_symbol_arity_is_read_from_its_graph(S):
    f = S.get_symbolic_func('+')
    assert f.arity == 2


def test_function_application_matches_the_operator(S):
    x, y, z = S.vars('x y z')
    f = S.get_symbolic_func('+')
    assert f(x, y).eq(z) == (x + y).eq(z)


def test_wrong_relation_arity_is_reported(S):
    x, y = S.vars('x y')
    with pytest.raises(CompileError, match='arity 3'):
        S.atom('A', [x, y]).evaluate()


def test_wrong_function_arity_is_reported(S):
    x, = S.vars('x')
    f = S.get_symbolic_func('+')
    with pytest.raises(SymbolicSymbolError, match='arity 2'):
        f(x)


def test_unknown_relation_is_reported(S):
    with pytest.raises(SymbolicSymbolError, match='no relation'):
        S.get_symbolic_rel('NoSuchRelation')


def test_unknown_function_is_reported(S):
    with pytest.raises(SymbolicSymbolError, match='no function'):
        S.get_symbolic_func('nope')


# ----------------------------------------------------------------------
# variable names
# ----------------------------------------------------------------------

def test_names_illegal_for_nltk_still_work(S):
    """`nltk` only accepts [a-df-z][0-9]* as an individual variable and
    silently drops anything else from the free-variable list. These names all
    fall outside that pattern."""
    alpha, beta, e = S.vars(['alpha', 'beta', 'e'])
    relation = (alpha + beta).eq(e).evaluate()
    assert relation.variables == ['alpha', 'beta', 'e']
    assert relation.contains(alpha=3, beta=4, e=7)
    assert not relation.contains(alpha=3, beta=4, e=8)


def test_tape_order_is_the_sorted_variable_names(S):
    b, a = S.vars('b a')
    assert (a + b).eq(S.const(3)).evaluate().variables == ['a', 'b']


def test_reorder_permutes_tapes(S):
    x, y, z = S.vars('x y z')
    relation = (x + y).eq(z).evaluate().reorder(['z', 'y', 'x'])
    assert relation.variables == ['z', 'y', 'x']
    assert relation.contains(7, 4, 3)
    assert relation.contains(x=3, y=4, z=7)


# ----------------------------------------------------------------------
# boolean algebra and quantification
# ----------------------------------------------------------------------

def test_conjunction(S):
    x, y, z = S.vars('x y z')
    relation = ((x + y).eq(z) & x.lt(y)).evaluate()
    assert relation.contains(x=2, y=5, z=7)
    assert not relation.contains(x=5, y=2, z=7)


def test_disjunction(S):
    x, y = S.vars('x y')
    relation = (x.lt(y) | x.eq(y)).evaluate()
    assert relation.contains(x=2, y=5)
    assert relation.contains(x=5, y=5)
    assert not relation.contains(x=6, y=5)


def test_complement(S):
    x, y = S.vars('x y')
    relation = (~x.lt(y)).evaluate()
    assert relation.contains(x=5, y=2)
    assert not relation.contains(x=2, y=5)


def test_drop_projects(S):
    x, y, z = S.vars('x y z')
    relation = ((x + y).eq(z)).drop(y).evaluate()
    assert relation.variables == ['x', 'z']
    assert relation.contains(x=3, z=9)


def test_universal_quantification(S):
    x, y = S.vars('x y')
    # every integer has a strictly larger one
    assert x.lt(y).drop(x).all(y).check()
    # but not every integer is below a fixed bound
    assert not x.lt(S.const(5)).all(x).check()


def test_implication_and_iff(S):
    x, y = S.vars('x y')
    assert x.lt(y).implies(~y.lt(x)).all([x, y]).check()
    assert x.eq(y).iff(~(x.lt(y) | y.lt(x))).all([x, y]).check()


def test_exinf(S):
    x, y = S.vars('x y')
    # infinitely many y exceed each x
    assert x.lt(y).exinf(y).check()
    # but only finitely many lie strictly between 0 and x
    assert not (S.const(0).lt(y) & y.lt(x)).exinf(y).check()


# ----------------------------------------------------------------------
# immutability and substitution
# ----------------------------------------------------------------------

def test_expressions_compare_structurally(S):
    x, y = S.vars('x y')
    assert (x + y).eq(S.const(3)) == (x + y).eq(S.const(3))
    assert {(x + y).eq(S.const(3)), (x + y).eq(S.const(3))}.__len__() == 1


def test_substitution_leaves_the_original_alone(S):
    x, y = S.vars('x y')
    base = (x + y).eq(S.const(9))
    renamed = base.substitute(y='w')
    assert base.variables() == ['x', 'y']
    assert renamed.variables() == ['w', 'x']
    assert renamed.evaluate().contains(w=4, x=5)


def test_substitution_avoids_capture(S):
    x, y, z = S.vars('x y z')
    # y is bound here, so substituting x by y must rename the binder rather
    # than let the incoming y fall under the quantifier
    formula = ((x + y).eq(z)).drop(y).substitute(x='y')
    assert formula.variables() == ['y', 'z']
    relation = formula.evaluate()
    assert relation.contains(y=3, z=9)


def test_nodes_are_immutable(S):
    x, y = S.vars('x y')
    with pytest.raises(AttributeError, match='immutable'):
        (x + y).func = 'other'


# ----------------------------------------------------------------------
# evaluation surface
# ----------------------------------------------------------------------

def test_check_and_emptiness(S):
    x, y = S.vars('x y')
    assert (x + y).eq(S.const(4)).check()
    assert (x.lt(y) & y.lt(x)).is_empty()


def test_finiteness(S):
    x, y = S.vars('x y')
    assert ((x + y).eq(S.const(4)) & x.lt(y) & S.const(0).lt(x)).is_finite()
    assert not x.lt(y).is_finite()


def test_finiteness_ignores_padding_spellings(S):
    """A relation automaton accepts each tuple followed by any number of
    all-padding columns, so a word-level cycle test calls every non-empty
    relation infinite. Finiteness has to be asked of the tuples."""
    x, y = S.vars('x y')
    single_point = x.eq(S.const(1)) & y.eq(S.const(3))
    assert single_point.is_finite()
    assert not single_point.evaluate().dfa.is_finite()  # the word language


def test_iteration_decodes_through_the_codec(S):
    x, y = S.vars('x y')
    formula = (x + y).eq(S.const(4)) & S.const(0).lt(x) & x.lt(y)
    solutions = sorted(islice(iter(formula), 3))
    assert solutions == [(1, 3)]


def test_materialize_splices_a_compiled_automaton(S):
    x, y, z = S.vars('x y z')
    shared = ((x + y).eq(z)).materialize('sum')
    combined = (shared & z.lt(S.const(100))).evaluate()
    assert combined.variables == ['x', 'y', 'z']
    assert combined.contains(x=3, y=4, z=7)
    assert not combined.contains(x=60, y=60, z=120)


def test_compile_reports_the_query_and_tape_order(S):
    x, y, z = S.vars('x y z')
    expression, variables = S.compile((x + y).eq(z))
    assert variables == ['x', 'y', 'z']
    assert 'A(' in str(expression) and 'Eq(' in str(expression)


# ----------------------------------------------------------------------
# structures without a signature
# ----------------------------------------------------------------------

def test_bare_structure_needs_no_signature():
    S = BuechiArithmeticZ().symbolic()
    x, y, z = S.vars('x y z')
    relation = S.atom('A', [x, y, z]).evaluate()
    assert relation.variables == ['x', 'y', 'z']


def test_constants_need_a_codec():
    S = BuechiArithmeticZ().symbolic()
    with pytest.raises(SymbolicSymbolError, match='no element codec'):
        S.const(3)


# ----------------------------------------------------------------------
# uniformly automatic classes
# ----------------------------------------------------------------------

@pytest.fixture(scope='module')
def depth2():
    from autstr.graphs import TreeDepthClass
    return TreeDepthClass(2)


def test_class_arities_exclude_the_advice_tape(depth2):
    K = depth2.symbolic()
    assert K.rel('E').arity == 2
    assert K.rel('Sing').arity == 1


def test_class_query_carries_the_advice_tape(depth2):
    K = depth2.symbolic()
    x, y = K.vars('x y')
    relation = (~K.rel('Sing')(x) | K.rel('E')(x, y).drop(y)).all(x).evaluate()
    assert relation.variables == ['advice']


def test_class_query_agrees_with_the_string_api(depth2):
    import networkx as nx
    from autstr.graphs import TreeDepthGraph

    K = depth2.symbolic()
    x, y = K.vars('x y')
    symbolic = (~K.rel('Sing')(x) | K.rel('E')(x, y).drop(y)).all(x)
    string = 'all x.((not Sing(x)) or exists y.(E(x,y)))'

    for graph in [nx.star_graph(1), nx.empty_graph(2), nx.path_graph(3)]:
        g = TreeDepthGraph.from_networkx(graph)
        assert K.check_member(symbolic, depth2.advice(g)) == depth2.check(string, g)


def test_class_member_becomes_an_ordinary_structure(depth2):
    import networkx as nx
    from autstr.graphs import TreeDepthGraph

    K = depth2.symbolic()
    g = TreeDepthGraph.from_networkx(nx.star_graph(1))
    member = K.get_structure(depth2.advice(g)).symbolic()
    a, b = member.vars('a b')
    assert member.rel('E')(a, b).evaluate().variables == ['a', 'b']


def test_class_constants_are_refused(depth2):
    K = depth2.symbolic()
    with pytest.raises(SymbolicSymbolError, match='no element codec'):
        K.const(3)


def test_class_reserves_the_advice_name(depth2):
    K = depth2.symbolic()
    with pytest.raises(SymbolicSymbolError, match='reserved'):
        K.var('advice')


def test_substitution_avoids_capture_by_inner_binders(S):
    """The fresh binder has to dodge names bound deeper in the body, not just
    the free ones."""
    x, y, z = S.vars('x y z')
    v0 = S.var('_v0')
    # `v0.eq(v0)` is trivially true, so the conjunct means exactly what it did
    # before; it is there so that '_v0' occurs and can be bound.
    inner = ((x + y).eq(z) & ((x + z).eq(y) & v0.eq(v0)).drop('_v0'))
    formula = inner.drop(y).substitute(x='y')
    assert formula.variables() == ['y', 'z']
    assert not formula.evaluate().is_empty()


def test_member_evaluation_is_refused_on_a_plain_structure(S):
    x, = S.vars('x')
    with pytest.raises(NotImplementedError, match='no members'):
        S.check_member(x.lt(S.const(3)), advice=[])


def test_class_implicit_and_explicit_checking_agree(depth2):
    """Implicit evaluation never builds a query automaton; it must still give
    the same answers as the compiled one."""
    import networkx as nx
    from autstr.graphs import TreeDepthGraph

    K = depth2.symbolic()
    depth2.cls.element_alphabet = ['0', '1']
    x, y = K.vars('x y')
    formula = (~K.rel('Sing')(x) | K.rel('E')(x, y).drop(y)).all(x)

    for graph in [nx.star_graph(1), nx.empty_graph(2)]:
        advice = depth2.advice(TreeDepthGraph.from_networkx(graph))
        assert (K.check_member(formula, advice, implicit=True)
                == K.check_member(formula, advice))


def test_whitespace_variable_names_are_split(S):
    """`.all('x y')` must bind both, the way `vars('x y')` does.

    Binding one variable literally named 'x y' left x and y free, and free
    variables are existentially closed by `check` -- so a universal silently
    became an existential and the query answered a different question.
    """
    x, y = S.vars('x y')
    # x + y = y holds exactly when x = 0, so the universal is false and the
    # existential is true; a mis-bound quantifier makes both come out true.
    assert not (x + y).eq(y).all('x y').check()
    assert (x + y).eq(y).drop('x y').check()


def test_quantifying_a_variable_that_is_not_free_is_an_error(S):
    x, y = S.vars('x y')
    with pytest.raises(SymbolicSymbolError, match='not free'):
        (x + y).eq(y).all('w')
    with pytest.raises(SymbolicSymbolError, match='not free'):
        (x + y).eq(y).drop(['x', 'w'])
