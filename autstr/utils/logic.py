import nltk
from nltk.sem.logic import Expression, AllExpression, ExistsExpression, NegatedExpression, ApplicationExpression, AndExpression, OrExpression
from typing import List


def get_free_elementary_vars(phi: Expression) -> List[str]:
    """
    Get an ordered list of all free elementary variables of phi.

    :param phi: The formula
    :return: Ordered list with all elementary variable names
    """
    types = phi.typecheck()
    free_vars = [
        str(v) for v in [x for x in phi.free() if isinstance(types[str(x)], nltk.sem.logic.EntityType)]
    ]
    free_vars.sort()

    return free_vars

def optimize_query(query: Expression) -> Expression:
    """
    Optimize a query for optimized automata construction.

    :param query: The query expression
    :return: Optimized query expression
    """
    if isinstance(query, AllExpression):
        query.term = optimize_query(query.term)
        if query.variable.name in get_free_elementary_vars(query):
            return query.term
        else:
            return query
    elif isinstance(query, ExistsExpression):
        query.term = optimize_query(query.term)
        if query.variable.name in get_free_elementary_vars(query):
            return query.term
        else:
            return query
    elif isinstance(query, NegatedExpression):
        if isinstance(query.term, AllExpression):
            term = optimize_query(NegatedExpression(query.term))
            return ExistsExpression(query.variable, term)
        elif isinstance(query.term, NegatedExpression):
            return optimize_query(query.term.term)
        else:
            return NegatedExpression(optimize_query(query.term))
    else:
        return query