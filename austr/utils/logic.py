import nltk
from nltk.sem.logic import Expression
from typing import Set, List


def get_free_elementary_vars(phi: Expression) -> List[str]:
    """
    Get an ordered list of all free elementary variables of phi
    :param phi: The formula
    :return: Ordered list with all elementary variable names
    """
    types = phi.typecheck()
    free_vars = [
        str(v) for v in [x for x in phi.free() if isinstance(types[str(x)], nltk.sem.logic.EntityType)]
    ]
    free_vars.sort()

    return free_vars

