from abc import ABC, abstractmethod
from copy import deepcopy, copy
from typing import List, Union, Tuple, Dict
import math

from austr.utils.automata_tools import lsbf_automaton, iterate_language
from austr.buildin.presentations import buechi_arithmetic
from austr.utils.misc import get_unique_id


class Relation(ABC):
    arithmetic = buechi_arithmetic()

    def __init__(self):
        self.presentation = None

    @abstractmethod
    def update_presentation(self):
        raise NotImplementedError

    def evaluate(self):
        """
        Returns automatic presentation of the relation
        :return:
        """
        if self.presentation is None:
            self.update_presentation()

        return self.presentation

    @abstractmethod
    def get_variables(self):
        raise NotImplementedError

    @abstractmethod
    def substitute(self, allow_collision: bool = False, **kwargs):
        raise NotImplementedError

    def __and__(self, other):
        return IntersectionRelation(self, other)

    def __or__(self, other):
        return UnionRelation(self, other)

    def __invert__(self):
        return ComplementRelation(self)

    def drop(self, variables: List):
        return ProjectionRealtion(self, variables)

    def isempty(self):
        if self.presentation is None:
            self.update_presentation()

        return self.presentation.isempty()

    def isfinite(self):
        if self.presentation is None:
            self.update_presentation()

        return self.presentation.isfinite()

    def __iter__(self):
        if self.presentation is None:
            self.update_presentation()

        for t in iterate_language(self.presentation, backward=True):
            yield tuple(int(n.replace('*', ''), base=2) for n in t)


class BaseRelation(Relation):
    def substitute(self, allow_collision: bool = False, **kwargs):
        kwargs = {
            str(x): Term.to_term(kwargs[x]) for x in kwargs
        }
        for i, t in enumerate(self.terms):
            if str(t) in kwargs.keys():
                self.terms[i] = kwargs[str(t)]
            else:
                t.substitute(allow_collision, **kwargs)

        self.presentation = None

        return self

    def __init__(self, relation_symbol, terms):
        super(BaseRelation, self).__init__()
        self.R = relation_symbol
        self.terms = [ConstantTerm(t) if isinstance(t, int) else t for t in terms]

    def get_variables(self):
        variables = []
        for t in self.terms:
            variables = variables + t.get_variables()
        variables = list(set(variables))
        variables.sort()
        return variables

    def update_presentation(self):
        phi, update = self.to_fo()

        update = {R: update[R].evaluate() for R in update}
        arithmetic = deepcopy(self.arithmetic)
        arithmetic.update(**update)
        self.presentation = arithmetic.evaluate(phi)

    def to_fo(self) -> Tuple[str, Dict]:
        """
        Creates the a translation of the atomic formula R(t1, ..., tn) into relational form with new predicates for
        the graphs of t1,...,tn.
        :return: The relational formula and the mapping of new relation symbols to terms
        """
        phi = self.R + '({})'

        arithmetic = deepcopy(self.arithmetic)

        unique_vars = get_unique_id(self.get_variables(), len(self.terms))
        unique_rels = get_unique_id(arithmetic.get_relation_symbols(), len(self.terms))

        final_vars = []

        updates = {}
        for R, t, x in zip(unique_rels, self.terms, unique_vars):
            if isinstance(t, VariableTerm):
                phi.format(t.get_name())
                final_vars.append(t.get_name())
            else:
                final_vars.append(x)
                guard = R + '(' + ','.join(t.get_variables() + [x]) + ')'
                phi = f'exists {x}.({guard} and {phi})'
                updates[R] = t

        phi = phi.format(','.join(final_vars))

        return phi, updates


class Term(Relation, ABC):
    @classmethod
    def to_term(self, x: Union[str, int, object]):
        return VariableTerm(x) if isinstance(x, str) else ConstantTerm(x) if isinstance(x, int) else x

    def __init__(self):
        super().__init__()
        self.presentation = None

    def eq(self, other):
        return BaseRelation('Eq', [self, other])

    def lt(self, other):
        return BaseRelation('Lt', [self, other])

    def gt(self, other):
        return BaseRelation('Lt', [other, self])

    def evaluate(self):
        if self.presentation is None:
            self.update_presentation()

        return self.presentation

    @abstractmethod
    def get_variables(self) -> List[str]:
        raise NotImplementedError

    def __add__(self, other):
        if isinstance(other, int):
            other = ConstantTerm(other)

        return AdditionTerm(self, other)

    def __mul__(self, other):
        if isinstance(other, int):
            # Reduce number of unique terms by base 2 decomposition
            if other == 0:
                n_bits = 1
            else:
                n_bits = math.floor(math.log(other, 2)) + 1
            power_multiples = None
            term = None
            for _ in range(n_bits):
                if power_multiples is None:
                    power_multiples = [self]
                else:
                    power_multiples.append(power_multiples[-1] + power_multiples[-1])
                if other % 2 == 1:
                    if term is None:
                        term = power_multiples[-1]
                    else:
                        term = term + power_multiples[-1]
                    other = int((other - 1) / 2)
                else:
                    other = int(other / 2)

            return term
        else:
            raise ValueError('Can multiply only with natural numbers')

    def __rmul__(self, other):
        return self.__mul__(other)

    def __or__(self, other):
        if isinstance(other, int):
            other = ConstantTerm(other)
        return BaseRelation(relation_symbol="B", terms=[self, other])

    @abstractmethod
    def update_presentation(self) -> None:
        raise NotImplementedError


class ConstantTerm(Term):
    def substitute(self, allow_collision: bool = False, **kwargs):
        return self

    def get_variables(self) -> List[str]:
        return []

    def update_presentation(self) -> None:
        self.presentation = lsbf_automaton(self.n)

    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def __hash__(self):
        return self.n


class VariableTerm(Term):
    def substitute(self, allow_collision: bool = False, **kwargs):
        return self

    def update_presentation(self):
        arithmetic = deepcopy(self.arithmetic)
        self.presentation = arithmetic.automata['U']

    def get_variables(self) -> List[str]:
        return [self.get_name()]

    def get_name(self) -> str:
        return self.name

    def __eq__(self, other):
        if isinstance(other, VariableTerm):
            return self.name == other.name
        else:
            return False

    def __init__(self, name: str):
        """
        Initialization.
        :param name: The name of the variable
        """
        super().__init__()
        self.name = name

    def __hash__(self):
        return int.from_bytes(self.name.encode(), 'little')

    def __str__(self):
        return self.name


class AdditionTerm(Term):
    def substitute(self, allow_collision: bool = False, **kwargs):
        kwargs = {
            str(x): Term.to_term(kwargs[x]) for x in kwargs
        }
        if str(self.left) in kwargs:
            self.left = kwargs[str(self.left)]
        else:
            self.left.substitute(allow_collision, **kwargs)

        if str(self.right) in kwargs:
            self.right = kwargs[str(self.right)]
        else:
            self.right.substitute(allow_collision, **kwargs)

        self.presentation = None

        return self

    def update_presentation(self) -> None:
        """

        :param base_presentation:
        :param recursive:
        :return:
        """

        left_is_var = isinstance(self.left, VariableTerm)
        right_is_var = isinstance(self.right, VariableTerm)

        phi = 'A({}, {}, {})'

        x0, y0, z = get_unique_id(self.get_variables(), 3)
        arithmetic = deepcopy(self.arithmetic)
        R0, R1 = get_unique_id(arithmetic.get_relation_symbols(), 2)

        if left_is_var:
            x = self.left.get_name()
        else:
            x = x0
            left_vars = self.left.get_variables()
            left_vars.sort()
            args = ','.join(left_vars + [x0])
            psi = f'{R0}({args})'

            phi = f'exists {x0}.({psi} and {phi})'

        if right_is_var:
            y = self.right.get_name()
        else:
            y = y0
            right_vars = self.right.get_variables()
            right_vars.sort()
            args = ','.join(right_vars + [y0])
            psi = f"{R1}({args})"

            phi = f'exists {y0}.({psi} and {phi})'

        phi = phi.format(x, y, z)
        arithmetic = deepcopy(self.arithmetic)
        updates = {R0: self.left.evaluate(), R1: self.right.evaluate()}
        arithmetic.update(**updates)
        self.presentation = arithmetic.evaluate(
            phi
        )

    def get_variables(self) -> List[str]:
        """
        get ordered list of all free variables in the term
        :return:
        """
        result = list(set(self.left.get_variables() + self.right.get_variables()))
        result.sort()
        return result

    def __eq__(self, other):
        if isinstance(other, AdditionTerm):
            return self.left == other.left and self.right == other.right
        else:
            return False

    def __init__(self, left: Term, right: Term):
        super().__init__()
        self.left = left
        self.right = right


class IntersectionRelation(Relation):
    def substitute(self, allow_collision: bool = False, **kwargs):
        self.left.substitute(allow_collision, **kwargs)
        self.right.substitute(allow_collision, **kwargs)
        self.presentation = None

        return self

    def get_variables(self):
        result = list(set(self.left.get_variables() + self.right.get_variables()))
        result.sort()
        return result

    def __init__(self, left: Relation, right: Relation):
        super().__init__()
        self.left = left
        self.right = right

    def update_presentation(self):
        arithmetic = deepcopy(self.arithmetic)
        R0, R1 = get_unique_id(arithmetic.get_relation_symbols(), 2)
        psi_R0 = R0 + '(' + ','.join(self.left.get_variables()) + ')'
        psi_R1 = R1 + '(' + ','.join(self.right.get_variables()) + ')'
        phi = f'({psi_R0}) and ({psi_R1})'
        arithmetic.update(**{R0: self.left.evaluate(), R1: self.right.evaluate()})
        self.presentation = arithmetic.evaluate(phi)


class UnionRelation(Relation):
    def substitute(self, allow_collision: bool = False, **kwargs):
        self.left.substitute(allow_collision, **kwargs)
        self.right.substitute(allow_collision, **kwargs)
        self.presentation = None

        return self

    def get_variables(self):
        result = list(set(self.left.get_variables() + self.right.get_variables()))
        result.sort()
        return result

    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def update_presentation(self):
        arithmetic = deepcopy(self.arithmetic)
        R0, R1 = get_unique_id(arithmetic.get_relation_symbols(), 2)
        psi_R0 = R0 + '(' + ','.join(self.left.get_variables()) + ')'
        psi_R1 = R1 + '(' + ','.join(self.right.get_variables()) + ')'
        phi = f'({psi_R0}) or ({psi_R1})'
        arithmetic.update(**{R0: self.left.evaluate(), R1: self.right.evaluate()})
        self.presentation = arithmetic.evaluate(phi)


class ComplementRelation(Relation):
    def substitute(self, allow_collision: bool = False, **kwargs):
        self.relation.substitute(allow_collision, **kwargs)
        self.presentation = None

        return self

    def __init__(self, relation):
        super().__init__()
        self.relation = relation

    def update_presentation(self):
        arithmetic = deepcopy(self.arithmetic)
        R0 = get_unique_id(arithmetic.get_relation_symbols(), 1)
        psi_R0 = R0 + '(' + ','.join(self.relation.get_variables()) + ')'
        phi = f'not ({psi_R0})'
        arithmetic.update(**{R0: self.relation.evaluate()})
        self.presentation = arithmetic.evaluate(phi)

    def get_variables(self):
        return self.relation.get_variables()


class ProjectionRealtion(Relation):
    def substitute(self, allow_collision: bool = False, **kwargs):
        kwrec = copy(kwargs)
        for x in self.variables:
            if x in kwargs:
                del kwrec[x]

        if not allow_collision:
            for i, v in enumerate(self.variables):
                if str(v) in kwargs.values():
                    v_new = get_unique_id(self.relation.get_variables(), 1)
                    self.variables[i] = v_new
                    self.relation.substitute(**{str(v): v_new})

        self.relation.substitute(**kwrec)
        self.presentation = None

        return self

    def update_presentation(self):
        ex_args = ' '.join(self.variables)
        R_args = ','.join(self.relation.get_variables())
        arithmetic = deepcopy(self.arithmetic)
        R0 = get_unique_id(arithmetic.get_relation_symbols(), 1)

        phi = f'exists {ex_args}.({R0}({R_args}))'
        arithmetic.update(**{R0: self.relation.evaluate()})
        self.presentation = arithmetic.evaluate(phi)

    def get_variables(self):
        result = [v for v in self.relation.get_variables() if v not in self.variables]
        result.sort()
        return result

    def __init__(self, relation, variables):
        super().__init__()
        self.relation = relation
        self.variables = [
            str(x) for x in variables
        ]
