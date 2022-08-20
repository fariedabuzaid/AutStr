from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy, copy
from typing import List, Union, Tuple, Dict
import math

from austr.utils.automata_tools import lsbf_automaton, iterate_language
from austr.buildin.presentations import buechi_arithmetic
from austr.utils.misc import get_unique_id


class Term(ABC):
    """
    Abstract class representing a term over the (base 2) Büchi arithmetic
    """
    arithmetic = buechi_arithmetic()

    def __init__(self):
        self.presentation = None

    @abstractmethod
    def update_presentation(self, recursive=True):
        """
        Updates the internal presentation of the term
        :param recursive: If True, recursively updates the presentation of all subrelations
        :return:
        """
        raise NotImplementedError

    def evaluate(self):
        """
        Returns automatic presentation of the relation.
        :return:
        """
        if self.presentation is None:
            self.update_presentation()

        return self.presentation

    @abstractmethod
    def get_variables(self):
        """
        Get all free variables of a term
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def substitute(self, allow_collision: bool = False, **kwargs):
        """
        Substitute variable names in the relation
        :param allow_collision: if True, does not check collision with quantified
        :param kwargs:
        :return:
        """
        raise NotImplementedError


class RelationalAlgebraTerm(Term, ABC):
    """
    Abstract class that
    """
    def __and__(self, other: RelationalAlgebraTerm):
        """
        Intersection
        :param other:
        :return: A term that presents the intersection of self and other
        """
        return IntersectionRATerm(self, other)

    def __or__(self, other: RelationalAlgebraTerm):
        """
        Union
        :param other:
        :return: A term that presents the union of self and other
        """
        return UnionRATerm(self, other)

    def __invert__(self):
        """
        Complement
        :return: A term that presents the complement of the current relation
        """
        return ComplementRATerm(self)

    def drop(self, variables: List):
        """
        Drop variables
        :param variables: The variables to drop
        :return: presentation of the projection of the current relation onto the variables self.get_variables without
            @variables
        """
        return DropRARelation(self, variables)

    def isempty(self):
        """
        Checks if the current relation is empty
        :return: True, if self presents an empty relation
        """
        if self.presentation is None:
            self.update_presentation()

        return self.presentation.isempty()

    def isfinite(self) -> bool:
        """
        checks if the number of solutions is finite.
        :return: True, if the relation contains only finitely many tuples
        """
        if self.presentation is None:
            self.update_presentation()

        return self.presentation.isfinite()

    def __iter__(self):
        """
        Iterates all solutions by successively enumerating all solution tuples smaller than (2^n,...,2^n) in
        lexicographic order. The procedure guarantees that every solution tuple is enumerated exactly once.
        :return:
        """
        if self.presentation is None:
            self.update_presentation()

        for t in iterate_language(self.presentation, backward=True):
            yield tuple(int(n.replace('*', ''), base=2) for n in t)


class BaseRATerm(RelationalAlgebraTerm):
    """
    Represents a term of the form R(t1,...,tn) for elementary terms t1,...,tn
    """
    def substitute(self, allow_collision: bool = False, **kwargs):
        kwargs = {
            str(x): ElementaryTerm.to_term(kwargs[x]) for x in kwargs
        }
        for i, t in enumerate(self.terms):
            if str(t) in kwargs.keys():
                self.terms[i] = kwargs[str(t)]
            else:
                t.substitute(allow_collision, **kwargs)

        self.presentation = None

        return self

    def __init__(self, relation_symbol, terms):
        super(BaseRATerm, self).__init__()
        self.R = relation_symbol
        self.terms = [ConstantETerm(t) if isinstance(t, int) else t for t in terms]

    def get_variables(self):
        variables = []
        for t in self.terms:
            variables = variables + t.get_variables()
        variables = list(set(variables))
        variables.sort()
        return variables

    def update_presentation(self, recursive=True, **kwargs):
        phi, update = self.to_fo()

        update = {R: update[R].evaluate() for R in update}
        arithmetic = deepcopy(self.arithmetic)
        arithmetic.update(**update)
        self.presentation = arithmetic.evaluate(phi)

    def to_fo(self) -> Tuple[str, Dict]:
        """
        Creates the a translation of the atomic formula R(t1, ..., tn) into a relational first-order formula with new
        predicates for T1,..., Tn for the graphs of t1,...,tn. The result will be of shape "exists y1,...,yn.(T1(y1)
        and ... and Tn(yn) and R(y1,...yn))". The method guarantees that the newly created relation symbols T1,...,Tn
        do not collide with already defined relation symbols.
        :return: The relational formula and the mapping of new relation symbols to terms
        """
        phi = self.R + '({})'

        arithmetic = deepcopy(self.arithmetic)

        unique_vars = get_unique_id(self.get_variables(), len(self.terms))
        unique_rels = get_unique_id(arithmetic.get_relation_symbols(), len(self.terms))

        final_vars = []

        updates = {}
        for R, t, x in zip(unique_rels, self.terms, unique_vars):
            if isinstance(t, VariableETerm):
                phi.format(t.get_name())
                final_vars.append(t.get_name())
            else:
                final_vars.append(x)
                guard = R + '(' + ','.join(t.get_variables() + [x]) + ')'
                phi = f'exists {x}.({guard} and {phi})'
                updates[R] = t

        phi = phi.format(','.join(final_vars))

        return phi, updates


class BinaryRATerm(RelationalAlgebraTerm, ABC):
    """
    Abstract class that represents binary relational algebra terms.
    """
    def substitute(self, allow_collision: bool = False, **kwargs):
        self.left.substitute(allow_collision, **kwargs)
        self.right.substitute(allow_collision, **kwargs)
        self.presentation = None

        return self

    def get_variables(self):
        result = list(set(self.left.get_variables() + self.right.get_variables()))
        result.sort()
        return result

    def __init__(self, left: RelationalAlgebraTerm, right: RelationalAlgebraTerm):
        super().__init__()
        self._template = None
        self.left = left
        self.right = right

    def update_presentation(self, recursive=True):
        """
            Builds presentation from the two sub-relations and combines then through a logical formula
            :param template: a first order formula with as str with 2 placeholders that will be replaced by the relations
                for the two sub-formulae.
            :return:
            """
        if recursive:
            self.left.update_presentation(recursive=recursive)
            self.right.update_presentation(recursive=recursive)

        arithmetic = deepcopy(self.arithmetic)
        R0, R1 = get_unique_id(arithmetic.get_relation_symbols(), 2)
        psi_R0 = R0 + '(' + ','.join(self.left.get_variables()) + ')'
        psi_R1 = R1 + '(' + ','.join(self.right.get_variables()) + ')'
        phi = self._template.format(psi_R0, psi_R1)
        arithmetic.update(**{R0: self.left.evaluate(), R1: self.right.evaluate()})
        self.presentation = arithmetic.evaluate(phi)

class IntersectionRATerm(BinaryRATerm):
    """
    Intersection of two relations.
    """
    def __init__(self, left: RelationalAlgebraTerm, right: RelationalAlgebraTerm):
        super(IntersectionRATerm, self).__init__(left, right)
        self._template = "(({} and {}))"



class UnionRATerm(BinaryRATerm):
    """
    Union of two relations.
    """
    def __init__(self, left: RelationalAlgebraTerm, right: RelationalAlgebraTerm):
        super(IntersectionRATerm, self).__init__(left, right)
        self._template = "(({} or {}))"


class ComplementRATerm(RelationalAlgebraTerm):
    """
    The complement of a relation
    """
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


class DropRARelation(RelationalAlgebraTerm):
    """
    Relation of the shape {(x1,...,xn) | (x1,...,xn,y1,...,yn) in R}
    """
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


class ElementaryTerm(Term, ABC):
    @classmethod
    def to_term(self, x: Union[str, int, ElementaryTerm]):
        """
        Classmethod for converting str and int into variables and constants, respectively
        :param x: The input parameter
        :return: the term tht presents x
        """
        return VariableETerm(x) if isinstance(x, str) else ConstantETerm(x) if isinstance(x, int) else x

    def __init__(self):
        super().__init__()
        self.presentation = None

    def eq(self, other: ElementaryTerm):
        """
        Creates the relation self = other
        :param other: the rhs of the equality
        :return: Equ
        """
        return BaseRATerm('Eq', [self, other])

    def lt(self, other):
        """
        Creates the relation self < other
        :param other: The term on the rhs
        :return:
        """
        return BaseRATerm('Lt', [self, other])

    def gt(self, other):
        """
        Creates the relation other < self
        :param other: The term on the lhs
        :return:
        """
        return BaseRATerm('Lt', [other, self])

    def evaluate(self):
        if self.presentation is None:
            self.update_presentation()

        return self.presentation

    def __add__(self, other: ElementaryTerm) -> AdditionETerm:
        """
        Creates the term self + other
        :param other:
        :return:
        """
        if isinstance(other, int):
            other = ConstantETerm(other)

        return AdditionETerm(self, other)

    def __mul__(self, other: ConstantETerm) -> AdditionETerm:
        """
        Creates a term that is equivalent to self*other in linear arithmetic. Note that other needs to be a constant.
        The method creates a nested addition and guarantees to create only O(log2(other)) many distinct terms on object
        level.
        :param other: The constant to multiply with
        :return: term that expresses the other-fold summation of self
        """
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
        """
        Creates a term equivalent to other*self. Uses commutativity.
        :param other: The Constant to multiply
        :return:
        """
        return self.__mul__(other)

    def __or__(self, other):
        """
        creates a relational algebra term that represents self | other. The semantics of | is given as x | y iff
        y = 2^n for some n and y divides x.
        :param other:
        :return:
        """
        if isinstance(other, int):
            other = ConstantETerm(other)
        return BaseRATerm(relation_symbol="B", terms=[self, other])

    @abstractmethod
    def update_presentation(self, **kwargs) -> None:
        raise NotImplementedError


class ConstantETerm(ElementaryTerm):
    def substitute(self, allow_collision: bool = False, **kwargs):
        return self

    def get_variables(self) -> List[str]:
        return []

    def update_presentation(self, **kwargs) -> None:
        self.presentation = lsbf_automaton(self.n)

    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def __hash__(self):
        return self.n


class VariableETerm(ElementaryTerm):
    def substitute(self, allow_collision: bool = False, **kwargs):
        return self

    def update_presentation(self, **kwargs):
        arithmetic = deepcopy(self.arithmetic)
        self.presentation = arithmetic.automata['U']

    def get_variables(self) -> List[str]:
        return [self.get_name()]

    def get_name(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        """
        equality is based on the name of the variable
        :param other: The other Variable
        :return:
        """
        if isinstance(other, VariableETerm):
            return self.name == other.name
        elif isinstance(str):
            return self.name == other
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


class AdditionETerm(ElementaryTerm):
    def substitute(self, allow_collision: bool = False, **kwargs):
        kwargs = {
            str(x): ElementaryTerm.to_term(kwargs[x]) for x in kwargs
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

    def update_presentation(self, **kwargs) -> None:
        left_is_var = isinstance(self.left, VariableETerm)
        right_is_var = isinstance(self.right, VariableETerm)

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
        if isinstance(other, AdditionETerm):
            return self.left == other.left and self.right == other.right
        else:
            return False

    def __init__(self, left: ElementaryTerm, right: ElementaryTerm):
        super().__init__()
        self.left = left
        self.right = right

