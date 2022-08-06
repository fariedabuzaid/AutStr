
from nltk.sem import logic
from automata.fa.dfa import DFA
from typing import Dict, Optional

from austr.buildin.automata import zero, one
from austr.utils.automata import stringlify_states, product, projection, expand, unpad, pad
from austr.utils.logic import get_free_elementary_vars


class AutomaticPresentation:
    """
    A presentation of a possibly infinite structure by finite state machines.
    """

    def __init__(self, automata: Dict[str, DFA]):
        """
        :param automata: dictionary containing the automata that recognize the domain and the relations of the structure.
            'U' is reserved key for the universe. All other keys are assumed to recognize relations over L(U)^k. They
            can be addressed by their keys in first-order queries.
        """
        universe = pad(automata['U']).minify()
        self.automata = {'U': universe}
        for R in automata:
            if R != 'U':
                arity = len(list(automata[R].input_symbols)[0])
                domain = product(universe, arity)
                self.automata[R] = pad(stringlify_states(automata[R])).intersection(domain).minify()

    def update(self, **kwargs) -> None:
        for key in kwargs:
            if isinstance(kwargs[key], DFA):
                self.automata[key] = self._prepare_automaton(kwargs[key])
            elif isinstance(kwargs[key], str):
                self.automata[key] = self._prepare_automaton(self._build_automaton(kwargs[key]))

    def _prepare_automaton(self, dfa) -> DFA:
        """Applies restriction to the universe and padding to the automaton"""
        arity = len(list(dfa.input_symbols)[0])
        domain = product(self.automata['U'], arity)
        return pad(stringlify_states(dfa)).intersection(domain).minify()

    def check(self, phi: logic.Expression) -> bool:
        """Checks if a given first-order formula holds on the presented structure. Free variables are assumed be
        implicitly existentially quantified.
        :param phi: the first order formula
        :returns: the truth value of the formula, if the formula where all free variables are existentially quantified.
        """
        return not self._build_automaton(phi).isempty()

    def evaluate(self, phi: logic.Expression) -> DFA:
        """Evaluates a given first-order query on the presented structure. Returns a presentation of the set of all
        satisfying assignments.
        :param phi: the first order formula.
        :returns: the truth value of the formula, if the formula where all free variables are existentially quantified.
        """
        if isinstance(phi, str):
            phi = logic.Expression.fromstring(phi)

        dfa_phi = self._build_automaton(phi)
        if len(get_free_elementary_vars(phi)) > 0:
            return unpad(self._build_automaton(phi))
        else:
            return dfa_phi

    def _build_automaton(self, phi: logic.Expression, verbose=False) -> DFA:
        """
        Creates a padded presentation of the satisfying assignments of phi.
        :param phi: The formula
        :param free_vars: Variable dictionary. All variables that scope the current formula. The result will be
            len(free_vars)-ary. The dictionary maps each variable to it's position in the tuple.
        :return: Padded presentation of the satisfying assignments of phi
        """
        if isinstance(phi, str):
            phi = logic.Expression.fromstring(phi)

        if isinstance(phi, logic.AllExpression):

            variable = str(phi.variable)
            free_vars = get_free_elementary_vars(phi.term)

            if variable not in free_vars:
                return self._build_automaton(phi.term)
            else:
                psi = (phi.term.negate()).simplify()
                dfa_rec = stringlify_states(self._build_automaton(psi).minify())

                pos = free_vars.index(variable)

                if len(free_vars) > 1:
                    domain = stringlify_states(product(self.automata['U'], len(free_vars) - 1))
                    result = projection(
                        dfa_rec,
                        pos
                    ).minify().complement().minify().intersection(domain).minify()
                else:
                    if dfa_rec.isempty():
                        result = one()
                    else:
                        result = zero()

                print(f'{str(phi)}: {len(result.states)} states')
                return result
        elif isinstance(phi, logic.ExistsExpression):
            psi = phi.term
            variable = str(phi.variable)
            free_vars = get_free_elementary_vars(psi)
            dfa_rec = stringlify_states(self._build_automaton(psi))

            if variable in free_vars:
                pos = free_vars.index(variable)

                if len(free_vars) > 1:
                    result = projection(dfa_rec, pos).minify()
                else:
                    print(f'{str(phi)}: 1 state')
                    if dfa_rec.isempty():
                        return zero()
                    else:
                        return one()
            else:
                result = stringlify_states(self._build_automaton(psi))

            result = pad(unpad(result))

            print(f'{str(phi)}: {len(result.states)} states')
            return result
        elif isinstance(phi, logic.AndExpression):
            left = phi.first
            right = phi.second

            free_vars = get_free_elementary_vars(phi)
            free_l = get_free_elementary_vars(left)
            free_r = get_free_elementary_vars(right)

            dfa_l = expand(self._build_automaton(left), len(free_vars), pos=[free_vars.index(v) for v in free_l])
            dfa_r = expand(self._build_automaton(right), len(free_vars), pos=[free_vars.index(v) for v in free_r])

            result = dfa_l.intersection(dfa_r).minify()
            print(f'{str(phi)}: {len(result.states)} states')
            return result

        elif isinstance(phi, logic.NegatedExpression):
            psi = phi.term
            free_vars = get_free_elementary_vars(phi)

            domain = product(self.automata['U'], len(free_vars))

            result = self._build_automaton(psi).complement()
            result = result.intersection(domain).minify()
            print(f'{str(phi)}: {len(result.states)} states')
            return result
        elif isinstance(phi, logic.OrExpression):
            left = phi.first
            right = phi.second

            free_vars = get_free_elementary_vars(phi)
            free_l = get_free_elementary_vars(left)
            free_r = get_free_elementary_vars(right)

            dfa_l = expand(self._build_automaton(left), len(free_vars), pos=[free_vars.index(v) for v in free_l])
            dfa_r = expand(self._build_automaton(right), len(free_vars), pos=[free_vars.index(v) for v in free_r])

            result = dfa_l.union(dfa_r).minify()
            print(f'{str(phi)}: {len(result.states)} states')
            return result
        elif isinstance(phi, logic.ApplicationExpression):
            R = str(phi.pred)
            variables = get_free_elementary_vars(phi)

            result = expand(
                self.automata[R],
                len(variables),
                [variables.index(v) for v in [str(v) for v in phi.args]]
            ).minify()

            print(f'{str(phi)}: {len(result.states)} states')
            return result
