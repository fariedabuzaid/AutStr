from nltk.sem import logic
from typing import Dict, Optional, Union, List

from autstr.sparse_automata import SparseDFA
from autstr.utils.automata_tools import pad, unpad, product, projection, expand, stack
from autstr.buildin.automata import zero, one  
from autstr.utils.logic import get_free_elementary_vars, optimize_query
class AutomaticPresentation:
    """
    A presentation of a possibly infinite structure by finite state machines.
    """

    def __init__(self, automata: Dict[str, SparseDFA], padding_symbol: Optional[any] = "*", enforce_consistency: bool = True) -> None:
        """
        :param automata: dictionary containing the automata that recognize the domain and the relations of the structure.
            'U' is reserved key for the universe. All other keys are assumed to recognize relations over L(U)^k. They
            can be addressed by their keys in first-order queries.
        """
        self.padding_symbol = padding_symbol
        universe = pad(automata['U'], padding_symbol=self.padding_symbol).minimize()
        self.automata = {'U': universe}
        # Prepare relation automata
        for R in automata:
            if R != 'U': 
                if enforce_consistency:
                    self.automata[R] = self._prepare_automaton(automata[R])
                else:
                    self.automata[R] = pad(automata[R]).minimize()

        self.sigma = automata['U'].base_alphabet  # Use base_alphabet for sigma

    def get_relation_symbols(self) -> List[str]:
        """
        Returns list of all defined relation symbols. The symbol 'U' must always be defined and denotes the Universe.

        :return: list of all defined relation symbols.
        """
        return list(self.automata.keys())

    def update(self, **kwargs) -> None:
        for key in kwargs:
            if isinstance(kwargs[key], SparseDFA):
                self.automata[key] = self._prepare_automaton(kwargs[key])
            elif isinstance(kwargs[key], str):
                query = optimize_query(logic.Expression.fromstring(kwargs[key]))
                self.automata[key] = self._prepare_automaton(self._build_automaton(query))

    def _prepare_automaton(self, dfa: SparseDFA) -> SparseDFA:
        """Applies restriction to the universe and padding to the automaton"""
        arity = dfa.symbol_arity  # Get arity from symbol_arity attribute
        domain = self.automata['U']
        for i in range(arity-1):
            domain_i = expand(domain, new_arity=arity, pos=[i]).minimize()
            dfa = dfa.intersection(domain_i).minimize()
        return pad(dfa, self.padding_symbol).minimize()

    def check(self, phi: logic.Expression | str) -> bool:
        """Checks if a given first-order formula holds on the presented structure. Free variables are assumed be
        implicitly existentially quantified.

        :param phi: the first order formula
        :returns: the truth value of the formula, if the formula where all free variables are existentially quantified.
        """
        if isinstance(phi, str):
            phi = logic.Expression.fromstring(phi)
        phi = phi.simplify()
        return not self._build_automaton(phi).is_empty()

    def evaluate(self, phi: Union[str, logic.Expression], updates: Optional[Dict[str, Union[SparseDFA, str]]] = None) -> SparseDFA:
        """Evaluates a given first-order query on the presented structure. Returns a presentation of the set of all
        satisfying assignments.

        :param phi: the first order formula.
        :param updates: Temporarily update the relations for the evaluation
        :returns: The truth value of the formula, if the formula where all free variables are existentially quantified.
        """
        if isinstance(phi, str):
            phi = logic.Expression.fromstring(phi)
        phi = optimize_query(phi)
        if updates is not None:
            for key in updates:
                if isinstance(updates[key], str):
                    query = optimize_query(logic.Expression.fromstring(updates[key]))
                    updates[key] = self._prepare_automaton(self._build_automaton(query))
                else:
                    updates[key] = self._prepare_automaton(updates[key])
            automata_backup = self.automata
            self.automata = dict(self.automata, **updates)

        if len(get_free_elementary_vars(phi)) > 0:
            dfa_phi = unpad(self._build_automaton(phi), self.padding_symbol).minimize()
        else:   
            dfa_phi = self._build_automaton(phi)

        if updates is not None:
            self.automata = automata_backup

        return dfa_phi

    def _build_automaton(self, phi: logic.Expression, verbose=True, init=True) -> SparseDFA:
        """
        Creates a padded presentation of the satisfying assignments of phi.

        :param phi: The formula
        :param free_vars: Variable dictionary. All variables that scope the current formula. The result will be
            len(free_vars)-ary. The dictionary maps each variable to it's position in the tuple.
        :return: Padded presentation of the satisfying assignments of phi
        """
        if init:
            if verbose:
                print(f'Building automaton for {str(phi)}')
            return self._build_automaton(phi, verbose=verbose, init=False)

        if isinstance(phi, str):
            phi = logic.Expression.fromstring(phi)

        if isinstance(phi, logic.AllExpression):

            variable = str(phi.variable)
            free_vars = get_free_elementary_vars(phi.term)

            if variable not in free_vars:
                return self._build_automaton(phi.term, verbose=verbose, init=False)
            else:
                psi = (phi.term.negate()).simplify()
                dfa_rec = self._build_automaton(psi, verbose=verbose, init=False).minimize()

                pos = free_vars.index(variable)

                if len(free_vars) > 1:
                    domain = product(self.automata['U'], len(free_vars) - 1)
                    result = projection(
                        dfa_rec,
                        pos
                    ).minimize().complement().minimize().intersection(domain).minimize()
                else:
                    if dfa_rec.is_empty():
                        result = one()
                    else:
                        result = zero()

                if verbose:
                    print(f'{str(phi)}: {result.num_states} states')
                return result
        elif isinstance(phi, logic.ExistsExpression):
            psi = phi.term
            variable = str(phi.variable)
            free_vars = get_free_elementary_vars(psi)
            dfa_rec = self._build_automaton(psi, verbose=verbose, init=False)

            if variable in free_vars:
                pos = free_vars.index(variable)

                if len(free_vars) > 1:
                    result = projection(dfa_rec, pos).minimize()
                else:
                    if verbose:
                        print(f'{str(phi)}: 1 state')
                    if dfa_rec.is_empty():
                        return zero()
                    else:
                        return one()
            else:
                result = self._build_automaton(psi, verbose=verbose, init=False)

            result = pad(unpad(result, self.padding_symbol).minimize(), self.padding_symbol).minimize()

            if verbose:
                print(f'{str(phi)}: {result.num_states} states')
            return result
        elif isinstance(phi, logic.AndExpression):
            left = phi.first
            right = phi.second

            free_vars = get_free_elementary_vars(phi)
            free_l = get_free_elementary_vars(left)
            free_r = get_free_elementary_vars(right)

            dfa_l = expand(self._build_automaton(left, verbose=verbose, init=False), len(free_vars), pos=[free_vars.index(v) for v in free_l])
            dfa_r = expand(self._build_automaton(right, verbose=verbose, init=False), len(free_vars), pos=[free_vars.index(v) for v in free_r])

            result = dfa_l.intersection(dfa_r).minimize()
            if verbose:
                print(f'{str(phi)}: {result.num_states} states')
            return result

        elif isinstance(phi, logic.NegatedExpression):
            psi = phi.term
            # Skip double negation
            if isinstance(psi, logic.NegatedExpression):
                return self._build_automaton(psi.term, verbose=verbose, init=False)
            else:
                free_vars = get_free_elementary_vars(phi)

                domain = product(self.automata['U'], len(free_vars))

                result = self._build_automaton(psi, verbose=verbose, init=False).complement()
                result = result.intersection(domain).minimize()
                if verbose:
                    print(f'{str(phi)}: {result.num_states} states')
                return result
        elif isinstance(phi, logic.OrExpression):
            left = phi.first
            right = phi.second

            free_vars = get_free_elementary_vars(phi)
            free_l = get_free_elementary_vars(left)
            free_r = get_free_elementary_vars(right)

            dfa_l = expand(self._build_automaton(left, verbose=verbose, init=False), len(free_vars), pos=[free_vars.index(v) for v in free_l])
            dfa_r = expand(self._build_automaton(right, verbose=verbose, init=False), len(free_vars), pos=[free_vars.index(v) for v in free_r])

            result = dfa_l.union(dfa_r).minimize()
            if verbose:
                print(f'{str(phi)}: {result.num_states} states')
            return result
        elif isinstance(phi, logic.ApplicationExpression):
            R = str(phi.pred)
            variables = get_free_elementary_vars(phi)

            result = expand(
                self.automata[R],
                len(variables),
                [variables.index(v) for v in [str(v) for v in phi.args]]
            ).minimize()

            if verbose:
                print(f'{str(phi)}: {result.num_states} states')
            return result