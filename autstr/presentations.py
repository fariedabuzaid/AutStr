from nltk.sem import logic
from typing import Dict, Optional, Union, List

from autstr.sparse_automata import SparseDFA, SparseDFASerializer
from autstr.utils.automata_tools import pad, unpad, projection, expand, stack
from autstr.buildin.automata import zero, one  
from autstr.utils.logic import get_free_elementary_vars, optimize_query

import json
import re
import struct
import zlib

class AutomaticPresentationSerializer:
    MAGIC = b'APRS'
    VERSION = 1
    HEADER_FORMAT = "4sB3sII"  # Magic, version, reserved, checksum, payload_size
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
    
    @classmethod
    def serialize(cls, presentation, filename: str) -> None:
        """Serialize AutomaticPresentation to binary file"""
        # Prepare payload components
        payload = cls._create_payload(presentation)
        
        # Create header
        checksum = zlib.crc32(payload)
        header = struct.pack(
            cls.HEADER_FORMAT,
            cls.MAGIC,
            cls.VERSION,
            b'\0\0\0',  # Reserved
            checksum,
            len(payload)
        )
        
        # Write to file
        with open(filename, 'wb') as f:
            f.write(header)
            f.write(payload)
    
    @classmethod
    def deserialize(cls, filename: str):
        """Deserialize AutomaticPresentation from binary file"""
        with open(filename, 'rb') as f:
            # Read and validate header
            header = f.read(cls.HEADER_SIZE)
            magic, version, _, checksum, payload_size = struct.unpack(cls.HEADER_FORMAT, header)
            
            if magic != cls.MAGIC:
                raise ValueError("Invalid file format (bad magic number)")
            if version != cls.VERSION:
                raise ValueError(f"Unsupported version: {version}")
            
            # Read and validate payload
            payload = f.read(payload_size)
            if zlib.crc32(payload) != checksum:
                raise ValueError("Data corruption detected (checksum mismatch)")
            
            return cls._parse_payload(payload)
    
    @classmethod
    def _create_payload(cls, presentation) -> bytes:
        """Create binary payload from AutomaticPresentation"""
        # Serialize metadata
        metadata = {
            "padding_symbol": presentation.padding_symbol,
            "enforce_consistency": True  # Not used in deserialization
        }
        metadata_json = json.dumps(metadata).encode('utf-8')
        metadata_len = len(metadata_json)
        
        # Serialize automata dictionary
        automata_data = {}
        for name, dfa in presentation.automata.items():
            # Use SparseDFA serialization to bytes
            dfa_bytes = SparseDFASerializer.to_bytes(dfa)
            automata_data[name] = list(dfa_bytes)  # Convert to list for JSON
        
        automata_json = json.dumps(automata_data).encode('utf-8')
        automata_len = len(automata_json)
        
        # Pack components
        return struct.pack("II", metadata_len, automata_len) + metadata_json + automata_json
    
    @classmethod
    def _parse_payload(cls, payload: bytes):
        """Parse binary payload into AutomaticPresentation"""
        # Read lengths
        metadata_len, automata_len = struct.unpack("II", payload[:8])
        offset = 8
        
        # Decode metadata
        metadata_json = payload[offset:offset+metadata_len]
        metadata = json.loads(metadata_json.decode('utf-8'))
        offset += metadata_len
        
        # Decode automata
        automata_json = payload[offset:offset+automata_len]
        automata_data = json.loads(automata_json.decode('utf-8'))
        
        # Convert back to SparseDFA instances
        automata = {}
        for name, dfa_bytes_list in automata_data.items():
            dfa_bytes = bytes(dfa_bytes_list)
            automata[name] = SparseDFASerializer.from_bytes(dfa_bytes)
        
        # Reconstruct presentation
        return AutomaticPresentation(
            automata,
            padding_symbol=metadata["padding_symbol"],
            enforce_consistency=False
        )

class DeferredRelations:
    """Relations declared up front and built on first use.

    Equality is definable in most presentations here -- from ``Leq`` in a
    lattice, from ``Subset`` on set-valued elements, from the operation in a
    group -- but defining it costs an automaton construction that most queries
    never need, and for the wider graph classes that construction is
    expensive. So such a relation is *registered* rather than built, and
    materializes when something asks for it.

    `materialize()` forces the construction, which is what you want before
    pickling or otherwise reusing a structure, and every constructor that
    registers deferred relations takes an ``eager`` flag for the same purpose.

    Subclasses say where their relations live (`_relations`) and how to install
    one (`_install_relation`). A definition is either a formula string over the
    existing signature or a callable returning an automaton.
    """

    def _declare_deferred(self, definitions: Dict[str, object],
                          eager: bool = False) -> None:
        self._deferred = dict(definitions)
        if eager:
            self.materialize()

    @property
    def _relations(self) -> dict:
        return self.automata

    def _install_relation(self, name: str, definition) -> None:
        raise NotImplementedError

    def get_relation_symbols(self) -> List[str]:
        """All relation symbols, including any not yet built."""
        names = list(self._relations.keys())
        names += [n for n in getattr(self, '_deferred', ())
                  if n not in self._relations]
        return names

    def relation(self, name: str):
        """The automaton for `name`, building it if it was deferred."""
        deferred = getattr(self, '_deferred', {})
        if name not in self._relations and name in deferred:
            self._install_relation(name, deferred[name])
        return self._relations.get(name)

    def materialize(self, *names: str):
        """Build the named deferred relations now, or all of them. Returns
        self, so it chains onto a constructor."""
        for name in (names or tuple(getattr(self, '_deferred', ()))):
            self.relation(name)
        return self

    def _materialize_for(self, phi) -> None:
        """Build any deferred relation the formula mentions."""
        deferred = getattr(self, '_deferred', None)
        if not deferred:
            return
        text = str(phi)
        for name in list(deferred):
            if name not in self._relations and re.search(
                    rf'\b{re.escape(name)}\b', text):
                self.relation(name)


class AutomaticPresentation(DeferredRelations):
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

    def automatic_presentation_to_file(self, filename: str) -> None:
        AutomaticPresentationSerializer.serialize(self, filename)

    @classmethod
    def automatic_presentation_from_file(cls, filename: str):
        return AutomaticPresentationSerializer.deserialize(filename)

    def _install_relation(self, name, definition):
        """Build a deferred relation: a formula over the current signature, or
        a callable returning an automaton."""
        self.update(**{name: definition() if callable(definition)
                       else definition})

    def symbolic(self, signature=None):
        """A symbolic interface to this structure: variables, relation and
        function symbols that build first-order expressions with Python
        operators instead of formula strings.

        :param signature: declared functions, operators and element codec.
            Relation arities are read from the automata, so a structure with
            no functions needs no signature at all.
        :return: a `autstr.symbolic.SymbolicContext`.
        """
        from autstr.symbolic.backends import StructureBackend
        from autstr.symbolic.context import SymbolicContext
        if signature is None:
            signature = self.default_signature()
        return SymbolicContext(StructureBackend(self), signature)

    def default_signature(self):
        """The signature `symbolic()` uses when none is given, or None for a
        structure that declares no operators and is addressed through its
        relation symbols. Structures that know their own vocabulary override
        this; see `autstr.symbolic.operation_signature`."""
        return None

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

    def _domain_product(self, arity: int) -> SparseDFA:
        """Universe automaton for `arity` tapes, built one tape at a time."""
        if arity <= 1:
            return self.automata['U']

        domain = expand(self.automata['U'], arity, [0]).minimize()
        for i in range(1, arity):
            domain = domain.intersection(
                expand(self.automata['U'], arity, [i]).minimize()
            ).minimize()
        return domain

    def check(self, phi: logic.Expression | str) -> bool:
        """Checks if a given first-order formula holds on the presented structure. Free variables are assumed be
        implicitly existentially quantified.

        :param phi: the first order formula
        :returns: the truth value of the formula, if the formula where all free variables are existentially quantified.
        """
        if isinstance(phi, str):
            phi = logic.Expression.fromstring(phi)
        phi = phi.simplify()
        self._materialize_for(phi)
        return not self._build_automaton(phi).is_empty()

    def evaluate(self, phi: Union[str, logic.Expression],
                 updates: Optional[Dict[str, Union[SparseDFA, str]]] = None,
                 prepared_updates: Optional[Dict[str, SparseDFA]] = None) -> SparseDFA:
        """Evaluates a given first-order query on the presented structure. Returns a presentation of the set of all
        satisfying assignments.

        :param phi: the first order formula.
        :param updates: Temporarily update the relations for the evaluation
        :param prepared_updates: like `updates`, for automata already known to
            be restricted to the universe -- results this presentation produced
            itself. They are only re-padded, skipping the domain intersection
            that `_prepare_automaton` would otherwise redo on every tape.
        :returns: The truth value of the formula, if the formula where all free variables are existentially quantified.
        """
        if isinstance(phi, str):
            phi = logic.Expression.fromstring(phi)
        phi = optimize_query(phi)
        self._materialize_for(phi)
        if prepared_updates:
            updates = dict(updates or {})
            for key, dfa in prepared_updates.items():
                updates[key] = pad(dfa, self.padding_symbol).minimize()
            prepared = set(prepared_updates)
        else:
            prepared = set()
        if updates is not None:
            for key in updates:
                if key in prepared:
                    continue
                if isinstance(updates[key], str):
                    query = optimize_query(logic.Expression.fromstring(updates[key]))
                    updates[key] = self._prepare_automaton(self._build_automaton(query))
                else:
                    updates[key] = self._prepare_automaton(updates[key])
            automata_backup = self.automata
            self.automata = dict(self.automata, **updates)

        try:
            if len(get_free_elementary_vars(phi)) > 0:
                dfa_phi = unpad(self._build_automaton(phi),
                                self.padding_symbol).minimize()
            else:
                dfa_phi = self._build_automaton(phi)
        finally:
            # Restore even if the build raises: otherwise a failed query would
            # leave the spliced relations installed, and every later
            # evaluation would silently answer against them.
            if updates is not None:
                self.automata = automata_backup

        return dfa_phi

    def _build_automaton(self, phi: logic.Expression, verbose=False, init=True) -> SparseDFA:
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

            psi = (phi.term.negate()).simplify()
            dfa_rec = self._build_automaton(psi, verbose=verbose, init=False).minimize()
            pos = free_vars.index(variable)

            if len(free_vars) > 1:
                domain = self._domain_product(len(free_vars) - 1)
                result = projection(dfa_rec, pos).minimize().complement().minimize().intersection(domain).minimize()
            else:
                result = one() if dfa_rec.is_empty() else zero()

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
                    return zero() if dfa_rec.is_empty() else one()
            else:
                result = dfa_rec

            result = pad(unpad(result, self.padding_symbol).minimize(), self.padding_symbol).minimize()

            if verbose:
                print(f'{str(phi)}: {result.num_states} states')
            return result
        elif isinstance(phi, logic.NegatedExpression):
            psi = phi.term
            # Skip double negation
            if isinstance(psi, logic.NegatedExpression):
                return self._build_automaton(psi.term, verbose=verbose, init=False)

            free_vars = get_free_elementary_vars(phi)
            domain = self._domain_product(len(free_vars))
            result = self._build_automaton(psi, verbose=verbose, init=False).complement()
            result = result.intersection(domain).minimize()

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
        elif isinstance(phi, logic.ImpExpression):
            return self._build_automaton(
                logic.OrExpression(logic.NegatedExpression(phi.first),
                                   phi.second),
                verbose=verbose, init=False)
        elif isinstance(phi, logic.IffExpression):
            return self._build_automaton(
                logic.AndExpression(
                    logic.OrExpression(logic.NegatedExpression(phi.first),
                                       phi.second),
                    logic.OrExpression(logic.NegatedExpression(phi.second),
                                       phi.first)),
                verbose=verbose, init=False)
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

        raise ValueError(f"Unsupported expression type: {type(phi)}")