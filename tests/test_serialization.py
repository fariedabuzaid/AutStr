import json
import struct
import zlib

import numpy as np

from autstr.arithmetic import BuechiArithmeticZ
from autstr.sparse_automata import SparseDFA, SparseDFASerializer


def _all_words(alphabet, length):
    if length == 0:
        yield ()
        return
    for prefix in _all_words(alphabet, length - 1):
        for letter in alphabet:
            yield prefix + (letter,)


def _same_language(a: SparseDFA, b: SparseDFA, max_length: int = 5) -> bool:
    """Brute-force language equality on all short words."""
    alphabet = sorted(a.base_alphabet)
    for length in range(max_length + 1):
        for word in _all_words(alphabet, length):
            symbols = [(letter,) * a.symbol_arity for letter in word]
            if a.accepts(symbols) != b.accepts(symbols):
                return False
    return True


class TestSerializer:
    def test_round_trip_preserves_the_automaton(self):
        """The payload stores the transition diagrams, so a reloaded automaton
        agrees with the original on structure and language."""
        dfa = SparseDFA(
            4, default_states=np.array([3, 3, 3, 3]),
            exception_symbols=np.array([[1, 2], [1, -1], [0, -1], [-1, -1]]),
            exception_states=np.array([[1, 2], [2, -1], [2, -1], [-1, -1]]),
            is_accepting=[False, False, True, False], start_state=0,
            symbol_arity=1, base_alphabet={'*', '0', '1'})

        reloaded = SparseDFASerializer.from_bytes(SparseDFASerializer.to_bytes(dfa))

        assert reloaded.num_states == dfa.num_states
        assert reloaded.start_state == dfa.start_state
        assert reloaded.symbol_arity == dfa.symbol_arity
        assert reloaded.base_alphabet == dfa.base_alphabet
        assert list(reloaded.is_accepting) == list(dfa.is_accepting)
        assert np.array_equal(reloaded.dense_next(), dfa.dense_next())
        assert _same_language(dfa, reloaded)

    def test_round_trip_of_a_multi_tape_automaton(self):
        dfa = BuechiArithmeticZ().automata['A']       # x + y = z over Z
        reloaded = SparseDFASerializer.from_bytes(SparseDFASerializer.to_bytes(dfa))
        assert reloaded.num_states == dfa.num_states
        assert np.array_equal(reloaded.dense_next(), dfa.dense_next())

    def test_the_legacy_flat_payload_still_loads(self):
        """Version 2 stored a default target plus flat exception rows, before
        transitions became shared diagrams. Files written by those versions
        must still load, so the payload is built here by hand -- the library
        no longer ships one to read.
        """
        # a two-state automaton over {0, 1, *}: accepts words ending in 1
        alphabet = sorted({'*', '0', '1'})
        code = {letter: i for i, letter in enumerate(alphabet)}
        num_states, max_exceptions = 2, 1
        defaults = np.array([0, 0], dtype=np.uint32)          # on anything else
        ex_symbols = np.array([[code['1']], [code['1']]], dtype=np.int32)
        ex_states = np.array([[1], [1]], dtype=np.int32)
        is_accepting = np.array([0, 1], dtype=np.uint8)

        alphabet_json = json.dumps(alphabet).encode('utf-8')
        payload = b''.join([
            struct.pack("IIIII", num_states, max_exceptions, 0, 1,
                        len(alphabet_json)),
            alphabet_json,
            defaults.tobytes(),
            ex_symbols.tobytes(),
            ex_states.tobytes(),
            is_accepting.tobytes(),
        ])
        data = struct.pack("4sB3sII", b'SDFA', 2, b'\0\0\0',
                           zlib.crc32(payload), len(payload)) + payload

        dfa = SparseDFASerializer.from_bytes(data)

        assert dfa.num_states == 2
        assert dfa.base_alphabet == {'*', '0', '1'}
        assert dfa.accepts([('1',)])
        assert dfa.accepts([('0',), ('1',)])
        assert not dfa.accepts([('1',), ('0',)])


class TestProductAlphabet:
    """A presentation whose letters are tuples.

    An interpretation of dimension k > 1 encodes an element as a word over the
    k-fold product alphabet, so both the alphabet's letters and the padding
    symbol are tuples. JSON has no tuples: without restoring them on load, the
    alphabet is a set of lists (unhashable) and the padding symbol is a list.
    """

    def _integers_from_naturals(self):
        from autstr.arithmetic import BuechiArithmetic
        from autstr.interpretations import interpret
        pair = ['x0', 'x1', 'y0', 'y1']
        same = 'exists z.(A(x0,y1,z) and A(y0,x1,z))'
        less = 'exists s.(exists t.(A(x0,y1,s) and A(y0,x1,t) and Lt(s,t)))'
        return interpret(
            BuechiArithmetic(),
            domain=('Eq(x0,x0) and Eq(x1,x1)', ['x0', 'x1']),
            relations={'Lt': (less, pair), 'Eq': (same, pair)},
            dimension=2, quotient=(same, pair))

    def test_round_trip_through_a_file(self, tmp_path):
        from autstr.presentations import AutomaticPresentation
        presentation = self._integers_from_naturals()
        path = str(tmp_path / 'integers.autstr')
        presentation.automatic_presentation_to_file(path)

        reloaded = AutomaticPresentation.automatic_presentation_from_file(path)

        assert set(reloaded.get_relation_symbols()) == \
            set(presentation.get_relation_symbols())
        assert isinstance(reloaded.padding_symbol, tuple)
        assert all(isinstance(letter, tuple) for letter in reloaded.sigma)
        # the order it presents is still the order of the integers:
        # no least element, and discrete rather than dense
        assert not reloaded.check(
            'exists x.(all y.((not Lt(y,x)) and (not Eq(x,y))))')
        assert not reloaded.check(
            'all x.(all y.(Lt(x,y) -> exists z.(Lt(x,z) and Lt(z,y))))')
        assert _same_language(reloaded.automata['Lt'],
                              presentation.automata['Lt'], max_length=3)
