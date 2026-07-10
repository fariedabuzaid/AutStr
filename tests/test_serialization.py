import numpy as np

from autstr.buildin.presentations import BuechiArithmeticZ
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
        """BuechiArithmeticZ ships a version-2 file (flat exception rows)."""
        presentation = BuechiArithmeticZ()
        assert presentation.check('exists x.(A(x,x,x))')       # 0 + 0 = 0
