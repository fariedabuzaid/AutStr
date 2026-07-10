"""Closure operations on automatic presentations.

Automatic structures over a common signature are closed under disjoint union
and under direct products, and the constructions are all statements about
*letters*: re-express the factors over a common alphabet, then combine them
with the Boolean operations the engine already has.

The letter work is done by `autstr.sparse_automata.recode`, which rewrites a
transition diagram for a new alphabet in one pass over its nodes. Widening an
alphabet therefore costs nothing that scales with the alphabet -- and that is
what makes products affordable, because the pair alphabet of a direct product
has ``|A| * |B|`` letters but only ``bits_A + bits_B`` variables. Letters
multiply; bits add.
"""
from typing import Dict, Optional, Sequence

import numpy as np

from autstr.presentations import AutomaticPresentation
from autstr.sparse_automata import SparseDFA, recode
from autstr.utils.automata_tools import _symbol_assignment
from autstr.utils.misc import encode_symbol


def prefix(dfa: SparseDFA, letters: Sequence) -> SparseDFA:
    """Accept exactly ``letters . w`` for the words ``w`` the automaton accepts.

    `letters` gives one letter per tape, so a k-ary relation is prefixed by one
    symbol of its convolution. A fresh start state reads that symbol and hands
    over; every other first symbol is rejected. This is how a disjoint union
    tags which side an element came from.
    """
    if len(letters) != dfa.symbol_arity:
        raise ValueError(f"expected {dfa.symbol_arity} letters, "
                         f"got {len(letters)}")
    store = dfa.store
    code = encode_symbol(tuple(letters), dfa.base_alphabet_frozen)
    assignment = _symbol_assignment(code, dfa.symbol_arity, dfa.m, dfa.bits)

    start, dead = dfa.num_states, dfa.num_states + 1
    dead_node = store.const(dead, dfa.symbol_arity, dfa.m, dfa.bits)
    start_node = store.set_path(dead_node, assignment, dfa.start_state)

    nodes = dfa.nodes.tolist() + [start_node, dead_node]
    return SparseDFA(dead + 1,
                     is_accepting=np.r_[dfa.is_accepting, False, False],
                     start_state=start, symbol_arity=dfa.symbol_arity,
                     base_alphabet=dfa.base_alphabet,
                     nodes=np.array(nodes, dtype=np.int64))


def _signatures(left: AutomaticPresentation, right: AutomaticPresentation):
    """The relation names and arities, checked to agree."""
    names = set(left.automata) | set(right.automata)
    for name in sorted(names):
        if name not in left.automata or name not in right.automata:
            raise ValueError(f"only one presentation has the relation {name!r}")
        if left.automata[name].symbol_arity != right.automata[name].symbol_arity:
            raise ValueError(f"relation {name!r} has different arities")
    return {name: left.automata[name].symbol_arity for name in names}


def disjoint_union(left: AutomaticPresentation, right: AutomaticPresentation,
                   tags: Sequence = ('<l>', '<r>')) -> AutomaticPresentation:
    """The disjoint union of two automatic structures over one signature.

    An element of the left factor is encoded as ``tags[0] . w`` and one of the
    right factor as ``tags[1] . w``, so the domains cannot collide and no tuple
    mixes the two sides -- which is exactly right, since the relations of a
    disjoint union never cross it.

    Note that a disjoint union is a *relational* construction: the disjoint
    union of two groups is not a group, because the multiplication becomes
    partial.
    """
    if left.padding_symbol != right.padding_symbol:
        raise ValueError("the factors must share a padding symbol")
    arities = _signatures(left, right)
    if len(set(tags)) != 2:
        raise ValueError("the two tags must differ")

    alphabet = (set(left.sigma) | set(right.sigma) | set(tags))
    if len(alphabet) != len(set(left.sigma) | set(right.sigma)) + 2:
        raise ValueError(f"the tags {tags} already occur in an alphabet")

    automata: Dict[str, SparseDFA] = {}
    for name, arity in arities.items():
        sides = []
        for presentation, tag in ((left, tags[0]), (right, tags[1])):
            widened = recode(presentation.automata[name], alphabet)
            sides.append(prefix(widened, (tag,) * arity))
        automata[name] = sides[0].union(sides[1]).minimize()

    # the factors were already domain-consistent, and tagging cannot break it
    return AutomaticPresentation(automata, padding_symbol=left.padding_symbol,
                                 enforce_consistency=False)


def direct_product(left: AutomaticPresentation, right: AutomaticPresentation,
                   kind: str = 'sync') -> AutomaticPresentation:
    """Not implemented yet -- see `disjoint_union` for the shape it will take.

    Both products live over the *pair* alphabet, where a letter carries one
    letter of each factor. Then

        sync   R((a,b), (a',b'))  iff  R_A(a,a') and R_B(b,b')
        async  R((a,b), (a',b'))  iff  (R_A(a,a') and b = b')
                                        or (R_B(b,b') and a = a')

    Both are Boolean combinations of the factors' relations embedded into the
    pair alphabet (a variable renaming into that half's bit block) together
    with equality on the other half.
    """
    raise NotImplementedError(
        "direct_product is the next step; disjoint_union is available")
