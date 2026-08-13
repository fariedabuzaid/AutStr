"""The finite subsets of the naturals -- the structure MSO0.

Finite subsets of :math:`\\mathbb{N}` under :math:`\\subseteq`, with singletons,
the successor on singletons and their order. A finite set is a
:math:`\\{0,1\\}`-word, position `i` set iff `i` is a member, so by Büchi's
theorem first-order logic over this structure is exactly **monadic second-order
logic** over :math:`(\\mathbb{N}, <)`: the definable sets of naturals are
precisely the regular ones.

    >>> M = MSO0().symbolic()
    >>> x, y = M.vars("x y")
    >>> ({0, 2}, {0, 1, 2}) in x.subset(y)
    True

That correspondence is why the structure is called MSO0, and why quantifying
over sets here costs no more than quantifying over elements elsewhere.
"""
from typing import List

import itertools as it

from autstr.presentations import (
    AutomaticPresentation, CompiledPresentation,
)
from autstr.utils.automata_tools import create_sparse_dfa


def _build_finite_powerset() -> AutomaticPresentation:
    """The finite subsets of the naturals, compiled from scratch."""
    # Base alphabet for the presentation
    base_alphabet = {'0', '1', '*'}
    
    # 1. Universe Automaton (U) - Includes empty set
    universe = create_sparse_dfa(
        states={'start', 'empty', 'zero', 'one', 'pad_one', 'pad_zero', 'reject'},
        input_symbols={'0', '1', '*'},
        transitions={
            'start': {
                '*': 'empty',
                '0': 'zero',
                '1': 'one'
            },
            'empty': {
                '*': 'empty',
                '0': 'reject',
                '1': 'reject'
            },
            'zero': {
                '*': 'pad_zero',
                '0': 'zero',
                '1': 'one'
            },
            'one': {
                '*': 'pad_one',
                '0': 'zero',
                '1': 'one'
            },
            'pad_zero': {
                '*': 'pad_zero',
                '0': 'reject',
                '1': 'reject'
            },
            'pad_one': {
                '*': 'pad_one',
                '0': 'reject',
                '1': 'reject'
            },
            'reject': {
                '0': 'reject',
                '1': 'reject',
                '*': 'reject'
            }
        },
        initial_state='start',
        final_states={'start', 'one','empty', 'pad_one'}  # Accepts empty set and non-empty sets
    )
    
    # 2. Subset Automaton (Subset) - Empty set is subset of all sets
    subset = create_sparse_dfa(
        states={'start', 'error'},
        input_symbols=set(it.product(base_alphabet, repeat=2)),
        transitions={
            'start': {
                ('0','0'): 'start',
                ('0','1'): 'start',
                ('0','*'): 'start',
                ('1','1'): 'start',
                ('*','0'): 'start',
                ('*','1'): 'start',
                ('*','*'): 'start',
                ('1','0'): 'error',
                ('1','*'): 'error'
            },
            'error': {k: 'error' for k in it.product(base_alphabet, repeat=2)}
        },
        initial_state='start',
        final_states={'start'}
    )
    
    # 3. Singleton Automaton (Sing) - Empty set is not a singleton
    sing = create_sparse_dfa(
        states={'start', 'one', 'one_pad', 'many', 'reject'},
        input_symbols=base_alphabet,
        transitions={
            'start': {
                '0': 'start',
                '1': 'one',
                '*': 'reject'  # Reject empty set immediately
            },
            'one': {
                '0': 'reject',
                '1': 'many',
                '*': 'one_pad'
            },
            'one_pad': {
                '0': 'many',
                '1': 'many',
                '*': 'one_pad'
            },
            'many': {
                '0': 'many',
                '1': 'many',
                '*': 'many'
            },
            'reject': {
                '0': 'reject',
                '1': 'reject',
                '*': 'reject'
            }
        },
        initial_state='start',
        final_states={'one','one_pad'}  # Only non-empty singletons
    )
    
    # 4. Successor Automaton (Succ) - Empty set has no successor
    succ = create_sparse_dfa(
        states={'start', 'after_x', 'after_y', 'error'},
        input_symbols=set(it.product(base_alphabet, repeat=2)),
        transitions={
            'start': {
                ('0','0'): 'start',
                ('1','0'): 'after_x',
                ('0','1'): 'error',
                ('1','1'): 'error',
                ('*','*'): 'error',  # Reject empty set
                ('0','*'): 'error',
                ('1','*'): 'error',
                ('*','0'): 'error',
                ('*','1'): 'error'
            },
            'after_x': {
                ('*','1'): 'after_y',
                ('0','0'): 'error',
                ('0','1'): 'error',
                ('1','0'): 'error',
                ('1','1'): 'error',
                ('*','0'): 'error',
                ('0','*'): 'error',
                ('1','*'): 'error',
                ('*','*'): 'error'
            },
            'after_y': {
                ('*','*'): 'after_y',
                ('0','0'): 'error',
                ('0','1'): 'error',
                ('1','0'): 'error',
                ('1','1'): 'error',
                ('*','0'): 'error',
                ('*','1'): 'error',
                ('0','*'): 'error',
                ('1','*'): 'error'
            },
            'error': {k: 'error' for k in it.product(base_alphabet, repeat=2)}
        },
        initial_state='start',
        final_states={'after_y'}  # Only consecutive singletons
    )
    
    # 5. Less-Than on Singletons Automaton (Lt_sing) - Empty set not involved
    lt_sing = create_sparse_dfa(
        states={'init', 'x_first', 'x_first_accept', 'error'},
        input_symbols=set(it.product(base_alphabet, repeat=2)),
        transitions={
            'init': {
                ('0','0'): 'init',
                ('0','1'): 'error',
                ('1','0'): 'x_first',
                ('1','1'): 'error',
                ('*','*'): 'error',
                ('0','*'): 'error',
                ('*','0'): 'error',
                ('1','*'): 'error',
                ('*','1'): 'error'
            },
            'x_first': {
                ('0','0'): 'x_first',
                ('0','1'): 'x_first_accept',
                ('1','0'): 'error',
                ('1','1'): 'error',
                ('*','*'): 'error',
                ('0','*'): 'error',
                ('*','0'): 'x_first',
                ('1','*'): 'error',
                ('*','1'): 'x_first_accept'
            },
            'x_first_accept': {
                ('*','*'): 'x_first_accept',
                ('0','0'): 'error',
                ('0','1'): 'error',
                ('1','0'): 'error',
                ('1','1'): 'error',
                ('*','0'): 'error',
                ('*','1'): 'error',
                ('0','*'): 'error',
                ('1','*'): 'error'
            },
            'error': {k: 'error' for k in it.product(base_alphabet, repeat=2)}
        },
        initial_state='init',
        final_states={'x_first_accept'}  # Only when first singleton < second
    )
    
    # Create the presentation with base automata
    presentation = AutomaticPresentation({
        'U': universe,
        'Subset': subset,
        'Sing': sing,
        'Succ': succ,
        'Lt_sing': lt_sing
    })
    
    # Define additional relations using formulas
    presentation.update(
        In="Sing(x) and Subset(x, y)",
        Eq_set="Subset(x, y) and Subset(y, x)",
        Leq_sing="Lt_sing(x, y) or (Eq_set(x, y) and Sing(x) and Sing(y))",
        Gt_sing="not Leq_sing(x, y) and Sing(x) and Sing(y)",
        Min="(forall z. not Subset(z, x)) or "  # Empty set case
             "(Sing(y) and Subset(y, x) and forall z. (-(Sing(z) and Subset(z, x)) or Leq_sing(y, z)))",
        Max="Sing(y) and Subset(y, x) and forall z. (-(Sing(z) and Subset(z, x)) or Leq_sing(z, y))",
        Intersect="forall a. (-Sing(a) or ((Subset(a, z) and (Subset(a, x) and Subset(a, y))) or (-Subset(a, z) and -(Subset(a, x) and Subset(a, y)))))",
        Union="forall a. (-Sing(a) or ((Subset(a, z) and (Subset(a, x) or Subset(a, y))) or (-Subset(a, z) and -(Subset(a, x) or Subset(a, y)))))",
        SetMinus="forall a. (-Sing(a) or ((Subset(a, z) and (Subset(a, x) and not Subset(a, y))) or (-Subset(a, z) and -(Subset(a, x) and not Subset(a, y)))))"
    )

    presentation.update(
        Geq_sing="not Lt_sing(x, y) and Sing(x) and Sing(y)",
    )
    
    return presentation




# --------------------------------------------------------------------------
# The presentation. It declares its own vocabulary, so `symbolic()` takes no
# argument; see `CompiledPresentation`.
# --------------------------------------------------------------------------

class MSO0(CompiledPresentation):
    """The finite subsets of :math:`\\mathbb{N}` under :math:`\\subseteq`, with
    singletons, successor and the order on singletons.

        >>> M = MSO0()
        >>> x, y = M.symbolic().vars("x y")
        >>> ({0, 2} , {0, 1, 2}) in x.subset(y)
        True
        >>> (x + y).eq({0, 1}).check()          # union
        True

    By Büchi's theorem, first-order logic over this structure is exactly
    monadic second-order logic over :math:`(\\mathbb{N}, <)`, so the definable
    sets are precisely the regular ones.
    """

    _BUILD = staticmethod(_build_finite_powerset)

    #: a set as a bitmask, position i set iff i is a member. The universe
    #: rejects trailing zeros, so the encoding is the CANONICAL one: {0} is
    #: `1`, {0, 2} is `101`, and the empty set is the empty word.
    PADDING = '*'

    @staticmethod
    def encode(s) -> List[str]:
        """The word encoding a finite set of naturals: position `i` carries
        ``1`` iff `i` is a member, up to the largest one."""
        s = set(s)
        if not s:
            return []
        if min(s) < 0:
            raise ValueError(f"not a set of naturals: {s}")
        return ['1' if i in s else '0' for i in range(max(s) + 1)]

    @staticmethod
    def decode(word) -> set:
        """The set encoded by a word, ignoring padding."""
        digits = ''.join(word).replace(MSO0.PADDING, '')
        return {i for i, c in enumerate(digits) if c == '1'}

    def default_signature(self):
        """Union as ``+``, intersection as ``*``, difference as binary ``-``,
        and the relations of the structure as methods, with sets written as
        Python sets."""
        from autstr.symbolic import FunctionCodec, Signature
        signature = Signature(codec=FunctionCodec(self.encode, self.decode))
        signature.function('union', graph='Union', out=2)
        signature.function('intersect', graph='Intersect', out=2)
        signature.function('minus', graph='SetMinus', out=2)
        signature.operator('+', 'union')
        signature.operator('*', 'intersect')
        signature.operator('-', 'minus')
        signature.operator('eq', 'Eq_set')
        signature.operator('subset', 'Subset')
        signature.operator('sing', 'Sing')
        # In(x, y): x is a singleton contained in y -- membership, for the
        # singletons that stand in for the elements of N.
        signature.operator('member_of', 'In')
        signature.operator('lt_sing', 'Lt_sing')
        return signature
