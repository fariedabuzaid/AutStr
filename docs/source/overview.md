# Overview

AutStr represents infinite mathematical structures as finite automata and decides
first-order and monadic second-order questions about them. This chapter is the
guided tour: it works through the two things the library represents — single
**structures** and whole **classes** — the operations that combine them, the
rank-width families, implicit evaluation, and finally how the machine works
underneath. Every part has a matching executable notebook, linked as we go.

The `README <https://github.com/fariedabuzaid/AutStr>`_ is the short version of
this page; the API reference is generated from the docstrings and linked in the
sidebar.

## Decision procedure and theorem prover

An `AutomaticPresentation` bundles automata for a domain and its relations, and
decides first-order statements about the presented structure:

```python
from autstr.buildin.presentations import BuechiArithmeticZ

Z = BuechiArithmeticZ()          # (ℤ, +, <, |) as automata

Z.check('all x.(exists y.(A(x,y,x)))')          # ∀x ∃y: x+y=x   — True
Z.check('exists x.(all y.(Lt(x,y)))')           # a least integer? — False
```

Because the first-order theory of an automatic structure is decidable, `check`
always terminates with a definite answer — a theorem prover for the fragment of
mathematics these structures capture. `evaluate` goes further and returns the
automaton of *all* satisfying assignments, which you can enumerate or reuse. The
{doc}`arithmetic & algebra notebook <notebooks/arithmetic_and_algebra>` walks
through Presburger and Büchi arithmetic in full.

## Symbolic expressions instead of formula strings

Formula strings are fine for a one-off question, but they are awkward to build
programmatically and give no help with arities or variable names. Every
structure and every class also hands out a **symbolic interface**: variables,
relation symbols and function symbols that compose with ordinary Python
operators.

```python
from autstr.arithmetic import integers

Z = integers()
x, y, z = Z.vars('x y z')

phi = (x + y).eq(z) & z.lt(100)      # a relation over x, y, z
phi.check()                          # satisfiable?                  — True
phi.evaluate().contains(x=3, y=4, z=7)                             # — True
(x + y).eq(4) & x.gt(0) & x.lt(y)    # ... and enumerate its solutions
```

Terms nest freely: `f(t)` for a declared function symbol compiles to an
existential over the graph relation, so `((x + y) + z).eq(10)` means what it
reads as. Arities come from the automata, so applying a relation to the wrong
number of arguments is an error rather than a silently wrong query, and
variables may be named anything — the compiler renames them internally and
restores the names in results.

What a structure offers is declared in a `Signature`: which relations are
function graphs, which operators they bind to, and how Python values encode as
elements.

```python
from autstr.symbolic import Signature, FunctionCodec

signature = (Signature(codec=FunctionCodec(encode, decode))
             .function('+', graph='A', out=2)     # A(x, y, z) is the graph of +
             .operator('+', '+')
             .operator('lt', 'Lt'))

S = my_presentation.symbolic(signature)
```

Results carry their tape order as variable *names*, so membership and
enumeration are keyed by name rather than position:

```python
relation = phi.evaluate()
relation.variables                   # ['x', 'y', 'z']
relation.contains(z=7, x=3, y=4)     # order does not matter
relation.is_finite()                 # finitely many satisfying tuples?
```

The same expressions compile against a uniformly automatic class, where they
define a relation across every member at once; `check_member` then evaluates
one member, explicitly or implicitly.

## Computer algebra over infinite domains

Structures need not be finitely generated. The localizations **ℤ[1/p]** — the
rationals whose denominator is a power of p — are infinite,
non-finitely-generated groups, yet each has an exact automatic presentation:

```python
from autstr.algebra import z1p_localization

z2 = z1p_localization(2)                       # (ℤ[1/2], +)

x = z2.from_fraction(1, 2)
y = z2.from_fraction(3, 4)
z2.check('A(x,y,z)', x=x, y=y, z=z2.add(x, y))                    # 1/2 + 3/4 = 5/4 — True
z2.check('all x.(exists y.(A(y,y,x)))')                           # 2-divisible?    — True
z2.check('all x.(exists y.(exists w.(A(y,y,w) and A(w,y,x))))')   # 3-divisible?    — False
```

The first-order divisibility theory even *distinguishes* the localizations: every
element of ℤ[1/2] is 2-divisible but not 3-divisible, and vice versa for ℤ[1/3].

## Uniformly automatic classes: one automaton for a whole family

A **uniformly automatic class** presents not one structure but an entire *family*,
by giving every automaton one extra tape that reads an **advice string**
synchronously with the elements. Fixing the advice instantiates one member; a
query is compiled once for the class and then decides any member by running its
advice word through the resulting automaton.

`autstr.graphs`, `autstr.algebra`, and `autstr.groups` ship ready-made classes:

```python
# Finite abelian groups — advice is the cyclic decomposition
from autstr.groups import FiniteAbelianGroups
ab = FiniteAbelianGroups()
ab.check('A(x,y,z)', [2, 3], x=(1, 1), y=(1, 2), z=(0, 0))   # (1,1)+(1,2)=(0,0) in Z2⊕Z3

# Non-abelian groups — dihedral, quaternion, semidihedral, modular, ...
from autstr.groups import IndexTwoCyclicGroups
G = IndexTwoCyclicGroups()
G.check('M(x,y,z)', G.dicyclic(4), x=(0, 1), y=(1, 0), z=(1, 1))   # i·j = k in Q₈

# Extraspecial p-groups — nilpotency class 2, order p^(1+2n)
from autstr.groups import ExtraspecialGroups
H = ExtraspecialGroups(3)
H.check('M(x,y,z)', 2, x=(1, (0, 0), (0, 0)), y=(0, (1, 0), (0, 0)),
        z=(1, (1, 0), (0, 0)))
```

The generic machinery in `autstr.uniform` turns *any* advice-indexed family of
automata into a class with relativized query evaluation, sentence checking, member
instantiation (`get_structure`), and a first-order `define` for bootstrapping
complex relations from primitives.

The same machinery runs over **trees** rather than words. Where an automatic
presentation encodes elements as strings and a word automaton reads them, a
*tree-automatic* presentation encodes them as finite trees read by a bottom-up
tree automaton — exactly the step from Büchi's theorem to Rabin's.
`autstr.tree_uniform` hosts the classes whose advice is naturally a tree: a tree
decomposition (bounded tree-width) or a k-expression (bounded clique-width), and
Skolem arithmetic (ℕ, ·) in `autstr.buildin.tree_presentations`, where a number
is the tree of its prime exponents.

The {doc}`graphs <notebooks/graphs>` and {doc}`groups <notebooks/groups>`
notebooks build these classes and query them; the
{doc}`arithmetic & algebra notebook <notebooks/arithmetic_and_algebra>` covers the
Boolean algebras, ℤ[1/p], and the MSO0 finite-powerset structure.

## Composing presentations

Automatic structures over a shared signature are closed under disjoint union and
direct products; uniformly automatic classes of **finite** structures are closed
under union and under taking all finite direct products of their members.
`autstr.composition` builds the new presentation for you.

```python
from autstr.composition import (
    class_union, direct_product_closure, blocks, tagged_advice,
)
from autstr.groups import ExtraspecialGroups, IndexTwoCyclicGroups
from autstr.uniform import UniformlyAutomaticClass

cyclic, extra = IndexTwoCyclicGroups(), ExtraspecialGroups(3)

def reduct(uniform):                     # the signature the two classes share
    return UniformlyAutomaticClass(
        {'U': uniform.class_automata['U'], 'M': uniform.class_automata['M']})

# Members of either family ...
both = class_union(reduct(cyclic.cls), reduct(extra.cls))
# ... and every finite direct product of them.
groups = direct_product_closure(both)

z4 = tagged_advice(cyclic.cyclic(4), '<l>')          # Z4, abelian
heis = tagged_advice(extra.advice(1), '<r>')         # extraspecial 3^(1+2)

abelian = 'all x.(all y.(all z.(M(x,y,z) -> M(y,x,z))))'
groups.check(abelian, blocks(z4, z4))       # True  — Z4 × Z4
groups.check(abelian, blocks(z4, heis))     # False — one nonabelian factor
```

| operation | on | construction |
|-----------|----|--------------|
| `disjoint_union(A, B)` | structures | tag each element with the side it came from |
| `direct_product(A, B, kind='sync')` | structures | `R_A(a,a') ∧ R_B(b,b')` |
| `direct_product(A, B, kind='async')` | structures | `(R_A(a,a') ∧ b=b') ∨ (R_B(b,b') ∧ a=a')` |
| `class_union(C, D)` | classes | tag the *advice*, so the advice languages are disjoint |
| `direct_product_closure(C)` | classes | advice `α₁\|…\|αₙ` presents `A_{α₁} × … × A_{αₙ}` |

Two of these are worth a word. The **direct product** encodes a pair over the
*pair alphabet*, where a letter carries one letter of each factor; each factor is
then embedded by a variable renaming into its half of the bits, and the two
products are Boolean combinations of the embeddings. That is affordable only
because the pair alphabet has `|Σ_A|·|Σ_B|` letters but `bits_A + bits_B`
variables — **letters multiply, bits add**, which is precisely what the decision
diagrams buy.

The **product closure** concatenates advices with a separator. Since an element of
a finite member is never longer than its advice, the blocks line up across every
tape, so a relation of the product is the original relation holding in every block
— one automaton with **one extra state**, where an interleaved encoding would need
one copy per component. `FiniteAbelianGroups` is this construction applied to the
cyclic groups, and it predates the module. The
{doc}`composition notebook <notebooks/composition>` walks through all five
operations.

## Bounded rank-width: groups and graphs from one linear algebra

The width notions above bound how much *combinatorial* structure crosses a cut.
Rank-width bounds the **linear-algebraic rank** of what crosses, and one body of
machinery — interface bases, streamed basis changes, and a two-sided factorization
of the sibling block (`autstr.chain_ring`) — presents both groups and graphs of
bounded rank-width uniformly.

**Class-2 groups** (`autstr.groups.CutRankGroups`,
`autstr.tree_groups.CutRankTreeGroups`): a member is a central extension of
(ℤ/pᵈ)ⁿ by (ℤ/pᵈ)ᵏ given by commutator labels; the advice spells out, cut by cut, a
rank-≤r factorization of the crossing block of the commutation form, and the
multiplication automaton carries r linear functionals instead of the digits it has
read. With `d = 1` this is the field F_p; with `d > 1` the center has exponent pᵈ
and the width is measured over the *chain ring* ℤ/pᵈ. Bounded pathwidth, bounded
vertex cover, the extraspecial matching and the complete graph are all special
layouts; the tree class recovers the word class on spine layouts.
`autstr.cocycle_groups.CocycleRankWidthGroups` generalizes further to *distributed*
centers — central generators scattered through the layout — via a six-register
protocol whose advice is microcode.

When a flat advice alphabet would be astronomical (it grows like q^(r²+r+kr)), the
classes switch to **factored letters**: one letter per ring entry of the
factorization, streamed through an accumulator — the advice alphabet drops to q+1
letters, and width r ≥ 2 over ℤ/4 goes from unrepresentable to explicitly
buildable this way.

**Graphs** (`autstr.tree_graphs.RankWidthClass`): the advice is a rank
decomposition annotated with the GF(2) factorization of its cuts; adjacency of two
vertices is a bilinear form applied to their r-bit interface vectors at the node
where their subtrees meet. Rank-width lower-bounds clique-width and stays bounded
on dense graphs (cliques have rank-width 1), and the class answers MSO queries like
every other graph class:

```python
from autstr.tree_graphs import RankWidthClass, RankWidthGraph

rw = RankWidthClass(1)
two_col = ('exists a.(all x.(all y.((not E(x,y)) or '
           '((Subset(x,a) and (not Subset(y,a))) or '
           '(Subset(y,a) and (not Subset(x,a)))))))')
rw.check(two_col, RankWidthGraph.complete_bipartite(2, 3))   # True  (rank-width 1)
rw.check(two_col, RankWidthGraph.clique(3))                  # False (odd cycle)
```

The bounded-rank-width groups appear in the {doc}`groups notebook <notebooks/groups>`
and the graphs in the {doc}`graphs notebook <notebooks/graphs>`.

## Implicit evaluation: members whose automata cannot be built

For the heavy ring members the *base* multiplication automaton is already
infeasible to construct — an O(|Σ|⁴) product over a huge advice alphabet. The
`autstr.implicit` layer decides first-order formulas on such members anyway, by
never building anything: Boolean connectives keep composite states and step the
base automata on the fly, `exists` is an on-the-fly powerset over the (tiny)
element alphabet, `not` flips acceptance. Because the advice is fixed input, the
cost is driven by quantifier alternation, not by alphabet size.

```python
from autstr.groups import CutRankGroups

G = CutRankGroups(2, d=3)         # commutation forms over Z/8 — automata unbuildable
advice = G.advice(3, G.clique_form(3))
x = ((5,), (3, 1, 6))

# model checking, implicitly:
G.check_implicit('exists y.(M(x,y,u))', advice, x=x, u=G.identity(3))   # True

# the satisfying SET, implicitly: exact count without enumeration, lazy iteration
inv = G.evaluate_implicit('M(x,y,u)', advice, x=x, u=G.identity(3))
len(inv)                          # 1 — the unique inverse
next(iter(inv))['y']              # ... and here it is, as a (b, a) tuple
```

Every uniform class offers `check_implicit` (model checking) and `evaluate_implicit`
(the satisfying assignments of a formula with open free variables, with the exact
solution count computed by dynamic programming — no enumeration — and lazy
iteration). `ImplicitClass` / `ImplicitTreeClass` package functional atoms and an
element alphabet as a first-class *fully implicit* presentation. The
{doc}`implicit-evaluation notebook <notebooks/implicit_evaluation>` also times the
implicit path against a compiled one.

## How it works

An **automatic presentation** encodes a countable structure so that its domain is a
regular language and each relation is recognized by a synchronous multi-tape
automaton reading its arguments letter-by-letter in lockstep. The foundational fact
is that this recognizability is *closed under first-order definability*: Boolean
combinations correspond to product automata, and quantifiers to projection followed
by determinization. Consequently the first-order theory of any automatic structure
is **decidable**, and every definable relation is again automatic.

### A concrete encoding

In the arithmetic package an integer is written **sign-magnitude, least-significant
bit first**: the first symbol is a sign bit (`0` for non-negative, `1` for
negative), followed by the binary digits of the magnitude from the lowest bit
upward, with `*` padding the shorter arguments of a multi-tape relation so all
tapes advance in lockstep. Under this encoding a definable set is a genuinely small
automaton — here, for instance, is the recognizer for the integers greater than 1
that are divisible by neither 2 nor 3:

```{image} _media/sieve_automaton-light.png
:alt: A 9-state automaton recognizing the integers greater than 1 divisible by neither 2 nor 3
:width: 760px
:align: center
:class: only-light
```

```{image} _media/sieve_automaton-dark.png
:alt: A 9-state automaton recognizing the integers greater than 1 divisible by neither 2 nor 3
:width: 760px
:align: center
:class: only-dark
```

The start state consumes the sign bit; the remaining states scan the magnitude bits
from least to most significant. Because "greater than 1" and "not divisible by 2 or
3" are both regular properties of that bit stream, a handful of states suffice —
doubled circles are accepting, and `*` marks the padding that ends the word.

### Advice and trees

Allowing the automata to read an additional fixed *advice* word widens the reach to
structures like (ℚ, +) and, using a *set* of advices, to whole parameterized classes
of finite structures. Deciding the first-order theory of a uniformly automatic class
reduces to the monadic second-order theory of its advice language (Abu
Zaid–Grädel–Reinhardt 2017; Abu Zaid 2018). The same programme runs over finite
*trees* read by bottom-up tree automata, which reaches structures no string encoding
captures naturally — (ℕ, ·) with a number written as the tree of its prime
exponents, and classes whose advice is inherently a tree.

### Why it is fast — and where it is hard

Evaluating a *fixed* formula on a structure is one linear pass of its advice through
the query automaton, so on any class of bounded width every fixed MSO property is
decided in linear time — a constructive, streaming form of Courcelle's theorem. The
cost lives entirely in *compiling* the automaton.

A transition is not a `symbol -> target` table but a **decision diagram over the
symbol's digits**, hash-consed and shared across states and automata
(`autstr.mtbdd`) — the representation MONA uses, for the same reason. A transition
that ignores a tape never tests that tape's variables, so cylindrification is a
variable renaming rather than a duplication of every row once per letter of every
new tape, complementation touches no diagram at all, and the alphabet's *width*
stops driving the cost. What remains is the subset explosion of determinizing an
existential quantifier: element quantifiers are cheap, and *set* quantifiers (MSO
proper) determinize over subsets of the intermediate automaton's states.
Connectedness and bipartiteness compile in seconds; 3-colourability — the minimal
NP-hard MSO query — is a genuinely large one-time compile. Once compiled, an
automaton can be serialized (diagrams and all) and reused forever.

Around the diagrams the engine is batched NumPy: frontier-batched constructions,
hashed partition refinement, and a subset construction that runs in a collectable
scratch store. JAX is an optional accelerator used only for bulk word processing.
```
