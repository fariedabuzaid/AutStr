# AutStr
Working with infinite data structures in Python.

## Introduction
What if your algorithms could process infinite structures — like complete infinite graphs or the entire set of natural numbers — with the same ease as manipulating a data frame? AutStr provides an intuitive interface for defining and manipulating exact representations of infinite relational structures. With AutStr, infinite structures become first-class citizens in Python.

Targeted at researchers in algorithmic model theory and curious practitioners alike, AutStr offers:
- **Symbolic representation** of infinite sets and relations
- **Automata-based computation** for efficient manipulation
- **First-order logic interface** for formal queries
- **Uniformly automatic classes** — one automaton deciding a property for a whole family of structures
- **Visualization tools** for insight into infinite structures

## What's new in 2.0

Version 2.0 is a near-complete overhaul of the computational core plus a new layer of
mathematics on top:

- **10²–10³× faster.** Every core algorithm (products, subset constructions,
  minimization, padding) was profiled and rewritten as batched, sparsity-aware NumPy:
  the reference benchmark query dropped from 85 s to 0.03 s, and the v1 test suite from
  200 s to under 2 s. Memory usage is linear — batched binary-search transition lookups
  and hashed partition refinement replaced all dense intermediate tensors.
- **NumPy-canonical, JAX optional.** The core depends only on NumPy and installs
  anywhere. JAX moved to the optional extra `autstr[jax]` and accelerates exactly one
  thing: `SparseDFA.accepts_batch`, bulk word processing on a fixed automaton
  (jit + scan, ~5× on CPU for 100k+ words, GPU-ready).
- **Uniformly automatic classes** (`autstr.uniform`): present a whole *class* of
  structures with one automaton tuple reading an advice string synchronously with the
  element encodings. Evaluate a first-order query once — model checking any member is
  then just running its advice word through a DFA. Includes relativized quantifiers,
  member-structure instantiation (`get_structure`), and a first-order bootstrap
  mechanism (`define`) for building complex relations from small primitives.
- **Graph classes** (`autstr.graphs`): bounded tree-depth and bounded pathwidth graphs
  as uniformly automatic classes with set-valued elements — full **monadic second-order
  logic** over the graphs. Conversion from/to networkx (exact decompositions for small
  graphs), graphviz rendering. A 6-state automaton deciding bipartiteness for *every*
  graph of tree-depth ≤ 3 compiles in seconds.
- **Algebraic classes** (`autstr.algebra`): finite Boolean algebras, finite abelian
  groups, and automatic presentations of the localizations **Z[1/p]** — with the
  first-order divisibility sentences that distinguish them.
- **Non-abelian group classes** (`autstr.groups`): the groups with a cyclic subgroup of
  index ≤ 2 (dihedral, dicyclic/quaternion, semidihedral, modular, …) under one advice
  format, and extraspecial p-groups — nilpotency class 2 with growing rank.
- **Notebooks**: `treedepth_graphs`, `algebra_showcase`, and `groups_showcase` walk
  through all of the above end to end.

### Installation

```bash
pip install autstr           # NumPy-only core
pip install autstr[jax]      # + JAX-accelerated batch word processing
pip install autstr[graphs]   # + networkx conversion for the graph classes
```

#### Verify Installation
```bash
python -c "from autstr import __version__; print(f'AutStr v{__version__} installed')"
# Should output: AutStr v2.0.0 installed
```

## Getting started
The most convenient way to get started is through the arithmetic package, which supports integers.

### Defining a relation
```python
from autstr.arithmetic import VariableETerm as Var
```
The most basic building blocks are variables:
```python
x = Var('x')
y = Var('y')
```
Variables can be added to integer constants or other variables. A variable can be multiplied by a constant but not by another variable (linear arithmetic):
```python
t0 = x + y + 3
t1 = 2 * x
```
To define a relation, compare terms using built-in operations. AutStr supports $<$ and $=$ comparisons:
```python
R = t0.lt(t1)  # < (less than) Defines binary relation R: (x, y) ∈ R ⇔ x + y + 3 < 2x
```
You can work with this representation of integer solutions similarly to an explicit set:
* Test for emptiness:
```python
R.is_empty()
```
* Test for finiteness:
```python
R.is_finite()
```
* Test tuple membership:
```python
(0, 4) in R
```
* Enumerate all solutions (AutStr guarantees each solution is enumerated exactly once):
```python
for solution, _ in zip(R, range(10)):  # First 10 pairs
    print(solution)
```

### Weak divisibility
In addition to ordinal comparisons, AutStr supports the weak divisibility predicate (base 2):
- $x|y \Leftrightarrow \exists n > 0: y = 2^n \land y \text{ divides } x$

This powerful predicate enables definitions like the powers of 2:
```python
Pt = x | x  # Set of powers of 2

assert 2**10 in Pt
assert 3 not in Pt
```

### Relational algebra
Combine relations using relational algebra operators:
* **Union**: $(x, y) \in E_0 \Leftrightarrow \text{Pt}(x) \lor R(x, y)$
```python
E0 = Pt | R 
```
* **Intersection**: $(x, y) \in E_1 \Leftrightarrow \text{Pt}(x) \land R(x, y)$
```python
E1 = Pt & R
```
* **Complement**: $(x, y) \in E_2 \Leftrightarrow \neg R(x, y)$
```python
E2 = ~R
```
* **Projection (Existential Quantification)**: $y \in E_3 \Leftrightarrow \exists x . R(x, y)$
```python
E3 = R.drop(['x'])  # Or R.ex('x')
```
* **Infinite Quantification**: $y \in E_4 \Leftrightarrow \exists^{\infty} x . E_0(x, y)$
```python
E4 = E0.exinf('x') 
```

### Automatic Structures
AutStr enables finite automata to represent infinite mathematical structures through **automatic presentations**. This powerful technique encodes:
- The domain as a regular language of strings (`automata['U']`)
- Relations as synchronous multi-tape automata recognizing valid tuples
- Enables efficient model checking for first-order queries

#### Key Features
```python
from autstr.presentations import AutomaticPresentation

# Initialize with automata for domain and relations
ap = AutomaticPresentation(automata={
    'U': universe_dfa,       # Domain automaton
    'R1': relation_dfa1,     # Relation automaton (arity k)
    'R2': relation_dfa2
})
```

1. **Automated Consistency Enforcement**
   - Automatically pads and minimizes automata
   - Ensures relations stay within domain bounds
   - Handles variable-length string encodings

2. **First-order Query Evaluation**
   ```python
   # Evaluate formula and get solution automaton
   solution_automaton = ap.evaluate("∃x.∀y.R1(x,y) ∧ R2(y)")
   
   # Check truth of closed formulas
   is_true = ap.check("∀x.∃y.R1(x,y)")
   ```

3. **Quantifier Support**
   - Existential (`∃`): `projection()`
   - Universal (`∀`): `complement(projection(complement()))`
   - Infinite quantification via automata operations

4. **Dynamic Relation Updates**
   ```python
   # Temporarily modify relations during evaluation
   result = ap.evaluate("R1(x,y)", updates={'R1': new_automaton})
   
   # Permanently update presentation
   ap.update(R1=new_automaton, R2="automaton_spec")
   ```

5. **Formal Logic Integration**
   - Seamless parsing of first-order formulas
   - Handles free variable expansion
   - Supports negation, conjunction, disjunction

#### Workflow Example
```python
# 1. Define domain automaton (e.g., base-2 integers)
universe = DFA(states=..., transitions=...)  

# 2. Create relation automata (e.g., addition)
plus_automaton = build_automaton_for("x+y=z")

# 3. Construct presentation
pres = AutomaticPresentation(automata={'U': universe, 'Plus': plus_automaton})

# 4. Query: "∃z. Plus(5,x,z) ∧ (z > 10)"
query = "∃z. Plus('5',x,z) ∧ ∃k. Plus(z,'10',k)"
solutions = pres.evaluate(query)

# 5. Enumerate solutions (x=6,7,8,...)
for sol in solutions:
    print(decode(sol))  # Output: 6, 7, 8, ...
```
### Uniformly Automatic Classes
New in 2.0: a single tuple of automata can present an entire *class* of structures.
Every automaton carries one extra tape holding an **advice string** that is read
synchronously with the element encodings; fixing an advice instantiates one member
structure. The payoff: a first-order (or, over set-valued elements, monadic
second-order) query is compiled **once per class** — deciding it for a member is then
just running that member's advice word through a DFA, in microseconds. The theory
behind this layer is developed in references [2] and [3] below; the built-in classes
implement the meta-theorems of [2].

#### Graphs of bounded tree-depth, with full MSO
```python
import networkx as nx
from autstr.graphs import TreeDepthClass, TreeDepthGraph

cls = TreeDepthClass(3)  # all graphs of tree-depth <= 3

bipartite = ('exists c.(all x.(all y.((not E(x,y)) or '
             '((Subset(x,c) and (not Subset(y,c))) or '
             '((not Subset(x,c)) and Subset(y,c))))))')
dfa, _ = cls.evaluate(bipartite)   # compiled once: a 6-state automaton

triangle = TreeDepthGraph.from_networkx(nx.cycle_graph(3))
dfa.accepts([(s,) for s in cls.advice(triangle)])   # False — in microseconds
```
`PathWidthClass(w)` does the same for bounded pathwidth. Both classes represent sets
of vertices (as in MSO0), support `check(phi, graph, x={...})` for concrete set
assignments, convert from/to networkx, and render via graphviz.

#### Algebra: from powerset algebras to Z[1/p]
```python
from autstr.algebra import FiniteAbelianGroups, z1p_localization

ab = FiniteAbelianGroups()          # advice = the cyclic decomposition
ab.check('A(x,y,z)', [2, 3], x=(1, 1), y=(1, 2), z=(0, 0))   # True in Z_2 + Z_3

z2 = z1p_localization(2)            # automatic presentation of (Z[1/2], +)
z2.check('A(x,y,z)', x=z2.from_fraction(1, 2),
         y=z2.from_fraction(3, 4), z=z2.from_fraction(5, 4))  # True
z2.check('all x.(exists y.(A(y,y,x)))')   # True: everything is 2-divisible
```

#### Non-abelian groups
```python
from autstr.groups import IndexTwoCyclicGroups, ExtraspecialGroups

G = IndexTwoCyclicGroups()   # dihedral, quaternion, semidihedral, modular, ...
q8 = G.dicyclic(4)           # the quaternion group Q_8
G.check('M(x,y,z)', q8, x=(0, 1), y=(1, 0), z=(1, 1))   # i * j = k

H = ExtraspecialGroups(3)    # nilpotency class 2, order 3^(1+2n)
H.check('Cen(x)', 2, x=(1, (0, 0), (0, 0)))              # central elements
```
The multiplication of the index-2 class is *defined first-order* from small primitive
automata via `UniformlyAutomaticClass.define` — the uniform analog of the Büchi
arithmetic bootstrap. See the `groups_showcase` notebook for the theory of where
uniform automaticity ends (spoiler: unbounded bilinearity).

### Algorithmic Design with Infinite Sets: The Sieve of Eratosthenes
AutStr enables novel algorithm design using infinite sets as first-class citizens. This implementation of the Sieve of Eratosthenes maintains the infinite candidate prime set symbolically:

```python
def infinite_sieve(steps):
    """Sieve of Eratosthenes over infinite integers"""
    candidates = (x.gt(1))  # Initial infinite set: {2,3,4,...}
    primes = []
    
    for _ in range(steps):
        # Find smallest candidate (symbolic operation)
        for p in candidates: # Elements are listed in ascending order (absolute values) 
            primes.append(p[0])
            break
        
        # Remove multiples: candidates = candidates \ {k·p | k>1}
        p = primes[-1]
        y = Var("y")
        multiples = (x.eq(p * y)).drop("y")
        candidates = candidates & ~multiples 
        
    return primes, candidates

# Execute first 3 sieving steps
primes, remainig = infinite_sieve(steps=3)
print(f"Primes found: {primes}")  # [2,3,5,7,11]
print(f"Remaining infinite set:") 
remaining.evaluate().show_diagram()
```

#### Key Algorithmic Features:
1. **Ordered Iteration**
   ```python
   for p in candidates:  # enumerates candidates in ascending order
   ```
2. **Infinite Set Operations**
   ```python
   multiples = (x.eq(p * y)).drop("y")
   candidates = candidates & ~multiples   
   ```
3. **Lazy Evaluation**
   - Relations remain symbolic until materialization
   - No explicit storage of infinite elements

#### Practical Guidelines:
1. **Prefer Deep/Narrow Formulas**
   ```python
   # Width=3 : 
   wide = "∃x.∃y.∃z.E(x,y) ∧ E(y, z)"
   
   # width=2 (easier):
   narrow = "∃x.∃y.E(x,y) ∧ (∃z.E(y,z))"
   ```
   - Depth can cause non-elementary(!) statespace explosion but scales often much better due to incremental minimization
   - Width causes exponential alphabet growth: $|\Sigma|^k$ but SparseDFAs can avoid explicit enumeration of entire alphabet in many cases.

2. **Complexity Boundaries**
   | Parameter        | Best Case       | Worst Case          |
   |------------------|-----------------|---------------------|
   | **Quantifier Depth** | Constant state space   | Non-elementary state space |
   | **Free Variables**  | constant number of exceptions  | Exponential alphabet size |

#### Theoretical Insight
While this infinite sieve beautifully demonstrates symbolic algorithm design, state complexity grows rapidly for sieved primes

> **Practical Recommendation**: Use infinite representations for conceptual modeling and verification, but switch to finite approximations with bounds for computational work. AutStr excels at proving properties about infinite structures, not processing them exhaustively.

This paradigm shift enables:
- Formal verification of infinite-state algorithms
- Symbolic exploration of hypothetical structures
- Correctness proofs for infinite data transformations

#### Theoretical Foundation
Automatic presentations leverage:
- **Regularity Preservation**: First-order operations maintain automata recognizability
- **Decidability**: First-order theories remain decidable for automatic structures

This implementation provides a complete toolkit for working with automatic structures over countable domains like ℤ, ℚ, and tree-like structures.

### Final Note
AutStr began as a summer passion project in 2022—a practical exploration of the automatic structures I'd studied theoretically during my PhD. This library represents the intersection of academic curiosity and hands-on implementation, born from a desire to make abstract model theory concepts tangible.

Released in July 2025 following a major update, the library gained significant new features beyond its original vision. That update was developed through a vibe coding session using DeepSeek, with extensive human testing and supervision throughout the process, and introduced the `sparse_dfa` backend, serialization, MSO0, and modernized packaging.

**Version 2.0 (July 2026) was designed and implemented in an intensive two-day
pair-programming session with Claude — specifically Anthropic's Fable 5 model —
working inside Claude Code.** Fable 5 profiled and rewrote the entire automata core
(the 10²–10³× speedups above), migrated the library from JAX to a NumPy-canonical
representation, and then built the whole uniformly automatic layer: the generic
advice-class machinery, bounded tree-depth and pathwidth graphs with MSO, finite
Boolean algebras, finite abelian groups, the Z[1/p] presentations, and the non-abelian
group classes — each verified against exhaustive ground-truth oracles, brute-force
semantics checks, or exact arithmetic models along the way. The mathematical direction,
the theory of uniform automaticity these packages implement, and the final review
remained human; the code is Fable 5's. It was a genuinely remarkable collaboration:
ideas from my dissertation that had waited a decade for an implementation became
running, tested code within hours of being sketched in conversation.

#### Performance and Maintenance
Version 2.0 removed the known performance bottlenecks:

- All core algorithms are batched, sparsity-aware NumPy (binary-search transition
  lookups, hashed partition refinement, frontier-batched constructions) with linear
  memory usage
- JAX is now an optional extra used solely for bulk word processing on fixed automata
- Efficient serialization allows storing precompiled results

Remaining optimization opportunities:
- Query optimization: advanced planning for first-order queries
- Caching of evaluated query automata across `check` calls

As this is a passion project developed outside my primary research, active maintenance will be limited. That said:

- Bug reports are welcome and will be prioritized
- Performance contributions are especially appreciated
- Research collaborations involving AutStr are encouraged


## References on Automatic Structures
1. **Abu Zaid, F.**  
   *Algorithmic Solutions via Model Theoretic Interpretations.*  
   Dissertation, RWTH Aachen University, 2016.  
   DOI: [10.18154/RWTH-2017-07663](https://doi.org/10.18154/RWTH-2017-07663)  

2. **Abu Zaid, F.**  
   *Uniformly Automatic Classes of Finite Structures.*  
   38th IARCS Annual Conference on Foundations of Software Technology and Theoretical Computer Science (FSTTCS 2018).  
   Leibniz International Proceedings in Informatics (LIPIcs), Volume 122, Pages 10:1–10:21.  
   DOI: [10.4230/LIPIcs.FSTTCS.2018.10](https://doi.org/10.4230/LIPIcs.FSTTCS.2018.10)  
   *The algorithmic meta-theorems for finite Boolean algebras, finite groups, and graphs of bounded tree-depth implemented by the `autstr.uniform`, `autstr.graphs`, `autstr.algebra`, and `autstr.groups` packages.*

3. **Abu Zaid, F., Grädel, E., & Reinhardt, F.**  
   *Advice Automatic Structures and Uniformly Automatic Classes.*  
   26th EACSL Annual Conference on Computer Science Logic (CSL 2017).  
   Leibniz International Proceedings in Informatics (LIPIcs), Volume 82, Pages 35:1–35:20.  
   DOI: [10.4230/LIPIcs.CSL.2017.35](https://doi.org/10.4230/LIPIcs.CSL.2017.35)  
   *Introduces automatic presentations with advice — the foundation of the uniformly automatic classes in this library; the advice-automatic presentation of (Q, +) is the blueprint for `z1p_localization`.*

4. **Blumensath, A., & Grädel, E.**  
   *Automatic Structures.*  
   Proceedings of the 15th Annual IEEE Symposium on Logic in Computer Science (LICS 2000).  
   Pages: 51–62.  
   URL: [LICS 2000 Proceedings](https://lics.siglog.org/2000/Grdel-AutomaticStructures.html)  

5. **Khoussainov, B., & Nerode, A.**  
   *Automatic presentations of structures.*  
   In D. Leivant (Ed.), Logic and Computational Complexity (LCC 1994). Lecture Notes in Computer Science, vol 960.  
   Springer. DOI: [10.1007/3-540-60178-3_93](https://doi.org/10.1007/3-540-60178-3_93)  

6. **Khoussainov, B., Rubin, S., & Stephan, F.**  
   *Automatic Structures: Richness and Limitations.*  
   Logical Methods in Computer Science, Volume 3, Issue 2 (2007).  
   arXiv: [cs/0703064](https://arxiv.org/abs/cs/0703064)  
   DOI: [10.2168/LMCS-3(2:2)2007](https://doi.org/10.2168/LMCS-3%282%3A2%292007)  
