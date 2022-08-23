# AutStr
Working with infinite data structures in python.

## Introduction
Ever wondered what would happen if you could input your algorithm an infinite structure, e.g. a graph, instead of a finite one?
Well, with AutStr and you can do exactly that at least for some infinite structures.
AutStr offers an easy to use interface to define (relational) structures over predefined infinite base structures. 
Currently, AutStr comes with buildin support for a strong extension of linear integer arithmetic but additional base structures over arbitrary countable domains are definable via the low level API.

## Getting started
The most convenient way to get started is through the arithmetic packages, which supports integers.
### Defining a relation
```{.py}
from autstr.arithmetic import VariableETerm as Var
```
The most basic building blocks are variables. 
```
x = Var('x')
y = Var('y')
```
They can be added with integer constants or other variables. A variable can also be multiplied with a constant but not with another variable (linear arithmetic).
```
t0 = x + y + 3
t1 = 2 * x
```
In order to define a relation we need to relate the terms somehow. AutStr has buildin support for $<$ and $=$ comparissons
```
R = t0.lt(t1)  # t0 < t1 
```
cmp defines a binary relation $R$ between $x$ and $y$ with $(x, y)\in R \Leftrightarrow x + y + 3 < 2x$.
We can use cmp as a representation of it's integer solution space in almost the same way as if we would have gotten it explicitly. In particular we can
* Test for emptyness
```
R.isempty()
```
* Test for finiteness
```
R.isfinite()
```
* Test if a tuple is contained
```
(0, 4) in cmp
```
* Enumerate all solutions. AutStr guarantees that every solution is enumerated exactly once (although this might of course take infinitely long).
```
for tuple, _ in zip(cmp, range(10)): # Iterate through first 10 pairs
  print(tuple)
```

### Weak divisibility
In addition to ordinal comparisons AutStr also defines the weak divisibility predicate (with base 2) where $x| y\Leftrightarrow\exists n > 0: y = 2^n \wedge y \text{ divides } x$. This is a very powerfull predicate. For instance, the following code defines the powers of $2$
```
Pt = x|x

assert 2**10 in Pt
assert 3 not in Pt
```

### Relational Algebra
AutStr allows to combine more complex relations from base relations using relational algebra. Implemented operators:
* Union: $(x, y)\in E_0 \Leftrightarrow \mathop{Pt}(x) \vee R(x, y)$
```
E0 = Pt | R 
```
* Join: $(x, y)\in E_1 \Leftrightarrow \mathop{Pt}(x) \wedge R(x, y)$
```
E1 = Pt & R
```
* Complement: $(x, y)\in E_2 \Leftrightarrow \neg R(x, y)$
```
E2 = ~R
```
* Drop (= existential quantification): $y\in E_3 \Leftrightarrow \exists x.R(x, y)$
```
E3 = R.drop(['x']) # R.ex('x')
```
* Infinite quantification: $y\in E_4 \Leftrightarrow \exists \text{ infinitely many } x(E_0(x, y))$
```
E4 = E0.exinf('x') 
```

### First-order interpretations

Variables and operators are evaluated in the structure $(\mathbb{Z}, +, |_2, <, 0, 1)$. 


## What's behind it?


