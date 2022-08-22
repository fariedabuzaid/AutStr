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
cmp = t0.lt(t1)  # t0 < t1 
```
cmp defines a binary relation $R$ between $x$ and $y$ with $(x, y)\in R \Leftrightarrow x + y + 3 < 2x$.
We can use cmp as a representation of it's integer solution space in almost the same way as if we would have gotten it explicitly. In particular we can
* Test for emptyness
```
cmp.isempty()
```
* Test for finiteness
```
cmp.isfinite()
```
* Test if a tuple is contained
```
(0, 4) in cmp
```
* Enumerate all solutions
```
for a, b in cmp:
  if abs(max([a, b])) > 5:
    break
```

### Relational Algebra

### First-order interpretations

Variables and operators are evaluated in the structure $(\mathbb{Z}, +, |_2, <, 0, 1, )$. 


## What's behind it?


