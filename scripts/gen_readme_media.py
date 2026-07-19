#!/usr/bin/env python3
"""Regenerate the static images embedded in the README.

Currently: the automaton recognizing the integers > 1 that are divisible by
none of a small set of primes — the "remaining candidates" of the infinite
Sieve of Eratosthenes (see notebooks/arithmetic_and_algebra.ipynb).

    python scripts/gen_readme_media.py

Writes PNGs into docs/media/. Run it whenever the rendered automaton or the
example changes; the file is committed so the README renders on GitHub.
"""
from pathlib import Path

from autstr.arithmetic import VariableETerm as Var

MEDIA = Path(__file__).resolve().parent.parent / "docs" / "media"


def coprime_automaton(primes):
    """The DFA for {x > 1 : no p in `primes` divides x}."""
    x = Var("x")
    s = x.gt(1)
    for p in primes:
        y = Var("y")
        s = s & ~((x.eq(p * y)).drop("y"))     # remove the multiples of p
    return s.evaluate()


def main():
    MEDIA.mkdir(parents=True, exist_ok=True)
    target = MEDIA / "sieve_automaton"
    dfa = coprime_automaton([2, 3])            # 9 states: clean and legible
    dot = dfa.show_diagram(filename=str(target), format="png", view=False)
    dot.attr(dpi="160", bgcolor="transparent")
    dot.render(filename=str(target), format="png", cleanup=True)
    print(f"wrote {target}.png  ({dfa.num_states} states)")


if __name__ == "__main__":
    main()
