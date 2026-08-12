"""Regenerate the serialized built-in presentations in `autstr/buildin/bin`.

The loaders (`BuechiArithmetic`, `BuechiArithmeticZ`, `MSO0`) deserialize these
artifacts with ``enforce_consistency=False`` — the stored automata are taken to
be already restricted to the universe. So whenever the construction of a
built-in presentation changes, or the consistency preparation itself does, the
artifacts have to be rebuilt from their generators:

    python scripts/gen_builtin_presentations.py [name ...]

Rewriting an artifact also re-encodes it in the current serializer format,
which is bulkier than the one some of them were written with — so rebuild only
the ones whose content actually changed.
"""
import sys
from pathlib import Path

from autstr.buildin.presentations import (
    buechi_arithmetic, buechi_arithmetic_Z, finite_powerset,
)

BIN = Path(__file__).resolve().parent.parent / 'autstr' / 'buildin' / 'bin'

TARGETS = {
    'buechi.autstr': buechi_arithmetic,
    'buechiZ.autstr': buechi_arithmetic_Z,
    'mso0.autstr': finite_powerset,
}


def main(names=()) -> None:
    for filename, build in TARGETS.items():
        if names and filename not in names and Path(filename).stem not in names:
            continue
        presentation = build()
        path = BIN / filename
        presentation.automatic_presentation_to_file(str(path))
        print(f'{filename}: {len(presentation.automata)} automata '
              f'({", ".join(sorted(presentation.automata))})')


if __name__ == '__main__':
    main(sys.argv[1:])
