"""Large-scale comparison of implicit vs explicit first-order evaluation.

Sweeps the bounded-rank-width group classes over growing ring size (q = p^d) and
generator count (n), running each first-order sentence two ways:

  * explicit  -- build the presentation and compile the query automaton
                 (`cls.check`), the classical path;
  * implicit  -- evaluate the formula on the fly over the base automata
                 (`check_implicit`), never materializing a product.

For every cell where both finish it asserts they agree; and it charts the
feasibility boundary -- where the explicit path times out or runs out of memory
while the implicit path keeps going.

Each experiment runs in its own subprocess with a wall-clock timeout and an
address-space cap (RLIMIT_AS), so a blow-up is recorded as `timeout`/`oom`
instead of taking down the driver. The grid is pruned monotonically (once a path
fails at some n, larger n for the same series is skipped) and a global time
budget stops the run early. Designed for a 16-vCPU / 64 GB box; safe to run
anywhere.

    python benchmarks/implicit_vs_explicit.py \
        --budget-seconds 10800 --mem-gb 55 \
        --explicit-timeout 240 --implicit-timeout 120 \
        --out benchmarks/implicit_vs_explicit_report.md

Nothing here is a pytest test (it is a long exploratory benchmark); it prints a
Markdown report and writes it to --out.
"""
import argparse
import subprocess
import sys
import time


# ------------------------------------------------------------------ workloads
# Each workload builds a member of a class and decides one sentence. They run in
# a spawned subprocess, so they import inside the function and take plain args.

SENTENCES = {
    # the clique is non-commutative; the zero form is commutative
    'nonabelian': 'exists x.(exists y.(exists z.(M(x,y,z) and (not M(y,x,z)))))',
    'commutative': 'all x.(all y.(all z.((M(x,y,z)) -> (M(y,x,z)))))',
    # every element cubed is trivial (exponent p, odd p) -- a deeper alternation
    'cube_trivial': ('all x.(exists a.(exists b.('
                     'M(x,x,a) and M(a,x,b) and M(b,b,b))))'),
    # nested alternation: for all x there is y that fails to commute with some z
    'alt3': ('all x.(exists y.(exists z.('
             '(M(x,y,z) and (not M(y,x,z))) or Eq(x,x))))'),
}


def _word_member(p, d, n):
    from autstr.groups import CutRankGroups
    c = CutRankGroups(p, d=d)
    advice = c.advice(n, c.clique_form(n))
    return c, advice


def _tree_member(p, d, n):
    from autstr.tree_groups import CutRankTreeGroups
    c = CutRankTreeGroups(p, d=d)
    shape = c.balanced(n)
    advice = c.advice(shape, c.clique_form(n))
    return c, advice


MEMBERS = {'word': _word_member, 'tree': _tree_member}


def _run_explicit(kind, p, d, n, sname):
    c, advice = MEMBERS[kind](p, d, n)
    return bool(c.check(SENTENCES[sname], advice))


def _run_implicit(kind, p, d, n, sname):
    c, advice = MEMBERS[kind](p, d, n)
    return bool(c.check_implicit(SENTENCES[sname], advice))


RUNNERS = {'explicit': _run_explicit, 'implicit': _run_implicit}


# ------------------------------------------------------------ isolation harness
# Each task runs as a fresh `python -m` subprocess (an in-process multiprocessing
# child mis-parses the nltk formulas; a clean interpreter does not). The child
# caps its own address space (RLIMIT_AS) before importing, so a blow-up is a
# clean MemoryError or an OS kill -- the driver survives either way.

RESULT_TAG = "__RESULT__"


def _run_task(path, kind, p, d, n, sname, mem_bytes):
    """Executed in the child interpreter: cap memory, run, print the result."""
    if mem_bytes:
        import resource
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except (ValueError, OSError):
            pass
    try:
        value = RUNNERS[path](kind, int(p), int(d), int(n), sname)
        print(f"{RESULT_TAG}\tok\t{value}")
    except MemoryError:
        print(f"{RESULT_TAG}\toom\t")
    except Exception as e:  # noqa: BLE001 -- report any failure as a cell
        print(f"{RESULT_TAG}\terror\t{type(e).__name__}: {e}")


def run_isolated(path, args, timeout, mem_bytes):
    """Run one workload in a fresh subprocess. Returns (status, value, secs).
    status in {ok, timeout, oom, killed, error}."""
    kind, p, d, n, sname = args
    cmd = [sys.executable, __file__, '--task', path, kind,
           str(p), str(d), str(n), sname, '--mem-bytes', str(mem_bytes)]
    start = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return 'timeout', None, time.time() - start
    elapsed = time.time() - start
    line = next((ln for ln in proc.stdout.splitlines()
                 if ln.startswith(RESULT_TAG)), None)
    if line is None:
        # no result line: OS OOM-kill (rc -9) or a hard crash
        return 'killed', None, elapsed
    _, status, value = line.split('\t', 2)
    if status == 'ok':
        value = {'True': True, 'False': False}.get(value, value)
    return status, value, elapsed


# --------------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--budget-seconds', type=float, default=10800)   # 3 hours
    ap.add_argument('--mem-gb', type=float, default=55.0)
    ap.add_argument('--explicit-timeout', type=float, default=240.0)
    ap.add_argument('--implicit-timeout', type=float, default=120.0)
    ap.add_argument('--out', default='benchmarks/implicit_vs_explicit_report.md')
    ap.add_argument('--max-n-word', type=int, default=8)
    ap.add_argument('--max-n-tree', type=int, default=6)
    # internal single-task mode (spawned by the driver, not called by hand)
    ap.add_argument('--task', nargs=6, default=None,
                    metavar=('PATH', 'KIND', 'P', 'D', 'N', 'SENTENCE'))
    ap.add_argument('--mem-bytes', type=int, default=0)
    args = ap.parse_args()

    if args.task is not None:
        _run_task(*args.task, args.mem_bytes)
        return

    mem_bytes = int(args.mem_gb * 1024 ** 3)
    rings = [(2, 1), (3, 1), (2, 2), (3, 2), (2, 3), (2, 4)]   # Z2,Z3,Z4,Z9,Z8,Z16
    kinds = [('word', args.max_n_word), ('tree', args.max_n_tree)]

    t0 = time.time()
    rows = []           # (kind, q, n, sentence, exp_status, exp_t, imp_status, imp_t, agree)
    # prune monotonically: once a path fails for a (kind, q, sentence) series at
    # some n, skip larger n for that path.
    dead = set()        # (path, kind, p, d, sentence)

    def budget_left():
        return args.budget_seconds - (time.time() - t0)

    print(f"# implicit vs explicit -- budget {args.budget_seconds:.0f}s, "
          f"mem cap {args.mem_gb:.0f} GB\n", flush=True)

    stop = False
    for (kind, max_n) in kinds:
        for (p, d) in rings:
            q = p ** d
            for sname, sentence in SENTENCES.items():
                for n in range(2, max_n + 1):
                    if budget_left() < 5:
                        stop = True
                        break
                    cell = {'kind': kind, 'q': q, 'p': p, 'd': d, 'n': n,
                            'sentence': sname}
                    for path, timeout in (('explicit', args.explicit_timeout),
                                          ('implicit', args.implicit_timeout)):
                        key = (path, kind, p, d, sname)
                        if key in dead:
                            cell[path] = ('skipped', None)
                            continue
                        if budget_left() < min(timeout, 30):
                            cell[path] = ('nobudget', None)
                            continue
                        status, value, secs = run_isolated(
                            path, (kind, p, d, n, sname),
                            min(timeout, budget_left()), mem_bytes)
                        cell[path] = (status, value, secs)
                        if status != 'ok':
                            dead.add(key)      # prune larger n for this series
                    rows.append(cell)
                    _print_cell(cell)
                if stop:
                    break
            if stop:
                break
        if stop:
            break

    report = _render(rows, args, time.time() - t0)
    with open(args.out, 'w') as f:
        f.write(report)
    print("\n" + report, flush=True)


def _cellstr(entry):
    if entry is None:
        return '-'
    status = entry[0]
    if status == 'ok':
        return f"ok({entry[1]}) {entry[2]:.1f}s"
    if len(entry) >= 3 and entry[2] is not None:
        return f"{status} {entry[2]:.0f}s"
    return status


def _agree(cell):
    e, i = cell.get('explicit'), cell.get('implicit')
    if e and i and e[0] == 'ok' and i[0] == 'ok':
        return 'YES' if e[1] == i[1] else '*** MISMATCH ***'
    return ''


def _print_cell(cell):
    print(f"[{cell['kind']:4s} Z/{cell['q']:<2d} n={cell['n']} "
          f"{cell['sentence']:12s}] explicit={_cellstr(cell.get('explicit'))} "
          f"| implicit={_cellstr(cell.get('implicit'))} {_agree(cell)}",
          flush=True)


def _render(rows, args, elapsed):
    lines = ["# Implicit vs explicit first-order evaluation",
             "",
             f"- budget: {args.budget_seconds:.0f}s, used {elapsed:.0f}s",
             f"- memory cap per task: {args.mem_gb:.0f} GB (RLIMIT_AS)",
             f"- timeouts: explicit {args.explicit_timeout:.0f}s, "
             f"implicit {args.implicit_timeout:.0f}s",
             "",
             "| class | ring | n | sentence | explicit | implicit | agree |",
             "|-------|------|---|----------|----------|----------|-------|"]
    mismatches = 0
    for c in rows:
        agree = _agree(c)
        if 'MISMATCH' in agree:
            mismatches += 1
        lines.append(
            f"| {c['kind']} | Z/{c['q']} | {c['n']} | {c['sentence']} | "
            f"{_cellstr(c.get('explicit'))} | {_cellstr(c.get('implicit'))} | "
            f"{agree} |")
    # boundary summary
    lines += ["", "## Feasibility boundary (largest n with status ok)", "",
              "| class | ring | explicit max n | implicit max n |",
              "|-------|------|----------------|----------------|"]
    series = {}
    for c in rows:
        key = (c['kind'], c['q'])
        s = series.setdefault(key, {'explicit': 0, 'implicit': 0})
        for path in ('explicit', 'implicit'):
            e = c.get(path)
            if e and e[0] == 'ok':
                s[path] = max(s[path], c['n'])
    for (kind, q), s in sorted(series.items()):
        lines.append(f"| {kind} | Z/{q} | {s['explicit'] or '-'} | "
                     f"{s['implicit'] or '-'} |")
    lines += ["", f"**mismatches: {mismatches}** "
              f"(implicit and explicit must agree wherever both finish)."]
    return "\n".join(lines) + "\n"


if __name__ == '__main__':
    main()
