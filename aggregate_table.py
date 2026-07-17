#!/usr/bin/env python
"""Collect the per-run LaTeX table rows into one comparison table across models.

Every ``run.py`` invocation logs one line per evaluated split:

    table test: <Model> & <frames> & <iters>[N trials] & <params> & <acc> & <auc>
                & <rej03> & <rej05> & <rej08> & <time>s & <flops> & <kNN> \\

For warm-started runs that line already carries ``mean +- std`` over the trials in
that run directory. This script walks a ``runs/`` tree, takes the latest such line
per run directory (highest ``run_idx`` = most trials), de-duplicates by
(model, frames, kNN) — so ablation variants of the same model keep separate rows,
and a true re-run of an identical variant supersedes older ones by the log file's
mtime (NOT by directory order: run-dir names end in a random suffix, so path order
says nothing about recency) — and assembles them into a single LaTeX ``tabular``.

    python aggregate_table.py                       # scans runs/, split=test
    python aggregate_table.py --runs runs/topt_local_debug --split test --out table.tex
"""

import argparse
import os
import re
from glob import glob

COLUMNS = (
    r"model & frames & iters & params & accuracy & AUC & "
    r"$1/\epsilon_B$(0.3) & (0.5) & (0.8) & time & FLOPs & kNN"
)


def latest_row(run_dir, split):
    """Return (row, mtime) for the `table <split>:` row from the highest-index log
    in `run_dir`; mtime is the modification time of the log the row came from
    (the recency stamp used for cross-run-dir dedup)."""
    logs = sorted(
        glob(os.path.join(run_dir, "out_*.log")),
        key=lambda p: int(re.search(r"out_(\d+)\.log$", p).group(1)),
    )
    pattern = re.compile(rf"table {re.escape(split)}:\s*(.*?)\s*\\\\\s*$")
    row, src = None, None
    for log in logs:  # later logs accumulate more trials -> keep the last match
        try:
            with open(log) as f:
                for line in f:
                    m = pattern.search(line)
                    if m:
                        row, src = m.group(1), log
        except OSError:
            continue
    if row is None:
        return None, 0.0
    try:
        mtime = os.path.getmtime(src)
    except OSError:
        mtime = 0.0
    return row, mtime


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runs", default="runs", help="root directory to scan (default: runs)")
    ap.add_argument("--split", default="test", help="which split's row to collect (default: test)")
    ap.add_argument("--out", default=None, help="optional path to write the .tex table")
    args = ap.parse_args()

    run_dirs = sorted(
        {
            os.path.dirname(p)
            for p in glob(os.path.join(args.runs, "**", "out_*.log"), recursive=True)
        }
    )
    # key on (model, frames, kNN) -- the variant axes that are table columns -- so
    # ablation runs of the SAME model (identity vs learnedpd frames, deltaR vs
    # minkowski kNN) each keep their own row. On a true re-run of the identical
    # variant the row with the NEWEST log mtime wins (run-dir names carry a random
    # suffix, so lexicographic path order says nothing about which run is current).
    rows = {}  # key -> (mtime, row, run_dir)
    for d in run_dirs:
        row, mtime = latest_row(d, args.split)
        if row is None:
            continue
        cells = [c.strip() for c in row.split("&")]
        model = cells[0]
        frames = cells[1] if len(cells) > 1 else ""
        knn = cells[-1] if len(cells) > 2 else ""
        key = (model, frames, knn)
        if key in rows:
            prev_mtime, prev_row, prev_dir = rows[key]
            if prev_row != row:
                newer, older = (d, prev_dir) if mtime >= prev_mtime else (prev_dir, d)
                print(f"[note] re-run of {key}: keeping {newer} (newer log), dropping {older}")
            if mtime < prev_mtime:
                continue
        rows[key] = (mtime, row, d)

    if not rows:
        print(f"No 'table {args.split}:' rows found under {args.runs}/")
        return

    body = " \\\\\n".join(rows[k][1] for k in sorted(rows)) + " \\\\"
    table = (
        "% columns: " + COLUMNS + "\n"
        "\\begin{tabular}{l l r r c c c c c r r l}\n"
        "\\hline\n" + body + "\n\\hline\n"
        "\\end{tabular}\n"
    )
    print(table)
    if args.out:
        with open(args.out, "w") as f:
            f.write(table)
        print(f"[wrote {args.out}]")


if __name__ == "__main__":
    main()
