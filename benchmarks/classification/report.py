"""Summarise a benchmark run with the statistics of Demšar (2006).

Usage::

    python report.py results.jsonl --reference DKLMRSLVQ > RESULTS.md
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

from prosemble.core.evaluation import (
    compare_to_reference, friedman_test, nemenyi_critical_difference,
)

METRICS = [  # (key, title, higher is better)
    ('accuracy', 'Accuracy', True),
    ('balanced_accuracy', 'Balanced accuracy', True),
    ('log_loss', 'Log-loss', False),
    ('ece', 'Expected calibration error', False),
    ('accuracy_at_80', 'Accuracy on the 80 % most confident samples', True),
]


def load(path):
    """dataset -> method -> metric -> list of fold values; and errors."""
    table = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    errors = defaultdict(int)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if 'error' in r:
                errors[(r['dataset'], r['method'])] += 1
                continue
            for key, value in r.items():
                if isinstance(value, float):
                    table[r['dataset']][r['method']][key].append(value)
    return table, errors


def score_table(table, methods, metric):
    """Mean over folds; datasets missing any method are left out."""
    datasets = [d for d in sorted(table) if all(table[d][m][metric] for m in methods)]
    scores = np.array([[np.mean(table[d][m][metric]) for m in methods] for d in datasets])
    return datasets, scores


def section(table, methods, metric, title, higher, reference):
    datasets, scores = score_table(table, methods, metric)
    if len(datasets) < 2:
        return f'## {title}\n\nToo few complete datasets.\n'
    friedman = friedman_test(scores, higher)
    cd = nemenyi_critical_difference(len(methods), len(datasets))
    order = np.argsort(friedman.mean_ranks)
    ref = methods.index(reference)
    pairwise = {r.method: r for r in compare_to_reference(scores, ref, higher)}
    lines = [f'## {title}', '',
             f'{len(datasets)} datasets. Friedman chi-squared = {friedman.statistic:.1f}, '
             f'p = {friedman.p_value:.2g}; Iman-Davenport F = {friedman.iman_davenport:.2f}, '
             f'p = {friedman.iman_davenport_p_value:.2g}. Nemenyi critical difference of '
             f'mean ranks (alpha = 0.05): {cd:.2f}.', '',
             f'| Method | Mean | Mean rank | {reference}: wins / ties / losses | Wilcoxon p (Holm) |',
             '|---|---|---|---|---|']
    for j in order:
        m = methods[j]
        if j == ref:
            versus = '—'
            p = '—'
        else:
            r = pairwise[j]
            versus = f'{r.wins} / {r.ties} / {r.losses}'
            p = f'{r.holm_p_value:.3g}'
        lines.append(f'| {m} | {scores[:, j].mean():.4f} | {friedman.mean_ranks[j]:.2f} | {versus} | {p} |')
    return '\n'.join(lines) + '\n'


def per_dataset(table, methods):
    datasets, scores = score_table(table, methods, 'accuracy')
    head = '| Dataset | ' + ' | '.join(methods) + ' |'
    lines = ['## Accuracy per dataset', '', head, '|' + '---|' * (len(methods) + 1)]
    for d, row in zip(datasets, scores):
        shown = [f'{v:.3f}' for v in row]
        best = f'{row.max():.3f}'
        cells = [f'**{s}**' if s == best else s for s in shown]
        lines.append(f'| {d} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('results')
    parser.add_argument('--reference', required=True,
                        help='method compared with every other by Wilcoxon tests')
    parser.add_argument('--methods', default=None)
    args = parser.parse_args(argv)
    table, errors = load(args.results)
    methods = (args.methods.split(',') if args.methods else
               sorted({m for d in table.values() for m in d}))
    if args.reference not in methods:
        sys.exit(f'reference {args.reference} is not among the methods')
    out = ['# Classification benchmark', '',
           f'Source: `{os.path.basename(args.results)}`. Wins, ties and losses count '
           f'datasets on which {args.reference} is better, equal or worse; '
           'p-values are two-sided Wilcoxon signed-rank tests, Holm-adjusted over '
           'the comparisons in each table.', '']
    for key, title, higher in METRICS:
        out.append(section(table, methods, key, title, higher, args.reference))
    out.append(per_dataset(table, methods))
    if errors:
        out.append('## Failed folds\n')
        out += [f'- {d}, {m}: {n}' for (d, m), n in sorted(errors.items())]
    print('\n'.join(out))


if __name__ == '__main__':
    main()
