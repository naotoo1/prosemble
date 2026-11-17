"""One-class time-series benchmark on the UCR and UEA archives.

Each class of a dataset in turn is the normal class: a model is fitted to
the training series of that class only, and scores every test series. The
measure is the area under the ROC curve (AUROC) for telling the normal
test series from the others, averaged over the classes of a dataset.

    python benchmarks/time_series/run_one_class.py --archive ucr --out ucr.jsonl
    python benchmarks/time_series/run_one_class.py --archive uea --out uea.jsonl
    python benchmarks/time_series/run_one_class.py --archive ucr --summary ucr.jsonl

UCR series are used as distributed (z-normalised). UEA channels are
standardised with the statistics of the normal training series.
``--classes`` runs some normal classes only, so that a dataset can be
split into jobs; rows are written per class and averaged by ``--summary``.
"""
import argparse
import importlib.util
import json
import os
import time
import warnings

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))


def _module(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_here, file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: Bytes allowed for the intermediates of one block of alignments.
BLOCK_BYTES = 3e8


def _archive(name):
    if name == 'ucr':
        ucr = _module('ucr_benchmark', 'run.py')
        load = lambda d: tuple(a[..., None] if a.ndim == 2 else a for a in ucr.load(d))
        return ucr.DATASETS, load
    uea = _module('uea_benchmark', 'run_multivariate.py')
    return uea.DATASETS, uea.load


def _blocked(fn, X, partners, T, C):
    rows = int(max(1, BLOCK_BYTES // (4.0 * T * T * (C + 3) * max(partners, 1))))
    return np.concatenate([np.asarray(fn(X[i:i + rows])) for i in range(0, len(X), rows)])


def _nn_euclidean(normal, X):
    a, b = normal.reshape(len(normal), -1), X.reshape(len(X), -1)
    d = (b ** 2).sum(1)[:, None] + (a ** 2).sum(1)[None, :] - 2 * b @ a.T
    return -d.min(1)


def _nn_dtw(normal, X):
    """Distance to the nearest normal training series under (dependent) DTW."""
    import jax.numpy as jnp
    from prosemble.core.soft_dtw import dtw_matrix
    n, T, C = normal.shape
    reference = jnp.asarray(normal)
    return -_blocked(lambda b: dtw_matrix(jnp.asarray(b), reference), X, n, T, C).min(1)


def _ocsvm(normal, X):
    from sklearn.svm import OneClassSVM
    return OneClassSVM(gamma='scale', nu=0.1).fit(normal.reshape(len(normal), -1)).score_samples(
        X.reshape(len(X), -1))


def _iforest(normal, X):
    from sklearn.ensemble import IsolationForest
    return IsolationForest(n_estimators=500, random_state=0).fit(
        normal.reshape(len(normal), -1)).score_samples(X.reshape(len(X), -1))


def _alignment_density(normal, X):
    from prosemble.models import OCAlignmentDensity
    model = OCAlignmentDensity(n_channels=normal.shape[2]).fit(normal)
    return np.asarray(model.score_samples(X))


def _methods():
    return {
        '1NN-ED': _nn_euclidean,
        '1NN-DTW': _nn_dtw,
        'OC-SVM': _ocsvm,
        'IsolationForest': _iforest,
        'OCAlignmentDensity': _alignment_density,
    }


def run(archive, out, datasets, methods, classes=None):
    from sklearn.metrics import roc_auc_score
    _, load = _archive(archive)
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            for r in map(json.loads, f):
                if 'auroc' in r:
                    done.add((r['dataset'], r['method'], r['normal_class']))
    table = _methods()
    for dataset in datasets:
        Xtr, ytr, Xte, yte = load(dataset)
        for c in np.unique(ytr):
            if classes is not None and int(c) not in classes:
                continue
            normal, test = Xtr[ytr == c], Xte
            if archive == 'uea':
                mu, sd = normal.mean(axis=(0, 1)), normal.std(axis=(0, 1))
                sd = np.where(sd > 0, sd, 1.0)
                normal, test = (normal - mu) / sd, (test - mu) / sd
            normal, test = normal.astype(np.float32), test.astype(np.float32)
            for method in methods:
                if (dataset, method, int(c)) in done:
                    continue
                start = time.time()
                try:
                    scores = table[method](normal, test)
                    row = {'auroc': float(roc_auc_score(yte == c, scores))}
                except Exception as error:                 # recorded, not fatal
                    row = {'error': f'{type(error).__name__}: {error}'[:300]}
                row.update(dataset=dataset, method=method, normal_class=int(c),
                           fit_time=time.time() - start)
                with open(out, 'a') as f:
                    f.write(json.dumps(row) + '\n')
                print(f"{dataset:28s} {int(c):3d} {method:20s} "
                      f"{row.get('auroc', row.get('error'))}", flush=True)


def summary(archive, paths, reference='1NN-DTW'):
    """Markdown table of the mean AUROC per dataset, and wins/ties/losses against ``reference``."""
    names, load = _archive(archive)
    per_class = {}
    for path in paths:
        with open(path) as f:
            for r in map(json.loads, f):
                if 'auroc' in r and r['dataset'] in names:
                    per_class.setdefault((r['dataset'], r['method']), {})[r['normal_class']] = r['auroc']
    n_classes = {d: len(np.unique(load(d)[1])) for d in {k[0] for k in per_class}}
    auc = {k: np.mean(list(v.values())) for k, v in per_class.items() if len(v) == n_classes[k[0]]}
    methods = [m for m in _methods() if any(k[1] == m for k in auc)]
    datasets = [d for d in names if methods and all((d, m) in auc for m in methods)]
    if not datasets:
        print('No dataset has results for every class and method yet.')
        return [], methods, np.zeros((0, len(methods)))
    A = np.array([[auc[d, m] for m in methods] for d in datasets])
    lines = ['| Dataset | ' + ' | '.join(methods) + ' |', '|---' * (len(methods) + 1) + '|']
    for d, row in zip(datasets, A):
        best = f'{row.max():.3f}'
        lines.append(f'| {d} | ' + ' | '.join(f'**{v:.3f}**' if f'{v:.3f}' == best else f'{v:.3f}'
                                               for v in row) + ' |')
    lines.append('| Mean | ' + ' | '.join(f'{v:.3f}' for v in A.mean(0)) + ' |')
    if reference in methods:
        ref = A[:, methods.index(reference)]
        diff = np.round(A - ref[:, None], 3)
        lines.append(f'| Better / equal / worse than {reference} | ' + ' | '.join(
            f'{(c > 0).sum()} / {(c == 0).sum()} / {(c < 0).sum()}' for c in diff.T) + ' |')
    print(f'{len(datasets)} datasets\n')
    print('\n'.join(lines))
    return datasets, methods, A


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--archive', choices=('ucr', 'uea'), default='ucr')
    parser.add_argument('--out')
    parser.add_argument('--summary', metavar='RESULTS', nargs='+')
    parser.add_argument('--datasets', default=None)
    parser.add_argument('--methods', default=None)
    parser.add_argument('--classes', default=None,
                        help='comma-separated normal classes (default: all)')
    args = parser.parse_args(argv)
    warnings.filterwarnings('ignore')
    if args.summary:
        summary(args.archive, args.summary)
        return
    if not args.out:
        parser.error('--out or --summary is required')
    names, _ = _archive(args.archive)
    datasets = args.datasets.split(',') if args.datasets else names
    methods = args.methods.split(',') if args.methods else list(_methods())
    classes = {int(c) for c in args.classes.split(',')} if args.classes else None
    run(args.archive, out=args.out, datasets=datasets, methods=methods, classes=classes)


if __name__ == '__main__':
    main()
