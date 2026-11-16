"""Time-series benchmark on datasets of the UCR archive.

Protocol
--------
- The datasets in ``DATASETS``, from the UCR time series archive (2018
  release), each with the archive's own train/test split. Every series is
  z-normalised, as is standard for the archive.
- Every method runs at fixed settings, the same on every dataset; nothing
  is tuned.
- One test accuracy per dataset and method is appended to a JSON-lines
  file. Lines already in the file are skipped, so an interrupted run
  resumes.

The 1-nearest-neighbour baselines reproduce the error rates published
with the archive, which checks the data and the DTW implementation.

Usage::

    python run.py --out results.jsonl                        # everything
    python run.py --out results.jsonl --shard 0 --shards 4   # a quarter
    python run.py --summary results.jsonl                    # table
"""

import argparse
import io
import json
import os
import time
import urllib.request
import warnings
import zipfile

import numpy as np

DATASETS = [
    'ArrowHead', 'BeetleFly', 'BirdChicken', 'CBF', 'Coffee',
    'DistalPhalanxOutlineCorrect', 'ECG200', 'ECGFiveDays', 'FaceFour',
    'GunPoint', 'Herring', 'ItalyPowerDemand', 'Lightning2', 'Lightning7',
    'Meat', 'MoteStrain', 'OliveOil', 'Plane', 'SonyAIBORobotSurface1',
    'SyntheticControl', 'ToeSegmentation1', 'Trace', 'TwoLeadECG', 'Wine',
]
URL = 'https://www.timeseriesclassification.com/aeon-toolkit/{}.zip'
CACHE = os.environ.get('PROSEMBLE_UCR_DATA',
                       os.path.join(os.path.expanduser('~'), '.cache', 'prosemble-ucr'))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _download(name):
    """Fetch the train and test files of one dataset into the cache."""
    folder = os.path.join(CACHE, name)
    wanted = [f'{name}_TRAIN.txt', f'{name}_TEST.txt']
    if all(os.path.exists(os.path.join(folder, f)) for f in wanted):
        return folder
    os.makedirs(folder, exist_ok=True)
    with urllib.request.urlopen(URL.format(name), timeout=120) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    members = {os.path.basename(m): m for m in archive.namelist()}
    for f in wanted:                       # only these two files, by exact name
        with open(os.path.join(folder, f), 'wb') as out:
            out.write(archive.read(members[f]))
    return folder


def load(name):
    """Train and test series (z-normalised, float32) and labels 0..C-1."""
    folder = _download(name)

    def read(split):
        a = np.loadtxt(os.path.join(folder, f'{name}_{split}.txt'))
        return a[:, 1:], a[:, 0]

    def znorm(X):
        X = np.nan_to_num(X)
        return ((X - X.mean(1, keepdims=True))
                / (X.std(1, keepdims=True) + 1e-8)).astype(np.float32)

    (Xtr, ytr), (Xte, yte) = read('TRAIN'), read('TEST')
    classes = np.unique(np.concatenate([ytr, yte]))
    encode = lambda y: np.searchsorted(classes, y).astype(np.int32)
    return znorm(Xtr), encode(ytr), znorm(Xte), encode(yte)


# ---------------------------------------------------------------------------
# Methods: each maps (X_train, y_train, X_test) to predicted test labels.
# ---------------------------------------------------------------------------

def _chunks(n, size=200):
    return [slice(i, i + size) for i in range(0, n, size)]


def _nn_euclidean(Xtr, ytr, Xte):
    nearest = [(((Xte[s, None] - Xtr[None]) ** 2).sum(-1)).argmin(1) for s in _chunks(len(Xte))]
    return ytr[np.concatenate(nearest)]


def _nn_dtw(band):
    def predict(Xtr, ytr, Xte):
        import jax.numpy as jnp
        from prosemble.core.soft_dtw import dtw_matrix
        nearest = [np.asarray(dtw_matrix(jnp.asarray(Xte[s]), jnp.asarray(Xtr), band)).argmin(1)
                   for s in _chunks(len(Xte))]
        return ytr[np.concatenate(nearest)]
    return predict


def _sklearn(make):
    return lambda Xtr, ytr, Xte: make().fit(Xtr, ytr).predict(Xte)


def _prosemble(cls, **settings):
    def predict(Xtr, ytr, Xte):
        import jax.numpy as jnp
        import prosemble.models as models
        extra = {}
        if cls == 'SoftDTWRSLVQ' and Xtr.shape[0] * Xtr.shape[1] > 20000:
            extra['batch_size'] = 64               # bounds the memory of training
        model = getattr(models, cls)(random_seed=0, **settings, **extra)
        model.fit(jnp.asarray(Xtr), jnp.asarray(ytr))
        return np.concatenate([np.asarray(model.predict(jnp.asarray(Xte[s])))
                               for s in _chunks(len(Xte))])
    return predict


def _methods():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    soft_dtw = dict(max_iter=150, lr=0.05)
    return {
        '1NN-ED': _nn_euclidean,
        '1NN-DTW': _nn_dtw(None),
        '1NN-DTW (band 0.1)': _nn_dtw(0.1),
        'SVM': _sklearn(SVC),
        'RandomForest': _sklearn(lambda: RandomForestClassifier(500, random_state=0)),
        'WRSLVQ': _prosemble('WRSLVQ', n_prototypes_per_class=2, max_iter=200, lr=0.01),
        'SoftDTWRSLVQ': _prosemble('SoftDTWRSLVQ', gamma=0.1, n_prototypes_per_class=1, **soft_dtw),
        'SoftDTWRSLVQ (gamma 1)': _prosemble('SoftDTWRSLVQ', gamma=1.0, n_prototypes_per_class=1, **soft_dtw),
        'SoftDTWRSLVQ (2 per class)': _prosemble('SoftDTWRSLVQ', gamma=0.1, n_prototypes_per_class=2, **soft_dtw),
    }


# ---------------------------------------------------------------------------
# Running and summarising
# ---------------------------------------------------------------------------

def run(out, datasets, methods):
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            done = {(r['dataset'], r['method']) for r in map(json.loads, f)}
    table = _methods()
    for dataset in datasets:
        data = None
        for method in methods:
            if (dataset, method) in done:
                continue
            if data is None:
                data = load(dataset)
            Xtr, ytr, Xte, yte = data
            start = time.time()
            try:
                row = {'accuracy': float(np.mean(table[method](Xtr, ytr, Xte) == yte))}
            except Exception as error:                     # recorded, not fatal
                row = {'error': f'{type(error).__name__}: {error}'[:300]}
            row.update(dataset=dataset, method=method, fit_time=time.time() - start)
            with open(out, 'a') as f:
                f.write(json.dumps(row) + '\n')
            print(f"{dataset:28s} {method:28s} "
                  + (f"{row['accuracy']:.3f}" if 'accuracy' in row else row['error'][:80]),
                  flush=True)


def summary(path, reference='1NN-DTW'):
    """Markdown table of accuracies, means, and wins/ties/losses against ``reference``."""
    acc = {}
    with open(path) as f:
        for r in map(json.loads, f):
            if 'accuracy' in r:
                acc[r['dataset'], r['method']] = r['accuracy']
    methods = [m for m in _methods() if any(k[1] == m for k in acc)]
    datasets = [d for d in DATASETS if all((d, m) in acc for m in methods)]
    A = np.array([[acc[d, m] for m in methods] for d in datasets])
    lines = ['| Dataset | ' + ' | '.join(methods) + ' |', '|---' * (len(methods) + 1) + '|']
    for d, row in zip(datasets, A):
        best = f'{row.max():.3f}'
        lines.append(f'| {d} | ' + ' | '.join(f'**{v:.3f}**' if f'{v:.3f}' == best else f'{v:.3f}'
                                               for v in row) + ' |')
    lines.append('| Mean | ' + ' | '.join(f'{v:.3f}' for v in A.mean(0)) + ' |')
    if reference in methods:
        ref = A[:, methods.index(reference)]
        diff = np.round(A - ref[:, None], 6)
        lines.append(f'| Better / equal / worse than {reference} | ' + ' | '.join(
            f'{(c > 0).sum()} / {(c == 0).sum()} / {(c < 0).sum()}' for c in diff.T) + ' |')
    print(f'{len(datasets)} datasets\n')
    print('\n'.join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out')
    parser.add_argument('--summary', metavar='RESULTS')
    parser.add_argument('--datasets', default=','.join(DATASETS))
    parser.add_argument('--methods', default=None)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args(argv)
    warnings.filterwarnings('ignore')
    if args.summary:
        return summary(args.summary)
    if not args.out:
        parser.error('--out or --summary is required')
    datasets = [d for i, d in enumerate(args.datasets.split(',')) if i % args.shards == args.shard]
    methods = args.methods.split(',') if args.methods else list(_methods())
    run(args.out, datasets, methods)


if __name__ == '__main__':
    main()
