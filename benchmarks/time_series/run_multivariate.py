"""Multivariate time-series benchmark on the UEA archive.

Compares the soft-DTW prototype models (SoftDTWRSLVQ, SoftDTWMRSLVQ)
with 1-nearest-neighbour classifiers under the Euclidean distance and
dimension-independent and dimension-dependent DTW, and with a random
forest, on 17 equal-length datasets of the UEA multivariate archive
(2018 release), using the archive's train/test splits.

    python benchmarks/time_series/run_multivariate.py --out results.jsonl
    python benchmarks/time_series/run_multivariate.py --summary results.jsonl
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

URL = 'https://www.timeseriesclassification.com/aeon-toolkit/{}.zip'
CACHE = os.environ.get('PROSEMBLE_UEA_DATA',
                       os.path.join(os.path.expanduser('~'), '.cache', 'prosemble-uea'))

#: Equal-length UEA datasets with fewer than 61 channels. Wider or longer
#: problems are too expensive for soft-DTW on a CPU; so is Cricket (12
#: classes of length 1197), where one fit of a soft model takes about
#: twelve hours on one CPU thread.
DATASETS = [
    'ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions',
    'Epilepsy', 'ERing', 'FingerMovements', 'HandMovementDirection', 'Handwriting',
    'Libras', 'LSST', 'NATOPS', 'PenDigits', 'RacketSports', 'SelfRegulationSCP1',
    'SelfRegulationSCP2', 'StandWalkJump', 'UWaveGestureLibrary',
]

#: Bytes allowed for the per-block intermediates of the pairwise distances.
BLOCK_BYTES = 3e8

#: Settings of the soft-DTW models, the same on every dataset.
SOFT = dict(n_prototypes_per_class=1, max_iter=150, lr=0.05, random_seed=0)


def _download(name):
    folder = os.path.join(CACHE, name)
    wanted = {f'{name}_TRAIN.ts', f'{name}_TEST.ts'}
    if all(os.path.exists(os.path.join(folder, f)) for f in wanted):
        return folder
    os.makedirs(folder, exist_ok=True)
    request = urllib.request.Request(URL.format(name), headers={'User-Agent': 'prosemble-benchmark'})
    with urllib.request.urlopen(request, timeout=300) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    for member in archive.namelist():
        base = os.path.basename(member)
        if base in wanted:
            with open(os.path.join(folder, base), 'wb') as f:
                f.write(archive.read(member))
    return folder


def _read_ts(path):
    """Series of a .ts file as an array (n, length, channels) and labels."""
    series, labels, in_data = [], [], False
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if not in_data:
                in_data = line.lower().startswith('@data')
                continue
            *channels, label = line.split(':')
            series.append([[float(v) if v != '?' else np.nan for v in c.split(',')] for c in channels])
            labels.append(label.strip())
    return np.array(series).transpose(0, 2, 1), np.array(labels)


def load(name):
    """Train and test series (n, length, channels) as distributed, float32,
    and labels 0..K-1.

    The series are not normalised: the archive's benchmark results for the
    nearest-neighbour classifiers were computed on these values, and
    normalising each series separately removes the position and scale that
    some problems (trajectories, for example) are about.
    """
    folder = _download(name)
    (Xtr, ytr), (Xte, yte) = (_read_ts(os.path.join(folder, f'{name}_{s}.ts')) for s in ('TRAIN', 'TEST'))
    classes = np.unique(np.concatenate([ytr, yte]))
    encode = lambda y: np.searchsorted(classes, y).astype(np.int32)
    as_float = lambda X: np.nan_to_num(X).astype(np.float32)
    return as_float(Xtr), encode(ytr), as_float(Xte), encode(yte)


def _blocks(n, size):
    return [slice(i, min(i + size, n)) for i in range(0, n, size)]


def _pairwise(fn, A, B, per_pair_bytes):
    """fn(A_block, B_block) over blocks small enough for BLOCK_BYTES."""
    pairs = max(1, int(BLOCK_BYTES // per_pair_bytes))
    b = min(len(B), pairs)
    a = max(1, min(len(A), pairs // b))
    return np.concatenate([np.concatenate([np.asarray(fn(A[i], B[j])) for j in _blocks(len(B), b)], axis=1)
                           for i in _blocks(len(A), a)], axis=0)


def _nn_euclidean(Xtr, ytr, Xte):
    a, b = Xtr.reshape(len(Xtr), -1), Xte.reshape(len(Xte), -1)
    d = (b ** 2).sum(1)[:, None] + (a ** 2).sum(1)[None, :] - 2 * b @ a.T
    return ytr[d.argmin(1)]


def _nn_dtw(dependent):
    def predict(Xtr, ytr, Xte):
        import jax.numpy as jnp
        from prosemble.core.soft_dtw import dtw_matrix
        T, C = Xtr.shape[1], Xtr.shape[2]
        if dependent:
            fn = lambda A, B: dtw_matrix(jnp.asarray(A), jnp.asarray(B))
            D = _pairwise(fn, Xte, Xtr, 4.0 * T * T * C * 3)
        else:
            D = sum(_pairwise(lambda A, B: dtw_matrix(jnp.asarray(A), jnp.asarray(B)),
                              Xte[:, :, c:c + 1], Xtr[:, :, c:c + 1], 4.0 * T * T * 3)
                    for c in range(C))
        return ytr[D.argmin(1)]
    return predict


def _forest(Xtr, ytr, Xte):
    from sklearn.ensemble import RandomForestClassifier
    m = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=1)
    return m.fit(Xtr.reshape(len(Xtr), -1), ytr).predict(Xte.reshape(len(Xte), -1))


def _soft_dtw(cls_name):
    def predict(Xtr, ytr, Xte):
        import jax.numpy as jnp
        import prosemble.models as models
        n, T, C = Xtr.shape
        k = len(np.unique(ytr))
        # Each channel is standardised with statistics of the training set,
        # so that one gamma and one learning rate suit every dataset.
        mu, sd = Xtr.mean(axis=(0, 1)), Xtr.std(axis=(0, 1)) + 1e-8
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        # Training: one batch keeps batch * k alignments of T x T (and the
        # T x T x C ground-cost intermediates) with their gradients.
        per_pair = 4.0 * T * T * (C + 3) * 6
        batch = int(min(n, max(1, BLOCK_BYTES // (per_pair * k))))
        model = getattr(models, cls_name)(
            gamma=float(C), n_channels=C, batch_size=batch if batch < n else None, **SOFT)
        model.fit(jnp.asarray(Xtr.reshape(n, -1)), jnp.asarray(ytr))
        if not np.isfinite(np.asarray(model.loss_history_)).all():
            raise FloatingPointError('non-finite training loss')
        chunk = int(max(1, BLOCK_BYTES // (4.0 * T * T * (C + 2) * k)))
        return np.concatenate([np.asarray(model.predict(jnp.asarray(Xte[s].reshape(len(Xte[s]), -1))))
                               for s in _blocks(len(Xte), chunk)])
    return predict


def _methods():
    return {
        '1NN-ED': _nn_euclidean,
        '1NN-DTW_I': _nn_dtw(dependent=False),
        '1NN-DTW_D': _nn_dtw(dependent=True),
        'RandomForest': _forest,
        'SoftDTWRSLVQ': _soft_dtw('SoftDTWRSLVQ'),
        'SoftDTWMRSLVQ': _soft_dtw('SoftDTWMRSLVQ'),
    }


def run(out, datasets, methods):
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            for r in map(json.loads, f):
                if 'accuracy' in r:
                    done.add((r['dataset'], r['method']))
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
            print(f"{dataset:28s} {method:16s} {row.get('accuracy', row.get('error'))}", flush=True)


def summary(path, reference='1NN-DTW_D'):
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
