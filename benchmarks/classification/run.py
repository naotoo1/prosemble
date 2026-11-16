"""Run the classification benchmark.

Protocol
--------
- Repeated stratified k-fold cross-validation (default: 2 repeats of 5
  folds) on every dataset of :mod:`suite`.
- Features are standardised with statistics of the training part of each
  fold only.
- Baselines are tuned by an inner 3-fold grid search on the training part;
  prosemble models run at fixed settings, the same on every dataset.
- The soft prosemble models are calibrated with ``calibrate(cv=3)`` on the
  training part; this changes their probabilities, never their labels.
- Per fold: accuracy, balanced accuracy, log-loss, expected calibration
  error, accuracy on the 80 % most confident test samples, and fit time.

Results are appended to a JSON-lines file, one line per dataset, fold and
method, and existing lines are skipped, so an interrupted run resumes.

Usage::

    python run.py --out results.jsonl                 # everything
    python run.py --out results.jsonl --shard 0 --shards 4   # a quarter of the datasets
    python run.py --datasets iris,wine --methods GLVQ,SVM
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from suite import DATASETS, load  # noqa: E402


# ---------------------------------------------------------------------------
# Methods: each returns a fitted object with predict and predict_proba.
# ---------------------------------------------------------------------------

def _tuned(estimator, grid):
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    def fit(X, y):
        # Shuffled: some datasets are stored sorted, and contiguous inner
        # folds then differ in distribution and pick poor settings.
        inner = StratifiedKFold(3, shuffle=True, random_state=0)
        search = GridSearchCV(estimator, grid, cv=inner, scoring='accuracy', n_jobs=1)
        return search.fit(X, y).best_estimator_
    return fit


def _svm(X, y):
    from sklearn.svm import SVC
    d = X.shape[1]
    best = _tuned(SVC(), {'C': [0.1, 1, 10, 100], 'gamma': [0.1 / d, 1.0 / d, 10.0 / d]})(X, y)
    # Refit with Platt scaling for probabilities.
    return SVC(C=best.C, gamma=best.gamma, probability=True, random_state=0).fit(X, y)


def _sklearn(name):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    return {
        'LogisticRegression': _tuned(LogisticRegression(max_iter=5000), {'C': [0.01, 0.1, 1, 10, 100]}),
        'kNN': _tuned(KNeighborsClassifier(), {'n_neighbors': [1, 3, 5, 9, 15, 25]}),
        'SVM': _svm,
        'RandomForest': lambda X, y: RandomForestClassifier(500, random_state=0, n_jobs=1).fit(X, y),
        'GradientBoosting': lambda X, y: HistGradientBoostingClassifier(random_state=0).fit(X, y),
    }[name]


#: prosemble model -> (class name, prototypes per class, calibrate)
PROSEMBLE = {
    'GLVQ': ('GLVQ', 2, False),
    'GMLVQ': ('GMLVQ', 2, False),
    'LGMLVQ': ('LGMLVQ', 2, False),
    'WRSLVQ': ('WRSLVQ', 2, True),
    'DKRSLVQ': ('DKRSLVQ', 2, True),
    'DKLMRSLVQ': ('DKLMRSLVQ', 1, True),
    'WLMRSLVQ': ('WLMRSLVQ', 2, True),
}
BASELINES = ['LogisticRegression', 'kNN', 'SVM', 'RandomForest', 'GradientBoosting']
METHODS = list(PROSEMBLE) + BASELINES


class _Prosemble:
    def __init__(self, name):
        self.cls, self.k, self.calibrate = PROSEMBLE[name]

    def __call__(self, X, y):
        import prosemble.models as models
        model = getattr(models, self.cls)(
            n_prototypes_per_class=self.k, max_iter=200, lr=0.01, random_seed=0)
        model.fit(X, y)
        if self.calibrate:
            model.calibrate(X, y, cv=3)
        return model


def fit_method(name, X, y):
    return _Prosemble(name)(X, y) if name in PROSEMBLE else _sklearn(name)(X, y)


# ---------------------------------------------------------------------------
# One fold
# ---------------------------------------------------------------------------

def evaluate(name, Xtr, ytr, Xte, yte, n_classes):
    from sklearn.metrics import balanced_accuracy_score
    from prosemble.core.evaluation import expected_calibration_error, selective_accuracy
    start = time.time()
    model = fit_method(name, Xtr, ytr)
    fit_time = time.time() - start
    pred = np.asarray(model.predict(Xte))
    proba = np.asarray(model.predict_proba(Xte), dtype=np.float64)
    if proba.shape[1] != n_classes:                       # a class absent from training
        full = np.zeros((len(yte), n_classes))
        full[:, np.unique(ytr)] = proba
        proba = full
    proba = np.clip(proba, 1e-12, 1.0)
    proba /= proba.sum(axis=1, keepdims=True)
    return {
        'accuracy': float(np.mean(pred == yte)),
        'balanced_accuracy': float(balanced_accuracy_score(yte, pred)),
        'log_loss': float(-np.mean(np.log(proba[np.arange(len(yte)), yte]))),
        'ece': expected_calibration_error(proba, yte),
        'accuracy_at_80': selective_accuracy(proba, yte, 0.8),
        'fit_time': fit_time,
    }


def folds(X, y, repeats, n_folds):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    for repeat in range(repeats):
        split = StratifiedKFold(n_folds, shuffle=True, random_state=repeat)
        for fold, (train, test) in enumerate(split.split(X, y)):
            scaler = StandardScaler().fit(X[train])
            yield (repeat, fold, scaler.transform(X[train]).astype(np.float32), y[train],
                   scaler.transform(X[test]).astype(np.float32), y[test])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out', required=True)
    parser.add_argument('--datasets', default=','.join(DATASETS))
    parser.add_argument('--methods', default=','.join(METHODS))
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args(argv)
    warnings.filterwarnings('ignore')

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                r = json.loads(line)
                done.add((r['dataset'], r['repeat'], r['fold'], r['method']))

    datasets = [d for i, d in enumerate(args.datasets.split(',')) if i % args.shards == args.shard]
    for dataset in datasets:
        X, y = load(dataset)
        n_classes = int(y.max()) + 1
        for repeat, fold, Xtr, ytr, Xte, yte in folds(X, y, args.repeats, args.folds):
            for method in args.methods.split(','):
                if (dataset, repeat, fold, method) in done:
                    continue
                try:
                    row = evaluate(method, Xtr, ytr, Xte, yte, n_classes)
                except Exception as error:                     # recorded, not fatal
                    row = {'error': f'{type(error).__name__}: {error}'[:300]}
                row.update(dataset=dataset, repeat=repeat, fold=fold, method=method)
                with open(args.out, 'a') as f:
                    f.write(json.dumps(row) + '\n')
                print(f"{dataset:24s} r{repeat} f{fold} {method:18s} "
                      + (f"acc {row['accuracy']:.3f}" if 'accuracy' in row else row['error'][:80]),
                      flush=True)


if __name__ == '__main__':
    main()
