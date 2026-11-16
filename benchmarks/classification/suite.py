"""The datasets of the classification benchmark.

36 OpenML datasets with numeric features only, no missing values, and at
least 10 samples in every class, so that each class appears in every fold
of a stratified 5-fold split. They are fixed by OpenML id, so a run can be
repeated exactly.
"""

import os
import warnings

import numpy as np

#: name -> OpenML dataset id
DATASETS = {
    'iris': 61, 'wine': 187, 'wdbc': 1510, 'diabetes': 37, 'ionosphere': 59,
    'sonar': 40, 'vehicle': 54, 'segment': 36, 'heart-statlog': 53,
    'balance-scale': 11, 'banknote-authentication': 1462,
    'blood-transfusion': 1464, 'climate-model-crashes': 1467,
    'qsar-biodeg': 1494, 'steel-plates-fault': 1504,
    'wall-robot-navigation': 1497, 'phoneme': 1489, 'ozone-level-8hr': 1487,
    'pc1': 1068, 'kc1': 1067, 'waveform-5000': 60, 'satimage': 182,
    'optdigits': 28, 'texture': 40499, 'wilt': 1570, 'mfeat-fourier': 14,
    'mfeat-karhunen': 16, 'mfeat-zernike': 22, 'spambase': 44,
    # OpenML 1500 is registered as "seismic-bumps" but holds the UCI seeds data.
    'led-display-7digit': 40496, 'seeds': 1500, 'parkinsons': 1488,
    'cardiotocography': 1466, 'hill-valley': 1479, 'breast-tissue': 1465,
    'wine-quality-red': 40691,
}

CACHE = os.environ.get('PROSEMBLE_BENCHMARK_DATA',
                       os.path.join(os.path.expanduser('~'), '.cache', 'prosemble-benchmark'))


def load(name):
    """Features as float32 and labels as 0..n_classes-1."""
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import LabelEncoder
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data = fetch_openml(data_id=DATASETS[name], data_home=CACHE,
                            as_frame=True, parser='liac-arff')
    X = data.data.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
    y = LabelEncoder().fit_transform(np.asarray(data.target).astype(str))
    if np.isnan(X).any():
        raise ValueError(f"{name} has missing values")
    return X, y.astype(np.int32)
