"""prosemble datasets"""


from dataclasses import dataclass
import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    make_moons,
    make_blobs
)

@dataclass
class DATASET:
    input_data: np.ndarray
    labels: np.ndarray


def breast_cancer_dataset() -> DATASET:
    data, labels = load_breast_cancer(return_X_y=True)
    return DATASET(data, labels)


def moons_dataset(n_samples:int,random_state: int = None) -> DATASET:
    data, labels = make_moons(
        n_samples=150, shuffle=True, noise=None, random_state=random_state
    )
    return DATASET(data, labels)


def bloobs(random_state: int = None) -> DATASET:
    data, labels = make_blobs(
        n_samples=[120, 80],
        centers=[[0.0, 0.0], [2.0, 2.0]],
        cluster_std=[1.2, 0.5],
        random_state=random_state,
        shuffle=False,
    )
    return DATASET(data, labels)


@dataclass
class DATA:
    random: int = 4
    sample_size: int = 1000

    @property
    def S_1(self) -> DATASET:
        return moons_dataset(self.random)

    @property
    def S_2(self) -> DATASET:
        return bloobs(self.random)

    @property
    def breast_cancer(self) -> DATASET:
        return breast_cancer_dataset()