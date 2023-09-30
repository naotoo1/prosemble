from enum import Enum
import random

import numpy as np
from dataclasses import dataclass
import torch
from sklearn.datasets import (
    load_breast_cancer,
    make_moons,
    make_blobs
)
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10


class Sampling(str, Enum):
    RANDOM = 'random'
    FULL = 'full'


def breast_cancer_dataset():
    data, labels = load_breast_cancer(
        return_X_y=True
    )
    return data, labels


def moons_dataset(random_state: int = None):
    data, labels = make_moons(
        n_samples=150,
        shuffle=True,
        noise=None,
        random_state=random_state
    )
    return data, labels


def bloobs(random_state: int = None):
    data, labels = make_blobs(
        n_samples=[120, 80],
        centers=[[0.0, 0.0], [2.0, 2.0]],
        cluster_std=[1.2, 0.5],
        random_state=random_state,
        shuffle=False,
    )
    return data, labels


def mnist_dataset():
    train_dataset = MNIST(
        "~/datasets",
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )
    test_dataset = MNIST(
        "~/datasets",
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )
    return torch.cat([
            train_dataset.data,
            test_dataset.data
        ]), torch.cat([
            train_dataset.targets,
            test_dataset.targets
        ])


def cifar_10(sample: Sampling, size: int):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(
             (0.5, 0.5, 0.5),
             (0.5, 0.5, 0.5))]
    )
    train_dataset = CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    full_train_ds = torch.cat(
        [torch.from_numpy(train_dataset.data),
         torch.from_numpy(test_dataset.data)]
    )
    full_train_labels = torch.cat(
        [torch.from_numpy(np.array(train_dataset.targets)),
         torch.from_numpy(np.array(test_dataset.targets))]
    )
    classwise_labels = get_classwise_labels(full_train_labels)
    Data,labels = get_random_inputs(
        full_train_ds,
        full_train_labels,
        classwise_labels,
        sample_size=size
    )

    if sample == 'full':
        return full_train_ds, full_train_labels
    if sample == 'random':
        return Data,labels
    raise RuntimeError("cifar-10:none of the cases match")


def get_classwise_labels(
        full_labels: torch.Tensor,
        num_class: int = 10
) -> np.ndarray:
    classwise_labels = []
    for class_label in range(num_class):
        for index, label in enumerate(full_labels):
            label = label.detach().cpu().numpy()
            if label == class_label:
                classwise_labels.append(index)
    return np.reshape(classwise_labels, (-1, 6000))


def get_random_inputs(
        full_train_ds: torch.Tensor,
        full_train_labels: torch.Tensor,
        classwise_labels: np.ndarray,
        sample_size: int = 1000
):
    random_labels = []
    for class_ in classwise_labels:
        random.shuffle(class_)
        random_labels.append(class_[:sample_size])
    random_label_indices = np.array(random_labels)
    random_label_indices = random_label_indices.flatten()

    return torch.from_numpy(
        np.array([
            full_train_ds[index] for index in random_label_indices
            ])
        ), torch.from_numpy(
        np.array([
            full_train_labels[index] for index in random_label_indices
            ])
        )

    
class DATA:
    def __init__(
            self,
            sample:Sampling=Sampling.FULL,
            random:int = 4,
            sample_size:int = 1000
            ):
        self.sample = sample
        self.random = random
        self.sample_size = sample_size

    @property
    def S_1(self):
        return moons_dataset(self.random)

    @property
    def S_2(self):
        return bloobs(self.random)

    @property
    def breast_cancer(self):
        return breast_cancer_dataset()

    @property
    def mnist(self):
        return mnist_dataset()

    @property
    def cifar_10(self):
        return cifar_10(self.sample, self.sample_size)

