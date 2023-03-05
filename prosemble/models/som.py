"""
Implementation of Self Organising Maps
"""

import numpy as np
import matplotlib.pyplot as plt
from prosemble.core.distance import (
    euclidean_distance,
    manhattan_distance
)


class SOM:
    def __init__(self, data, c, num_iter, init_lr, sigma):
        self.data = data
        self.number_cluster = c
        self.num_iter = num_iter
        self.learning_rate = init_lr
        self.sigma = sigma
        self.final_map = []
        self.grid = int(np.sqrt(5 * np.sqrt(self.data.shape[0])))
        self.som = np.random.random_sample(
            size=(self.grid, self.grid, self.data.shape[1])
        )
        if self.num_iter is None:
            self.num_iter = 500 * self.grid * self.grid

    def lr_decay(self, num_iter, distance_n):
        decay = 1 - (num_iter / self.num_iter)
        learning_rate = decay * self.learning_rate
        sigma_rate = np.ceil(decay * distance_n)
        return learning_rate, sigma_rate

    # Best Matching Unit search
    def winning_neuron(self, data, t):
        winner = [0, 0]
        shortest_distance = np.sqrt(data.shape[1])  # initialise with max distance
        for row in range(self.grid):
            for col in range(self.grid):
                distance = euclidean_distance(self.som[row][col], data[t])
                if distance < shortest_distance:
                    shortest_distance = distance
                    winner = [row, col]
        return winner

    def fit(self):
        for step in range(self.num_iter):
            learning_rate, neighbourhood_range = self.lr_decay(num_iter=step, distance_n=4)
            rand_sample_index = np.random.randint(0, high=self.data.shape[0])  # random index of traing data
            winner = self.winning_neuron(data=self.data, t=rand_sample_index)
            for row in range(self.grid):
                for col in range(self.grid):
                    if manhattan_distance([row, col], winner) <= neighbourhood_range:
                        self.som[row][col] += learning_rate * (
                                self.data[rand_sample_index] - self.som[row][col])  # update neighbour's weight
        return self.som

    def get_cluster(self, y):

        label_data = y
        map = np.empty(shape=(self.grid, self.grid), dtype=object)

        for row in range(self.grid):
            for col in range(self.grid):
                map[row][col] = []  # empty list to store the label

        for t in range(self.data.shape[0]):
            if (t + 1) % 1000 == 0:
                print("sample data: ", t + 1)
            winner = self.winning_neuron(self.data, t)
            map[winner[0]][winner[1]].append(label_data[t])  # label of winning neuron

        label_map = np.zeros(shape=(self.grid, self.grid), dtype=np.int64)
        for row in range(self.grid):
            for col in range(self.grid):
                label_list = map[row][col]
                if len(label_list) == 0:
                    label = 1
                else:
                    label = max(label_list, key=label_list.count)
                label_map[row][col] = label
        self.final_map.append(label_map)
        title = f'Iteration {self.num_iter}'
        plt.imshow(label_map)
        plt.colorbar()
        plt.title(title)
        plt.pause(10)
        return label_map, self.final_map[0]

    def predict(self, x):
        """

        :param x: array-like: input vector
        :return: predict the label of the input vector
        """
        winner_labels = []
        final_map = self.final_map[0]
        for t in range(np.array(x).shape[0]):
            winner = self.winning_neuron(x, t)
            row = winner[0]
            col = winner[1]
            predicted = final_map[row][col]
            winner_labels.append(predicted)
        return winner_labels
