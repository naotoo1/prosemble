import numpy as np
from sklearn.metrics import accuracy_score
from operator import itemgetter
from label_security1 import LabelSecurity


class Hybrid:
    """
    A module for  pro-trainning uitilization of LVQ , GLVQ and Matrix GLVQ learned prototypes
    :param
    new_prototypes: array-like, shape=[num_prototypes, num_features]
           Prototypes from the trained model using train-set, where num_prototypes refers to the number of prototypes

    proto_classes: array-like, shape=[num_prototypes, num_features]
           Prototypes from the trained model using train-set, where num_prototypes refers to the number of prototypes

    mm: array-like, shape=[num_classes]
       Class labels of prototypes

    omega_matrix: array-like, shape=[dim, num_features]
         Omega_matrix from the trained model, where dim is an int refers to the maximum rank

    matrix: "yes" for matrix GLVQ and "no" otherwise , string

    """

    def __init__(self, new_prototypes, proto_classes, mm, omega_matrix, matrix):
        self.new_prototypes = new_prototypes
        self.proto_classes = proto_classes
        self.mm = mm
        self.omega_matrix = omega_matrix
        self.matrix = matrix

    def distance(self, x, v):
        # computes the distances between each data point and the prototypes
        empty = []
        for i in range(len(x)):
            for j in range(len(v)):
                if self.matrix == "n":
                    d = (x[i] - v[j])
                    d = d.T.dot(d)
                    empty.append([i, j, d])
                if self.matrix == "y":
                    d = (x[i] - v[j])
                    d = d.T.dot(self.omega_matrix.T).dot(self.omega_matrix).dot(d)
                    empty.append([i, j, d])
        return empty

    def reshape(self, x):
        x = np.array(x)
        x = x.reshape(len(x), len(self.proto_classes))
        return x

    def min_val(self, l, i):  # returns a list of the index of data points with minimum distance from prototypes
        empty = []
        d = (min(enumerate(map(itemgetter(i), l)), key=itemgetter(1)))
        empty.append(d[0])
        return empty  # d

    def max_val_1(self, l, i):  # returns a list of the index of data points with maximum distance from the prototypes
        empty = []
        d = (max(enumerate(map(itemgetter(i), l)), key=itemgetter(1)))
        empty.append(d[0])
        return empty  # d

    def predict(self, x_test):
        """
        Predicts class membership index for each input data
        :param x_test:
        Test set from the data set or the data set to be classified
        :return:
        Predicted class labels.
        """
        empty = []
        n = 0
        mm = self.mm
        for i in range(len(x_test)):
            d = self.distance(x_test, self.new_prototypes)[n:mm]  # try_now[n:mm]
            n = mm
            mm += self.mm
            dd = self.min_val(d, -1)  # index of the minimum value
            empty.append(dd)
        dd = np.array(empty)
        dd = dd.flatten()
        return dd

    def get_security(self, x):
        predict = self.predict(x)
        security = LabelSecurity(x_test=x, class_labels=self.proto_classes, predict_results=predict,
                                 model_prototypes=self.new_prototypes, x_dat=x)
        sec = security.label_sec_f(predict)
        return sec

    def accuracy(self, x, y):
        """
        Computes the accuracy of the predicted classifications
        :param x: True labels
        y-test
        :param y: classification results using the max votes scheme
        y-pred
        :return: int
        accuracy score
        """
        d = accuracy_score(x, y)
        return d

    def spredict_final(self, x, y):
        """
        computes the  soft votes  and probabilities of each predicted label for all the models
        :param x: set
        :param y: lists of list containing securities from the models
        :return:
        List containing the label,votes and probabilities of the label
        """
        empty = []
        for j in range(len(x)):
            for label in self.proto_classes:
                c = 0
                ts_ = 0
                for k in y:
                    if k[j][0] == label:
                        ts_ += k[j][1]
                        c += 1
                try:
                    r = ts_ / c
                except ZeroDivisionError:
                    r = 0
                empty.append([label, c, r])
        return empty

    def predict_final(self, x, y):
        """
        computes the hard votes  and probabilities of each predicted label for all the models
        :param x: set
        :param y: lists of list containing predictions from the models

        :return:
        List containing the label, votes and probabilities of the label
        """
        empty = []
        z = len(y)
        for j in range(len(x)):
            for label in self.proto_classes:
                c = 0
                for i in y:
                    if i[j] == label:
                        c += 1
                r = c / z
                empty.append([label, c, r])
        return empty

    def predict_sprob(self, x, y):
        """

        :param x: Test-set
        :param y: list containing all the securities from the model
        :return: List of the probabilities of the predicted labels
        """

        empty = []
        p = len(self.proto_classes)
        t = self.spredict_final(x, y)
        for i in t:
            z = i[2]
            empty.append(z)
        new = np.array(empty)
        new = new.reshape(len(x), p)
        return new

    def predict_prob(self, x, y):
        """

        :param x: Test-set
        :param y: list containing all the predictions from the model
        :return: List of the probabilities of the predicted labels
        """

        empty = []
        p = len(self.proto_classes)
        t = self.predict_final(x, y)
        for i in t:
            z = i[2]
            empty.append(z)
        new = np.array(empty)
        new = new.reshape(len(x), p)
        return new

    def pred_sprob(self, x, y):  # pred ensemble
        """
        :param x: Test set
        :param y: list containing all the securities from the model
        :return: The classification labels with highest soft votes
        """
        empty = []
        t = self.predict_sprob(x, y)
        for i in t:
            d = max(enumerate(i), key=itemgetter(1))[0]
            empty.append(d)
        dd = np.array(empty)
        dd = dd.flatten()
        return dd

    def pred_prob(self, x, y):  # pred ensemble
        """
        :param x: Test set
        :param y: list containing all the predictions from the model
        :return: The classification labels with maximum votes
        """
        empty = []
        t = self.predict_prob(x, y)
        for i in t:
            d = max(enumerate(i), key=itemgetter(1))[0]
            empty.append(d)
        dd = np.array(empty)
        dd = dd.flatten()
        return dd

    def sprob(self, x, y):
        """

        :param x: Test set
        :param y: list containing all the securities from the model
        :return: returns the  probabilities of the classification based on soft voting
        """
        empty = []
        t = self.predict_sprob(x, y)
        for i in t:
            d1 = max(enumerate(i), key=itemgetter(1))[1]
            d1 = np.round(d1, 4)
            empty.append(d1)
        dd = np.array(empty)
        dd = dd.flatten()
        return dd

    def prob(self, x, y):
        """

        :param x: Test set
        :param y: list containing all the predictions from the model
        :return: returns the  probabilities of the classification based on hard voting
        """
        empty = []
        t = self.predict_prob(x, y)
        for i in t:
            d1 = max(enumerate(i), key=itemgetter(1))[1]
            d1 = np.round(d1, 4)
            empty.append(d1)
        dd = np.array(empty)
        dd = dd.flatten()
        return dd
