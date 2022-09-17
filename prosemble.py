import numpy as np
from sklearn.metrics import accuracy_score
from operator import itemgetter
from scipy.spatial import distance
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde


def accuracy_rate1(x, y):
    """

    :param x: True labels
    :param y: predicted labels
    :return:  accuracy score
    """
    return accuracy_score(x, y)


def rejection_rate(x, y):
    """

    :param x: predicted labels below the thresh-hold confidence
    :param y: predicted labels regardless of thresh-hold confidence
    :return: rejection rate
    """
    z = len(x) / len(y)
    return z


def get_array(x):
    return [np.array(i) for i in x]


def get_pdf(x, y, z):
    return 1 / (x * np.sqrt(2 * np.pi)) * np.exp(-(y - z) ** 2 / (2 * x ** 2))


def get_iqr(x, y):
    return 1.5 * (x - y)


def get_percentile(x, y):
    return np.percentile(x, y)


class LabelSecurity:
    """
    Label Security
    :params

    x_test: array, shape=[num_data,num_features]
            Where num_data is the number of samples and num_features refers to the number of features.
    class_labels: array-like, shape=[num_classes]
       Class labels of prototypes

    predict_results:  array-like, shape=[num_data]
        Predicted labels of the test-set

    model_prototypes: array-like, shape=[num_prototypes, num_features]
           Prototypes from the trained model using train-set, where num_prototypes refers to the number of prototypes

    x_dat : array, shape=[num_data, num_features]
       Input data

    fuzziness_parameter: int, optional(default=2)
    """

    def __init__(self, x_test, class_labels, predict_results, model_prototypes, x_dat, fuzziness_parameter=2):
        self.x_test = x_test
        self.class_labels = class_labels
        self.predict_results = predict_results
        self.model_prototypes = model_prototypes
        self.x_dat = x_dat
        self.fuzziness_parameter = fuzziness_parameter

    def label_sec_f(self, x):
        """
        Computes the labels security of each prediction from the model using the test_set

        :param x
        predicted labels from the model using test-set
        :return:
        labels with their security
        """

        security = float

        # Empty list to populate with certainty of labels
        my_label_sec_list = []

        # loop through the test_set
        for i in range(self.x_test.shape[0]):

            # consider respective class labels
            for label in range(self.class_labels.shape[0]):

                # checks where the predicted label equals the class label
                if self.predict_results[i] == label:

                    # computes the certainty/security per predicted label
                    ed_dis: float = distance.euclidean(self.x_test[i, 0:self.x_dat.shape[1]],
                                                       self.model_prototypes[label, 0:self.x_dat.shape[1]])
                    sum_dis = 0
                    for j in range(self.model_prototypes.shape[0]):
                        sum_dis += np.power(ed_dis / distance.euclidean(self.x_test[i, 0:self.x_dat.shape[1]],
                                                                        self.model_prototypes[j,
                                                                        0:self.x_dat.shape[1]]),
                                            2 / (self.fuzziness_parameter - 1))
                        security = 1 / sum_dis

                    my_label_sec_list.append(np.round(security, 4))  # add the computed label certainty to list above
        my_label_sec_list = np.array(my_label_sec_list)
        my_label_sec_list = my_label_sec_list.reshape(len(my_label_sec_list), 1)  # reshape list to 1-D array
        x = np.array(x)
        x = x.reshape(len(x), 1)  # reshape predicted labels into 1-D array
        labels_with_certainty = np.concatenate((x, my_label_sec_list), axis=1)
        return labels_with_certainty


class LabelSecurityM:
    """
    label security for matrix GLVQ
    :parameters

    x_test: array, shape=[num_data, num_features]
        Where num_data refers to the number of samples and num_features refers to the number of features

    class_labels: array-like, shape=[num_classes]
       Class labels of prototypes

    model_prototypes: array-like, shape=[num_prototypes, num_features]
           Prototypes from the trained model using train-set, where num_prototypes refers to the number of prototypes

    model_omega: array-like, shape=[dim, num_features]
         Omega_matrix from the trained model, where dim is an int refers to the maximum rank

    x:  array, shape=[num_data, num_features]
       Input data

    fuzziness_parameter=int, optional(default=2)
    """

    def __init__(self, x_test, class_labels, model_prototypes, model_omega, x, fuzziness_parameter=2):
        self.x_test = x_test
        self.class_labels = class_labels
        self.model_prototypes = model_prototypes
        self.model_omega = model_omega
        self.x = x
        self.fuzziness_parameter = fuzziness_parameter

    def label_security_m_f(self, x):
        """
        Computes the label security of each prediction from the model using the test_set
        :param x: predicted labels from the model using X_test
        :return: labels with their securities
        """
        security = " "
        # Empty list to populate with the label certainty
        my_label_security_list = []
        # loop through the test set
        for i in range(len(self.x_test)):
            # considers respective class labels of prototypes
            for label in range(len(self.class_labels)):
                # checks if predicted label equals class label of prototypes
                if x[i] == label:

                    # computes the label certainty per predicted label
                    standard_ed = (self.x_test[i, 0:self.x.shape[1]]
                                   - self.model_prototypes[label, 0:self.x.shape[1]])
                    squared_ed = standard_ed.T.dot(self.model_omega.T).dot(self.model_omega).dot(standard_ed)
                    sum_dis = 0
                    for j in range(len(self.model_prototypes)):
                        standard_ed1 = (self.x_test[i, 0:self.x.shape[1]] - self.model_prototypes[j, 0:self.x.shape[1]])
                        sum_dis += np.power(
                            squared_ed / (
                                standard_ed1.T.dot(self.model_omega.T).dot(self.model_omega).dot(standard_ed1)),
                            1 / (self.fuzziness_parameter - 1))
                        security = 1 / sum_dis

                    # adds the computed certainty to the list
                    my_label_security_list.append(np.round(security, 4))
        my_label_security_list = np.array(my_label_security_list)
        my_label_security_list = my_label_security_list.reshape(len(my_label_security_list), 1)  # 1-D array reshape
        x = np.array(x)
        x = x.reshape(len(x), 1)  # reshape the predicted labels into 1-D array
        labels_with_certainty = np.concatenate((x, my_label_security_list), axis=1)
        return labels_with_certainty


class LabelSecurityLM:
    """
            label security for local matrix GLVQ
            :parameters

            x_test: array, shape=[num_data, num_features]
                Where num_data refers to the number of samples and num_features refers to the number of features

            class_labels: array-like, shape=[num_classes]
               Class labels of prototypes

            model_prototypes: array-like, shape=[num_prototypes, num_features]
                   Prototypes from the trained model using train-set, where num_prototypes refers to the number
                   of prototypes

            model_omega: array-like, shape=[dim, num_features]
                 Omega_matrix from the trained model, where dim is an int refers to the maximum rank

            x:  array, shape=[num_data, num_features]
               Input data

            fuzziness_parameter=int, optional(default=2)
            """

    def __init__(self, x_test, class_labels, model_prototypes, model_omega, x, fuzziness_parameter=2):
        self.x_test = x_test
        self.class_labels = class_labels
        self.model_prototypes = model_prototypes
        self.model_omega = model_omega
        self.x = x
        self.fuzziness_parameter = fuzziness_parameter

    def label_security_lm_f(self, x):
        """
            computes the label security of each prediction from the model using the test_set
            and returns only labels their corresponding security.
            :param x: predicted labels from the model using X_test
            :return: labels with  security
            """
        security = " "
        # Empty list to populate with the label security
        my_label_security_list = []
        # loop through the test set
        for i in range(len(self.x_test)):
            # considers respective class labels of prototypes
            for label in range(len(self.class_labels)):
                # checks if predicted label equals class label of prototypes
                if x[i] == label:

                    # computes the label certainty per predicted label
                    standard_ed = (self.x_test[i, 0:self.x.shape[1]]
                                   - self.model_prototypes[label, 0:self.x.shape[1]])
                    squared_ed = standard_ed.T.dot(self.model_omega[label].T).dot(self.model_omega[label]) \
                        .dot(standard_ed)
                    sum_dis = 0
                    for j in range(len(self.model_prototypes)):
                        standard_ed1 = (
                                self.x_test[i, 0:self.x.shape[1]] - self.model_prototypes[j, 0:self.x.shape[1]])
                        sum_dis += np.power(
                            squared_ed / (
                                standard_ed1.T.dot(self.model_omega[j].T).dot(self.model_omega[j]).dot(standard_ed1)),
                            1 / (self.fuzziness_parameter - 1))
                        security = 1 / sum_dis

                    # adds the computed certainty to the list
                    my_label_security_list.append(np.round(security, 4))
        my_label_security_list = np.array(my_label_security_list)
        my_label_security_list = my_label_security_list.reshape(len(my_label_security_list), 1)  # 1-D array reshape
        x = np.array(x)
        x = x.reshape(len(x), 1)  # reshape the predicted labels into 1-D array
        labels_with_certainty = np.concatenate((x, my_label_security_list), axis=1)
        return labels_with_certainty


class Hybrid:
    """
    A module prototype based ensemble learning based on LVQs
    :param
    model_prototypes: array-like, shape=[num_prototypes, ,num_features]
           Prototypes from the trained model using train-set, where num_prototypes refers to the number of prototypes

    proto_classes: array-like, shape=[num_prototypes, num_features]
           Prototypes from the trained model using train-set, where num_prototypes refers to the number of prototypes

    mm: array-like, shape=[num_classes]
       Class labels of prototypes

    omega_matrix: array-like, shape=[dim, num_features]
         Omega_matrix from the trained model, where dim is an int refers to the maximum rank

    matrix: "yes" for matrix GLVQ and "no" otherwise , string

    """

    def __init__(self, model_prototypes, proto_classes, mm, omega_matrix, matrix):
        self.model_prototypes = model_prototypes
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
            d = self.distance(x_test, self.model_prototypes)[n:mm]  # try_now[n:mm]
            n = mm
            mm += self.mm
            dd = self.min_val(d, -1)  # index of the minimum value
            empty.append(dd)
        dd = np.array(empty)
        dd = dd.flatten()
        return dd

    def get_security(self, x, y):
        predict = self.predict(x)
        security = LabelSecurity(x_test=x, class_labels=self.proto_classes, predict_results=predict,
                                 model_prototypes=self.model_prototypes, x_dat=x, fuzziness_parameter=y)
        sec = security.label_sec_f(predict)
        return sec

    def get_security_m(self, x, y):
        predict = self.predict(x)
        security = LabelSecurityM(x_test=x, class_labels=self.proto_classes, model_prototypes=self.model_prototypes,
                                  model_omega=self.omega_matrix, x=x, fuzziness_parameter=y)
        sec = security.label_security_m_f(predict)
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


class ThreshT:
    """
    Class Related Thresholds for multiple reject classifications
    :params
    Y_test : array: labels of the test_set
    class_labels: array-like, shape=[num_classes]
       Class labels of prototypes
    predict_results:  array-like, shape=[num_data]
        Predicted labels of the test-set
    reject_rate1: float: maximum rejection rate to be considered in the optimal search.

    """

    def __init__(self, predict_results, reject_rate1):
        self.predict_results = predict_results
        self.rejection_rate1 = reject_rate1

    def threshh(self, d1, protocert_1, j):
        """

        :param d1: The computed classification labels securities
        :param protocert_1: Class needed to do the sorting for the class_label_security.
        :param j: class_label under consideration for the optimal search
        :return:
        optimised list of all class related threshold, accuracy and rejection rate.
        """
        should_continue = True
        empty1 = []
        empty2 = []
        empty3 = []
        # y = thresh_hold
        j_ = 0
        for i in self.predict_results:
            if i == j:
                j_ += 1
        y = 1 / j_

        while should_continue:
            y = y + 0.01
            # list of predicted labels whose confidence is less than the thresh-hold for a given class
            index_listgl = protocert_1.thresh_function(x=d1, y=y, y_='<', y__='l', y___=j)
            # list of predicted labels regardless of the confidence thresh-hold for a given class
            index_listgl_ = protocert_1.thresh_function(x=d1, y=0, y_='>', y__='l', y___=j)
            # list containing index of predicted labels greater than or equal a given confidence thresh-hold
            index_listgi = protocert_1.thresh_function(x=d1, y=y, y_='>=', y__='i', y___=j)
            # list containing predicted labels greater than or equal to a given confidence thresh-hold
            index_listgi_ = protocert_1.thresh_function(x=d1, y=y, y_='>=', y__='l', y___=j)
            # computes the rejection rate based on the confidence thresh-hold
            z = rejection_rate(index_listgl, index_listgl_)
            # Actual labels of non-rejected classifications using the using their indexes
            true_labels = protocert_1.thresh_y_test(x=index_listgi)
            # computes accuracy of non-rejected classification
            z_ = accuracy_rate1(true_labels, index_listgi_)
            empty1.append([y, z_, z])
            empty2.append(z)
            empty3.append(z_)
            if z > self.rejection_rate1:
                should_continue = False
        return empty1[:-1], empty2[:-1], empty3[:-1]

    def thresh_new(self, d1, protocert_1, j):
        """
        :param d1: he computed classification labels securities
        :param protocert_1: Class needed to do the sorting for the class_label_security.
        :param j: class_label under consideration for the optimal search
        :return:
        optimised class related thresh-hold security. The thresh-hold at which we have
        minimum rejection and max accuracy
        """
        empty = []
        y = self.threshh(d1=d1, protocert_1=protocert_1, j=j)
        for i in range(len(y[1])):
            z = y[1][i]
            z_ = y[2][i]
            with np.errstate(divide='ignore', invalid='ignore'):
                z__ = z_ // z
                empty.append(z__)
        d = max([(i, v) for i, v in enumerate(empty)])
        return y[0][d[0]][0]


class ProtoCert:
    """
    Class to determine a unique and multiple reject options for improving classification reliability
    :param
    x_test: array, shape=[num_data,num_features]
            Where num_data is the number of samples and num_features refers to the number of features.

    class_labels: array-like, shape=[num_classes]
       Class labels of prototypes

    predict_results:  array-like, shape=[num_data]
        Predicted labels of the test-set

    """

    def __init__(self, y_test):
        self.y_test = y_test

    def thresh_function(self, x, y, y_, y__, y___):
        """
        :param x: predicted labels with their corresponding securities
        :param y: float: security threshold
        :param y_: string: '>', '<' ,'>=' to indicate the threshold security
        :param y__: string: 's' for securities , 'i' for index of data point, 'l' for label, 'all' for list with
                (securities,indexes,labels)
        :param y___: class label under consideration (None for all labels)
        :return: List containing securities( greater than or less than a given security thresh-hold),
        """
        empty = []
        empty2 = []
        empty3 = []
        empty4 = []
        for i in range(len(x)):
            if y_ == '>' and x[i][1] > y and x[i][0] == y___:
                empty.append(x[i][1])
                empty2.append(i)
                empty3.append(x[i][0])
                empty4.append([i, x[i][0], x[i][1]])
            if y_ == '>' and x[i][1] > y and y___ is None:
                empty.append(x[i][1])
                empty2.append(i)
                empty3.append(x[i][0])
                empty4.append([i, x[i][0], x[i][1]])
            if y_ == '>=' and x[i][1] >= y and x[i][0] == y___:
                empty.append(x[i][1])
                empty2.append(i)
                empty3.append(x[i][0])
                empty4.append([i, x[i][0], x[i][1]])
            if y_ == '>=' and x[i][1] >= y and y___ is None:
                empty.append(x[i][1])
                empty2.append(i)
                empty3.append(x[i][0])
                empty4.append([i, x[i][0], x[i][1]])
            if y_ == '=' and x[i][1] == y and x[i][0] == y___:
                empty.append(x[i][1])
                empty3.append(x[i][0])
                empty2.append(i)
                empty4.append([i, x[i][0], x[i][1]])
            if y_ == '=' and x[i][1] == y and y___ is None:
                empty.append(x[i][1])
                empty2.append(i)
                empty3.append(x[i][0])
                empty4.append([i, x[i][0], x[i][1]])
            if y_ == '<' and x[i][1] < y and x[i][0] == y___:
                empty.append(x[i][1])
                empty2.append(i)
                empty3.append(x[i][0])
                empty4.append([i, x[i][0], x[i][1]])
            if y_ == '<' and x[i][1] < y and y___ is None:
                empty.append(x[i][1])
                empty2.append(i)
                empty3.append(x[i][0])
                empty4.append([i, x[i][0], x[i][1]])
        if y__ == 'i':
            return empty2
        if y__ == 's':
            return empty
        if y__ == 'l':
            return empty3
        if y__ == 'all':
            return empty4

    def thresh_y_test(self, x):
        """

        :param x: thresh hold index list
        :return: labels with the thresh hold security
        """
        empty = []
        y = self.y_test
        for index in x:
            for i in range(len(y)):
                if i == index:
                    empty.append(y[i])
        return empty


class ProtoCertt:
    """
    Class to determine a unique and multiple reject options for improving classification reliability
    :param
    x_test: array, shape=[num_data,num_features]
            Where num_data is the number of samples and num_features refers to the number of features.

    class_labels: array-like, shape=[num_classes]
       Class labels of prototypes

    predict_results:  array-like, shape=[num_data]
        Predicted labels of the test-set

    """

    def __init__(self, y_test):
        self.y_test = y_test

    def thresh_function(self, x, y, y_, y__, l3):
        """
        sorting function for class related threshold in multiple reject classification
        :param x: predicted labels with their corresponding securities
        :param y: list: security threshold
        :param y_: string: '>', '<' ,'>=' to indicate the threshold security
        :param y__: string: 's' for securities , 'i' for index of data point, 'l' for label, 'all' for list with
                (securities,indexes,labels)
        :param y___: class label under consideration (None for all labels)
        :param l3: list of class labels
        :return: List containing securities( greater than or less than a given security thresh-hold),
        """
        empty = []
        empty2 = []
        empty3 = []
        empty4 = []
        # l3 = [0, 1, 2]
        for i in range(len(x)):
            for j3 in range(len(l3)):
                if y_ == '>' and x[i][0] == l3[j3] and x[i][1] > y[j3]:
                    empty.append(x[i][1])
                    empty2.append(i)
                    empty3.append(x[i][0])
                    empty4.append([i, x[i][0], x[i][1]])
                if y_ == '>=' and x[i][0] == l3[j3] and x[i][1] >= y[j3]:
                    empty.append(x[i][1])
                    empty2.append(i)
                    empty3.append(x[i][0])
                    empty4.append([i, x[i][0], x[i][1]])
                if y_ == '=' and x[i][0] == l3[j3] and x[i][1] == y[j3]:
                    empty.append(x[i][1])
                    empty3.append(x[i][0])
                    empty2.append(i)
                    empty4.append([i, x[i][0], x[i][1]])
                if y_ == '<' and x[i][0] == l3[j3] and x[i][1] < y[j3]:
                    empty.append(x[i][1])
                    empty2.append(i)
                    empty3.append(x[i][0])
                    empty4.append([i, x[i][0], x[i][1]])
        if y__ == 'i':
            return empty2
        if y__ == 's':
            return empty
        if y__ == 'l':
            return empty3
        if y__ == 'all':
            return empty4

    def thresh_y_test(self, x):
        """

        :param x: thresh hold index list
        :return: labels with the thresh hold security
        """
        empty = []
        y = self.y_test
        for index in x:
            for i in range(len(y)):
                if i == index:
                    empty.append(y[i])
        return empty


class visualize:
    """
    confidence_list : list
    2D array: Contains list of lists for all label securities

    num_classes : int :
    The cardinality of the class labels

    color: list :
    contains colors and cardinality of this list be same as the num_classes

    class_labels : list
    list of class labels

    :return a visualization of the label securities
    """

    def __init__(self, confidence_list, num_classes, colors, class_labels, n_bins=100):
        self.confidence = confidence_list
        self.num_classes = num_classes
        self.colors = colors
        self.labels = class_labels
        self.n_bins = n_bins

        if type(self.confidence) != list:
            raise TypeError('Improper data type passed for confidence_list.')

        if num_classes < 1:
            raise ValueError('Number of classes must be atleast 1.')

        if type(self.labels) != list:
            raise TypeError('Improper data type passed for class labels.')

        if len(colors) != num_classes:
            raise ValueError(f'The must be {len(num_classes)} different colors')

    def get_vis(self, x):
        """
        :param x: str: title of the plot
        :return: visualization of the classification label securities
        """

        # Creating dataset for box plot
        data = get_array(x=self.confidence)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)

        # Creating axes instance
        bp = ax.boxplot(data, patch_artist=True, notch='True', vert=0)
        colors = self.colors
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        # changing color and linewidth of whiskers
        for whisker in bp['whiskers']:
            whisker.set(color='#8B008B', linewidth=1.5, linestyle=":")

        # changing color and linewidth of caps
        for cap in bp['caps']:
            cap.set(color='#8B008B', linewidth=2)

        # changing color and linewidth of medians
        for median in bp['medians']:
            median.set(color='red', linewidth=3)

        # changing style of fliers
        for flier in bp['fliers']:
            flier.set(marker='D', color='#e7298a', alpha=0.5)

        # x-axis labels
        ax.set_yticklabels(self.labels)

        # Adding title
        plt.title(x)

        # Removing top axes and right axes ticks
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        # show plot
        return plt.show()

    def get_vis_(self, x, y, z):
        """
        :param x: title of the x-axis
        :param y: title of y-axis
        :param z: title of the plot
        :return: visualize the distribution of the classification label securities of all the classes
        """

        for i in range(len(self.confidence)):
            mu = np.mean(self.confidence[i])
            sigma = np.std(self.confidence[i])
            density = gaussian_kde(self.confidence[i])
            fig, axes = plt.subplots(nrows=2, ncols=1, sharex='all', figsize=(10, 7))

            # histogram
            axes[1].hist(self.confidence[i], self.n_bins, density=True, alpha=.1, edgecolor='black')

            median = np.percentile(self.confidence[i], 50)
            q1 = np.percentile(self.confidence[i], 25)
            q3 = np.percentile(self.confidence[i], 75)

            # probability density function
            x_ = np.linspace(np.percentile(self.confidence[i], 0), np.percentile(self.confidence[i], 100),
                             self.n_bins)

            density.covariance_factor = lambda: 100

            # KDE : non-parametric estimation of PDF of random variable
            axes[1].plot(x_, density(x_))

            # add text to bottom graph.
            axes[1].annotate('q1', xy=(q1, norm(mu, sigma).pdf(q1)), ha='center')
            axes[1].annotate('q3', xy=(q3, norm(mu, sigma).pdf(q3)), ha='center')

            # dashed lines
            plt.axvline(np.percentile(self.confidence[i], 0), color='b', linestyle='-.', alpha=.6, linewidth=.5)
            plt.axvline(np.percentile(self.confidence[i], 25), color='g', linestyle='--', alpha=.6, linewidth=.5)
            plt.axvline(np.percentile(self.confidence[i], 50), color='orange', linestyle='--', alpha=.6,
                        linewidth=.5)
            plt.axvline(np.percentile(self.confidence[i], 75), color='b', linestyle='--', alpha=.6, linewidth=.5)
            plt.axvline(np.percentile(self.confidence[i], 100), color='r', linestyle='-.', alpha=.6, linewidth=.5)

            axes[1].set_ylabel(y)
            axes[1].set_xlabel(x)

            # top boxplot
            axes[0].set_title(f'{z} {self.labels[i]}')
            axes[0].boxplot(self.confidence[i], 0, 'gD', vert=False)
            axes[0].axvline(median, color='orange', alpha=.6, linewidth=.5)
            axes[0].axvline(q1, color='green', alpha=.6, linewidth=.5)
            axes[0].axvline(q3, color='b', alpha=.6, linewidth=.5)
            axes[0].axvline(np.percentile(self.confidence[i], 0), color='b', alpha=.6, linewidth=.5)
            axes[0].axvline(np.percentile(self.confidence[i], 100), color='r', alpha=.6, linewidth=.5)
            axes[0].axis('off')

            plt.subplots_adjust(hspace=0)
        return plt.show()

    def get_vis__(self, x, y, z):
        """
        :param x: str: label of the x-axis
        :param y: str: label of the y-axis
        :param z: str: title of plot
        :return: visualization the distribution(pdf) for the classification labels securities of all classes

        """

        for i in range(len(self.confidence)):
            mu = np.mean(self.confidence[i])
            sigma = np.std(self.confidence[i])

            fig, axes = plt.subplots(nrows=2, ncols=1, sharex='all', figsize=(10, 7))

            # histogram
            n, bins, patches = axes[1].hist(self.confidence[i], self.n_bins, density=True, alpha=.1,
                                            edgecolor='black')

            pdf = get_pdf(x=sigma, y=bins, z=mu)

            median = np.percentile(self.confidence[i], 50)
            q1 = np.percentile(self.confidence[i], 25)
            q3 = np.percentile(self.confidence[i], 75)

            # probability density function
            axes[1].plot(bins, pdf, color='orange', alpha=.6)
            iqr = get_iqr(x=q3, y=q1)
            x1 = np.linspace(q1 - iqr, q1)
            x2 = np.linspace(q3, q3 + iqr)
            pdf1 = get_pdf(x=sigma, y=x1, z=mu)
            pdf2 = get_pdf(x=sigma, y=x2, z=mu)

            axes[1].fill_between(x1, pdf1, 0, alpha=.6, color='orange')
            axes[1].fill_between(x2, pdf2, 0, alpha=.6, color='orange')

            # add text to bottom graph.
            axes[1].annotate("{:.1f}%".format(100 * (norm(mu, sigma).cdf(q1) - norm(mu, sigma).cdf(q1 - iqr))),
                             xy=(q1 - iqr / 2, 0), ha='center')
            axes[1].annotate("{:.1f}%".format(100 * (norm(mu, sigma).cdf(q3) - norm(mu, sigma).cdf(q1))),
                             xy=(median, 0), ha='center')
            axes[1].annotate("{:.1f}%".format(100 * (norm(mu, sigma).cdf(q3 + iqr) - norm(mu, sigma).cdf(q3))),
                             xy=(q3 + iqr / 2, 0), ha='center')
            axes[1].annotate('q1', xy=(q1, norm(mu, sigma).pdf(q1)), ha='center')
            axes[1].annotate('q3', xy=(q3, norm(mu, sigma).pdf(q3)), ha='center')

            # dashed lines
            plt.axvline(np.percentile(self.confidence[i], 0), color='b', linestyle='-.', alpha=.6, linewidth=.5)
            plt.axvline(np.percentile(self.confidence[i], 25), color='g', linestyle='--', alpha=.6, linewidth=.5)
            plt.axvline(np.percentile(self.confidence[i], 50), color='orange', linestyle='--', alpha=.6, linewidth=.5)
            plt.axvline(np.percentile(self.confidence[i], 75), color='b', linestyle='--', alpha=.6, linewidth=.5)
            plt.axvline(np.percentile(self.confidence[i], 100), color='r', linestyle='-.', alpha=.6, linewidth=.5)

            axes[1].set_ylabel(y)
            axes[1].set_xlabel(x)

            # top boxplot
            axes[0].set_title(f'{z} {self.labels[i]}')
            axes[0].boxplot(self.confidence[i], 0, 'gD', vert=False)
            axes[0].axvline(median, color='orange', alpha=.6, linewidth=.5)
            axes[0].axvline(q1, color='green', alpha=.6, linewidth=.5)
            axes[0].axvline(q3, color='b', alpha=.6, linewidth=.5)
            axes[0].axvline(np.percentile(self.confidence[i], 0), color='b', alpha=.6, linewidth=.5)
            axes[0].axvline(np.percentile(self.confidence[i], 100), color='r', alpha=.6, linewidth=.5)
            axes[0].axis('off')

            plt.subplots_adjust(hspace=0)
        return plt.show()


if __name__ == '__main__':
    print('import module to use')
