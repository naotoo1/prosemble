"""Module to Determine classification Label Security/Certainty"""
import numpy as np
from scipy.spatial import distance


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


if __name__ == '__main__':
    print('import module to use')
