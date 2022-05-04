from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from hybrid import Hybrid
import matplotlib.pyplot as plt
import numpy as np

# Data_set and scaling
scaler = StandardScaler()
X, y = load_breast_cancer(return_X_y=True)
X = X[:, 0:2]


#  select test set for the simulation
def select_(x, i_):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=i_)
    scaler.fit(X_train)
    # X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_test, y_test


# summary of input parameters
proto_classes = np.array([0, 1])

# Transferred learned prototypes for glvq, gmlvq and celvq respectively
new_prototypes_1 = np.array([[1.8352698, 0.6309362], [-0.71516573, -0.6174223]])
new_prototypes_2 = np.array([[1.195601, 0.86342883], [-0.61568695, -0.48220304]])
new_prototypes_3 = np.array([[2.7108397, 0.74844754], [-1.9635255, -0.9237849]])

# object of the Hybrid class
tryy_1 = Hybrid(new_prototypes_1, proto_classes, 2, omega_matrix=None, matrix='n')
tryy_2 = Hybrid(new_prototypes_2, proto_classes, 2, omega_matrix=None, matrix='n')
tryy_3 = Hybrid(new_prototypes_3, proto_classes, 2, omega_matrix=None, matrix='n')


#  simulate ensemble lvq based on transfer learning with glvq, gmlvq and celvq learned prototypes
def simulation(x):
    """
    :param x: x_test
    :return:
    Simulated accuracy  for both hard and soft voting
    """
    accuracy_list1 = []
    simu_list1 = []
    accuracy_list = []
    simu_list = []
    # test_ss = np.arange(0.2, 1, 0.1)
    for i in range(20):
        a, b = select_(x, i_=0.2)
        pred1 = tryy_1.predict(x_test=a)
        pred2 = tryy_2.predict(x_test=a)
        pred3 = tryy_3.predict(x_test=a)
        sec1 = tryy_1.get_security(x=a)
        sec2 = tryy_2.get_security(x=a)
        sec3 = tryy_3.get_security(x=a)
        all_pred = [pred1, pred2, pred3]
        all_sec = [sec1, sec2, sec3]
        final_pred1 = tryy_1.pred_sprob(x=a, y=all_sec)
        accuracy1 = tryy_1.accuracy(x=b, y=final_pred1)
        final_pred = tryy_1.pred_prob(x=a, y=all_pred)
        accuracy = tryy_1.accuracy(x=b, y=final_pred)
        accuracy_list1.append(accuracy1)
        simu_list1.append(i)
        accuracy_list.append(accuracy)
        simu_list.append(i)
    return accuracy_list, simu_list, accuracy_list1, simu_list1


# simulation results
p, r, s, t = simulation(x=X)

# summary simulation results for soft voting
simulated_accuracy1 = s
simulated_list1 = t

# summary simulation results for hard voting
simulated_accuracy = p
simulated_list = r

# plot simulated results of transfer learning in ensemble lvq
plt.plot(simulated_list1, simulated_accuracy1, label='soft voting', marker='o')
plt.plot(simulated_list, simulated_accuracy, label='hard voting', marker='v')

plt.xlabel('Simulations')
plt.ylabel('Accuracy(%)')
plt.legend()
plt.show()
