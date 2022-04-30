from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from hybrid1 import Hybrid
import matplotlib.pyplot as plt

# Data_set and scaling
scaler = StandardScaler()
X, y = load_iris(return_X_y=True)
X = X[:, 0:2]


#  select test set for the simulation
def select_(x, i_):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=i_)
    scaler.fit(X_train)
    # X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_test, y_test


# summary of input parameters
proto_classes = [0, 1, 2]

# Transferred learned prototypes from the iris for glvq, gmlvq and celvq respectively
new_prototypes_1 = (
    [[-1.1442775, 0.8578415], [-0.22741659, -1.3264811], [1.4355098, 0.14914764]])
new_prototypes_2 = (
    [[-0.9664123, 0.8180202], [-0.14138626, -0.7539766], [1.0968374, 0.0320867]])
new_prototypes_3 = (
    [[-1.601125, 1.136284], [0.22535476, - 1.4961139], [1.7082069 - 0.34369266]])

# object of the Hybrid class
tryy_1 = Hybrid(new_prototypes_1, proto_classes, 3, omega_matrix=None, matrix='n')
tryy_2 = Hybrid(new_prototypes_2, proto_classes, 3, omega_matrix=None, matrix='n')
tryy_3 = Hybrid(new_prototypes_3, proto_classes, 3, omega_matrix=None, matrix='n')


#  simulate ensemble lvq based on transfer learning with glvq, gmlvq and celvq learned prototypes
def simulation(x):
    """
    :param x: x_test
    :return:
    Simulated accuracy and  list
    """
    accuracy_list = []
    simu_list = []
    # test_ss = np.arange(0.2, 1, 0.1)
    for i in range(20):
        ss = select_(x, i_=0.2)
        pred1 = tryy_1.predict(x_test=ss[0])
        pred2 = tryy_2.predict(x_test=ss[0])
        pred3 = tryy_3.predict(x_test=ss[0])
        all_pred = [pred1, pred2, pred3]
        final_pred = tryy_1.pred_prob(x=ss[0], y=all_pred)
        accuracy = tryy_1.accuracy(x=ss[1], y=final_pred)
        accuracy_list.append(accuracy)
        simu_list.append(i)
    return accuracy_list, simu_list


# summary results of the transferred prototypes (new_prototypes_1)
simulated_results = simulation(x=X)
simulated_accuracy = simulated_results[0]
simulated_list = simulated_results[1]


# plot simulated results of transfer learning in ensemble lvq
plt.plot(simulated_list, simulated_accuracy, label='hybrid', marker='o')

plt.xlabel('Simulations')
plt.ylabel('Accuracy(%)')
plt.legend()
plt.show()
