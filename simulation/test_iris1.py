from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from prosemble import Hybrid
import matplotlib.pyplot as plt
import numpy as np

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
proto_classes = np.array([0, 1, 2])

# Transferred learned prototypes for glvq, gmlvq and celvq respectively
glvq_prototypes = np.array([[-1.092267, 0.9856019], [-0.29071018, -1.230379], [1.5310693, 0.08934504]])
gmlvq_prototypes = np.array([[-0.97786397, 0.8252505], [-0.25761604, -0.49248296], [1.2729689, 0.05621301]])
celvq_prototypes = np.array([[-1.654047, 1.1912421], [0.06487547, -1.4322541], [1.6647131, -0.4211262]])
omega_matrix = np.array([[1.3414, -0.6254], [-0.5219, 0.2435]])

# object of the Hybrid class
glvq = Hybrid(model_prototypes=glvq_prototypes, proto_classes=proto_classes, mm=3, omega_matrix=None, matrix='n')
gmlvq = Hybrid(model_prototypes=gmlvq_prototypes, proto_classe=proto_classes,mm=3, omega_matrix=omega_matrix, matrix='y')
celvq = Hybrid(model_prototypes=celvq_prototypes, proto_classes=proto_classes, mm=3, omega_matrix=None, matrix='n')


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
        pred1 = glvq.predict(x_test=a)
        pred2 = gmlvq.predict(x_test=a)
        pred3 = celvq.predict(x_test=a)
        sec1 = glvq.get_security(x=a, y=2)
        sec2 = gmlvq.get_security_m(x=a, y=2)
        sec3 = celvq.get_security(x=a, y=2)
        all_pred = [pred1, pred2, pred3]
        all_sec = [sec1, sec2, sec3]
        final_pred1 = glvq.pred_sprob(x=a, y=all_sec)
        accuracy1 = glvq.accuracy(x=b, y=final_pred1)
        final_pred = glvq.pred_prob(x=a, y=all_pred)
        accuracy = glvq.accuracy(x=b, y=final_pred)
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
