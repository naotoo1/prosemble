from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from hybrid12 import Hybrid
import matplotlib.pyplot as plt
import numpy as np


def simulate_m(x, y_):
    """

    :param x: Test set
    :param y_:list : set of possible fuzzifiers
    :return: classification label securities based on the fuzzifiers
    """
    sample_dat = [tryy_1.get_security(x, i) for i in y_]
    sample_dat2 = [tryy_2.get_security_m(x, i) for i in y_]
    sample_dat3 = [tryy_3.get_security(x, i) for i in y_]
    return sample_dat, sample_dat2, sample_dat3


def optimise_m(x):
    """
    :param x: int:  model evaluation performance measure(eg average accuracy from CV)
    :return:  returns optimized fuzzier for the classification label securities
    """
    m = round((1 / (x * x)) + 1)
    return round(m)


def sim_mlist(x, y_, z):
    """

    :param x: Test set
    :param y_:List : set of possible fuzzifiers
    :param z: index of sample data from the test set
    :return: list: Classification label securities for a sample data based on the set list of fuzzifiers
    """
    simulated_mlist = [i_[1][1] for i_ in simulate_m(x, y_)[z]]
    return simulated_mlist


def sim_m():
    """

    :return: List: Optimized fuzzifiers based on simulated list of model performance evaluation measure.
    """
    r = np.arange(0.4, 1.1, 0.1)
    m_list = [optimise_m(i) for i in r]
    return m_list, r


if __name__ == '__main__':
    # Data_set and scaling
    scaler = StandardScaler()
    X, y = load_iris(return_X_y=True)
    X = X[:, 0:2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler.fit(X_train)
    # X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # summary of input parameters
    proto_classes = np.array([0, 1, 2])

    # Transferred learned prototypes for glvq, gmlvq and celvq respectively
    new_prototypes_1 = np.array([[-1.092267, 0.9856019], [-0.29071018, -1.230379], [1.5310693, 0.08934504]])
    new_prototypes_2 = np.array([[-0.97786397, 0.8252505], [-0.25761604, -0.49248296], [1.2729689, 0.05621301]])
    new_prototypes_3 = np.array([[-1.654047, 1.1912421], [0.06487547, -1.4322541], [1.6647131, -0.4211262]])
    omega_matrix = np.array([[1.3414, -0.6254], [-0.5219, 0.2435]])

    # object of the Hybrid class
    tryy_1 = Hybrid(new_prototypes_1, proto_classes, 3, omega_matrix=None, matrix='n')
    tryy_2 = Hybrid(new_prototypes_2, proto_classes, 3, omega_matrix=omega_matrix, matrix='y')
    tryy_3 = Hybrid(new_prototypes_3, proto_classes, 3, omega_matrix=None, matrix='n')

    #  simulate ensemble lvq based on transfer learning with glvq, gmlvq and celvq learned prototypes
    ym = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    a, b = sim_m()
    glvq_sim_sec = sim_mlist(x=X_test, y_=ym, z=0)
    gmlvq_sim_sec = sim_mlist(x=X_test, y_=ym, z=1)
    celvq_sim_sec = sim_mlist(x=X_test, y_=ym, z=2)

    print(glvq_sim_sec)
    print(a)
    f = plt.figure(1)
    plt.plot(ym, glvq_sim_sec, label='GLVQ')
    plt.plot(ym, gmlvq_sim_sec, label='GMLVQ')
    plt.plot(ym, celvq_sim_sec, label='CELVQ')
    plt.xlabel('m')
    plt.ylabel('classification label security')
    plt.legend()

    p = plt.figure(2)
    plt.plot(a, b)
    plt.xlabel('optimised m')
    plt.ylabel('Test accuracy')

    plt.show()
