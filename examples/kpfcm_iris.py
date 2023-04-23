"""Kernel Possibilistic Fuzzy C-Means clustering example using Iris Data."""

# import prosemble package
import prosemble as ps
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# load some data
X, y = load_iris(return_X_y=True)
print(X.shape[0])

# Get data split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=10)

# Setup the model
kpfcm = ps.models.KPFCM(
    data=X_train,
    c=3,
    num_iter=100,
    epsilon=0.0001,
    ord='fro',
    m=2,
    a=2,
    b=1,
    k=0.02,
    sigma=1,
    eta=2,
    set_centroids=None,
    set_U_matrix='kfcm',
    plot_steps=True
)

# fit the model
kpfcm.fit()

# summary of the objective function
print(kpfcm.get_objective_function())

# Get the clustering results of the input vector
print(kpfcm.predict())

# Make new prediction
print(kpfcm.predict_new(x=X_test))

# Get the learned centroids
print(kpfcm.final_centroids())

