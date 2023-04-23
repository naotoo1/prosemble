"""Kernel Fuzzy C-Means clustering example using Iris Data."""

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
kfcm = ps.models.KFCM(
    data=X_train,
    c=3, num_iter=1000,
    epsilon=0.001,
    ord='fro',
    set_prototypes=None,
    m=2,
    sigma=1,
    set_U_matrix=None,
    plot_steps=True
)

# fit the model
kfcm.fit()

# summary of the objective function
print(kfcm.get_objective_function())

# Get the clustering results of the input vector
print(kfcm.predict())

# Make new prediction
print(kfcm.predict_new(x=X_test))

# Get the learned centroids
print(kfcm.final_centroids())

