"""Kernel Allied Fuzzy C-Means clustering example using Iris Data."""

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
kafcm = ps.models.KAFCM(
    data=X_train,
    c=3,
    num_iter=1000,
    epsilon=0.00001,
    ord='fro',
    m=2,
    a=3,
    b=3,
    k=1,
    sigma=1,
    set_centroids='kfcm',
    set_U_matrix='kfcm',
    plot_steps=True
)

# fit the model
kafcm.fit()

# summary of the objective function
print(kafcm.get_objective_function())

# Get the clustering results of the input vector
print(kafcm.predict())

# Make new prediction
print(kafcm.predict_new(x=X_test))

# Get the learned centroids
print(kafcm.final_centroids())
