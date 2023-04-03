"""Fuzzy C-Means clustering example using Iris Data."""

# import prosemble package
import prosemble as ps
from prosemble.models.fcm import FCM
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# load some data
X, y = load_iris(return_X_y=True)

# Get data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Setup the model
fcm = ps.models.FCM(
    data=X_train,
    c=3,
    m=2,
    num_iter=100,
    epsilon=0.00001,
    ord='fro',
    set_U_matrix=None,
    plot_steps=True
)

# fit the model
fcm.fit()

# summary of the objective function
print(fcm.get_objective_function())

# Get the clustering results of the input vector
print(fcm.predict())

# Make new prediction
print(fcm.predict_new(x=X_test))

# Get the learned centroids
print(fcm.final_centroids())
