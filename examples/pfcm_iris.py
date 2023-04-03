"""Possibilistic Fuzzy C-Means clustering example using Iris Data."""

# import prosemble package
import prosemble as ps
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# load some data
X, y = load_iris(return_X_y=True)

# Get data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Setup the model
pfcm = ps.models.PFCM(
    data=X_train,
    c=3,
    m=2,
    eta=2,
    k=1,
    a=2,
    b=2,
    num_iter=1000,
    epsilon=0.00001,
    ord='fro',
    set_U_matrix='fcm',
    plot_steps=True
)

# fit the model
pfcm.fit()

# summary of the objective function
print(pfcm.get_objective_function())

# Get the clustering results of the input vector
print(pfcm.predict())

# Make new prediction
print(pfcm.predict_new(x=X_test))

# Get the learned centroids
print(pfcm.final_centroids())
