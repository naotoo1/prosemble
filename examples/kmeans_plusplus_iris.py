"""KMeans_plusplus clustering example using Iris Data."""

# import prosemble package
import prosemble as ps
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# load some data
X, y = load_iris(return_X_y=True)

# Get data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Setup the model
kmeans = ps.models.kmeans_plusplus(
    data=X_train,
    c=3,
    num_inter=100,
    epsilon=0.00001,
    ord='fro',
    plot_steps=True
)

# fit the model
kmeans.fit()

# summary of the objective function
print(kmeans.get_objective_function())

# Get the clustering results of the input vector
print(kmeans.predict())

# Make new prediction
print(kmeans.predict_new(x=X_test))

# Get the learned centroids
print(kmeans.get_centroids())
