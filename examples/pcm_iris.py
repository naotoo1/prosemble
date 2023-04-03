"""Possibilistic C-Means clustering example using Iris Data."""

# import prosemble package
import prosemble as ps
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# load some data
X, y = load_iris(return_X_y=True)

# Get data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Setup the model
pcm = ps.models.PCM(
    data=X_train,
    c=3,
    m=2,
    k=0.001,
    num_iter=1000,
    epsilon=0.00001,
    ord='fro',
    set_U_matrix='fcm',
    plot_steps=True
)

# fit the model
pcm.fit()

# summary of the objective function
print(pcm.get_objective_function())

# Get the clustering results of the input vector
print(pcm.predict())

# Make new prediction
print(pcm.predict_new(x=X_test))

# Get the learned centroids
print(pcm.final_centroids())
