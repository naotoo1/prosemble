"""Basic Graded Possibilistic clustering example using Iris Data."""

# import prosemble package
import prosemble as ps
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# load some data
X, y = load_iris(return_X_y=True)

# Get data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Setup the model
bgpc = ps.models.BGPC(
    data=X_train,
    c=3,
    a_f=2,
    b_f=0.5,
    num_iter=100,
    epsilon=0.00001,
    ord='fro',
    set_U_matrix='fcm',
    plot_steps=True
)

# fit the model
bgpc.fit()

# Get the clustering results of the input vector
print(bgpc.predict())

# Make new prediction
print(bgpc.predict_new(x=X_test))

# Get the learned centroids
print(bgpc.final_centroids())
