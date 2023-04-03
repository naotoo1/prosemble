"""Fuzzy Possibilistic C-Means clustering example using Iris Data."""

# import prosemble package
import prosemble as ps
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# load some data
X, y = load_iris(return_X_y=True)

# Get data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Setup the model
fpcm = ps.models.FPCM(
    data=X_train,
    c=3,
    m=2,
    eta=2,
    num_iter=1000,
    epsilon=0.00001,
    ord='fro',
    set_U_matrix='fcm',
    plot_steps=True
)

# fit the model
fpcm.fit()

# summary of the objective function
print(fpcm.get_objective_function())

# Get the clustering results of the input vector
print(fpcm.predict())

# Make new prediction
print(fpcm.predict_new(x=X_test))

# Get the learned centroids
print(fpcm.final_centroids())
