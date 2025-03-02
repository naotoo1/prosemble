"""Allied Fuzzy C-Means clustering example using Iris Data."""

# import prosemble package
import prosemble as ps
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# load some data
X, y = load_iris(return_X_y=True)
print(X.shape[0])

# Get data split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=30)

# Setup the model
afcm = ps.models.AFCM(
    data=X_train,
    c=3,
    num_iter=10000,
    epsilon=0.00001,
    ord='fro',
    m=2,
    a=2,
    b=2,
    k=1,
    set_U_matrix='fcm',
    plot_steps=True
)

# fit the model
afcm.fit()

# summary of the objective function
print(afcm.get_objective_function())

# Get the clustering results of the input vector
print(afcm.predict())

# Make new prediction
print(afcm.predict_new(x=X_test))

# Get the learned centroids
print(afcm.final_centroids())
