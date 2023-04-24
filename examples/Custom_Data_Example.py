"""Kernel Fuzzy C-Means clustering example using Iris Data."""

# import prosemble package
import prosemble as ps
from sklearn.model_selection import train_test_split
import pickle as pkl

PATH = open(r'Path to Pickle file', 'rb')

X = pkl.load(PATH)

print(X.shape)

# Get data split
X_train, X_test = train_test_split(X, test_size=0.2)

# Setup the model
fcm = ps.models.KFCM(
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
fcm.fit()

# summary of the objective function
print(fcm.get_objective_function)

# Get the clustering results of the input vector
print(fcm.predict)

# Make new prediction
print(fcm.predict_new(x=X_test))

# Get the learned centroids
print(fcm.final_centroids)
