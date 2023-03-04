# import prosemble package
from prosemble.models.fcm import FCM
from sklearn.model_selection import train_test_split
import pickle as pkl

PATH = open(r'Path to Pickle file', 'rb')

X = pkl.load(PATH)

print(X.shape)
# Get data split
X_train, X_test = train_test_split(X, test_size=0.2)

# Instantiate an object from a given class
fcm = fcm.FCM(data=X_train, c=3, m=2, num_iter=100, epsilon=0.00001,
              ord=None, set_U_matrix=None, plot_steps=True)

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
