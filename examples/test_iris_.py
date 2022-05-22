from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from prosemble import Hybrid
import numpy as np

# Data_set and scaling
scaler = StandardScaler()
X, y = load_iris(return_X_y=True)
X = X[:, 0:2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler.fit(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape, y_train.shape)

# summary of prototype labels
proto_classes = np.array([0, 1, 2])

# Transferred learned prototypes for glvq, gmlvq and celvq models respectively
new_prototypes_1 = np.array([[-1.092267, 0.9856019], [-0.29071018, -1.230379], [1.5310693, 0.08934504]])
new_prototypes_2 = np.array([[-0.97786397, 0.8252505], [-0.25761604, -0.49248296], [1.2729689, 0.05621301]])
new_prototypes_3 = np.array([[-1.654047, 1.1912421], [0.06487547, -1.4322541], [1.6647131, -0.4211262]])
omega_matrix = np.array([[1.3414, -0.6254], [-0.5219, 0.2435]])

# object of the Hybrid Class
tryy_1 = Hybrid(new_prototypes_1, proto_classes, 3, omega_matrix=None, matrix='n')
tryy_2 = Hybrid(new_prototypes_2, proto_classes, 3, omega_matrix=omega_matrix, matrix='y')
tryy_3 = Hybrid(new_prototypes_3, proto_classes, 3, omega_matrix=None, matrix='n')

# predictions with transfered learned prototypes of glvq, gmlvq and celvq
pred1 = tryy_1.predict(x_test=X_test)
pred2 = tryy_2.predict(x_test=X_test)
pred3 = tryy_3.predict(x_test=X_test)

# certainty of predicted results with hyperparameter m chosen as 2
sec1 = tryy_1.get_security(x=X_test, y=2)
sec2 = tryy_2.get_security_m(x=X_test, y=2)
sec3 = tryy_3.get_security(x=X_test, y=2)

# list with all respective predictions from the trained prototypes.
all_pred = [pred1, pred2, pred3]
all_sec = [sec1, sec2, sec3]

# predictions based on max votes
final_pred = tryy_1.pred_prob(x=X_test, y=all_pred)
print(final_pred)

# predictions based on soft voting
final_pred_1 = tryy_1.pred_sprob(x=X_test, y=all_sec)
print(final_pred_1)

# summary results of the hard voting accuracy
print(tryy_1.accuracy(y_test, final_pred))
#
# summary results of the soft voting accuracy
print(tryy_1.accuracy(y_test, final_pred_1))

# summary of prediction probability Hard Voting
print(tryy_1.prob(x=X_test, y=all_pred))

# summary of prediction probability Soft Voting
print(tryy_1.sprob(x=X_test, y=all_sec))
