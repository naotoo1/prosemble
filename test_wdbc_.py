from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from hybrid12 import Hybrid
import numpy as np

# Data_set and scaling
scaler = StandardScaler()
X, y = load_breast_cancer(return_X_y=True)
X = X[:, 0:2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler.fit(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape, y_train.shape)

# summary of prototype labels
proto_classes = np.array([0, 1])

# Transferred learned prototypes for glvq, gmlvq and celvq models respectively
new_prototypes_1 = np.array([[1.8352698, 0.6309362], [-0.71516573, -0.6174223]])
new_prototypes_2 = np.array([[1.195601, 0.86342883], [-0.61568695, -0.48220304]])
new_prototypes_3 = np.array([[2.7108397, 0.74844754], [-1.9635255, -0.9237849]])

# object of the Hybrid Class
tryy_1 = Hybrid(new_prototypes_1, proto_classes, 2, omega_matrix=None, matrix='n')
tryy_2 = Hybrid(new_prototypes_2, proto_classes, 2, omega_matrix=None, matrix='n')
tryy_3 = Hybrid(new_prototypes_3, proto_classes, 2, omega_matrix=None, matrix='n')

# predictions with transferred learned prototypes of glvq, gmlvq and celvq
pred1 = tryy_1.predict(x_test=X_test)
pred2 = tryy_2.predict(x_test=X_test)
pred3 = tryy_3.predict(x_test=X_test)

sec1 = tryy_1.get_security(x=X_test)
sec2 = tryy_2.get_security(x=X_test)
sec3 = tryy_3.get_security(x=X_test)

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
