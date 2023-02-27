from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from prosemble import Hybrid

# load trained models
rslvq = pickle.load(open('rslvq.pkl', 'rb'))
mrslvq = pickle.load(open('mrslvq.pkl', 'rb'))
lmrslvq = pickle.load(open('lmrslvq.pkl', 'rb'))
grlvq = pickle.load(open('grlvq.pkl', 'rb'))
grmlvq = pickle.load(open('grmlvq.pkl', 'rb'))

# summary of models
print(rslvq.get_params())
print(mrslvq.get_params())
print(lmrslvq.get_params())
print(grlvq.get_params())
print(grmlvq.get_params())


# Data_set and scaling
scaler = StandardScaler()
X, y = load_iris(return_X_y=True)
X = X[:, 0:2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler.fit(X_train)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train.shape, y_train.shape)

# Summary of predictions from the rslvq, mrslvq ,lmrslvq, grlvq and grmlvq trained models
predict1 = rslvq.predict(X_test)
predict2 = mrslvq.predict(X_test)
predict3 = lmrslvq.predict(X_test)
predict4 = grlvq.predict(X_test)
predict5 = grmlvq.predict(X_test)
omega_matrix1 = mrslvq.w_
omega_matrix2 = grmlvq.w_

# class labels of prototypes
proto_classes = np.array([0, 1, 2])

# object for the hybrid class
# ensemble = Hybrid(None, proto_classes, 3, omega_matrix=None, matrix='n')

rslvq_ = Hybrid(model_prototypes=rslvq.w_, proto_classes=proto_classes, mm=3, omega_matrix=None, matrix='n')
mrslvq_ = Hybrid(model_prototypes=mrslvq.w_, proto_classes=proto_classes,mm=3, omega_matrix=omega_matrix1, matrix='y')
lmrslvq_ = Hybrid(model_prototypes=lmrslvq.w_, proto_classes=proto_classes,mm=3, omega_matrix=None, matrix='n')
grlvq_ = Hybrid(model_prototypes=grlvq.w_,proto_classes=proto_classes,m=3,omega_matrix=None, matrix='n')
grmlvq_ = Hybrid(model_prototypes=grmlvq.w_, proto_classes=proto_classes,mm=3, omega_matrix=None, matrix='n')

# certainty of predicted results with hyperparameter m chosen as 2
sec1 = rslvq_.get_security(x=X_test, y=2)
sec2 = mrslvq_.get_security_m(x=X_test, y=2)
sec3 = lmrslvq_.get_security(x=X_test, y=2)
sec4 = grlvq_.get_security(x=X_test, y=2)
sec5 = grmlvq_.get_security(x=X_test, y=2)

# list with all respective predictions from the trained prototypes.
all_pred = [predict1, predict2, predict3, predict4, predict5]
all_sec = [sec1, sec2, sec3, sec4, sec5]

# predictions based on max voting
final_pred = rslvq_.pred_prob(x=X_test, y=all_pred)
print(final_pred)

# predictions based on soft voting
final_pred_1 = rslvq_.pred_sprob(x=X_test, y=all_sec)
print(final_pred_1)

# summary results of the hard voting accuracy
print(rslvq_.accuracy(y_test, final_pred))

# summary results of the soft voting accuracy
print(rslvq_.accuracy(y_test, final_pred_1))


# summary of prediction probability Hard Voting
print(rslvq_.prob(x=X_test, y=all_pred))

# summary of prediction probability Soft Voting
print(rslvq_.sprob(x=X_test, y=all_sec))
