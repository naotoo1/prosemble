from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from prosemble import Hybrid

# load models to be used in prosemble
svc = pickle.load(open('svc.pkl', 'rb'))
knn = pickle.load(open('knn.pkl', 'rb'))
dtc = pickle.load(open('dtc.pkl', 'rb'))

# Data_set and scaling
scaler = StandardScaler()
X, y = load_breast_cancer(return_X_y=True)
X = X[:, 0:2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler.fit(X_train)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train.shape, y_train.shape)

# Summary of predictions from the svc, knn and dtc trained models
predict1 = svc.predict(X_test)
predict2 = knn.predict(X_test)
predict3 = dtc.predict(X_test)
print(svc.predict_proba(X_test))
print(knn.predict_proba(X_test))
print(dtc.predict_proba(X_test))


# function to get securities of classifications from models to be used in prosemble
def get_posterior(x, y_, z_):
    z1 = z_.predict_proba(x)
    certainties = [np.max(i) for i in z1]
    cert = np.array(certainties).flatten()
    cert = cert.reshape(len(cert), 1)
    y_ = y_.reshape(len(y_), 1)
    labels_with_certainty = np.concatenate((y_, cert), axis=1)
    return np.round(labels_with_certainty, 4)


# class labels of prototypes
proto_classes = np.array([0, 1])

# summary of the certainties from the rslvq, mrslvq and lmrslvq models
print(get_posterior(X_test, predict1, svc))
print(get_posterior(X_test, predict2, knn))
print(get_posterior(X_test, predict3, dtc))
#
# certainty of predicted results
sec1 = get_posterior(X_test, predict1, svc)
sec2 = get_posterior(X_test, predict2, knn)
sec3 = get_posterior(X_test, predict3, dtc)

# list with all respective predictions from the trained prototypes.
all_pred = [predict1, predict2, predict3]
all_sec = [sec1, sec2, sec3]

# object for the hybrid class
ensemble = Hybrid(model_prototypes=None, proto_classes=proto_classes, mm=2, omega_matrix=None, matrix='n')

# predictions based on max voting
final_pred = ensemble.pred_prob(x=X_test, y=all_pred)
print(final_pred)

# predictions based on soft voting
final_pred_1 = ensemble.pred_sprob(x=X_test, y=all_sec)
print(final_pred_1)

# summary results of the hard voting accuracy
print(ensemble.accuracy(y_test, final_pred))

# summary results of the soft voting accuracy
print(ensemble.accuracy(y_test, final_pred_1))

# summary of prediction probability Hard Voting
print(ensemble.prob(x=X_test, y=all_pred))

# summary of prediction probability Soft Voting
print(ensemble.sprob(x=X_test, y=all_sec))
