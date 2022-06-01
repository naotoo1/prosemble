from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# models in the ensemble
svc = SVC(probability=True)
knn = KNeighborsClassifier()
dtc = DecisionTreeClassifier()

# summary of models in the ensemble
models = [svc, knn, dtc]
for model_ in models:
    print(model_.get_params())

# Data_set and scaling
scaler = StandardScaler()
X, y = load_breast_cancer(return_X_y=True)
X = X[:, 0:2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape, y_train.shape)

# Train, test and save models
model_names = ['svc.pkl', 'knn.pkl', 'dtc.pkl']
test_acc = []
for i in range(len(models)):
    models[i].fit(X_train, y_train)
    test_acc.append(accuracy_score(y_test, models[i].predict(X_test)))
    pickle_out = open(model_names[i], 'wb')
    pickle.dump(models[i], pickle_out)
    pickle_out.close()

# summary of trained accuracy
print(test_acc)
