from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn_lvq import MrslvqModel, RslvqModel, LmrslvqModel, GrmlvqModel, GrlvqModel
import pickle
from sklearn.metrics import accuracy_score


# models
model1 = RslvqModel()
model2 = MrslvqModel()
model3 = LmrslvqModel()
model4 = GrlvqModel()
model5 = GrmlvqModel()

# summary of models
models = [model1, model2, model3, model4, model5]
for model_ in models:
    print(model_.get_params())

# Data_set and scaling
scaler = StandardScaler()
X, y = load_iris(return_X_y=True)
X = X[:, 0:2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train.shape, y_train.shape)

# Train, test and save models
model_names = ['rslvq.pkl', 'mrslvq.pkl', 'lmrslvq.pkl', 'grlvq.pkl', 'grmlvq.pkl']
test_acc = []
for i in range(len(models)):
    models[i].fit(X_train, y_train)
    test_acc.append(accuracy_score(y_test, models[i].predict(X_test)))
    pickle.dump(models[i], open(model_names[i], 'wb'))

# summary of test accuracy
print(test_acc)
