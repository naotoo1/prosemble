# Prosemble
[![python: 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![github](https://img.shields.io/badge/version-0.0.2-yellow.svg)](https://github.com/naotoo1/Prosemble)
[![pypi](https://img.shields.io/badge/pypi-0.0.2-orange.svg)](https://pypi.org/project/prosemble)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

ML package for both prototype and non-prototype-based classifiers that utilizes ensemble learning by soft and hard voting.

## why?
In ML the convention has been to save a trained model for future use or deployment. An alternative way in the case of prototype-based models(learning vector quantization) would be to access learned prototypes from pre-trained models for use in deployment.

This project implements the harnessing of trained models and learned prototypes in ensemble learning for both prototype-based and non protype-based classification models. In this regard the hard voting and soft voting scheme is applied to achieve the classification results. 

## Installation
```python
pip install prosemble
```

## How to use
### Prototype-based example with LVQs
Import libraries
```python
import pickle
import numpy as np
from prosemble import Hybrid
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
```
Load some ```trained models``` from your working directory to be used in the ensemble. The models were trained using LVQs with the Iris test set.
```python
rslvq = pickle.load(open('rslvq.pkl', 'rb'))
mrslvq = pickle.load(open('mrslvq.pkl', 'rb'))
lmrslvq = pickle.load(open('lmrslvq.pkl', 'rb'))
grlvq = pickle.load(open('grlvq.pkl', 'rb'))
grmlvq = pickle.load(open('grmlvq.pkl', 'rb'))

```
In this example we consider the iris data set
```python
# Data_set and scaling
scaler = StandardScaler()
X, y = load_iris(return_X_y=True)
X = X[:, 0:2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler.fit(X_train)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train.shape, y_train.shape)
```
Summary of predictions from the ```rslvq```, ```mrslvq``` ,```lmrslvq```, ```grlvq``` and ```grmlvq``` trained models
```python
predict1 = rslvq.predict(X_test)
predict2 = mrslvq.predict(X_test)
predict3 = lmrslvq.predict(X_test)
predict4 = grlvq.predict(X_test)
predict5 = grmlvq.predict(X_test)
omega_matrix1 = mrslvq.w_
omega_matrix2 = grmlvq.w_
```
In the object for the ```Hybrid class``` we utilize the learned prototypes accessed from the trained models in the ensemble
```python
# class labels of prototypes
proto_classes = np.array([0, 1, 2])

# object for the Hybrid class
# ensemble = Hybrid(None, proto_classes, 3, omega_matrix=None, matrix='n')

rslvq_ = Hybrid(model_prototypes=rslvq.w_, proto_classes=proto_classes, mm=3, omega_matrix=None, matrix='n')
mrslvq_ = Hybrid(model_prototypes=mrslvq.w_,proto_classes=proto_classes, mm=3, omega_matrix=omega_matrix1, matrix='y')
lmrslvq_ = Hybrid(model_prototypes=lmrslvq.w_, proto_classes=proto_classes, mm=3, omega_matrix=None, matrix='n')
grlvq_ = Hybrid(model_prototypes=grlvq.w_, proto_classes=proto_classes,mm=3, omega_matrix=None, matrix='n')
grmlvq_ = Hybrid(model_prototypes=grmlvq.w_, proto_classes=proto_classes,mm=3, omega_matrix=None, matrix='n')
```
The posterior of the predicted labels from the trained models are determined using the model_prototypes by calling the method ```get_security``` on the model instance from the ```Hybrid Class```.
```python
# certainty of predicted results with hyperparameter m chosen as 2
sec1 = rslvq_.get_security(x=X_test, y=2)
sec2 = mrslvq_.get_security_m(x=X_test, y=2)
sec3 = lmrslvq_.get_security(x=X_test, y=2)
sec4 = grlvq_.get_security(x=X_test, y=2)
sec5 = grmlvq_.get_security(x=X_test, y=2)
```
Make predictions using the hard and soft voting scheme. Use the ```pred_porb``` and ```pred_sprob methods``` for the hard and soft voting schemes respectively.
NB that these methods are avaible for all objects of the Hybrid Class instance and can be called and applied to all the ensemble models.
```python
# list with all respective predictions from the trained prototypes.
all_pred = [predict1, predict2, predict3, predict4, predict5]
all_sec = [sec1, sec2, sec3, sec4, sec5]

# predictions based on max voting
final_pred = rslvq_.pred_prob(x=X_test, y=all_pred)
print(final_pred)

# predictions based on soft voting
final_pred_1 = rslvq_.pred_sprob(x=X_test, y=all_sec)
print(final_pred_1)

```
For performance evaluation call on the performance metric methods or you may used any other metric of your choice.

```python

# summary results of the hard voting accuracy
print(rslvq_.accuracy(y_test, final_pred))

# summary results of the soft voting accuracy
print(rslvq_.accuracy(y_test, final_pred_1))
```

For the ensemble prediction certainties/probabilities call on the ```prob``` and ```sprob``` recall proceedure for the soft and hard voting respectively.
```python

# summary of prediction probability Hard Voting
print(rslvq_.prob(x=X_test, y=all_pred))

# summary of prediction probability Soft Voting
print(rslvq_.sprob(x=X_test, y=all_sec))
```
## Non Prototype-based example
For non prototype-based examplen the proceedure remains almost the same. The only change comes in the method that will be used in getting the prediction probabilities does not involve the use of learned prototypes but is based on the models that are used in the ensemble.

## Bibtex
If you would like to cite the package, please use this:
```python
@misc{Otoo_Prosemble_2022,
author = {Otoo, Nana Abeka},
title = {Prosemble},
year = {2022},
publisher = {GitHub},
journal = {GitHub repository},
howpublished= {\url{https://github.com/naotoo1/Prosemble}},
}
```



