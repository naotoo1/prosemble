# Ensemble-lvq
[![python: 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![github](https://img.shields.io/badge/version-0.0.1-yellow.svg)](https://github.com/naotoo1/Prosemble)
[![pypi](https://img.shields.io/badge/pypi-0.0.1-orange.svg)](https://pypi.org/project/prosemble)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A Prototype based ML project implementation which utilizes learned protototypes from LVQs in ensemble learning by soft and hard voting

## why?
In ML the convention has been to save a trained model for future use or deployment. An alternative way would be to access learned prototypes from pre-trained models
for use in deployemnt.

This project implements the harnessing of pre-trained prototypes in ensemble learning with lvq models. In this regard the hard voting and soft voting scheme is applied to achieve the classification results. 

## Installation
```python
pip install prosemble
```

## How to use
To exemplify refer to the test_iris_.py and test_wdbc_.py in the examples folder


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



