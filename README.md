# Ensemble-lvq
A Prototype based ML project implementation which utilizes learned protototypes from LVQs in ensemble learning by soft and hard voting

# why?
In ML the convention has been to save a trained model for future use or deployment. An alternative way would be to access learned prototypes from pre-trained models
for use in deployemnt.

This project implements the harnessing of pre-trained prototypes in ensemble learning with lvq models. In this regard the hard voting and soft voting scheme is applied to achieve the classification results. 

# How to use
To use ensemble-lvq, you must access learned prototypes from of lvq models. So for a data set to be classified, apply the methods in ```hybrid.py``` to obtain the predicated labels and the prediction probabilities as well as the votes which supports the outcomes. To exemplify, refer to ```test_iris_.py```  and ```test_wdbc_.py```. 


# simulated results
Below is a simulation of the technique with the iris data set using the ```test_iris1.py``` based on transferred learned prototypes from ```GLVQ```, ```GMLVQ``` and ``` CELVQ``` models.


![smh2](https://user-images.githubusercontent.com/82911284/166449818-27cad0f9-2b59-4159-9bbc-635d8b2a3edb.png)


Below is a simulation of the technique with the wdbc data set using the ```test_wdbc1.py``` based on transferred transferred pre-trained model prototypes from ```GLVQ```, ```GMLVQ``` and ```CELVQ``` models.

![wdbc1](https://user-images.githubusercontent.com/82911284/166610528-93c3dd63-fdd4-4b53-89e5-bce4bf932fe5.png)



# Questions?
For questions contact abekaotoo@gmail.com


