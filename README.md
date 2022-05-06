# ensemble-lvq
Ensemble lvq is based on the idea of transfer learning cum ensemble technique with lvq models.

# why?
The idea explores the possibility of harnessing transfer learning cum ensemble learning in lvq models. In this regard the hard voting and soft voting scheme is applied to achieve the classification results. 

# How to use
To use ensemble-lvq, you must access learned prototypes from different models of lvq as transfered prototypes. So for a data set to be classified, apply the methods in ```hybrid.py``` script to obtain the predicated labels and the prediction probabilities as well as the votes which supports the outcomes. To exemplify, refer to ```test_iris_.py```  and ```test_wdbc_.py```. To try a simulation of performance accuracy refer to ```test_iris1.py``` and ```test_wdbc1.py```. Please note that optimal performance in this regard is based on how well either your transfered prototypes or pre-trained models are trained. In the case of soft voting scheme, additionally, we have to optimise the classification label certainty hyperparamter(m) which is linked to the reliablity of the model in use. Inorder to optimise ```m```, refer to the  method  introduced in ```opt_m.py```.



# simulated results
Below is a simulation of the technique with the iris data set using the ```test_iris1.py``` based on transferred learned prototypes from ```GLVQ```, ```GMLVQ``` and ``` CELVQ``` models.


![smh2](https://user-images.githubusercontent.com/82911284/166449818-27cad0f9-2b59-4159-9bbc-635d8b2a3edb.png)


Below is a simulation of the technique with the wdbc data set using the ```test_wdbc1.py``` based on transferred transferred pre-trained model prototypes from ```GLVQ```, ```GMLVQ``` and ```CELVQ``` models.

![wdbc1](https://user-images.githubusercontent.com/82911284/166610528-93c3dd63-fdd4-4b53-89e5-bce4bf932fe5.png)

Below is a plot which explains why an optimised hyperparameter(m) for the computation of the classification label security must used based on the reliability of the model.  The reliability of the model is thus linked to the performance evaluation measures used. 


Below is a plot of an optimal search of the hyperparater m based on the method in ```opt_m.py```

# Questions?
For questions contact abekaotoo@gmail.com


