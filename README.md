# Dataset description 
The pickle file contains the datasets concerning the 88 days of rainfall measurements described in the article "An Online Training Procedure for Rain Detection
Models Applied to Satellite Microwave Links". The data have been collected using sensors mounted on two parabolic dishes having a diameter of 60cm and 85cm, respectively. 

Open the datasets using the pickle library for Python with the following commands:

```ruby
import pickle as pk 

datasetPath = "SRS_dataset_88days.pkl"  
with open(datasetPath, "rb") as f:  
   dataset = pk.load(f) 
   f.close()  
```

The *dataset* is a dictionary containing the 88 days SRS measurements expressed in mV for both dishes, the 88 days TBRG measurements expressed in mm/min, and the timestamps of the measurements.

The days of observations are listed in the table which summarizes the dataset, displaying the ID assigned to each day, the total accumulation of rain in mm (h), the maximum rainfall
intensity in mm/h (max RI) observed by the TBRG, the number of rainy minutes (d), the freezing level in m. The elevation angle of the sensors with respect to the Turksat dataset is 29.2048°.

![image](https://github.com/cosmiclabunige/SRS_dataset_88_days/assets/114477377/9bbb4f3d-c98d-43a9-9fb2-d91c373d428c)

# Experiments conducted in the article

The anomaly detection algorithm (ADA) was trained offline by using the first non-rainy days of the dataset, i.e. days 4, 6, 8, 9, 11. 
The machine learning algorithms (MLAs) were trained on 10 days, the same 5 of ADA and 5 containing rain observations, i.e. days 1, 2, 3, 5, 7.
Details about the offline training procedure are provided in the article.

## Algorithms

### Anomaly Detection Algorithm (ADA)
It works as an anomaly detection algorithm where we consider as normal data the non-rainy observations and as anomalies the rainy ones. Thus the training dataset consists of only non-rainy data used to compute some thresholds on the normal data to identify the anomalies. In particular, ADA relies on the heuristic evaluation of three parameters: the minimum signal power $P_R^{min}$, the maximum difference between two consecutive observations $\Delta P_R$, and the standard deviation $stdev$, which are computed over a window of previous observations (further details are provided in the article). 

A new SRS observation (in dBm) $x_t$ is classified as follows, based on the previous label $y_{t-1} \in [True; False]$ indicating wether it was raining or not: 

<span style="color: red;">first_cond</span> = (($x_t$ - $x_{t-1}$) $<-\Delta P_R$ ) | ($x_t < P_R^{min}$)

<span style="color: red;">second_cond</span> = std($[x_{t-9}, x_{t-8}, x_{t-7}, x_{t-6}, x_{t-5}, x_{t-4}, x_{t-3}, x_{t-2}, x_{t-1}, x_{t}]$) > $stdev$

<span style="color: red;">third_cond</span> = std($[x_{t-9}, x_{t-8}, x_{t-7}, x_{t-6}, x_{t-5}, x_{t-4}, x_{t-3}, x_{t-2}, x_{t-1}, x_{t}]$) < $stdev$/4

<span style="color: red;">fourth_cond</span> = std($[x_{t-19},x_{t-18},x_{t-17},x_{t-16},x_{t-15},x_{t-14},x_{t-13},x_{t-12},x_{t-11},x_{t-10},x_{t-9}, x_{t-8}, x_{t-7}, x_{t-6}, x_{t-5}, x_{t-4}, x_{t-3}, x_{t-2}, x_{t-1}, x_{t}]$) < $stdev$/2

if (first_cond | second_cond) and not $y_{t-1}$:
   
&emsp; $y_{t}$ = True

elif (third_cond | fourth_cond) and $y_{t-1}$:

&emsp; $y_{t}$ = False

#### Computation of parameters

The computation of the three parameters is depicted in the following procedure, where the input $\widetilde{\mathcal{T}}$ represents the dataset containing the observations $x_t$.

![ADAParametersUpdating](https://github.com/cosmiclabunige/SRS_dataset_88_days/assets/114477377/a317de05-bd86-465b-8894-8dfd6bac99c1)




### Machine Learning Algorithms (MLAs) 
Two MLAs have been adopted along with the ADA: one artificial neural network (ANN) and one convolutional neural network (CNN). The ANN consists of only one hidden layer, while the CNN consists of two convolutional layers, each followed by a pooling layer with a size of 2. Moreover, a fully connected layer with 10 neurons has been stacked after the second pooling. The ReLU function has been adopted as an activation function in the convolutional and fully connected layer. Details about the hyperparameters list of the two MLAs are provided in the article. 

The ANN requires an array of features as input. Thus, for each observation $x_t$, 8 features based on the 29 previous observations and the current one have been extracted (details in the article). While the CNN receives as input the time series consisting of the current observation $x_t$ and the previous 29, without requiring any features extraction procedure. 

#
The algorithms have been tested on the other 78 days not used during the offline training, i.e. 45 non-rainy days and 33 rainy days. 

# Example of testing code
You can find the code to train the models and visualize the results here: https://colab.research.google.com/drive/1dI-YVUWwamK7uMeWmz-XSzjyuDcetw9g#scrollTo=Oe1pdErmsruG.
Inside the *Main.py*, one can change the radius of the dish (either 60cm or 85cm), the classifier choosing between ANN, CNN, or ADA, and the event one wants to test from 0 to 77 (corresponding to one day in the test set).
