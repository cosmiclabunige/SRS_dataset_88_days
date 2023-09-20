import pickle
import matplotlib.pyplot as plt
import numpy as np
from Classifiers_Classes import ANNClass, CNNClass, ADAClass
from Results_Class import ResultsClass
from pathlib import Path 

# dish diameter 
dishDiameter = "85" 
# wether filtering the SRS signal or not
bool_filterData = False
# Number of days taken for the offline training. 
# For ANN and CNN 5 days containing rain events and 5 days not containing rain events will be taken.
# For ADA only 5 days of non rainy events will be taken. 
offlineDays = 5 
# Time window to compute the features for the ANN or to extract time series for CNN
timeWindow = 30
# Dataset path 
datasetPath = Path("Dataset") / "SRS_dataset_88days.pkl"
# Path for saving the classification results
resultsPath =  Path("Results")
# Path for saving the boxplots, the barplots, and the classification plots 
imagesPath = Path("Images") 


# Train offline and test online the models. If already trained comments these lines, or just comments the algorithm you don't want to train
# ADA = ADAClass(datasetPath=datasetPath, resultsPath=resultsPath, dishDiameter=dishDiameter, offlineDays=offlineDays, bool_filterData=bool_filterData)
# ADA.Online()

# CNN = CNNClass(datasetPath=datasetPath, resultsPath=resultsPath, dishDiameter=dishDiameter, offlineDays=offlineDays, bool_filterData=bool_filterData, timeWindow=timeWindow, 
#                 _filters=[(4, 8), (8, 8), (8, 16), (4, 8, 16), (8, 8, 8), (8, 16, 32)], _kernel_size=[3, 5, 7], _learning_rate=5e-4, _rolls=3)
# CNN.Online()

ANN = ANNClass(datasetPath=datasetPath, resultsPath=resultsPath, dishDiameter=dishDiameter, offlineDays=offlineDays, bool_filterData=bool_filterData, timeWindow=timeWindow, 
               _neurons=[25, 50, 75], _lam=[10 ** i for i in range(-4, 5)])
ANN.Online()


### Compute and visualize the results. If you want to only train the models comments these lines
# RC = ResultsClass(resultsPath=resultsPath,
#                   saveImagesPath=imagesPath,
#                   dishDiameter=dishDiameter,
#                   whichModels=["ADA"])

# RC.plot_classification_results("ADA", 0)

















