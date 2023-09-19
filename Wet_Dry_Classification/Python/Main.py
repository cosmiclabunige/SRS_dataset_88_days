import pickle
import matplotlib.pyplot as plt
import numpy as np
from Classifiers_Classes import ANNClass, CNNClass, ADAClass
from Results_Class import ResultsClass
from pathlib import Path 
#################################################
#                 Dataset Choice                #
#################################################

dishRadius = "85" # dish diameter 
bool_filterData = False # weather filtering the data or not 
offlineDays = 5 # number of days taken for the offline training
timeWindow = 30
datasetPath = Path("Wet_Dry_Classification") / "Dataset"
resultsPath =  Path("Wet_Dry_Classification") / "Results"
imagesPath = Path("Wet_Dry_Classification") / "Images"
notRainyDays = [7, 13, 16, 17, 26, 27, 28, 38, 39, 51, 54, 55, 56, 59, 62, 63, 64, 71, 72, 81, 84, 128, 141, 167, 170, 180,
                      188, 190, 191, 192, 195, 206, 207, 210, 212, 214, 215, 220, 221, 222, 228, 236, 238, 240, 244, 246, 249, 253,
                      256, 257]

rainyDays = [4, 5, 6, 8, 14, 21, 58, 60, 61, 73, 86, 89, 97, 127, 133, 136, 149, 151, 182, 184, 185, 189, 202,
                   229, 235, 239, 242, 247, 255, 261, 265, 287, 297, 319, 322, 326, 328, 337]



ADA = ADAClass(datasetPath=datasetPath, resultsPath=resultsPath, dishRadius=dishRadius, offlineDays=offlineDays, notRainyDays=notRainyDays,
               rainyDays=rainyDays, timeWindow=timeWindow)
ADA.Online()

# CNN = CNNClass(datasetPath=datasetPath, resultsPath=resultsPath, dishRadius=dishRadius, offlineDays=offlineDays, notRainyDays=notRainyDays,
#                rainyDays=rainyDays, bool_filterData=bool_filterData, timeWindow=timeWindow, 
#                 _filters=[(4, 8), (8, 8), (8, 16), (4, 8, 16), (8, 8, 8), (8, 16, 32)], _kernel_size=[3, 5, 7], _learning_rate=5e-4, _rolls=3)
# CNN.Online()

# ANN = ANNClass(datasetPath=datasetPath, resultsPath=resultsPath, dishRadius=dishRadius, offlineDays=offlineDays, notRainyDays=notRainyDays,
#                rainyDays=rainyDays, bool_filterData=bool_filterData, timeWindow=timeWindow, 
#                _neurons=[25, 50, 75], _lam=[10 ** i for i in range(-4, 5)])
# ANN.Online()

RC = ResultsClass(resultsPath,
                  saveImagesPath=imagesPath,
                  dishRadius=dishRadius,
                  whichModels=["ADA"])


RC.plot_classification_results("ADA", 0)

















