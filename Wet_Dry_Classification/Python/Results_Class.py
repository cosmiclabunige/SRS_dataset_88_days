import os
import pickle as pk
import numpy as np
from glob import glob
from pathlib import Path
from sklearn.metrics import confusion_matrix, recall_score
import matplotlib.pyplot as plt
import matplotlib


def compute_metrics(y_true, y_pred):
    """
    Functions to compute the three metrics for each day. If a day does not contain rain events the recall and hm are = -1
    """
    Spec = []
    Rec = []
    Hm = []
    for i in range(len(y_true)):
        tn, fp, fn, tp = confusion_matrix(y_true=y_true[i], y_pred=y_pred[i], labels=[0, 1]).ravel()
        spe = tn / (tn + fp)
        if np.count_nonzero(y_true[i] == 1) == 0:
            rec = -1
            hm = -1
        else:
            rec = tp / (tp + fn)
            hm = 2 * spe * rec / (spe + rec)
        Spec.append(spe)
        Rec.append(rec)
        Hm.append(hm)

    spec = np.asarray(Spec)
    rec = np.asarray(Rec)
    hm = np.asarray(Hm)
    return spec, rec, hm


def compute_outliers_percentage(x):
    """
    Compute the percentages of data ``x`` above the upper wisker and below the lower wisker.
    """
    # First quartile
    Q1 = np.percentile(x, 25) 
    # Third quartile
    Q3 = np.percentile(x, 75)
    # Interquartile difference
    IQR = Q3 - Q1
    # Compute the two wiskers
    lower_whisker = Q1 - 1.5 * IQR
    higher_whisker = Q3 + 1.5 * IQR
    # Count data points below the lower whisker and above the upper
    count_below_lower_whisker = np.sum(x < lower_whisker)
    count_above_higher_whisker = np.sum(x > higher_whisker)
    # Calculate the percentages
    percentage_below_lower_whisker = (count_below_lower_whisker / len(x)) * 100
    percentage_above_upper_whisker = (count_above_higher_whisker / len(x)) * 100
    return percentage_below_lower_whisker, percentage_above_upper_whisker


class ResultsClass():
    """
    Class to visualize for each algorithm: 

    - the bloxplot and barplots of metrics for non-rainy (Specificity) and rainy days (Specificity, Recall, and Harmonic Mean)
    - the results of the classification.

    The input parameters are:

    - ``resultsPath``: the path where the results of the algorithms classification are saved.
    - ``saveImagesPath``: the path where the images are saved
    - ``dishDiameter``: the diameter of the dish whose results one wants to visualize the results.
    - ``whichModels``: list of the models whose one wants to visualize the results. The models must be trained in advance.

    """
    def __init__(self,
                 resultsPath: str,
                 saveImagesPath : str,
                 dishDiameter: str,
                 whichModels: list):
        # create the folder for the images if it does not exist
        if not os.path.exists(saveImagesPath):
            os.makedirs(saveImagesPath)

        # Check if the required model has been trained
        files = []
        for f in resultsPath.iterdir():
            if f.is_file():
                files.append(str(f))
        for m in whichModels:
            assert any(m in s for s in files), f"The model {m} has not been trained"
        
        # Statement of all the variables needed for the computation
        # Specificity for non-rainy days 
        self.__Spe_noRain = [] 
        # Specificity for rainy days
        self.__Spe_rain = [] 
        # Recall for only rainy days
        self.__Rec = [] 
        # Harmonic mean for only rainy days
        self.__HM = [] 
        # List of predicted values
        self.__y_pred = [] 
        # list of true values
        self.__y_true = [] 
        # Images path
        self.__resPath = saveImagesPath 
        # Dish diameter
        self.__dishDiameter = dishDiameter
        # Models to evaluate
        self.__models = whichModels
        # Number of classifiers beeing evaluated
        self.__numOfClassifiers = len(whichModels)
        # Specificity temporary list
        Spe = []
        # Recall temporary list 
        Rec = []
        # Harmonic temporary list 
        HM = [] 
        # For each model compute the metrics. If a day does not contain rainy events the compute metrics function returns -1 for Rec and HM
        for mo in whichModels:
            # File path containing the results
            model = mo + f"_Predictions_{dishDiameter}"
            file = Path(resultsPath) / model
            # Check if the file exists
            assert file.exists(), "The required file does not exist check the dish radius" 
            # Open the results file
            with open(file, 'rb') as f:
                results = pk.load(f)
                f.close()
            # Extract the dates of the days
            self.__dates = results["dates"]
            # Extract the true labels  
            y_true = results["y_true"] 
            # Extract the predicted labels
            y_pred = results["y_pred"] 
            # Append to the corresponding list
            self.__y_pred.append(y_pred) 
            self.__y_true.append(y_true)
            # Extract the tbrg measurements
            self.__TBRG = results["TBRG"] 
            # Extract the SRS signals
            self.__SRS = results["SRS"] 
            # Compute the metrics
            spe, rec, hm = compute_metrics(y_true, y_pred) 
            # Append the metrics to the corresponding list
            Spe.append(spe)
            Rec.append(rec)
            HM.append(hm)
        
        # Properly assign the three metrics to the class variables for the further processing
        for S, R, H in zip(Spe, Rec, HM):
            # Temporay lists of the metrics
            Spe_noRain_tmp = []
            Spe_rain_tmp = []
            Rec_tmp = []
            HM_tmp = []
            for s,r,h in zip(S,R,H):
                # If the recall from the previous computation is -1 then append only the Specificity
                if r == -1:
                    Spe_noRain_tmp.append(s)
                else:
                    Spe_rain_tmp.append(s)
                    Rec_tmp.append(r)
                    HM_tmp.append(h)
            # append the metrics to the corresponding class variable
            self.__Spe_noRain.append(np.asarray(Spe_noRain_tmp))
            self.__Spe_rain.append(np.asarray(Spe_rain_tmp))
            self.__Rec.append(np.asarray(Rec_tmp))
            self.__HM.append(np.asarray(HM_tmp))
        
        # Call the function for computing the boxplot and the barplots of the metrics for non-rainy days, rainy days
        self.__boxplot_noRain()
        self.__boxplot_rain()
        self.__barplots_metrics_TBRG_intervals()
    
    def __boxplot_noRain(self):
        """
        Function to compute the boxplot on the Specificity metric in non-rainy days.
        """
        # Position of the boxplots in the x-axis
        positions = np.arange(1, self.__numOfClassifiers+1, 1)
        # Labels that will be visualized on the x-axis. They contain the metric name and the model name
        labels = []
        for i in self.__models:
            stri = "Spe\n" + i
            labels.append(stri)
        fig, ax = plt.subplots(figsize=(16, 8))
        # Properties of the boxplot. Check documentation at https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.boxplot.html
        boxprops = dict(linewidth=4, color='blue')
        medianprops = dict(linewidth=4, color='blue')
        whiskerprops = dict(linewidth=3, color='blue')
        flierprops = dict(marker='o', markerfacecolor='blue', markersize=7,
                        markeredgecolor='none')
        capprops = dict(linewidth=3, color='blue')
        meanprops = dict(marker='D', markerfacecolor='blue', markersize=10,
                        markeredgecolor='none')
        # Plot the Specificity boxplot for each model
        ax.boxplot(self.__Spe_noRain, positions=positions, labels=labels, boxprops=boxprops, medianprops=medianprops,
                    whiskerprops=whiskerprops, flierprops=flierprops, capprops=capprops, meanprops=meanprops,
                    showmeans=True, showfliers=False)
        # Properties of the plot
        plt.yticks(np.arange(0.50, 1.01, 0.05))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
        ax.text(np.mean(positions), 1.04, self.__dishDiameter + "cm", color='black', fontsize=20, fontweight='bold')
        ax.tick_params(axis='x', labelsize=22)
        ax.tick_params(axis='y', labelsize=24)
        fig.tight_layout()
        # Save the plot as a pdf
        boxplotpath = self.__resPath / f"Boxplot_Non_Rainy_{self.__dishDiameter}.pdf"
        plt.savefig(str(boxplotpath), format='pdf')

        # Print the percetanges of data below the lower wisker and above the upper wisker for each evaluated model
        for i, m in enumerate(self.__models):
            data = self.__Spe_noRain[i]
            pblw, pauw = compute_outliers_percentage(data)
            print(f"MODEL {m} Specificity No Rain: percentage below lower wisk {pblw:.2f}, percentage above higher wisk {pauw:.2f}")

    def __boxplot_rain(self):
        """
        Function to compute the boxplot on the Specificity, Recall, and HM metrics in rainy days.
        """
        # Positions to plot the boxplots for each model
        r = np.arange(1, 3.2 * self.__numOfClassifiers-1, 3.2)
        x = np.asarray([i for i in r])
        w1 = 0.9
        w2 = 0.9
        positionsSpec = x - w1
        positionsRec = x
        positionsHM = x + w2
        # Labels that will be visualized on the x-axis. They contain the metrics name and the models name
        labelsSpec = ["Spe"]*self.__numOfClassifiers
        labelsRec = []
        for i in self.__models:
            stri = "Rec\n" + i
            labelsRec.append(stri)
        labelsHm = ["HM"]*self.__numOfClassifiers

        fig, ax = plt.subplots(figsize=(16, 8))
        # ########## SPECIFICITY  ##########
        # Properties of the specificity boxplots. Check documentation at https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.boxplot.html
        boxprops = dict(linewidth=4, color='blue')
        medianprops = dict(linewidth=4, color='blue')
        whiskerprops = dict(linewidth=3, color='blue')
        flierprops = dict(marker='o', markerfacecolor='blue', markersize=7,
                        markeredgecolor='none')
        capprops = dict(linewidth=3, color='blue')
        meanprops = dict(marker='D', markerfacecolor='blue', markersize=10,
                        markeredgecolor='none')
        # Plot the boxplot of the Specificity for each model
        ax.boxplot(self.__Spe_rain, positions=positionsSpec, boxprops=boxprops, medianprops=medianprops,
                labels=labelsSpec,
                whiskerprops=whiskerprops, flierprops=flierprops, capprops=capprops, meanprops=meanprops,
                showmeans=True, showfliers=False)
        
        # ########## RECALL  ##########
        # Properties of the recall boxplots.
        boxprops = dict(linewidth=4, color='black')
        flierprops = dict(marker='o', markerfacecolor='black', markersize=7,
                        markeredgecolor='none')
        medianprops = dict(linewidth=4, color='black')
        whiskerprops = dict(linewidth=3, color='black')
        capprops = dict(linewidth=3, color='black')
        meanprops = dict(marker='D', markerfacecolor='black', markersize=10,
                        markeredgecolor='none')
        # Plot the boxplot of the Recall for each model
        ax.boxplot(self.__Rec, positions=positionsRec, labels=labelsRec, boxprops=boxprops, medianprops=medianprops,
                    whiskerprops=whiskerprops, flierprops=flierprops, capprops=capprops, meanprops=meanprops,
                    showmeans=True, showfliers=False)
        # ########### HARMONIC MEAN ###########
        # Properties of the harmonic mean boxplots.
        boxprops = dict(linewidth=4, color='green')
        flierprops = dict(marker='o', markerfacecolor='green', markersize=7,
                        markeredgecolor='none')
        medianprops = dict(linewidth=4, color='green')
        whiskerprops = dict(linewidth=3, color='green')
        capprops = dict(linewidth=3, color='green')
        meanprops = dict(marker='D', markerfacecolor='green', markersize=10,
                        markeredgecolor='none')
        # Plot the boxplot of the HM for each model
        ax.boxplot(self.__HM, positions=positionsHM, labels=labelsHm, boxprops=boxprops, medianprops=medianprops,
                whiskerprops=whiskerprops, flierprops=flierprops, capprops=capprops, meanprops=meanprops,
                showmeans=True, showfliers=False)

        # ######## GENERAL ###########
        # Change the color of the plot background for better visualization
        pos = np.arange(-0.6, 3.2*self.__numOfClassifiers, 3.2) 
        for xmin, xmax in zip(pos[::2], pos[1::2]):
            ax.axvspan(xmin, xmax, color='gainsboro')
        # Properties of the plot
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
        plt.yticks(np.arange(0.4, 1.01, 0.1))
        plt.xlim([pos[0], pos[-1]])
        ax.text(np.mean(positionsRec), 1.04, self.__dishDiameter + "cm", color='black', fontsize=20, fontweight='bold')
        ax.tick_params(axis='x', labelsize=22)
        ax.tick_params(axis='y', labelsize=24)
        fig.tight_layout()
        # Save the plot as a pdf
        boxplotpath = self.__resPath / f"Boxplot_Rainy_{self.__dishDiameter}.pdf"
        plt.savefig(str(boxplotpath), format='pdf')
        # Print the percetanges of data below the lower wisker and above the upper wisker for each evaluated model and for each metric
        for i, m in enumerate(self.__models):
            spe = self.__Spe_rain[i]
            rec = self.__Rec[i]
            hm = self.__HM[i]
            pblw_spe, pauw_spe = compute_outliers_percentage(spe)
            pblw_rec, pauw_rec = compute_outliers_percentage(rec)
            pblw_hm, pauw_hm = compute_outliers_percentage(hm)
            print(f"MODEL {m} Specificity Rain: percentage below lower wisk {pblw_spe:.2f}, percentage above higher wisk {pauw_spe:.2f}")
            print(f"MODEL {m} Recall Rain: percentage below lower wisk {pblw_rec:.2f}, percentage above higher wisk {pauw_rec:.2f}")
            print(f"MODEL {m} HM Rain: percentage below lower wisk {pblw_hm:.2f}, percentage above higher wisk {pauw_hm:.2f}")


    def __barplots_metrics_TBRG_intervals(self):
        """
        Function to plot the bars representing the metrics' values in different interval of rain intensity (RI).
        Specificically the intervals are RI < 2 mm/min, 2 mm/min <=RI < 6 mm/min, and RI >= 6 mm/min.
        """
        # List to compute the Specificity metric in case of not-rainy events in the rainy days
        hist0 = []
        # List to compute the Recall metric in the first interval
        hist1 = []
        # List to compute the Recall metric in the second interval
        hist2 = []
        # List to compute the Recall metric in the third interval
        hist3 = []
        # List to compute the HM metric in the case of rainy days
        histHM = []
        # Name of the models for the x-axis
        labels = self.__models
        # Loop to divide the predicted labels based on the intervals for each model
        for i in range(len(self.__models)):
            y_pred = self.__y_pred[i]
            # List that contain the predicted labels in case of not-rainy events during rainy days for each model
            yPred0 = []
            # List that contain the predicted labels for the first interval for each model
            yPred1 = []
            # List that contain the predicted labels for the second interval for each model
            yPred2 = []
            # List that contain the predicted labels for the third interval for each model
            yPred3 = []
            # List that contain the predicted labels for the all the rainy events for each model
            yPredAll = []
            # Loop over all the days
            for j, x in enumerate(self.__TBRG):
                x = np.asarray(x)
                # Skip the day if it is not a rainy day 
                if np.count_nonzero(x>0)==0:
                    continue
                # Find the indexes of the TBRG measurements in each interval
                indTBRG0 = np.where(x==0.0)[0]
                indTBRG1 = np.where((x > 0) & (x < 2))[0]
                indTBRG2 = np.where((x >=2) & (x < 6))[0]
                indTBRG3 = np.where(x >= 6)[0]
                tmp = y_pred[j]
                # Extract the predicted labels in each interval
                yPred0_tmp = np.asarray([tmp[ii] for ii in indTBRG0])
                yPred1_tmp = np.asarray([tmp[ii] for ii in indTBRG1])
                yPred2_tmp = np.asarray([tmp[ii] for ii in indTBRG2])
                yPred3_tmp = np.asarray([tmp[ii] for ii in indTBRG3])
                # Append to the proper list only if the label array is not empty
                if len(yPred0_tmp)>0:
                    yPred0.extend(yPred0_tmp)
                if len(yPred1_tmp)>0:
                    yPred1.extend(yPred1_tmp)
                    yPredAll.extend(yPred1_tmp)
                if len(yPred2_tmp)>0:
                    yPred2.extend(yPred2_tmp)
                    yPredAll.extend(yPred2_tmp)
                if len(yPred3_tmp)>0:
                    yPred3.extend(yPred3_tmp)
                    yPredAll.extend(yPred3_tmp) 
            # Compute the metrics for each interval
            spec = recall_score(np.asarray([0] * len(yPred0)), yPred0, pos_label=0)
            recAll = recall_score(np.asarray([1]*len(yPredAll)), yPredAll)
            rec1 = recall_score(np.asarray([1]*len(yPred1)), yPred1)
            rec2 = recall_score(np.asarray([1]*len(yPred2)), yPred2)
            rec3 = recall_score(np.asarray([1]*len(yPred3)), yPred3)
            # Append the metrics to the proper list
            hist0.append(spec)
            hist1.append(rec1)
            hist2.append(rec2)
            hist3.append(rec3)
            histHM.append(2*spec*recAll/(spec+recAll))
        
        # ########## RECALL ############
        # Positions of the bars and their width
        x = np.arange(len(labels))
        width = 0.25  
        fig, ax = plt.subplots(figsize=(14, 6.5))
        # Round the metrics to have only one decimal digit
        hist1 = [round(item*1000)/10 for item in hist1]
        hist2 = [round(item * 1000) / 10 for item in hist2]
        hist3 = [round(item * 1000) / 10 for item in hist3]
        # Temporary list to set the limits on the y-axis in the plot
        tmp = []
        tmp.extend(hist1)
        tmp.extend(hist2)
        tmp.extend(hist3)
        ma = max(tmp) + 3
        mi = min(tmp)-10
        # Plot the bars concerning the Recall for each interval
        rects1 = ax.bar(x - width, hist1, width, label='TBRG<2 mm/h')
        rects2 = ax.bar(x, hist2, width, label='2<=TBRG<6 mm/h')
        rects3 = ax.bar(x + width, hist3, width, label='TBRG>=6 mm/h')
        # Properties of the plot
        ax.set_xticks(x, labels)
        ax.tick_params(labelsize=24)
        ax.legend(loc='lower left', fontsize=24)
        ax.text(np.mean(x), ma+0.3, self.__dishDiameter+"cm", color='black', fontsize=20, fontweight='bold')
        ax.bar_label(rects1, padding=3, fontsize=20, rotation=0)
        ax.bar_label(rects2, padding=3, fontsize=20, rotation=0)
        ax.bar_label(rects3, padding=3, fontsize=20, rotation=0)
        fig.tight_layout()
        plt.ylim([mi, ma])
        plt.yticks([])
        # Save the plot in a pdf
        filename = self.__resPath / f"Rec_{self.__dishDiameter}.pdf"
        plt.savefig(filename, format='pdf')

        # ########## SPECIFICITY AND HM ############
        # Thw width of the bars
        width = 0.4  
        fig, ax = plt.subplots(figsize=(14, 6.5))
        hist0 = [round(item * 1000) / 10 for item in hist0]
        histHM = [round(item * 1000) / 10 for item in histHM]
        # Plot the bars for Spec and HM
        rectsSpec = ax.bar(x-width/2, hist0, width, label='TBRG=0 mm/h')
        rectsHM = ax.bar(x+width/2, histHM, width, color='green')
        ax.set_xticks(x, labels)
        # Compute the limits on the y-axis
        tmp = []
        tmp.extend(hist0)
        tmp.extend(histHM)
        ma = max(tmp) + 3
        mi = min(tmp)-10
        # Plot properties
        ax.tick_params(labelsize=24)
        ax.legend(loc='lower right', fontsize=24)
        ax.text(np.mean(x), ma+0.3, self.__dishDiameter+"cm", color='black', fontsize=20, fontweight='bold')
        ax.bar_label(rectsSpec, padding=3, fontsize=24, rotation=0)
        ax.bar_label(rectsHM, padding=3, fontsize=24, rotation=0)
        fig.tight_layout()
        plt.ylim([mi, ma])
        plt.yticks([])
        # Save the plot in a pdf
        filename = self.__resPath / f"Spe_HM_{self.__dishDiameter}.pdf"
        plt.savefig(filename, format='pdf')


    def plot_classification_results(self, model, index):
        """
        Function to plot the results of the classification. The plot consists of a day of observations with the following notation:
        - True Negative (TN) in blue: non-rainy observations correctly classified
        - False Positive (FP) in purple: non-rainy observations incorrectly classified
        - True Positive (TP) in green: rainy observation correctly classified
        - False Negative (FN) in red: rainy observation incorrectly classified

        The function receives as input two parameters:
        - ``model``: the model to be tested
        - ``index``: the index of the day that one wants to visualize   
        """
        # Check if the index does not exceed the number of possible days and if the model exists
        assert index < len(self.__y_pred[0]), f"The day selected does not exist, the maximum index is {len(self.__y_pred[0])-1}"
        assert model in self.__models, f"The model {model} has not been trained or it does not exist"
        # Close all the previous plots
        plt.close('all')
        # Extract the index of the model
        i = self.__models.index(model)
        # Extract the predicted and true labels of the required model and the required day
        y_pred = self.__y_pred[i][index]
        y_true = self.__y_true[i][index]
        # Take the TBRG measurements, SRS signals, and the date of the required day 
        tbrg = self.__TBRG[index]
        srs = self.__SRS[index]
        day = self.__dates[index]
        # Compute the Specificity to be visualized on the plot
        spec = recall_score(y_true, y_pred, pos_label=0)
        # Extract indeces of rainy events
        tbrg = np.asarray(tbrg)
        indTBRG = np.where(tbrg > 0)[0]
        # Compute the indeces of correct predictions
        indexOk = np.where(y_pred == y_true)[0]
        # Compute the indeces of incorrect predictions
        indexNotOk = np.where(y_pred != y_true)[0]
        # Initialize the plot
        fig, ax = plt.subplots(figsize=(19.2, 10.8))
        ax.tick_params(labelsize=24)
        # Dimension of the plot fontisizes
        d = 30
        if len(indTBRG) > 0:
            # If a day contain rainy events compute the recall and the hm
            rec = recall_score(y_true, y_pred)
            hm = 2*spec*rec/(spec+rec)
            indTBRG = np.asarray(indTBRG)
            # Take the values of the TBRG signals when differ from 0
            xTBRGplot = np.asarray([i for i in tbrg if i > 0])
            # Indeces of TP, TN, FP, FN
            indexTP = np.asarray([iii for iii in indexOk if iii in indTBRG])
            indexTN = np.asarray([iii for iii in indexOk if iii not in indTBRG])
            indexFN= np.asarray([iii for iii in indexNotOk if iii in indTBRG])
            indexFP = np.asarray([iii for iii in indexNotOk if iii not in indTBRG])
            # Plot the TBRG as vertical lines only when fiffers from 0 
            ax2 = ax.twinx()
            ax2.tick_params(labelsize=24)
            ax.vlines(indTBRG, ymin=[0] * len(indTBRG), ymax=xTBRGplot, color="cyan", linewidth=4)
            ax.set_ylabel("TBRG [mm/h]", fontsize=32)
            # Plot the classified data giving the color based on the classification results
            xSRStmp = [srs[ii] for ii in indexTN]
            ax2.scatter(indexTN, xSRStmp, s=d, facecolors='none', edgecolors='blue', marker='s')
            xSRStmp = [srs[ii] for ii in indexFP]
            ax2.scatter(indexFP, xSRStmp, s=d, facecolors='none', edgecolors='purple', marker='s')
            xSRStmp = [srs[ii] for ii in indexTP]
            ax2.scatter(indexTP, xSRStmp, s=d, facecolors='none', edgecolors='green', marker='o')
            xSRStmp = [srs[ii] for ii in indexFN]
            ax2.scatter(indexFN, xSRStmp, s=d, facecolors='none', edgecolors='red', marker='o')
            ax2.set_ylabel("SRS [dBm]", fontsize=32)
            # Visualize the metric on the plot
            ax.set_xlabel("Spec= {:.4f}   Rec= {:.4f}   HM= {:.4f}".format(spec, rec, hm), fontsize=32)
            # Properties of the plot
            ax.spines['top'].set_color('black')
            ax2.spines['top'].set_color('black')
            ax.spines['bottom'].set_color('black')
            ax2.spines['bottom'].set_color('black')
            ax.spines['left'].set_color('black')
            ax2.spines['left'].set_color('black')
            ax.spines['right'].set_color('black')
            ax2.spines['right'].set_color('black')
            ax.set_facecolor('xkcd:white')
            ax2.set_facecolor('xkcd:white')
            pos = 'lower left'
            ax2.legend(["True Negative", "False Positive", "True Positive", "False Negative"], scatterpoints=1, labelspacing=1,
                handletextpad=0.1, handlelength=0.8, markerscale=2, fontsize=24, loc=pos)
        else:
            # Same procedure but applied on non-rainy days, thus only the TN and Fp are computed
            indexTN = np.asarray([iii for iii in indexOk])
            indexFP = np.asarray([iii for iii in indexNotOk])
            xSRStmp = [srs[ii] for ii in indexTN]
            ax.scatter(indexTN, xSRStmp, s=d, facecolors='none', edgecolors='blue', marker='s')
            xSRStmp = [srs[ii] for ii in indexFP]
            ax.scatter(indexFP, xSRStmp, s=d, facecolors='none', edgecolors='purple', marker='s')
            ax.set_ylabel("SRS [dBm]", fontsize=32)
            ax.set_xlabel("Specificity = {:.4f}".format(spec), fontsize=32)
            ax.spines['top'].set_color('black')
            ax.spines['bottom'].set_color('black')
            ax.spines['left'].set_color('black')
            ax.spines['right'].set_color('black')
            ax.set_facecolor('xkcd:white')
            pos = 'lower left'
            ax.legend(["True Negative", "False Positive"], scatterpoints=1, labelspacing=1,
                handletextpad=0.1, handlelength=0.8, markerscale=2, fontsize=24, loc=pos)
        # Visualize the plot
        plt.xlim([-1, 1440])
        plt.yticks(fontsize=24)
        plt.xticks([0, 479, 959, 1439], ['00:00', '08:00', '16:00', '23:59'])
        plt.title(f"{model} CLASSIFICATION RESULTS OF {day} DISH {self.__dishDiameter}cm ", fontsize=32)
        plt.show()
        

