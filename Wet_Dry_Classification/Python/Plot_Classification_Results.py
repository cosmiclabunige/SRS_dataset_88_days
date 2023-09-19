import pickle as pk
import datetime
import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, precision_score
import matplotlib
import matplotlib.pyplot as plt


def ConvertIndBm(X):
    ydBm = X / 20000 - 53
    return ydBm


def plot_results_rainy(xSRS, xTBRG, y_pred, y_true, day, classifier, radius):
    # prec = precision_score(y_true, y_pred, pos_label=1)
    spec = recall_score(y_true, y_pred, pos_label=0)
    rec = recall_score(y_true, y_pred)
    hm = 2*spec*rec/(spec+rec)
    # f1 = (2 * prec * rec) / (prec + rec)
    xTBRG = np.asarray(xTBRG)
    indTBRG = np.where(xTBRG > 0)[0]
    indTBRG = np.asarray(indTBRG)
    fig, ax = plt.subplots(figsize=(19.2, 10.8))
    xTBRGplot = np.asarray([i for i in xTBRG if i > 0])
    indexOk = np.where(y_pred == y_true)[0]
    indexTBRGOk = np.asarray([iii for iii in indexOk if iii in indTBRG])
    indexSRSOk = np.asarray([iii for iii in indexOk if iii not in indTBRG])
    indexNotOk = np.where(y_pred != y_true)[0]
    indexTBRGNotOk = np.asarray([iii for iii in indexNotOk if iii in indTBRG])
    indexSRSNotOk = np.asarray([iii for iii in indexNotOk if iii not in indTBRG])
    ax.vlines(indTBRG, ymin=[0] * len(indTBRG), ymax=xTBRGplot, color="cyan", linewidth=4)
    ax.set_ylabel("TBRG [mm/h]", fontsize=32)
    ax.tick_params(labelsize=24)
    ax2 = ax.twinx()
    xSRS = ConvertIndBm(xSRS)
    xSRStmp = [xSRS[ii] for ii in indexSRSOk]
    d =30
    ax2.scatter(indexSRSOk, xSRStmp, s=d, facecolors='none', edgecolors='blue', marker='s')
    xSRStmp = [xSRS[ii] for ii in indexSRSNotOk]
    ax2.scatter(indexSRSNotOk, xSRStmp, s=d, facecolors='none', edgecolors='purple', marker='s')
    xSRStmp = [xSRS[ii] for ii in indexTBRGOk]
    ax2.scatter(indexTBRGOk, xSRStmp, s=d, facecolors='none', edgecolors='green', marker='o')
    xSRStmp = [xSRS[ii] for ii in indexTBRGNotOk]
    ax2.scatter(indexTBRGNotOk, xSRStmp, s=d, facecolors='none', edgecolors='red', marker='o')
    ax2.set_ylabel("SRS [dBm]", fontsize=32)
    ax2.tick_params(labelsize=24)
    ax.set_xlabel("Spec= {:.4f}   Rec= {:.4f}   HM= {:.4f}".format(spec, rec, hm), fontsize=32)
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
    #matplotlib.use('TkAgg')
    plt.xlim([-1, 1440])
    #plt.ylim([-18, -10.5])
    plt.yticks(fontsize=24)
    plt.xticks([0, 479, 959, 1439], ['00:00', '08:00', '16:00', '23:59'])
    if day != '2017-09-09':
        pos = 'lower right'
    else:
        pos = 'lower left'
    ax2.legend(["True Negative", "False Positive", "True Positive", "False Negative"], scatterpoints=1, labelspacing=1,
              handletextpad=0.1, handlelength=0.8, markerscale=2, fontsize=24, loc=pos)
    # plt.title("{} CLASSIFICATION RESULTS OF {} DISH {}cm ".format(classifier, day, radius), fontsize=32)
    # plt.show()



def plot_results_notrainy(xSRS, y_pred, y_true, day, classifier, radius):
    spec = recall_score(y_true, y_pred, pos_label=0)
    fig, ax = plt.subplots(figsize=(19.2,10.8))
    indexOk = np.where(y_pred == y_true)[0]
    indexSRSOk = np.asarray([iii for iii in indexOk])
    indexNotOk = np.where(y_pred != y_true)[0]
    indexSRSNotOk = np.asarray([iii for iii in indexNotOk])
    xSRS = ConvertIndBm(xSRS)
    xSRStmp = [xSRS[ii] for ii in indexSRSOk]
    d = 30
    ax.scatter(indexSRSOk, xSRStmp, s=d, facecolors='none', edgecolors='blue', marker='s')
    xSRStmp = [xSRS[ii] for ii in indexSRSNotOk]
    ax.scatter(indexSRSNotOk, xSRStmp, s=d, facecolors='none', edgecolors='purple', marker='s')
    ax.set_ylabel("SRS [dBm]", fontsize=32)
    ax.tick_params(labelsize=24)
    ax.set_xlabel("Specificity = {:.4f}".format(spec), fontsize=32)
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.set_facecolor('xkcd:white')
    #matplotlib.use('TkAgg')
    plt.xlim([-1, 1440])
    #plt.ylim([-18, -10.5])
    plt.yticks(fontsize=24)
    plt.xticks([0, 479, 959, 1439], ['00:00', '08:00', '16:00', '23:59'], fontsize=24)
    if day != '2017-09-09':
        pos = 'lower right'
    else:
        pos = 'lower left'
    ax.legend(["True Negative", "False Positive", "True Positive", "False Negative"], scatterpoints=1, labelspacing=1,
           handletextpad=0.1, handlelength=0.8, markerscale=2, fontsize=24, loc=pos)
    # plt.title("{} CLASSIFICATION RESULTS OF {} DISH {}cm ".format(classifier, day, radius), fontsize=32)


def plot_SRS_TBRG_rainy(xSRS, xTBRG):
    xTBRG = np.asarray(xTBRG)
    indTBRG = np.where(xTBRG > 0)[0]
    indTBRG = np.asarray(indTBRG)
    fig, ax = plt.subplots()
    xTBRGplot = np.asarray([i for i in xTBRG if i > 0])
    ax.vlines(indTBRG, ymin=[0] * len(indTBRG), ymax=xTBRGplot, color="cyan", linewidth=3)
    ax.set_ylabel("TBRG [mm/h]", fontsize=32)
    ax2 = ax.twinx()
    xSRSrain = [xSRS[i]  for i in indTBRG]
    # ax2.plot(xSRS*1e-6, color="black", linewidth=1)
    ax2.scatter(np.arange(0,len(xSRS)), np.asarray(xSRS), color="blue", linewidth=3, marker="*")
    ax2.scatter(indTBRG, np.asarray(xSRSrain), color="red", linewidth=3, marker="*")
    ax2.set_ylabel("SRS [dBm]", fontsize=28)
    plt.xlim([-1, 1440])
    plt.xticks([0, 479, 959, 1439], ['00:00:00', '08:00:00', '16:00:00', '23:59:59'])
    ax.xaxis.set_tick_params(labelsize=20, rotation=30)
    ax.yaxis.set_tick_params(labelsize=20)
    ax2.yaxis.set_tick_params(labelsize=20)
    matplotlib.use('TkAgg')
    plt.title("Example of SRS signal with TBRG measure", fontsize=32)
    #plt.show()

#################################################
#                 Dataset Choice                #
#################################################


dataset2use = "85"  # 60, 85
filteredData = False
cl = "ANN"
classifier = "../Results/{}_Adaptive_{}".format(cl, dataset2use)

with open(classifier, 'rb') as f:
    classifier = pk.load(f)
    f.close()
#################################################
#                 Dataset Path                  #
#################################################
radius = "85"
datasetPath85 = "../Dataset/Processed_SRS_{}.pkl".format(radius)

radius = "60"
datasetPath60 = "../Dataset/Processed_SRS_{}.pkl".format(radius)

with open(datasetPath85, "rb") as f:
    datasetDict85 = pk.load(f)
    f.close()

with open(datasetPath60, "rb") as f:
    datasetDict60 = pk.load(f)
    f.close()

SRSDatesTmp = datasetDict60["listOfSRSTimeStamps"]
SRSDatesTmp = [SRSDatesTmp[i][0].split(" ")[0] for i in range(len(SRSDatesTmp))]

TBRGValues = datasetDict60["listOfTBRGValues"]

####################################################
#               Take Only Interesting Days         #
####################################################

indexToTakeNotRain = [7, 13, 16, 17, 26, 27, 28, 38, 39, 51, 54, 55, 56, 59, 62, 63, 64, 71, 72, 81, 84, 128, 141, 167, 170, 180,
                      188, 190, 191, 192, 195, 206, 207, 210, 212, 214, 215, 220, 221, 222, 228, 236, 238, 240, 244, 246, 249, 253,
                      256, 257]
indexToTakeRain = [4, 5, 6, 8, 14, 21, 58, 60, 61, 73, 86, 89, 97, 127, 133, 136, 149, 151, 182, 184, 185, 189, 202,
                   229, 235, 239, 242, 247, 255, 261, 265, 287, 297, 319, 322, 326, 328, 337]

iii = 5
index_test = indexToTakeNotRain[iii:]
index_test.extend(indexToTakeRain[iii:])
index_test = np.asarray(index_test)
index_test.sort()

for i, id in enumerate(index_test):
    day = SRSDatesTmp[id]
    if (day != '2018-01-01') and (day != '2019-04-07') and (day != '2017-09-09'):
        continue
    tmp = TBRGValues[id]
    tmp = np.asarray(tmp)
    l = len(np.where(tmp>0)[0])
    title_img = "../Images/Classification_{}_{}_{}.pdf".format(cl, day, dataset2use)
    if l > 0:
        plot_results_rainy(classifier["X_test"][i], tmp, np.asarray(classifier["y_pred"][i]), np.asarray(classifier["y_test"][i]), day, cl, dataset2use)
        plt.savefig(title_img, bbox_inches='tight', format='pdf')
        plt.close()
    else:
        plot_results_notrainy(classifier["X_test"][i], np.asarray(classifier["y_pred"][i]), np.asarray(classifier["y_test"][i]), day, cl, dataset2use)
        plt.savefig(title_img, bbox_inches='tight', format='pdf')
        plt.close()
