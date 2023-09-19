import pickle as pk
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, recall_score
import matplotlib.pyplot as plt
import matplotlib

# Convert the signals in dbm
def ConvertInDb(X):
    ydBm = X / 20000 - 53
    return ydBm


def compute_metrics(y_true, y_pred):
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
    Q1 = np.percentile(x, 25)
    Q3 = np.percentile(x, 75)
    IQR = Q3 - Q1
    lower_whisker = Q1 - 1.5 * IQR
    higher_whisker = Q3 + 1.5 * IQR
    # Count data points below the lower whisker and above upper
    count_below_lower_whisker = np.sum(x < lower_whisker)
    count_above_higher_whisker = np.sum(x > higher_whisker)
    # Calculate the percentage
    percentage_below_lower_whisker = (count_below_lower_whisker / len(x)) * 100
    percentage_above_upper_whisker = (count_above_higher_whisker / len(x)) * 100
    return percentage_below_lower_whisker, percentage_above_upper_whisker


class ResultsClass():
    def __init__(self,
                 resultsPath: str,
                 dishRadius: str,
                 whichModels: list):

        self.__Spe_noRain = []
        self.__Spe_rain = []
        self.__Rec = []
        self.__HM = []
        self.__y_pred = []
        self.__y_true = []
        self.__path = resultsPath
        self.__dishRadius = dishRadius
        self.__models = whichModels
        self.__numOfClassifiers = len(whichModels)
        Spe = []
        Rec = []
        HM = []
        for mo in whichModels:
            model = mo + f"_Predictions_{dishRadius}"
            file = Path(resultsPath) / model

            with open(file, 'rb') as f:
                results = pk.load(f)
                f.close()
            
            self.__dates = results["dates"]
            y_true = results["y_true"]
            y_pred = results["y_pred"]
            self.__y_pred.append(y_pred)
            self.__y_true.append(y_true)
            self.__TBRG = results["TBRG"]
            self.__SRS = results["SRS"]
            spe, rec, hm = compute_metrics(y_true, y_pred)
            Spe.append(spe)
            Rec.append(rec)
            HM.append(hm)
            
        for S, R, H in zip(Spe, Rec, HM):
            Spe_noRain_tmp = []
            Spe_rain_tmp = []
            Rec_tmp = []
            HM_tmp = []
            for s,r,h in zip(S,R,H):
                if r == -1:
                    Spe_noRain_tmp.append(s)
                else:
                    Spe_rain_tmp.append(s)
                    Rec_tmp.append(r)
                    HM_tmp.append(h)
            self.__Spe_noRain.append(np.asarray(Spe_noRain_tmp))
            self.__Spe_rain.append(np.asarray(Spe_rain_tmp))
            self.__Rec.append(np.asarray(Rec_tmp))
            self.__HM.append(np.asarray(HM_tmp))
        
        self.__boxplot_noRain()
        self.__boxplot_rain()
        self.__barplots_metrics_TBRG_intervals()
    
    def __boxplot_noRain(self):
        positions = np.arange(1, self.__numOfClassifiers+1, 1)
        labels = []
        for i in self.__models:
            stri = "Spe\n" + i
            labels.append(stri)
        fig, ax = plt.subplots(figsize=(16, 8))
        boxprops = dict(linewidth=4, color='blue')
        medianprops = dict(linewidth=4, color='blue')
        whiskerprops = dict(linewidth=3, color='blue')
        flierprops = dict(marker='o', markerfacecolor='blue', markersize=7,
                        markeredgecolor='none')
        capprops = dict(linewidth=3, color='blue')
        meanprops = dict(marker='D', markerfacecolor='blue', markersize=10,
                        markeredgecolor='none')
        ax.boxplot(self.__Spe_noRain, positions=positions, labels=labels, boxprops=boxprops, medianprops=medianprops,
                    whiskerprops=whiskerprops, flierprops=flierprops, capprops=capprops, meanprops=meanprops,
                    showmeans=True, showfliers=False)
        plt.yticks(np.arange(0.50, 1.01, 0.05))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
        ax.text(np.mean(positions), 1.04, self.__dishRadius + "cm", color='black', fontsize=20, fontweight='bold')
        ax.tick_params(axis='x', labelsize=22)
        ax.tick_params(axis='y', labelsize=24)
        fig.tight_layout()
        boxplotpath = Path("Images") / "Boxplot_Non_Rainy.pdf"
        plt.savefig(str(boxplotpath), format='pdf')

        for i, m in enumerate(self.__models):
            data = self.__Spe_noRain[i]
            pblw, pauw = compute_outliers_percentage(data)
            print(f"MODEL {m} Specificity No Rain: percentage below lower wisk {pblw:.2f}, percentage above higher wisk {pauw:.2f}")

    def __boxplot_rain(self):
        r = np.arange(1, 3.2 * self.__numOfClassifiers-1, 3.2)
        x = np.asarray([i for i in r])
        w1 = 0.9
        w2 = 0.9
        positionsSpec = x - w1
        positionsRec = x
        positionsHM = x + w2
        labelsSpec = ["Spe"]*self.__numOfClassifiers
        labelsRec = []
        for i in self.__models:
            stri = "Rec\n" + i
            labelsRec.append(stri)
        labelsHm = ["HM"]*self.__numOfClassifiers

        # ########## SPECIFICITY  ##########
        fig, ax = plt.subplots(figsize=(16, 8))
        boxprops = dict(linewidth=4, color='blue')
        medianprops = dict(linewidth=4, color='blue')
        whiskerprops = dict(linewidth=3, color='blue')
        flierprops = dict(marker='o', markerfacecolor='blue', markersize=7,
                        markeredgecolor='none')
        capprops = dict(linewidth=3, color='blue')
        meanprops = dict(marker='D', markerfacecolor='blue', markersize=10,
                        markeredgecolor='none')
        ax.boxplot(self.__Spe_rain, positions=positionsSpec, boxprops=boxprops, medianprops=medianprops,
                labels=labelsSpec,
                whiskerprops=whiskerprops, flierprops=flierprops, capprops=capprops, meanprops=meanprops,
                showmeans=True, showfliers=False)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
        plt.yticks(np.arange(0.35, 1.0, 0.2))
        # ########## RECALL  ##########
        boxprops = dict(linewidth=4, color='black')
        flierprops = dict(marker='o', markerfacecolor='black', markersize=7,
                        markeredgecolor='none')
        medianprops = dict(linewidth=4, color='black')
        whiskerprops = dict(linewidth=3, color='black')
        capprops = dict(linewidth=3, color='black')
        meanprops = dict(marker='D', markerfacecolor='black', markersize=10,
                        markeredgecolor='none')
        ax.boxplot(self.__Rec, positions=positionsRec, labels=labelsRec, boxprops=boxprops, medianprops=medianprops,
                    whiskerprops=whiskerprops, flierprops=flierprops, capprops=capprops, meanprops=meanprops,
                    showmeans=True, showfliers=False)
        # ######## HARMONIC MEAN #######
        boxprops = dict(linewidth=4, color='green')
        flierprops = dict(marker='o', markerfacecolor='green', markersize=7,
                        markeredgecolor='none')
        medianprops = dict(linewidth=4, color='green')
        whiskerprops = dict(linewidth=3, color='green')
        capprops = dict(linewidth=3, color='green')
        meanprops = dict(marker='D', markerfacecolor='green', markersize=10,
                        markeredgecolor='none')
        
        ax.boxplot(self.__HM, positions=positionsHM, labels=labelsHm, boxprops=boxprops, medianprops=medianprops,
                whiskerprops=whiskerprops, flierprops=flierprops, capprops=capprops, meanprops=meanprops,
                showmeans=True, showfliers=False)

        # ######## GENERAL ###########

        pos = np.arange(-0.6, 3.2*self.__numOfClassifiers, 3.2) # , 2.6, 5.8, 9, 12.2, 15.4, 18.6, 21.8]
        for xmin, xmax in zip(pos[::2], pos[1::2]):
            ax.axvspan(xmin, xmax, color='gainsboro')

        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
        plt.yticks(np.arange(0.4, 1.01, 0.1))
        plt.xlim([pos[0], pos[-1]])
        ax.text(np.mean(positionsRec), 1.04, self.__dishRadius + "cm", color='black', fontsize=20, fontweight='bold')
        ax.tick_params(axis='x', labelsize=22)
        ax.tick_params(axis='y', labelsize=24)
        # ax1.set_title('BOX PLOT RAINY DAYS DISH {}cm'.format(radius), fontsize=32)
        fig.tight_layout()
        boxplotpath = Path("Images") / "Boxplot_Rainy.pdf"
        plt.savefig(str(boxplotpath), format='pdf')

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
        hist0 = []
        hist1 = []
        hist2 = []
        hist3 = []
        histHM = []
        labels = self.__models
        for i in range(len(self.__models)):
            y_pred = self.__y_pred[i]
            yPred0 = []
            yPred1 = []
            yPred2 = []
            yPred3 = []
            yPredAll = []
            for j, x in enumerate(self.__TBRG):
                x = np.asarray(x)
                if np.count_nonzero(x>0)==0:
                    continue
                indTBRG0 = np.where(x==0.0)[0]
                indTBRG1 = np.where((x > 0) & (x < 2))[0]
                indTBRG2 = np.where((x >=2) & (x < 6))[0]
                indTBRG3 = np.where(x >= 6)[0]
                tmp = y_pred[j]
                yPred0_tmp = np.asarray([tmp[ii] for ii in indTBRG0])
                yPred1_tmp = np.asarray([tmp[ii] for ii in indTBRG1])
                yPred2_tmp = np.asarray([tmp[ii] for ii in indTBRG2])
                yPred3_tmp = np.asarray([tmp[ii] for ii in indTBRG3])
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

            spec = recall_score(np.asarray([0] * len(yPred0)), yPred0, pos_label=0)
            recAll = recall_score(np.asarray([1]*len(yPredAll)), yPredAll)
            rec1 = recall_score(np.asarray([1]*len(yPred1)), yPred1)
            rec2 = recall_score(np.asarray([1]*len(yPred2)), yPred2)
            rec3 = recall_score(np.asarray([1]*len(yPred3)), yPred3)
            hist0.append(spec)
            hist1.append(rec1)
            hist2.append(rec2)
            hist3.append(rec3)
            histHM.append(2*spec*recAll/(spec+recAll))
            
        x = np.arange(len(labels))  # the label locations
        width = 0.25  # the width of the bars
        fig, ax = plt.subplots(figsize=(14, 6.5))
        
        hist1 = [round(item*1000)/10 for item in hist1]
        hist2 = [round(item * 1000) / 10 for item in hist2]
        hist3 = [round(item * 1000) / 10 for item in hist3]
        tmp = []
        tmp.extend(hist1)
        tmp.extend(hist2)
        tmp.extend(hist3)
        ma = max(tmp) + 3
        mi = min(tmp)-10
        rects1 = ax.bar(x - width, hist1, width, label='TBRG<2 mm/h')
        rects2 = ax.bar(x, hist2, width, label='2<=TBRG<6 mm/h')
        rects3 = ax.bar(x + width, hist3, width, label='TBRG>=6 mm/h')

        ax.set_xticks(x, labels)
        ax.tick_params(labelsize=24)
        ax.legend(loc='lower left', fontsize=24)
        ax.text(np.mean(x), ma+0.3, self.__dishRadius+"cm", color='black', fontsize=20, fontweight='bold')
        ax.bar_label(rects1, padding=3, fontsize=20, rotation=0)
        ax.bar_label(rects2, padding=3, fontsize=20, rotation=0)
        ax.bar_label(rects3, padding=3, fontsize=20, rotation=0)
        fig.tight_layout()
        plt.ylim([mi, ma])
        plt.yticks([])
        filename = Path("Images") / "Rec.pdf"
        plt.savefig(filename, format='pdf')
        ###########################################
       
        width = 0.4  # the width of the bars
        fig, ax = plt.subplots(figsize=(14, 6.5))
        hist0 = [round(item * 1000) / 10 for item in hist0]
        histHM = [round(item * 1000) / 10 for item in histHM]
        rectsSpec = ax.bar(x-width/2, hist0, width, label='TBRG=0 mm/h')
        rectsHM = ax.bar(x+width/2, histHM, width, color='green')
        ax.set_xticks(x, labels)
        tmp = []
        tmp.extend(hist0)
        tmp.extend(histHM)
        ma = max(tmp) + 3
        mi = min(tmp)-10
        ax.tick_params(labelsize=24)
        ax.legend(loc='lower right', fontsize=24)
        ax.text(np.mean(x), ma+0.3, self.__dishRadius+"cm", color='black', fontsize=20, fontweight='bold')
        ax.bar_label(rectsSpec, padding=3, fontsize=24, rotation=0)
        ax.bar_label(rectsHM, padding=3, fontsize=24, rotation=0)
        fig.tight_layout()
        plt.ylim([mi, ma])
        plt.yticks([])
        filename = Path("Images") / "Spe_HM.pdf"
        plt.savefig(filename, format='pdf')


    def plot_classification_results(self, model, index):
        assert index < len(self.__y_pred[0]), f"The day selected does not exist, the maximum index is {len(self.__y_pred[0])-1}"
        assert model in self.__models, f"The model {model} has not been trained or it does not exist"
        plt.close('all')
        i = self.__models.index(model)
        y_pred = self.__y_pred[i][index]
        y_true = self.__y_true[i][index]
        tbrg = self.__TBRG[index]
        srs = self.__SRS[index]
        xSRS = ConvertInDb(srs)
        day = self.__dates[index]
        spec = recall_score(y_true, y_pred, pos_label=0)
        
        tbrg = np.asarray(tbrg)
        indTBRG = np.where(tbrg > 0)[0]
        indexOk = np.where(y_pred == y_true)[0]
        indexNotOk = np.where(y_pred != y_true)[0]
        fig, ax = plt.subplots(figsize=(19.2, 10.8))
        ax.tick_params(labelsize=24)
        d = 30
        if len(indTBRG) > 0:
            rec = recall_score(y_true, y_pred)
            hm = 2*spec*rec/(spec+rec)
            indTBRG = np.asarray(indTBRG)
            xTBRGplot = np.asarray([i for i in tbrg if i > 0])
            indexTBRGOk = np.asarray([iii for iii in indexOk if iii in indTBRG])
            indexSRSOk = np.asarray([iii for iii in indexOk if iii not in indTBRG])
            indexTBRGNotOk = np.asarray([iii for iii in indexNotOk if iii in indTBRG])
            indexSRSNotOk = np.asarray([iii for iii in indexNotOk if iii not in indTBRG])
            ax2 = ax.twinx()
            ax2.tick_params(labelsize=24)
            ax.vlines(indTBRG, ymin=[0] * len(indTBRG), ymax=xTBRGplot, color="cyan", linewidth=4)
            ax.set_ylabel("TBRG [mm/h]", fontsize=32)
            xSRStmp = [xSRS[ii] for ii in indexSRSOk]
            ax2.scatter(indexSRSOk, xSRStmp, s=d, facecolors='none', edgecolors='blue', marker='s')
            xSRStmp = [xSRS[ii] for ii in indexSRSNotOk]
            ax2.scatter(indexSRSNotOk, xSRStmp, s=d, facecolors='none', edgecolors='purple', marker='s')
            xSRStmp = [xSRS[ii] for ii in indexTBRGOk]
            ax2.scatter(indexTBRGOk, xSRStmp, s=d, facecolors='none', edgecolors='green', marker='o')
            xSRStmp = [xSRS[ii] for ii in indexTBRGNotOk]
            ax2.scatter(indexTBRGNotOk, xSRStmp, s=d, facecolors='none', edgecolors='red', marker='o')
            ax2.set_ylabel("SRS [dBm]", fontsize=32)
            ax2.set_xlabel("Spec= {:.4f}   Rec= {:.4f}   HM= {:.4f}".format(spec, rec, hm), fontsize=32)
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
            indexSRSOk = np.asarray([iii for iii in indexOk])
            indexSRSNotOk = np.asarray([iii for iii in indexNotOk])
            xSRStmp = [xSRS[ii] for ii in indexSRSOk]
            ax.scatter(indexSRSOk, xSRStmp, s=d, facecolors='none', edgecolors='blue', marker='s')
            xSRStmp = [xSRS[ii] for ii in indexSRSNotOk]
            ax.scatter(indexSRSNotOk, xSRStmp, s=d, facecolors='none', edgecolors='purple', marker='s')
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

        plt.xlim([-1, 1440])
        plt.yticks(fontsize=24)
        plt.xticks([0, 479, 959, 1439], ['00:00', '08:00', '16:00', '23:59'])
        plt.title(f"{model} CLASSIFICATION RESULTS OF {day} DISH {self.__dishRadius}cm ", fontsize=32)
        plt.show()
        

