import os
import pickle
import enlighten
import collections
import numpy as np
import tensorflow as tf
from pathlib import Path
from scipy.ndimage import filters
from scipy.signal.windows import gaussian
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

gpu_available = tf.test.is_gpu_available()
if gpu_available:
    print("GPU AVAILABLE")
else:
    print("TRAINING ON CPU")

# Convert the signals in dbm
def ConvertInDb(X):
    if isinstance(X, list):
        X = np.asarray(X)
    ydBm = X / 20000 - 53
    return list(ydBm)

# Filter the data with a Gaussian window
def FilterData(X, width=12, std=6):
    b = gaussian(width, std)
    c = b[:int(width)]
    ga = filters.convolve1d(X, c / c.sum(), axis=0)
    return list(ga)

# Extract the Features for ML Classifiers
def ComputeFeatures(X, verbose=False):
    if verbose:
        print("COMPUTE FEATURES")
    fea = []
    for tmp in X:
        x1 = np.mean(tmp)
        x2 = np.std(tmp)
        x3 = np.mean(np.diff(tmp))
        x4 = np.std(np.diff(tmp))
        x5 = np.min(tmp)
        x6 = np.max(tmp)
        x7 = np.min(np.diff(tmp))
        x8 = np.max(np.diff(tmp))
        arr = np.asarray([x1, x2, x3, x4, x5, x6, x7, x8])
        fea.append(arr)
    feaArr = np.asarray(fea)
    return feaArr

# Balance the dataset for the training
def BalanceDataset(X, y, verbose=False):
    if verbose:
        print("BALANCING THE DATASET")
    rainEvents = np.count_nonzero(y == 1)
    notRainEvents = np.count_nonzero(y == 0)
    dataPerClass = np.min([rainEvents, notRainEvents])
    indexesRain = np.where(y == 1)[0]
    indexesNotRain = np.where(y == 0)[0]
    np.random.seed(777)
    np.random.shuffle(indexesRain)
    np.random.shuffle(indexesNotRain)
    indexesRain = indexesRain[0:dataPerClass]
    indexesNotRain = indexesNotRain[0:dataPerClass]
    X_bal = []
    y_bal = []

    for i in indexesRain:
        X_bal.append(X[i])
        y_bal.append(1)
    for i in indexesNotRain:
        X_bal.append(X[i])
        y_bal.append(0)
    return np.asarray(X_bal), np.asarray(y_bal)

# Create the subsequences
def CreateSubseq(X, tw, y=None, online=False, verbose=False):
        if verbose:
            print("CREATING SUB-SEQUENCES")
        sub_seq = []
        if not online:
            y_seq = []
        for j in range(len(X) - tw):
            sub_seq.append(X[j:j + tw])
            if not online:
                y_seq.append(y[j + tw])
        if not online:
            return np.asarray(sub_seq), np.asarray(y_seq)
        else:
            return np.asarray(sub_seq) 

# Compute the evaluation metrics
def ComputeMetrics(yTrue, yPred):
    tn, fp, fn, tp = confusion_matrix(yTrue, yPred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    spec = tn / (tn + fp)
    FPR = fp / (fp + tn)
    FNR = fn / (fn + tp)
    acc = (tp + tn) / (tp + tn + fn + fp)
    print("Precision: {:.4f}, Recall: {:.4f}, Specifity: {:.4f},\n FPR: {:.4f}, FNR: {:.4f}, Acc: {:.4f}".format(prec,
                                                                                                                 rec,
                                                                                                                 spec,
                                                                                                                 FPR,
                                                                                                                 FNR,
                                                                                                                 acc))

# Reject outliers in the ADA algorithm
def reject_outliers(data, m=2.):
    tmp = np.asarray(data)
    d = np.abs(tmp - np.median(tmp, axis=0))
    mdev = np.median(d, axis=0)
    s = d / mdev if np.all(mdev, axis=0) else 0.
    if len(s.shape) == 1:
        res = tmp[np.where(s < m)[0]]
    else:
        res = tmp[np.all(s < m, axis=1), :]
    return res


class ReadDataClass():
    def __init__(self,
                 datasetPath: str,
                 dishDiameter: str,
                 offlineDays : int,
                 bool_filterData: bool):
        
        assert os.path.exists(datasetPath), f"Dataset path {datasetPath} does not exist"
        with open(datasetPath, "rb") as f:
            datasetDict = pickle.load(f)
        f.close()

        signal2take = "SRS"+ dishDiameter
        self.__Dates = datasetDict["Dates"]
        try:
            self.__SRS = datasetDict[signal2take]
        except:
            raise TypeError("The dish diameter does not exist")
        self.__TBRG = datasetDict["TBRG"]
        self.dishDiameter = dishDiameter

        assert offlineDays <= 10, "For an effective training and test of the algorithms select a number of offline days lower than 11"
        self.offlineDays = offlineDays
        
        self.rainyDays = []
        self.notRainyDays = []
        for i, tbrg in enumerate(self.__TBRG):
            tbrg = np.asarray(tbrg)
            isRain = np.any(np.where(tbrg>0)[0])
            if isRain:
                self.rainyDays.append(i)
            else:
                self.notRainyDays.append(i)

        self.bool_filterData = bool_filterData
        

    def __extract_signals_and_labels(self, listOfIndex):
        SRS = [self.__SRS[i] for i in listOfIndex]
        TBRG = [self.__TBRG[i] for i in listOfIndex]
        labels = []
        for tbrg in TBRG:
            y = [0]*len(TBRG[0])
            indexes = np.where(np.asarray(tbrg)>0)[0]
            for j in indexes:
                y[j] = 1
            labels.append(y)
        return SRS, TBRG, labels


    def create_training_set(self, takeRainyDays=True):
        index_train = self.notRainyDays[0:self.offlineDays]
        if takeRainyDays:
            index_train.extend(self.rainyDays[0:self.offlineDays])
        index_train = np.asarray(index_train)
        index_train.sort()
        SRS, _, labels = self.__extract_signals_and_labels(index_train)
        SRS_db = [ConvertInDb(x) for x in SRS]
        if self.bool_filterData:
            SRS_filt = [FilterData(x) for x in SRS_db]
        else:
            SRS_filt = SRS_db
        X_train = np.squeeze(np.asarray(SRS_filt))
        y_train = np.asarray(labels)
        print("TRAIN NON-RAINY/RAINY EVENTS: {}/{}".format(np.count_nonzero(np.asarray(labels) == 0), 
                                                           np.count_nonzero(np.asarray(labels) == 1)))
        return X_train, y_train


    def create_test_set(self):
        index_test = self.notRainyDays[self.offlineDays:]
        index_test.extend(self.rainyDays[self.offlineDays:])
        index_test = np.asarray(index_test)
        index_test.sort()
        SRS_Dates = [self.__Dates[i] for i in index_test]
        SRS, TBRG, labels = self.__extract_signals_and_labels(index_test)
        SRS_db = [ConvertInDb(x) for x in SRS]
        if self.bool_filterData:
            SRS_filt = [FilterData(x) for x in SRS_db]
        else:
            SRS_filt = SRS_db
        X_test= np.squeeze(np.asarray(SRS_filt))
        y_test = np.asarray(labels)
        print("TEST NON-RAINY/RAINY EVENTS: {}/{}".format(np.count_nonzero(np.asarray(labels) == 0),
            np.count_nonzero(np.asarray(labels) == 1)))
        return X_test, y_test, SRS_Dates, TBRG


class ADADataClass(ReadDataClass):
    def __init__(self, 
                 datasetPath : str,
                 resultsPath : str,
                 dishDiameter : str,
                 offlineDays: int,
                 bool_filterData: bool):
        super().__init__(datasetPath=datasetPath, 
                         dishDiameter=dishDiameter, 
                         offlineDays=offlineDays, 
                         bool_filterData=bool_filterData)
        if not os.path.exists(resultsPath):
            os.makedirs(resultsPath)
        self.resPath = resultsPath
        self.dataForOnline = collections.deque(maxlen=2*offlineDays)
        self.dataForSTD = collections.deque(maxlen=20)
        X_train, _ = self.create_training_set(takeRainyDays=False)
        for i in X_train:
            self.dataForOnline.append(i)
        for i in range(-10, 0, 1):
            self.dataForSTD.append(self.dataForOnline[-1][i])  # initialize the array for STD with the last 10 samples of the training dataset
        print("COMPUTING THRESHOLDS ON TRAINING DATA")
        self.min, self.diff, self.std = self.Compute_Thresholds()
        print("THRESHOLDS COMPUTED")

    def Compute_Thresholds(self):
        _min = self.__Compute_Min()
        diff = self.__Compute_Diff()
        stdev = self.__Compute_Std()
        return _min, diff, stdev

    def __Compute_Min(self):
        tmp = np.asarray(self.dataForOnline)
        min_tmp = [np.min(i, axis=0) for i in tmp]
        minTmp = reject_outliers(min_tmp)
        return np.min(minTmp, axis=0)

    def __Compute_Diff(self):
        tmp = np.asarray(self.dataForOnline)
        tmp_list = [np.diff(i, axis=0) for i in tmp]
        maxTmp = []
        for i in tmp_list:
            tmp2 = np.asarray(i)
            tmp2 = np.abs(tmp2[np.where(tmp2 < 0)[0]])
            tmp2 = reject_outliers(tmp2, 7)
            maxTmp.append(np.max(tmp2, axis=0))
        return np.max(maxTmp, axis=0) * 1.5

    def __Compute_Std(self):
        tmp = np.asarray(self.dataForOnline)
        stdTmp = []
        for i in tmp:
            tmp_std = [np.std(i[j:j + 10], ddof=1, axis=0) for j in range(len(i) - 10)]
            tmp_std = reject_outliers(tmp_std, 7)
            stdTmp.append(np.max(tmp_std, axis=0))
        return np.max(stdTmp, axis=0)


class ADAClass(ADADataClass):
    def __init__(self, 
                 datasetPath : str,
                 resultsPath : str,
                 dishDiameter : str,
                 offlineDays: int,
                 bool_filterData: bool):
        super().__init__(datasetPath=datasetPath,
                         resultsPath=resultsPath, 
                         dishDiameter=dishDiameter, 
                         offlineDays=offlineDays, 
                         bool_filterData=bool_filterData)

    def Online(self):
        X_online, y_online, days_date, TBRG = self.create_test_set()
        print("ONLINE INFERENCE")
        y_pred = []
        resultsPath = self.resPath  / f"ADA_Metrics_{self.dishDiameter}.txt"
        with open(resultsPath, 'w') as f:
            f.writelines('Ada Results\n')
        eventPrec = self.dataForOnline[-1][-1]  # take the last minute of the training dataset
        rainMinuteFlag = False  # flag keeping count if in the previous minute it was rainining
        for i, x in enumerate(X_online):
            y = y_online[i]
            y_tmp = []
            for ev in x:
                self.dataForSTD.append(ev)  # collect the SRS events
                tmpArrayStd = np.asarray(self.dataForSTD)
                first_cond = ((ev - eventPrec) < -self.diff) | (ev < self.min)
                second_cond = np.std(tmpArrayStd[-10:], ddof=1, axis=0) > self.std
                third_cond = np.std(tmpArrayStd, ddof=1, axis=0) < self.std / 4
                fourth_cond = np.std(tmpArrayStd[-10:], ddof=1, axis=0) < self.std / 1.7
                if np.all(first_cond | second_cond) and not rainMinuteFlag:
                    rainMinuteFlag = True
                    y_tmp.append(1)
                elif np.all(third_cond | fourth_cond) and rainMinuteFlag:
                    rainMinuteFlag = False
                    y_tmp.append(0)
                else:
                    if rainMinuteFlag:
                        y_tmp.append(1)
                    else:
                        y_tmp.append(0)
                eventPrec = ev
            rainMinuteFlag = y[-1]
            y_pred.append(np.asarray(y_tmp))
            tn, fp, fn, tp = confusion_matrix(y, y_tmp, labels=[0, 1]).ravel()
            if np.count_nonzero(y == 1) == 0:
                rec = -1
            else:
                rec = tp / (tp + fn)
            spec = tn / (tn + fp)
            stri = "DAY {}, NON-RAINY/RAINY: {}/{}, SPE: {:.4f} REC: {:.4f}\n".format(days_date[i], np.count_nonzero(y == 0),
                                                                                 np.count_nonzero(y == 1), spec, rec)
        
            with open(resultsPath, 'a') as f:
                f.writelines(stri)

            print(stri)
            if np.count_nonzero(y == 1) == 0: 
                self.dataForOnline.append(x)
                self.min, self.diff, self.std = self.Compute_Thresholds()
        data2save = {"SRS": X_online, "dates":days_date, "y_true":y_online, "y_pred": y_pred, "TBRG":TBRG}
        data2savePath = self.resPath / f"ADA_Predictions_{self.dishDiameter}"
        pickle.dump(data2save, open(data2savePath, "wb"))
        print("END Online Testing")


class ClassifiersDataClass(ReadDataClass):
    def __init__(self,
                 datasetPath : str,
                 resultsPath : str,
                 dishDiameter : str,
                 offlineDays: int,
                 timeWindow: int, 
                 bool_filterData: bool,
                 bool_deepNetworks: bool):
        super().__init__(datasetPath=datasetPath, 
                         dishDiameter=dishDiameter, 
                         bool_filterData=bool_filterData,
                         offlineDays=offlineDays)
        if not os.path.exists(resultsPath):
            os.makedirs(resultsPath)
        self.resPath = resultsPath
        self.__X_train, self.__y_train = self.create_training_set()
        self.X_val = None
        self.y_val = None
        self.X_train = None
        self.y_train = None
        self.bool_deepNetworks = bool_deepNetworks
        self.timeWindow = timeWindow
        assert len(np.asarray(self.__X_train).shape) == 2, "Input tensor must be 2D: days x events"
        
        self.rainDataForOnline = collections.deque(maxlen=offlineDays*2)
        self.y_rain = collections.deque(maxlen=offlineDays*2)
        self.notRainDataForOnline = collections.deque(maxlen=offlineDays*2)
        self.y_not_rain = collections.deque(maxlen=offlineDays*2)

        self.last_events = self.__X_train[-1][-timeWindow:]
        for i, xf in enumerate(self.__X_train):
            y_tmp = self.__y_train[i]
            if np.count_nonzero(y_tmp == 1) == 0:
                self.notRainDataForOnline.append(xf)
                self.y_not_rain.append(y_tmp)
            else:
                self.rainDataForOnline.append(xf)
                self.y_rain.append(y_tmp)
        self.PrepareTrainingDataset()


    def PrepareTrainingDataset(self, online=False):
        X_tr = []
        y_tr = []
        for i, x in enumerate(self.rainDataForOnline):
            X_tr.append(x)
            y_tr.append(self.y_rain[i])
        for i, x in enumerate(self.notRainDataForOnline):
            X_tr.append(x)
            y_tr.append(self.y_not_rain[i])
        X_tr = np.asarray(X_tr)
        y_tr = np.asarray(y_tr)
        X_tr = np.reshape(X_tr, (X_tr.shape[0] * X_tr.shape[1]))
        y_tr = np.reshape(y_tr, (y_tr.shape[0] * y_tr.shape[1]))
        X_tr_seq, y_tr_seq = CreateSubseq(X_tr, tw=self.timeWindow, y=y_tr)
        if  not self.bool_deepNetworks:
            X_tr_fea = ComputeFeatures(X_tr_seq)
        else:
            X_tr_fea = X_tr_seq

        X_tr_bal, y_tr_bal = BalanceDataset(X_tr_fea, y_tr_seq)

        if not self.bool_deepNetworks:
            scaler = MinMaxScaler()
            scaler.fit(X_tr_fea)
            scalerpath = self.resPath / "scaler.pkl"
            pickle.dump(scaler, open(scalerpath, "wb"))
            X_tr_bal = scaler.transform(X_tr_bal)

        if not online:
            X_train, X_val, y_train, y_val = train_test_split(X_tr_bal, y_tr_bal, stratify=y_tr_bal,
                                                              test_size=0.2, random_state=666)
        else:
            X_train = X_tr_bal
            y_train = y_tr_bal
            X_val = None
            y_val = None

        self.X_train = X_train
        self.X_val = X_val
        self.y_train = tf.keras.utils.to_categorical(y_train, 2)
        if y_val is not None:
            self.y_val = tf.keras.utils.to_categorical(y_val, 2)


class ANNClass(ClassifiersDataClass):
    def __init__(self, 
                 datasetPath : str,
                 resultsPath : str,
                 dishDiameter : str,
                 offlineDays: int,
                 bool_filterData: bool = False,
                 timeWindow: int = 30, 
                 **kwargs):
        super().__init__(datasetPath=datasetPath, 
                         resultsPath=resultsPath,
                         dishDiameter=dishDiameter, 
                         offlineDays=offlineDays, 
                         timeWindow=timeWindow, 
                         bool_filterData=bool_filterData, 
                         bool_deepNetworks=False)
        self.__manager = None
        self.__close_man = None
        self.__learning_rate = None
        self.__rolls = None
        self.__seed = None
        self.__patience = None
        self.__epochs = None
        self.__batch_size = None
        self.__lam = None
        self.__neu = None
        self.__define_hyperparams(kwargs)
        self.__best_model = self.__TrainingANN()

    def __define_hyperparams(self, kwargs):
        neu = kwargs.get('_neurons', None)
        if neu is not None:
            self.__neu = neu
        else:
            self.__neu = [100]

        lam = kwargs.get('_lam', None)
        if lam is not None:
            self.__lam = lam
        else:
            self.__lam = [0]

        batch_size = kwargs.get("_batch_size", None)
        if batch_size is not None:
            self.__batch_size = int(batch_size)
        else:
            self.__batch_size = 64

        epochs = kwargs.get("_epochs", None)
        if epochs is not None:
            self.__epochs = int(epochs)
        else:
            self.__epochs = 100

        patience = kwargs.get("_patience", None)
        if patience is not None:
            self.__patience = int(patience)
        else:
            self.__patience = 10

        seed = kwargs.get("_seed", None)
        if seed is not None:
            self.__seed = int(seed)
        else:
            self.__seed = 666

        rolls = kwargs.get("_rolls", None)
        if rolls is not None:
            self.__rolls = int(rolls)
        else:
            self.__rolls = 1

        learning_rate = kwargs.get("_learning_rate", None)
        if learning_rate is not None:
            self.__learning_rate = learning_rate
        else:
            self.__learning_rate = 0.001

        manager = kwargs.get("_manager", None)
        if manager is not None:
            self.__manager = manager
            self.__close_man = False
        else:
            self.__close_man = True
            self.__manager = enlighten.get_manager()


    def __TrainingANN(self):
        score_best = 0
        rolls_mana = self.__manager.counter(total=self.__rolls, desc="Rolls", unit="num", color="blue", leave=False)
        neu_mana = self.__manager.counter(total=len(self.__neu), desc="Neurons", unit="num", color="green", leave=False)
        lam_mana = self.__manager.counter(total=len(self.__lam), desc="Lambda", unit="num", color="yellow", leave=False)
        model_best = None
        lambda_best = None
        neurons_best = None
        num_feat = self.X_train.shape[1]  # define the number of features

        callbacks_list = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                           patience=self.__patience)]  # early stop criterion during the training phase

        neu_mana.count = 0
        print("TRAINING ANN OFFLINE")
        for nn in self.__neu:
            neu_mana.update()
            lam_mana.count = 0
            for la in self.__lam:
                lam_mana.update()
                rolls_mana.count = 0
                for ro in range(self.__rolls):
                    rolls_mana.update()
                    seed = self.__seed + ro * 123
                    model = self.__create_model(num_feat=num_feat, neurons=nn, lam=la, seed=seed)
                    hist = model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val),
                                     epochs=self.__epochs, batch_size=self.__batch_size, verbose=0,
                                     callbacks=callbacks_list)
                    score = hist.history["val_accuracy"][-1]
                    if score > score_best:
                        score_best = score
                        model_best = model
                        lambda_best = la
                        neurons_best = nn
        print("ANN best model lambda/neurons:{}/{}".format(lambda_best, neurons_best))
        y_tmp = model_best.predict(self.X_train)
        y_tmp = np.argmax(y_tmp, axis=1)
        y_train = np.argmax(self.y_train, axis=1)
        y_val = np.argmax(self.y_val, axis=1)
        tn, fp, fn, tp = confusion_matrix(y_train, y_tmp, labels=[0, 1]).ravel()
        rec = tp / (tp + fn)
        spe = tn / (tn + fp)
        print("Num of NotRain/Rain Data Train Set Best Model: {}/{}, Rec: {:.4f}, Spe: {:.4f}".format(np.count_nonzero(y_train == 0), 
                                                                                 np.count_nonzero(y_train == 1), 
                                                                                 rec,
                                                                                 spe))
        y_tmp = model_best.predict(self.X_val)
        y_tmp = np.argmax(y_tmp, axis=1)
        tn, fp, fn, tp = confusion_matrix(y_val, y_tmp, labels=[0, 1]).ravel()
        rec = tp / (tp + fn)
        spe = tn / (tn + fp)
        print("Num of NotRain/Rain Data Val Set Best Model: {}/{}, Rec: {:.4f}, Spe: {:.4f}".format(np.count_nonzero(y_val == 0), 
                                                                                 np.count_nonzero(y_val == 1), 
                                                                                 rec,
                                                                                 spe))
        rolls_mana.close()
        neu_mana.close()
        lam_mana.close()
        if self.__close_man:
            self.__manager.stop()
        return model_best


    def __create_model(self, num_feat, neurons, lam, seed):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.__learning_rate)  # usually from 10^-4 to 10^-3
        input_shape = (num_feat,)
        inp = tf.keras.layers.Input(shape=input_shape)
        x = tf.keras.layers.Dense(neurons, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lam),
                                  kernel_initializer=tf.keras.initializers.RandomUniform(minval=-1, maxval=1,
                                                                                         seed=seed),
                                  bias_initializer=tf.keras.initializers.RandomUniform(minval=-1, maxval=1, seed=seed))(
            inp)
        out = tf.keras.layers.Dense(2, activation='softmax')(x)
        model = tf.keras.models.Model(inputs=[inp], outputs=[out])
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        model.trainable = True
        return model


    def __TrainingOnlineFC(self):
        self.__best_model.fit(self.X_train, self.y_train, epochs=20, batch_size=self.__batch_size, verbose=0)


    def Online(self):
        X_online, y_online, days_date, TBRG,  = self.create_test_set()
        print("ONLINE INFERENCE")
        y_pred = []
        if self.bool_filterData:
            resultsPath = self.resPath  / f"ANN_Metrics_Filtered_{self.dishDiameter}.txt"
            with open(resultsPath, 'w') as f:
                f.writelines('ANN Filtered Data Results\n')
        else:
            resultsPath = self.resPath  / f"ANN_Metrics_{self.dishDiameter}.txt"
            with open(resultsPath, 'w') as f:
                f.writelines('ANN Results\n')

        for i, x in enumerate(X_online):
            x = np.concatenate((self.last_events, x))
            self.last_events = x[-self.timeWindow:]
            y = y_online[i]
            X_seq = CreateSubseq(x, tw=self.timeWindow, online=True)
            X_fea = ComputeFeatures(X_seq, verbose=0)
            scalerpath = self.resPath  / "scaler.pkl"
            scaler = pickle.load(open(scalerpath, "rb"))
            X_test = scaler.transform(X_fea)
            y_tmp = self.__best_model.predict(X_test, verbose=0)
            y_tmp = np.argmax(y_tmp, axis=1)
            y_pred.append(y_tmp)
            tn, fp, fn, tp = confusion_matrix(y, y_tmp, labels=[0, 1]).ravel()
            if np.count_nonzero(y == 1) == 0:
                rec = -1
            else:
                rec = tp / (tp + fn)
            spec = tn / (tn + fp)
            stri = "DAY {}, NON-RAINY/RAINY: {}/{}, SPE: {:.4f} REC: {:.4f}\n".format(days_date[i], np.count_nonzero(y == 0),
                                                                                 np.count_nonzero(y == 1), spec, rec)
            if self.bool_filterData:
                with open(resultsPath, 'w') as f:
                    f.writelines(stri)
            else:
                with open(resultsPath, 'w') as f:
                    f.writelines(stri)
            print(stri)
            x = x[self.timeWindow:]
            if np.count_nonzero(y == 1) == 0:
                self.notRainDataForOnline.append(x)
                self.y_not_rain.append(y)
            else:
                self.rainDataForOnline.append(x)
                self.y_rain.append(y)
            self.PrepareTrainingDataset(online=True)
            self.__TrainingOnlineFC()
        data2save = {"SRS": X_online, "dates":days_date, "y_true":y_online, "y_pred": y_pred, "TBRG": TBRG}
        if self.bool_filterData:
            data2savePath = self.resPath  / f"ANN_Predictions_Filtered_{self.dishDiameter}"
            pickle.dump(data2save, open(data2savePath, "wb"))
        else:
            data2savePath = self.resPath  / f"ANN_Predictions_{self.dishDiameter}"
            pickle.dump(data2save, open(data2savePath, "wb"))
        print("END Online Testing")
        

class CNNClass(ClassifiersDataClass):
    def __init__(self, 
                 datasetPath : str,
                 resultsPath : str,
                 dishDiameter : str,
                 offlineDays: int,
                 bool_filterData: bool = False,
                 timeWindow: int = 30,  
                 **kwargs):
        super().__init__(datasetPath=datasetPath, 
                         resultsPath=resultsPath,
                         dishDiameter=dishDiameter, 
                         offlineDays=offlineDays, 
                         timeWindow=timeWindow, 
                         bool_filterData=bool_filterData,
                         bool_deepNetworks=True)

        self.__manager = None
        self.__close_man = None
        self.__learning_rate = None
        self.__rolls = None
        self.__seed = None
        self.__patience = None
        self.__epochs = None
        self.__batch_size = None
        self.filt = None
        self.ker = None
        self.define_hyperparams(kwargs)
        self.__best_model = self.TrainingCNN()


    def define_hyperparams(self, kwargs):
        filt = kwargs.get('_filters', None)
        if filt is not None:
            self.filt = filt
        else:
            self.filt = [(4, 8)]

        ker = kwargs.get('_kernel_size', None)
        if ker is not None:
            self.ker = ker
        else:
            self.ker = [5]

        batch_size = kwargs.get("_batch_size", None)
        if batch_size is not None:
            self.__batch_size = int(batch_size)
        else:
            self.__batch_size = 64

        epochs = kwargs.get("_epochs", None)
        if epochs is not None:
            self.__epochs = int(epochs)
        else:
            self.__epochs = 100

        patience = kwargs.get("_patience", None)
        if patience is not None:
            self.__patience = int(patience)
        else:
            self.__patience = 8

        seed = kwargs.get("_seed", None)
        if seed is not None:
            self.__seed = int(seed)
        else:
            self.__seed = 666

        rolls = kwargs.get("_rolls", None)
        if rolls is not None:
            self.__rolls = int(rolls)
        else:
            self.__rolls = 1

        learning_rate = kwargs.get("_learning_rate", None)
        if learning_rate is not None:
            self.__learning_rate = learning_rate
        else:
            self.__learning_rate = 0.001

        manager = kwargs.get("_manager", None)
        if manager is not None:
            self.__manager = manager
            self.__close_man = False
        else:
            self.__close_man = True
            self.__manager = enlighten.get_manager()


    def TrainingCNN(self):
        score_best = 0
        rolls_mana = self.__manager.counter(total=self.__rolls, desc="Rolls", unit="num", color="blue", leave=False)
        filt_mana = self.__manager.counter(total=len(self.filt), desc="Filters", unit="num", color="green", leave=False)
        ker_mana = self.__manager.counter(total=len(self.ker), desc="Kernels", unit="num", color="green", leave=False)
        model_best = None
        num_feat = self.X_train.shape[1]  # define the number of features

        callbacks_list = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                           patience=self.__patience)]  # early stop criterion during the training phase

        filt_mana.count = 0
        print("TRAINING CNN OFFLINE")
        for fil in self.filt:
            filt_mana.update()
            ker_mana.count = 0
            for ke in self.ker:
                ker_mana.update()
                rolls_mana.count = 0
                for ro in range(self.__rolls):
                    rolls_mana.update()
                    seed = self.__seed + ro * 123
                    model = self.create_model(num_feat=num_feat, filt=fil, kern=ke, seed=seed)
                    hist = model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val),
                                     epochs=self.__epochs, batch_size=self.__batch_size, verbose=0,
                                     callbacks=callbacks_list)
                    score = hist.history["val_accuracy"][-1]
                    if score > score_best:
                        score_best = score
                        model_best = model
        model_best.summary()
        y_tmp = model_best.predict(self.X_train)
        y_tmp = np.argmax(y_tmp, axis=1)
        y_train = np.argmax(self.y_train, axis=1)
        y_val = np.argmax(self.y_val, axis=1)
        tn, fp, fn, tp = confusion_matrix(y_train, y_tmp, labels=[0, 1]).ravel()
        rec = tp / (tp + fn)
        spe = tn / (tn + fp)
        print("Num of NotRain/Rain Data Train Set Best Model: {}/{}, Rec: {:.4f}, Spe: {:.4f}".format(np.count_nonzero(y_train == 0), 
                                                                                 np.count_nonzero(y_train == 1), 
                                                                                 rec,
                                                                                 spe))
        y_tmp = model_best.predict(self.X_val)
        y_tmp = np.argmax(y_tmp, axis=1)
        tn, fp, fn, tp = confusion_matrix(y_val, y_tmp, labels=[0, 1]).ravel()
        rec = tp / (tp + fn)
        spe = tn / (tn + fp)
        print("Num of NotRain/Rain Data Val Set Best Model: {}/{}, Rec: {:.4f}, Spe: {:.4f}".format(np.count_nonzero(y_val == 0), 
                                                                                 np.count_nonzero(y_val == 1), 
                                                                                 rec,
                                                                                 spe))
        rolls_mana.close()
        filt_mana.close()
        ker_mana.close()
        if self.__close_man:
            self.__manager.stop()
        return model_best


    def create_model(self, num_feat, filt, kern, seed):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.__learning_rate)  # usually from 10^-4 to 10^-3
        input_shape = (num_feat, 1)
        inp = tf.keras.layers.Input(shape=input_shape)
        x = None
        for i, f in enumerate(filt):
            if i == 0:
                x = tf.keras.layers.Conv1D(filters=f, kernel_size=kern, activation='relu', padding="same")(inp)
            else:
                x = tf.keras.layers.Conv1D(filters=f, kernel_size=kern, activation='relu', padding="same")(x)
            x = tf.keras.layers.MaxPooling1D(pool_size=2, padding="same")(x)
            x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
        x = tf.keras.layers.GlobalMaxPooling1D()(x) #GlobalAveragePooling1D
        x = tf.keras.layers.Dense(10, activation='relu')(x)
        out = tf.keras.layers.Dense(2, activation='softmax')(x)
        model = tf.keras.models.Model(inputs=[inp], outputs=[out])
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        model.trainable = True
        return model


    def TrainingOnlineCNN(self):
        self.__best_model.fit(self.X_train, self.y_train, epochs=20, batch_size=self.__batch_size, verbose=0)


    def Online(self):
        X_online, y_online, days_date, TBRG = self.create_test_set()
        print("ONLINE INFERENCE")
        y_pred = []
        if self.bool_filterData:
            resultsPath = self.resPath / f"CNN_Metrics_Filtered_{self.dishDiameter}.txt"
            with open(resultsPath, 'w') as f:
                f.writelines('CNN Filtered Data Results\n')
        else:
            resultsPath = self.resPath / f"CNN_Metrics_{self.dishDiameter}.txt"
            with open(resultsPath, 'w') as f:
                f.writelines('CNN Results\n')

        for i, x in enumerate(X_online):
            x = np.concatenate((self.last_events, x))
            self.last_events = x[-self.timeWindow:]
            y = y_online[i]
            X_seq = CreateSubseq(x, tw=self.timeWindow, online=True)
            X_fea = X_seq
            X_test = X_fea
            y_tmp = self.__best_model.predict(X_test)
            y_tmp = np.argmax(y_tmp, axis=1)
            y_pred.append(y_tmp)
            tn, fp, fn, tp = confusion_matrix(y, y_tmp, labels=[0, 1]).ravel()
            if np.count_nonzero(y == 1) == 0:
                rec = -1
            else:
                rec = tp / (tp + fn)
            spec = tn / (tn + fp)
            stri = "DAY {}, NON-RAINY/RAINY: {}/{}, SPE: {:.4f} REC: {:.4f}\n".format(days_date[i], np.count_nonzero(y == 0),
                                                                                 np.count_nonzero(y == 1), spec, rec)
            if self.bool_filterData:
                with open(resultsPath, 'w') as f:
                    f.writelines(stri)
            else:
                with open(resultsPath, 'w') as f:
                    f.writelines(stri)
            print(stri)
            x = x[self.timeWindow:]
            if np.count_nonzero(y == 1) == 0:
                self.notRainDataForOnline.append(x)
                self.y_not_rain.append(y)
            else:
                self.rainDataForOnline.append(x)
                self.y_rain.append(y)
            self.PrepareTrainingDataset(online=True)
            self.TrainingOnlineCNN()
        data2save = {"SRS": X_online, "dates":days_date, "y_true":y_online, "y_pred": y_pred, "TBRG": TBRG}
        if self.bool_filterData:
            data2savePath = self.resPath / f"CNN_Predictions_Filtered_{self.dishDiameter}"
            pickle.dump(data2save, open(data2savePath, "wb"))
        else:
            data2savePath = self.resPath / f"CNN_Predictions_{self.dishDiameter}"
            pickle.dump(data2save, open(data2savePath, "wb"))
        print("END Online Testing")