import pandas as pd
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm

"""
Global Variable
"""
AF = 15.3725
RP = 0.05
NP = 20
WBC = 449569
MSSC = 1265040
CFC = 22380
ASSC = 319205
SC = 5870878
GSPC = 17640
GSTC = 126521
CEC = 85459
LSPC = 217551
CEPCI_2017 = 567.500
CEPCI_2010 = 550.800
CEPCI_2002 = 390.400

"""
The Method calculating EAC
"""


def cal_EAC(ULS, UOS):
    OSTC = (CEPCI_2017 / CEPCI_2002) * (-0.000000005 * ((UOS * 720) ** 2) + (0.0388 * (UOS * 720)) + 15055)
    LSTC = (CEPCI_2017 / CEPCI_2002) * (-0.000000003 * ((ULS * 720) ** 2) + (0.0317 * (ULS * 720)) + 15055)
    OCPC = (CEPCI_2017 / CEPCI_2010) * (170000 * (UOS / 676.6955) ** (0.6))
    DC = (CEPCI_2017 / CEPCI_2002) * (UOS * 96.096 + 2458.6)

    C_equip = OSTC + LSTC + OCPC + DC + WBC + MSSC + CFC + ASSC + SC + GSPC + GSTC + CEC + LSPC
    FCI = C_equip * 100 / 30
    SUC = FCI * 0.1
    WCI = FCI * 0.2

    TCI = FCI + SUC + WCI
    EAC = TCI / 15.3725

    return FCI, EAC


"""
The Method calculating TPC
"""


def cal_TPC(FCI, UHLS, ULLS, UOS, sensitivity):
    FC = FCI * 0.04
    materials = (0.03 * UHLS + sensitivity * ULLS) * 24 * 365
    water = (UOS * 0.0308 * 8760 - 0.00001) * 0.01703435
    electricity = (0.0308 * UOS * 8760 - 0.000006) * 0.065
    maintenance = FCI * 0.06
    operating = maintenance * 0.15

    temp = FC + materials + water + electricity + (1.6 * maintenance) + operating

    TPC = temp / 0.49425

    return TPC


"""
The Method calculating TAC // return type : numpy array
"""


def calc_TAC(given_X_data, sensitivity):
    tac_list = list()
    for data in given_X_data:
        uls = np.sum(data[0:2])
        uos = np.sum(data[2:7])
        fci, eac = cal_EAC(uls, uos)

        uhls = data[0]
        ulls = data[1]

        tpc = cal_TPC(fci, uhls, ulls, uos, sensitivity)

        tac = eac + tpc
        tac_list.append(tac)

    tac_list = np.array(tac_list)
    tac_list = tac_list.reshape((15836, 1))

    return tac_list


def run(sensitivity):
    """
    df_given : given data [DataFrame]
    np_given : df_given to numpy array
    tac_list : With LIME, PAEGAK, calculated [numpy array]
    np_added_TAC : Preprocessed data ["HIGHLIME", "LOWLIME", "OYSTER", "SCALLOP", "COCKLE", "CLAM", "MUSSEL", "GP", "TAC"]

    """
    df_given = pd.read_csv("./data/given_data.csv", index_col=0)
    np_given = df_given.to_numpy()
    tac_list = calc_TAC(np_given, sensitivity)
    np_added_TAC = np.concatenate((np_given, tac_list), axis=1)

    df_add_TAC = pd.DataFrame(np_added_TAC)
    df_add_TAC.columns = ["HIGHLIME", "LOWLIME", "OYSTER", "SCALLOP", "COCKLE", "CLAM", "MUSSEL", "GP", "TAC"]

    """
    Calculate Mean & Variance
    """
    var_X = ["HIGHLIME", "LOWLIME", "OYSTER", "SCALLOP", "COCKLE", "CLAM", "MUSSEL"]
    var_y = ["GP", "TAC"]

    X = df_add_TAC[var_X]
    y = df_add_TAC[var_y]

    scaler_X = preprocessing.StandardScaler().fit(X)
    X_sc = scaler_X.transform(X)
    scaler_y = preprocessing.StandardScaler().fit(y)
    y_sc = scaler_y.transform(y)

    """
    Train Surrogate Model
    """
    flag = True
    while flag:
        train_X, test_X, train_y, test_y = train_test_split(X_sc, y_sc, test_size=0.25, shuffle=True)

        model = Sequential()
        model.add(Dense(100, activation="relu", input_dim=len(var_X)))
        model.add(Dense(100, activation="relu"))
        model.add(Dense(100, activation="sigmoid"))
        model.add(Dense(len(var_y)))

        model.compile(optimizer="adam", loss="mse")
        model.fit(train_X, train_y, epochs=200, verbose=1, validation_data=(test_X, test_y))

        """
        Print R2 Performance
        """

        pred_y = model.predict(test_X)

        Actual1 = test_y[:, 0]
        Actual2 = test_y[:, 1]
        Predict1 = pred_y[:, 0]
        Predict2 = pred_y[:, 1]

        GP_R2 = metrics.r2_score(Actual1, Predict1)
        TAC_R2 = metrics.r2_score(Actual2, Predict2)

        if GP_R2 >= 0.99 and TAC_R2 >= 0.99:
            flag = False

        print("R2_GP : {}".format(metrics.r2_score(Actual1, Predict1)))
        print("R2_TAC : {}".format(metrics.r2_score(Actual2, Predict2)))
        print("R2_AVG : {}".format(metrics.r2_score(test_y, pred_y)))
    # os.system("echo R2_GP : {}".format(metrics.r2_score(Predict1, Actual1)))
    # os.system("echo R2_TAC : {}".format(metrics.r2_score(Predict2, Actual2)))
    # os.system("echo R2_AVG : {}".format(metrics.r2_score(Predict, Actual)))

    #######################################################################################################
    res = list()
    i = 0

    ## 2,121,625,880 ##
    for high in range(0, 3711, 200):
        for low in range(0, 3711, 200):
            for oyster in range(0, 3711, 200):
                for scallop in range(0, 9, 1):
                    for cockle in range(0, 38, 1):
                        for clam in range(0, 549, 10):
                            for mussel in range(0, 3711, 200):
                                i += 1
                                if i % 1000000 == 0:
                                    print("{} : {}".format(i, len(res)))
                                    # os.system("echo iter : {} || len(data) : {}".format(i, len(res)))
                                value = high + low + oyster + scallop + cockle + clam + mussel
                                if value == 3710:
                                    res.append([high, low, oyster, scallop, cockle, clam, mussel])

    virtual_X = np.array(res)

    # virtual_X_df = pd.DataFrame(virtual_X)
    # virtual_X_df.columns = ["HIGHLIME", "LOWLIME", "OYSTER", "SCALLOP", "COCKLE", "CLAM", "MUSSEL"]

    """
    scaling & df -> numpy
    """
    # virtual_X_sc = (virtual_X_df - mean_X) / std_X
    # virtual_X_sc = np.array(virtual_X_sc)
    virtual_X_sc = scaler_X.transform(virtual_X)

    virtual_y_sc = model.predict(virtual_X_sc)

    virtual_y = scaler_y.inverse_transform(virtual_y_sc)

    # numpy
    pumping_data = np.concatenate((virtual_X, virtual_y), axis=1)
    pumping_data = pd.DataFrame(pumping_data)
    pumping_data.columns = ["HIGHLIME", "LOWLIME", "OYSTER", "SCALLOP", "COCKLE", "CLAM", "MUSSEL", "GP", "TAC"]

    # Remain GP >= 0.93
    pumping_data = pumping_data[pumping_data["GP"] >= 0.93]

    sorted_df = pumping_data.sort_values(by=["TAC"])

    sorted_df.to_csv("./res/l{}.csv".format(str(sensitivity)))


if __name__ == "__main__":
    sensitivity = [0.012, 0.0135, 0.0165, 0.018]
    for sense in sensitivity:
        run(sense)
