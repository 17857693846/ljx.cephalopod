import pandas as pd
import numpy as np

def read_file(feature_li, target_li, month=4):
    d4 = pd.read_csv("For16.csv", encoding="gbk")
    d4 = d4.loc[d4["Month"] == month]
    fl = []
    for feature in feature_li:
        f = d4.iloc[:, feature]
        fl.append(f)
    if len(fl) > 1:
        flist = pd.concat(fl, axis=1)
    else:
        flist = fl[0]

    tl = []
    for target in target_li:
        t = d4.iloc[:, target]
        tl.append(t)
    if len(tl) > 1:
        tlist = pd.concat(tl, axis=1)
    else:
        tlist = tl[0]
    return flist, tlist

dataset = [4, 7]
ins = ["Loliginidae", "Sepiolidae"]
feature_names = ["Year", "Month", "Lat", "Lon", "SST", "CHLA", "SSH", "SOI"]
feature_ins = [0, 1, 2, 3, 12, 13, 14, 15]
models = ["RF", "XGBoost", "GBDT"]

#This function can get the test part by year
def get_train_test(species=4, year=2014, month=4):
    f, t = read_file(feature_ins, [species], month=month)
    data = pd.concat([f, t], axis=1)
    train = data.loc[data["Year"] != year]
    test = data.loc[data["Year"] == year]

    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]
    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]
    print(X_train)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    get_train_test()
