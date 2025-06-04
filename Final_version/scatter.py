import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from Data_Mining.Random_Forest.Two_spe import PlotFigure as pf
import heapq
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 16
plt.rcParams["font.sans-serif"] = "Times New Roman"
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.dpi"] = 300

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
years = [2014, 2015, 2016, 2017, 2018, 2019]

#This function can get the test part by year
def get_train_test(species=4, year=2014, month=4):
    f, t = read_file(feature_ins, [species], month=month)
    #f = f.reset_index(drop=True)
    y = np.log(t+1)
    data = pd.concat([f, y], axis=1)
    train = data.loc[data["Year"] != year]
    test = data.loc[data["Year"] == year]

    X_train = train.iloc[:, 2:-1]
    y_train = train.iloc[:, -1]
    X_test = test.iloc[:, 2:-1]
    y_test = test.iloc[:, -1]
    return X_train, X_test, y_train, y_test

def predict_2019(species=4, month=4):
    year = 2019
    X_train, X_test, y_train, y_test = get_train_test(species=species, year=year, month=month)
    # print(X_train)
    mo1 = RandomForestRegressor()
    mo1.fit(X_train, y_train)
    p1 = mo1.predict(X_test)
    print(f"RF of {year}:\n", r2_score(y_test, p1))

    mo2 = XGBRegressor()
    mo2.fit(X_train, y_train)
    p2 = mo2.predict(X_test)
    print(f"XGBoost of {year}:\n", r2_score(y_test, p2))

    mo3 = RandomForestRegressor()
    mo3.fit(X_train, y_train)
    p3 = mo3.predict(X_test)
    print(f"GBDT of {year}:\n", r2_score(y_test, p3))

    df_all = pd.DataFrame(np.array([y_test, p1, p2, p3]).T, columns=["o", "rf", "xgb", "gbdt"])
    print(df_all)
    return df_all

def sactter(species=4, year=2019, month=4):
    df = predict_2019(species=species, month=month)
    fig, ax = plt.subplots()
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c="#EA5514", marker="+", label="rf")
    plt.scatter(df.iloc[:, 0], df.iloc[:, 2], c="#4595FF", marker="1", label="xgb")
    plt.scatter(df.iloc[:, 0], df.iloc[:, 3], c="black", marker="o", label="gbdt")
    ax.legend(loc="lower right")
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--")
    plt.xlabel("Observed ln(CPUE+1)")
    plt.ylabel("Predicted ln(CPUE+1)")
    plt.title("Population quantity prediction of XGBoost " + str(year) + "/" + str(month))
    plt.savefig("ind_" + str(species)+str(year) + str(month) + ".tif")
    plt.show()
    plt.cla()

if __name__=="__main__":
    sactter(species=4, year=2019, month=4)
    sactter(species=4, year=2019, month=11)
    sactter(species=7, year=2019, month=4)
    sactter(species=7, year=2019, month=11)




