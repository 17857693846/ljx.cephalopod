import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from Data_Mining.Random_Forest.Final_version import PlotFigure as pf
import heapq
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = [8, 6]
plt.rcParams["font.size"] = 20
plt.rcParams["font.sans-serif"] = "Times New Roman"
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.dpi"] = 300

def read_file(feature_li, target_li, month=4):
    d4 = pd.read_csv("two_species.csv", encoding="gbk")
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

dataset = [11, 12]
ins = ["Loliginidae", "Sepiolidae"]
feature_names = ["Year", "Month", "Lat", "Lon", "SST", "Chl-a", "SSH", "SOI", "SSS", "DO", "BT"]
feature_ins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
models = ["RF", "XGBoost", "GBDT"]
years = [2014, 2015, 2016, 2017, 2018, 2019]

def model(f, t, model="RF"):
    y = np.log(t+1)
    #X_train, X_test, y_train, y_test = train_test_split(f, y, test_size=0.2)
    #y=minmax_scale(t)
    if model=="RF":
        mo = RandomForestRegressor()
        par = {"n_estimators": range(65, 195, 10),
               "min_samples_split": [2],
               "min_samples_leaf": [1,3]
               }
    elif model=="XGBoost":
        mo = XGBRegressor(n_estimators=200, eta=0.02)
        par = {"n_estimators": range(10, 20, 1)}
    elif model=="GBDT":
        mo = GradientBoostingRegressor(n_estimators=200, learning_rate=0.02)
    else:
        print("now not have this model, please check")
    mo.fit(f, y)
    imp = mo.feature_importances_
    print("r2:", mo.score(f, y))
    return imp
#spidx=0, 1 for loli and sep; m="RF", "XGBoost", "GBDT"
def plot_importances(spidx=0, m="RF"):
    print("species name:", ins[spidx])
    f, t = read_file(feature_ins[2:], [dataset[spidx]], month=4)
    imp = model(f, t, model=m)
    f1, t1 = read_file(feature_ins[2:], [dataset[spidx]], month=11)
    imp1 = model(f1, t1, model=m)
    print(f"importance in spring by {m}:", imp)
    print(f"importance in autumn by {m}:", imp1)
    colors = ["#9ac9db", "#f8ac8c"]

    ind = np.arange(len(feature_ins) - 2)
    width = 0.4
    plt.bar(ind, imp, width, color=colors[0], edgecolor="black", hatch="++", label="Spring")
    plt.bar(ind+width, imp1, width, color=colors[1], edgecolor="black", hatch="//", label="Autumn")
    plt.xticks(ind + width/2, feature_names[2:])
    plt.legend()
    plt.grid(True, linestyle="--", color="gray", linewidth="0.8", axis="y")
    plt.tick_params(left=False, bottom=False)
    plt.ylabel("Importance")
    plt.xlabel("Factors")
    plt.title(f"Importance of {m} for {ins[spidx]}")
    plt.savefig(f"Importance of {m} for {ins[spidx]}.tif")
    plt.show()

if __name__=="__main__":
    plot_importances(spidx=0, m="RF")
    plot_importances(spidx=0, m="XGBoost")
    plot_importances(spidx=0, m="GBDT")
    plot_importances(spidx=1, m="RF")
    plot_importances(spidx=1, m="XGBoost")
    plot_importances(spidx=1, m="GBDT")
