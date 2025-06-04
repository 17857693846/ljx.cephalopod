import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
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
    return d4, flist, tlist

np.random.seed(100)
dataset = [11, 12]
ins = ["Loliginidae", "Sepiolidae"]
feature_names = ["Year", "Month", "Lat", "Lon", "SST", "Chl-a", "SSH", "SOI", "SSS", "DO", "Depth"]
feature_ins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
models = ["RF", "XGBoost", "GBDT"]
years = [2014, 2015, 2016, 2017, 2018, 2019]

def model(f, t):
    f.reset_index(drop=True)
    y = np.log(t+1)

    X_train, X_test, y_train, y_test = train_test_split(f, y, test_size=0.2)
    #y=minmax_scale(t)
    mo = RandomForestRegressor()
    par = {"n_estimators": range(65, 195, 10),
           "min_samples_split": [2],
           "min_samples_leaf": [1, 3]
           }
    mo.fit(X_train, y_train)
    p = mo.predict(f)
    pre = np.exp(p)-1
    dfp = pd.DataFrame(pre, columns=["predict_CPUE"])
    return dfp

if __name__=="__main__":
    data, f, t = read_file(feature_ins[2:], [dataset[0]], month=11)
    pre = model(f, t)
    d = pd.concat([data.reset_index(drop=True), pre], axis=1, ignore_index=False)
    d.to_csv("loli_autumn.csv")


