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
import seaborn as sns
from scipy import stats

plt.rcParams["font.size"] = 16
plt.rcParams["font.sans-serif"] = "Times New Roman"
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.dpi"] = 300

np.random.seed(99)
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
feature_names = ["Year", "Month", "Lat", "Lon", "SST", "Chl-a", "SSH", "SOI", "SSS", "DO", "Depth"]
feature_ins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
models = ["RF", "XGBoost", "GBDT"]
years = [2014, 2015, 2016, 2017, 2018, 2019]

def standard(spid=0, month=4):
    f, t = read_file(feature_ins, [dataset[spid]], month=month)
    f = f.reset_index(drop=True)
    t = t.reset_index(drop=True)

    if month == 4:
        season = "spring"
    elif month == 11:
        season = "autumn"
    name = ins[spid]+" "+season
    print(name)
    X_train, X_test, y_train, y_test = train_test_split(f, t, test_size=0.2, random_state=100)
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    p1 = rf.predict(f)
    preRF = pd.DataFrame(p1, columns=["RF"])

    rf = XGBRegressor(n_estimators=200, eta=0.02)
    rf.fit(X_train, y_train)
    p2 = rf.predict(f)
    preXGB = pd.DataFrame(p2, columns=["XGBoost"])

    rf = GradientBoostingRegressor(n_estimators=150, learning_rate=0.02)
    rf.fit(X_train, y_train)
    p3 = rf.predict(f)
    preGBDT = pd.DataFrame(p3, columns=["GBDT"])

    tx1, px1 = stats.ttest_ind(a=t, b=p1)
    print(f"TTest RF:T({tx1}),P({px1})")
    tx2, px2 = stats.ttest_ind(a=t, b=p2)
    print(f"TTest XGB:T({tx2}),P({px2})")
    tx3, px3 = stats.ttest_ind(a=t, b=p3)
    print(f"TTest GBDT:T({tx3}),P({px3})")

    rr1, pp1 = stats.pearsonr(t, p1)
    print(f"Cor RF:T({rr1}),P({pp1})")
    rr2, pp2 = stats.pearsonr(t, p2)
    print(f"Cor XGB:T({rr2}),P({pp2})")
    rr3, pp3 = stats.pearsonr(t, p3)
    print(f"Cor GBDT:T({rr3}),P({pp3})")


    '''con = pd.concat([f["Year"], f["Month"],
                     pd.DataFrame(np.array(t), columns=["Nominal CPUE"]),
                     preRF, preXGB, preGBDT], axis=1, ignore_index=False)'''
    #con.to_csv(name+"_concat.csv")

    #con.to_csv(name+".csv")


if __name__=="__main__":
    standard(spid=0, month=4)
    standard(spid=0, month=11)
    standard(spid=1, month=4)
    standard(spid=1, month=11)
