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
plt.rcParams["font.size"] = 16
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
feature_names = ["Year", "Month", "Lat", "Lon", "SST", "Chl-a", "SSH", "SOI", "SSS", "DO", "Depth"]
feature_ins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
models = ["RF", "XGBoost", "GBDT"]
years = [2014, 2015, 2016, 2017, 2018, 2019]

#This function can get the test part by year
def get_train_test(species=11, year=2014, month=4):
    f, t = read_file(feature_ins, [species], month=month)
    #f = f.reset_index(drop=True)
    y = np.log(t+1)
    data = pd.concat([f, y], axis=1)
    train = data.loc[data["Year"] == year]
    test = data.loc[data["Year"] == year]
    tn = int(0.5*len(y))
    X_train = train.iloc[:tn, 2:-1]
    y_train = train.iloc[:tn, -1]
    X_test = test.iloc[:, 2:-1]
    y_test = test.iloc[:, -1]
    return X_train, X_test, y_train, y_test

def model(species=11, month=4):
    values = []
    for year in years:
        X_train, X_test, y_train, y_test = get_train_test(species=species, year=year, month=month)
        #print(X_train)
        mo1 = RandomForestRegressor()
        mo1.fit(X_train, y_train)
        p1 = mo1.predict(X_test)
        print(f"RF of {year}:\n", r2_score(y_test, p1))

        mo2 = XGBRegressor(n_estimators=200, eta=0.02)
        mo2.fit(X_train, y_train)
        p2 = mo2.predict(X_test)
        print(f"XGBoost of {year}:\n", r2_score(y_test, p2))

        mo3 = GradientBoostingRegressor(n_estimators=150, learning_rate=0.02)
        mo3.fit(X_train, y_train)
        p3 = mo3.predict(X_test)
        print(f"GBDT of {year}:\n", r2_score(y_test, p3))

        values.append([np.mean(np.exp(y_test)-1), np.mean(np.exp(p1)-1), np.mean(np.exp(p2)-1), np.mean(np.exp(p3)-1)])
    print("values:\n", values)
    return values

def plot_lines_by_season(spid=0, month=4):
    if month == 4:
        season = "spring"
    elif month == 11:
        season = "autumn"
    value = model(species=dataset[spid], month=month)
    values = np.array(value)
    o = values[:, 0]
    rf = values[:, 1]
    xgb = values[:, 2]
    gbdt = values[:, 3]
    plt.title(f"{ins[spid]}'s nominal CPUE lines in {season}")
    x = ["2014", "2015", "2016", "2017", "2018", "2019"]
    plt.plot(x, o, color="blue", marker="o", label="Origin")
    plt.plot(x, rf, color="green", marker="s", label="RF")
    plt.plot(x, xgb, color="black", marker="D", label="XGBoost")
    plt.plot(x, gbdt, color="red", marker="^", label="GBDT")
    plt.legend(fontsize=10)
    plt.yticks(np.arange(0, int(2*max(o)), int(2*max(o))/5))
    plt.ylabel("CPUE(ind./km$^{2}$)")
    plt.xlabel("Year")
    plt.grid(linestyle='--', linewidth=0.8)
    plt.savefig(f"{ins[spid]} in {season}.tif")
    plt.show()

if __name__=="__main__":
    plot_lines_by_season(spid=0, month=4)
    plot_lines_by_season(spid=0, month=11)
    plot_lines_by_season(spid=1, month=4)
    plot_lines_by_season(spid=1, month=11)




