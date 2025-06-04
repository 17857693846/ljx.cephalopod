import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from Data_Mining.Random_Forest.Final_version import PlotFigure as pf
import heapq

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

def model(f, t, model="RF"):
    y = np.log(t+1)
    X_train, X_test, y_train, y_test = train_test_split(f, y, test_size=0.2)
    #print("shape X train", X_train.shape)
    #y=minmax_scale(t)
    if model=="RF":
        mo = RandomForestRegressor()
        par = {"n_estimators": range(65, 195, 10),
               "min_samples_split": [2],
               "min_samples_leaf": [1,3]
               }
    elif model=="XGBoost":
        mo = XGBRegressor(n_estimators=150, eta=0.02)
        par = {"n_estimators": range(10, 20, 1)}
    elif model=="GBDT":
        mo = GradientBoostingRegressor()
    else:
        print("now not have this model, please check")
    mo.fit(X_train, y_train)
    p = mo.predict(X_test)
    mse = pow(mean_squared_error(y_test, p), 1)
    r2 = r2_score(y_test, p)
    #c = cross_validate(mo, f, y, cv=5)
    '''clf = GridSearchCV(mo, param_grid=par, n_jobs=-1, cv=10)
    clf.fit(f, y)
    print(clf.best_params_)
    print(clf.best_score_)'''
    return mse, r2

def research(feature_select, month=4, iterate=50, sect=20):
    if month == 4:
        season = "spring"
    elif month == 11:
        season = "autumn"
    mses_total = []
    r2s_total = []
    for i in dataset:
        print(i)
        f, t = read_file(feature_ins, [i], month=month)
        mses = []
        r2s = []
        for m in models:
            print(m)
            mm = []
            mr = []
            for _ in range(iterate):

                mse, r2 = model(f.iloc[:, feature_select], t, model=m)
                mm.append(round(mse, 3))
                mr.append(round(r2, 3))
                print("r2", r2)
        #print("cross", c["test_score"])
            lis = map(mr.index, heapq.nlargest(sect, mr))
            ids = list(lis)
            print(ids)
            max20 = heapq.nlargest(sect, mr)
            min20 = list(np.array(mm)[ids])

            mses.append(min20)
            r2s.append(max20)
            print("mses", mses)
            print("r2s", r2s)
        mses_total.append(mses)
        r2s_total.append(r2s)

    pf.plot_boxs_two(mses_total[0], mses_total[1], "MSE", title="(B)", figname="Autumn")
    pf.plot_boxs_two(r2s_total[0], r2s_total[1], "R$^{2}$", title="(D)", figname="Autumn")
    return mses_total, r2s_total

def model_no_selection(f, t, model="RF"):
    y = np.log(t+1)
    X_train, X_test, y_train, y_test = train_test_split(f, y, test_size=0.2)
    if model=="RF":
        mo = RandomForestRegressor()
        par = {"n_estimators": range(65, 195, 10),
               "min_samples_split": [2],
               "min_samples_leaf": [1,3]
               }
    elif model=="XGBoost":
        mo = XGBRegressor()
    elif model=="GBDT":
        mo = GradientBoostingRegressor()
    else:
        print("now not have this model, please check")
    mo.fit(f, y)
    p = mo.predict(X_test)
    rmse = pow(mean_squared_error(y_test, p), 0.5)
    r2 = r2_score(y_test, p)
    return rmse, r2

#60, 140
def main_model():
    rmseos = []
    r2tos = []
    for i in dataset:
        f, t = read_file(feature_ins, [i])
        r1to = []
        r2to = []
        for m in models:
            rmses = []
            r2s = []
            for i in range(10):
                rmse, r2 = model_no_selection(f, t, model=m)
                rmses.append(rmse)
                r2s.append(r2)
            r1to.append(rmses)
            r2to.append(r2s)
        rmseos.append(r1to)
        r2tos.append(r2to)
    return rmseos, r2tos


if __name__=="__main__":
    '''a, b = main_model()
    pf.plot_boxs_two(a[0], a[1], "RMSE")
    pf.plot_boxs_two(b[0], b[1], "R$^{2}$")'''
    feature_select = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    research(feature_select, month=11, iterate=20, sect=10)
    print("Autumn")
    '''research(feature_select, month=11, iterate=50, sect=20)
    print("Autumn")'''
    '''pf.plot_boxs_two(a[0], a[1], "MSE")
    pf.plot_boxs_two(b[0], b[1], "R$^{2}$")'''
