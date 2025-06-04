import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA, PCA, TruncatedSVD, FactorAnalysis
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import minmax_scale, normalize
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV, LearningCurveDisplay
from Data_Mining.Random_Forest.Final_version import PlotFigure as pf
import heapq
import matplotlib.pyplot as plt

def read_file(feature_li, target_li, month=4):
    d4 = pd.read_csv("two_species.csv", encoding="gbk")
    d4 = d4.loc[d4["Month"] == month]
    d4 = d4.loc[d4["Q"] != 0]
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
    #y = minmax_scale(t)
    #kernel={'linear', 'cosine', 'precomputed', 'sigmoid', 'poly', 'rbf'}
    '''pca = KernelPCA(n_components=4)
    f = pca.fit_transform(f)'''

    X_train, X_test, y_train, y_test = train_test_split(f, y, test_size=0.2)
    #y=minmax_scale(t)
    if model=="RF":
        mo = RandomForestRegressor(n_estimators=100)
        par = {"n_estimators": range(65, 195, 10),
               "min_samples_split": [2],
               "min_samples_leaf": [1,3]
               }
    elif model=="XGBoost":
        #when eta=0.02, 0.03, 0.04, the rate will higher and stable
        mo = XGBRegressor(n_estimators=150, eta=0.05)
    elif model=="GBDT":
        mo = GradientBoostingRegressor(n_estimators=150, learning_rate=0.02, subsample=1)
    else:
        print("now not have this model, please check")
    mo.fit(X_train, y_train)
    p = mo.predict(X_test)
    mse = pow(mean_squared_error(y_test, p), 1)
    r2 = r2_score(y_test, p)
    #print(mo.oob_score_)
    #c = cross_validate(mo, f, y, cv=5)
    '''clf = GridSearchCV(mo, param_grid=par, n_jobs=-1, cv=10)
    clf.fit(f, y)
    print(clf.best_params_)
    print(clf.best_score_)'''
    return mse, r2

if __name__=="__main__":
    f, t = read_file(feature_ins[2:], [dataset[0]], month=4)
    #fe = minmax_scale(f)
    for _ in range(10):
        mse, r2 = model(f, t, model="RF")
        print(r2)
    '''model1 = RandomForestRegressor(n_estimators=120)
    model2 = XGBRegressor(n_estimators=150, eta=0.02)
    a = LearningCurveDisplay.from_estimator(
        model1, fe, t, score_type="train", cv=10)
    a.plot()
    plt.show()'''
