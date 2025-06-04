import Data_Mining.Random_Forest.Final_version.Predict_by_singular_factor as data_generate
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from Data_Mining.Random_Forest.Final_version import PlotFigure as pf
import heapq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MultipleLocator
matplotlib.rcParams['figure.figsize'] = [8, 6]
plt.rcParams["font.size"] = 20
plt.rcParams["font.sans-serif"] = "Times New Roman"
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.dpi"] = 300

def read_file(feature_li, target_li, month=4):
    d4 = pd.read_csv("two_species.csv", encoding="gbk")
    d4 = d4.loc[d4["Month"] == month]
    '''d4 = d4.loc[d4["Q"] != 0]
    d4 = d4.loc[d4["E"] !=0]'''
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
feature_names = ["Year", "Month", "Lat", "Lon", "SST", "Chla", "SSH", "SOI", "SSS", "DO", "BT"]
feature_ins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
models = ["RF", "XGBoost", "GBDT"]
years = [2014, 2015, 2016, 2017, 2018, 2019]

def model(f, t, model="RF", environment_factor="SST"):
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
    print(f)
    mo.fit(f, y)

    validation = data_generate.generate_test_dataset(environment_factor, season=4, step=50).drop_duplicates()
    level = mo.predict(validation)
    factors = validation[environment_factor]
    return np.array(factors), level

if __name__=="__main__":
    environment_factor = "SST"
    colors = ["lightsalmon", "lightsteelblue"]
    for m in [4, 11]:
        for i in [0, 1]:
            f, t = read_file(feature_ins[2:], [dataset[i]], month=m)
            x, y = model(f, t, model="RF", environment_factor=environment_factor)
            cubic_interpolation_model = interp1d(x, y, kind="cubic")
            X_ = np.linspace(x.min(), x.max(), 500)
            Y_ = cubic_interpolation_model(X_)
            plt.plot(X_, Y_, label=ins[i], color=colors[i])
        axes = plt.subplot()
        axes.minorticks_on()
        axes.xaxis.set_minor_locator(MultipleLocator(0.4))
        plt.legend()
        #$^\circ$C mg/m$^{2}$
        if environment_factor=="SST":
            plt.xlabel(environment_factor+' ($^\circ$C)')
        elif environment_factor=="Chla":
            plt.xlabel("Chl-a" + ' (mg/m$^{3}$)')
        plt.ylabel("Abundance(ln(CPUE+1))")
        if m == 4:
            ti = 'Spring'
        else:
            ti = 'Autumn'
        plt.title(f"{ti}")
        plt.savefig(f"{environment_factor} in {ti}.svg")
        plt.savefig(f"{environment_factor} in {ti}.tif")
        plt.show()





