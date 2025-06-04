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
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MultipleLocator
matplotlib.rcParams['figure.figsize'] = [6, 5]
plt.rcParams["font.size"] = 16
plt.rcParams["font.sans-serif"] = "Times New Roman"
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.dpi"] = 300
#plt.style.use(['science'])

np.random.seed(100)
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
feature_names = ["Year", "Month", "Lat", "Lon", "SST", "Chl-a", "SSH", "ONI", "SSS", "DO", "Depth"]
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
    #print(f)
    mo.fit(f, y)

    validation = data_generate.generate_test_dataset(environment_factor, season=4, step=50).drop_duplicates()
    level = mo.predict(validation)
    factors = validation[environment_factor]
    return np.array(factors), level

if __name__=="__main__":
    #SST Chla
    environment_factor = "Chla"
    colors = ["lightsalmon", "lightsteelblue"]
    for m in [4, 11]:
        dataAll = pd.DataFrame()
        for i in [0, 1]:
            f, t = read_file(feature_ins[2:], [dataset[i]], month=m)
            xs = []
            ys = []
            zs = []
            for _ in range(10):
                x, y = model(f, t, model="RF", environment_factor=environment_factor)
                xs.append(x)
                ys.append(y)
                strs = [ins[i] for k in range(len(x))]
                zs.append(strs)
            xx = [item for sublst in xs for item in sublst]
            yy = [item for sublst in ys for item in sublst]
            zz = [item for sublst in zs for item in sublst]
            dd = {"x": xx, "y": yy, "Species": zz}
            data = pd.DataFrame(dd)
            dataAll = pd.concat([dataAll, data])
        print("data", dataAll)
        sns.relplot(x="x", y="y", data=dataAll, hue="Species", kind="line", label=ins[i], color=colors[i])
        axes = plt.subplot()
        axes.minorticks_on()
        axes.xaxis.set_minor_locator(MultipleLocator(1))
        #plt.legend()
        #$^\circ$C mg/m$^{2}$
        if environment_factor=="SST":
            plt.xlabel(environment_factor+' ($^\circ$C)')
        elif environment_factor=="Chla":
            plt.xlabel("Chl-a" + ' (mg/m$^{3}$)')
        plt.ylim([0, 3])
        plt.ylabel("ln(CPUE+1)")
        if m == 4:
            ti = '(C) Spring  '
        else:
            ti = '(D) Autumn  '
        plt.title(f"{ti}")
        plt.savefig(f"{ti} for {environment_factor}.svg", bbox_inches="tight")
        plt.savefig(f"{ti} for {environment_factor}.tiff", bbox_inches="tight")
        plt.show()

