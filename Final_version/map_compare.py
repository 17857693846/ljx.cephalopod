import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from Data_Mining.Random_Forest.Final_version.linemap import plot_map_line_area
import heapq
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
    return flist, tlist

dataset = [11, 12]
ins = ["Loliginidae", "Sepiolidae"]
feature_names = ["Year", "Month", "Lat", "Lon", "SST", "Chl-a", "SSH", "SOI", "SSS", "DO", "Depth"]
feature_ins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
models = ["RF", "XGBoost", "GBDT"]
years = [2014, 2015, 2016, 2017, 2018, 2019]

def model(f, t, model="RF"):
    f.reset_index(drop=True)
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
    p = mo.predict(f)
    print("r2:", mo.score(f, y))
    return t, pd.Series(p, name="P")

def union(speidx=0, month=4, year=2019):
    f, t = read_file(feature_ins, [dataset[speidx]], month=month)
    o, p = model(f.iloc[:, 2:], t, model="RF")
    op = pd.concat([o.reset_index(drop=True), p], axis=1)

    all = pd.concat([f.iloc[:, :4].reset_index(drop=True), op], axis=1)
    all = all.loc[all["Year"] == year].reset_index(drop=True)
    print(all)
    return all

def map(year=2019):
    '''lo4 = union(speidx=0, month=4, year=year)
    plot_map_line_area(lo4.iloc[:, 3], lo4.iloc[:, 2], lo4["P"], lo4["Q"], title="RF for Loliginidae",
                       subtime=f"{year}/4", label="Predicted ln(CPUE+1)", figure_name=f"{year} RF loli 194.tif")
    '''
    lo11 = union(speidx=0, month=11, year=year)
    plot_map_line_area(lo11.iloc[:, 3], lo11.iloc[:, 2], lo11["P"], lo11["Q"], title="RF for Loliginidae",
                       subtime=f"{year}/11", label="Predicted ln(CPUE+1)", figure_name=f"{year} RF loli 1911.tif")
    '''se4 = union(speidx=1, month=4, year=year)
    plot_map_line_area(se4.iloc[:, 3], se4.iloc[:, 2], se4["P"], se4["E"], title="RF for Sepiolidae",
                       subtime=f"{year}/4", label="Predicted ln(CPUE+1)", figure_name=f"{year} RF sep 194.tif")
    se11 = union(speidx=1, month=11, year=year)
    plot_map_line_area(se11.iloc[:, 3], se11.iloc[:, 2], se11["P"], se11["E"], title="RF for Sepiolidae",
                       subtime=f"{year}/11", label="Predicted ln(CPUE+1)", figure_name=f"{year} RF sep 1911.tif")
    '''

def all_union(speidx=0, month=4):
    f, t = read_file(feature_ins, [dataset[speidx]], month=month)
    o, p = model(f.iloc[:, 2:], t, model="RF")
    op = pd.concat([np.log(o.reset_index(drop=True)+1), p], axis=1)

    all = pd.concat([f.iloc[:, :4].reset_index(drop=True), op], axis=1)
    all = all.loc[all.iloc[:, -2] != 0].reset_index(drop=True)
    print(all)
    return all

def scatter_line():
    sis = [0, 1]
    month = [4, 11]
    colors = ["cornflowerblue", "grey"]
    for si in sis:
        if month == 4:
            season = "spring"
        elif month == 11:
            season = "autumn"
        lo4 = all_union(speidx=si, month=month[0])
        lo11 = all_union(speidx=si, month=month[1])
        if si == 0:
            x = "Q"
        elif si == 1:
            x = "E"
        fig, ax = plt.subplots()
        sns.regplot(x=x, y="P", data=lo4, marker="+",
                    label="Spring", scatter_kws={'s':50,'color':colors[0], 'alpha':0.6},
                    line_kws={'linestyle':'--','color':'dodgerblue'})
        sns.regplot(x=x, y="P", data=lo11, marker=".",
                    label="Autumn", scatter_kws={'s':50,'color':colors[1], 'alpha':0.6},
                    line_kws={'linestyle':'-','color':'dimgray'})
        #plt.scatter(np.log(x+1), y, marker="+", color="red")
        plt.legend()
        plt.xticks([0, 2, 4, 6])
        plt.yticks([0, 2, 4, 6])
        plt.grid(True, linestyle="-.", color="gray", linewidth="0.8", axis="y")
        plt.grid(True, linestyle="-.", color="gray", linewidth="0.8", axis="x")
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="-.", color="red")
        plt.xlabel("Observed ln(CPUE+1)")
        plt.ylabel("Predicted ln(CPUE+1)")
        plt.title(f"Comparison of observed and prediction \nfor {ins[si]} with RF")
        plt.savefig(f"{ins[si]} with RF.tif")
        plt.show()

if __name__=="__main__":
    for year in [2019]:
        map(year)
