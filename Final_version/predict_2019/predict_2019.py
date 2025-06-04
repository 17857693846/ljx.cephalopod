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

np.random.seed(100)
dataset = [11, 12]
ins = ["Loliginidae", "Sepiolidae"]
feature_names = ["Year", "Month", "Lat", "Lon", "SST", "Chl-a", "SSH", "SOI", "SSS", "DO", "Depth"]
feature_ins = [2, 3, 4, 5, 6, 7, 8, 9, 10]
models = ["RF", "XGBoost", "GBDT"]
years = [2014, 2015, 2016, 2017, 2018, 2019]

all_data = pd.read_csv("two_species.csv")
spring_data = all_data.loc[all_data["Month"] == 4]
da1418 = spring_data.loc[all_data["Year"] != 2019]
da19 = spring_data.loc[all_data["Year"] == 2019]

X_train = da1418.iloc[:, feature_ins]
q_train = da1418.iloc[:, -2]
e_train = da1418.iloc[:, -1]

X_test = da19.iloc[:, feature_ins]
q_test = da19.iloc[:, -2]
e_test = da19.iloc[:, -1]

rf1 = RandomForestRegressor(n_estimators=200, min_samples_split=2,
                           min_samples_leaf=1, ccp_alpha=0.01)
rf1.fit(X_train, q_train)
q_predict = rf1.predict(X_test)

rf2 = RandomForestRegressor(n_estimators=200, min_samples_split=2,
                           min_samples_leaf=1, ccp_alpha=0.01)
rf2.fit(X_train, e_train)
e_predict = rf2.predict(X_test)

X_test["QPS"] = minmax_scale(q_predict)
X_test["EPS"] = minmax_scale(e_predict)

X_test.to_csv("predict_2019_4.csv")



