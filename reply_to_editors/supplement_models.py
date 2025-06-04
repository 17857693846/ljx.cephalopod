import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import normalize, minmax_scale
from sklearn.model_selection import train_test_split
from tqdm import trange
from sklearn.metrics import mean_squared_error, r2_score

two = pd.read_csv("two_species.csv", encoding="gbk")
sp_env = two.loc[two["Month"] == 4].iloc[:, 1: 10] #Spring environment
au_env = two.loc[two["Month"] == 11].iloc[:, 1: 10] #Autumn environment
sp_loli = two.loc[two["Month"] == 4].iloc[:, -2] #Spring Loliginidae
au_loli = two.loc[two["Month"] == 11].iloc[:, -2] #Autumn Loliginidae
sp_sep = two.loc[two["Month"] == 4].iloc[:, -1] #Spring Sepiolidae
au_sep = two.loc[two["Month"] == 11].iloc[:, -1] #Autumn Sepiolidae

def explain_var(a, b):
    residual = a-b
    return 1-np.var(residual)/np.var(a)

def output_results(x, y, sp_season="Loli_Spring"):
    result1 = pd.DataFrame()
    result2 = pd.DataFrame()
    result3 = pd.DataFrame()
    result4 = pd.DataFrame()
    result5 = pd.DataFrame()
    result6 = pd.DataFrame()
    for _ in trange(100):
        train_X, test_X, train_y, test_y = train_test_split(x, np.log(y+1), test_size=0.2)
        rf = RandomForestRegressor(n_estimators=300, min_samples_split=10, min_samples_leaf=4)
        rf.fit(train_X, train_y)
        predict_y1 = rf.predict(test_X)
        r21 = r2_score(test_y, predict_y1)
        mse1 = mean_squared_error(test_y, predict_y1)
        expv1 = explain_var(test_y, predict_y1)
        result1 = result1._append({"model": "RF", "r2": r21,
                                   "mse": mse1, "explainvar": expv1}, ignore_index=True)

        xgb = XGBRegressor(n_estimators=400, learning_rate=0.02, max_depth=3)
        xgb.fit(train_X, train_y)
        predict_y2 = xgb.predict(test_X)
        r22 = r2_score(test_y, predict_y2)
        mse2 = mean_squared_error(test_y, predict_y2)
        expv2 = explain_var(test_y, predict_y2)
        result2 = result2._append({"model": "XGB", "r2": r22,
                                   "mse": mse2, "explainvar": expv2}, ignore_index=True)

        gbdt = GradientBoostingRegressor(n_estimators=500, learning_rate=0.01, max_depth=3)
        gbdt.fit(train_X, train_y)
        predict_y3 = gbdt.predict(test_X)
        r23 = r2_score(test_y, predict_y3)
        mse3 = mean_squared_error(test_y, predict_y3)
        expv3 = explain_var(test_y, predict_y3)
        result3 = result3._append({"model": "GBDT", "r2": r23,
                                   "mse": mse3, "explainvar": expv3}, ignore_index=True)

        ann = MLPRegressor(hidden_layer_sizes=(20, 20), activation="logistic", learning_rate=0.01)
        ann.fit(train_X, train_y)
        predict_y4 = ann.predict(test_X)
        r24 = r2_score(test_y, predict_y4)
        mse4 = mean_squared_error(test_y, predict_y4)
        expv4 = explain_var(test_y, predict_y4)
        result4 = result4._append({"model": "ANN", "r2": r24,
                                   "mse": mse4, "explainvar": expv4}, ignore_index=True)

        knn = KNeighborsRegressor(n_neighbors=9, p=1, weights="uniform")
        knn.fit(train_X, train_y)
        predict_y5 = knn.predict(test_X)
        r25 = r2_score(test_y, predict_y5)
        mse5 = mean_squared_error(test_y, predict_y5)
        expv5 = explain_var(test_y, predict_y5)
        result5 = result5._append({"model": "KNN", "r2": r25,
                                   "mse": mse5, "explainvar": expv5}, ignore_index=True)

        svm = SVR(C=1, epsilon=0.5, kernel="linear")
        svm.fit(train_X, train_y)
        predict_y6 = svm.predict(test_X)
        r26 = r2_score(test_y, predict_y6)
        mse6 = mean_squared_error(test_y, predict_y6)
        expv6 = explain_var(test_y, predict_y6)
        result6 = result6._append({"model": "SVM", "r2": r26,
                                   "mse": mse6, "explainvar": expv6}, ignore_index=True)

        time.sleep(0.5)

    result = pd.concat([result1, result2, result3, result4, result5, result6], axis=0)
    print(result)
    result.to_csv(f"{sp_season}_result.csv")

if __name__=="__main__":
    output_results(sp_env, sp_loli, sp_season="spring_loli")
    output_results(sp_env, sp_sep, sp_season="spring_sep")
    output_results(au_env, au_loli, sp_season="autumn_loli")
    output_results(au_env, au_sep, sp_season="autumn_sep")
