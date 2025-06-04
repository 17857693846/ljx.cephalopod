# -*- coding: utf-8 -*-
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from tqdm import trange
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats
BASE_SEED = 100
np.random.seed(BASE_SEED)
BEST_PARAMS = {
    "RF": {"min_samples_leaf": 4, "min_samples_split": 10, "n_estimators": 300},
    "XGB": {"learning_rate": 0.02, "max_depth": 3, "n_estimators": 400},
    "GBDT": {"learning_rate": 0.01, "max_depth": 3, "n_estimators": 500},
    "ANN": {"alpha": 0.01, "hidden_layer_sizes": (20, 20)},
    "KNN": {"n_neighbors": 9, "p": 1, "weights": "uniform"},
    "SVM": {"C": 1, "epsilon": 0.5, "kernel": "linear"}
}

two = pd.read_csv("two_species.csv", encoding="gbk")
sp_env = two.loc[two["Month"] == 4].iloc[:, 1:10]  # Spring environment
au_env = two.loc[two["Month"] == 11].iloc[:, 1:10]  # Autumn environment
sp_loli = two.loc[two["Month"] == 4].iloc[:, -2]  # Spring Loliginidae
au_loli = two.loc[two["Month"] == 11].iloc[:, -2]  # Autumn Loliginidae
sp_sep = two.loc[two["Month"] == 4].iloc[:, -1]  # Spring Sepiolidae
au_sep = two.loc[two["Month"] == 11].iloc[:, -1]  # Autumn Sepiolidae


def explain_var(a, b):
    residual = a - b
    return 1 - np.var(residual) / np.var(a)


def run_experiment(x, y, season, species, n_iter=100):
    all_results = []

    for i in trange(n_iter, desc=f"{season}_{species}"):
        iter_seed = BASE_SEED + i

        train_X, test_X, train_y, test_y = train_test_split(
            x, np.log(y + 1), test_size=0.2, random_state=iter_seed
        )

        iteration_results = []

        rf = RandomForestRegressor(**BEST_PARAMS["RF"], random_state=iter_seed)
        rf.fit(train_X, train_y)
        predict_y = rf.predict(test_X)
        iteration_results.append({
            "model": "RF",
            "r2": r2_score(test_y, predict_y),
            "mse": mean_squared_error(test_y, predict_y),
            "explainvar": explain_var(test_y, predict_y)
        })

        xgb = XGBRegressor(**BEST_PARAMS["XGB"], random_state=iter_seed)
        xgb.fit(train_X, train_y)
        predict_y = xgb.predict(test_X)
        iteration_results.append({
            "model": "XGB",
            "r2": r2_score(test_y, predict_y),
            "mse": mean_squared_error(test_y, predict_y),
            "explainvar": explain_var(test_y, predict_y)
        })

        gbdt = GradientBoostingRegressor(**BEST_PARAMS["GBDT"], random_state=iter_seed)
        gbdt.fit(train_X, train_y)
        predict_y = gbdt.predict(test_X)
        iteration_results.append({
            "model": "GBDT",
            "r2": r2_score(test_y, predict_y),
            "mse": mean_squared_error(test_y, predict_y),
            "explainvar": explain_var(test_y, predict_y)
        })

        ann = MLPRegressor(**BEST_PARAMS["ANN"], random_state=iter_seed, max_iter=1000)
        ann.fit(train_X, train_y)
        predict_y = ann.predict(test_X)
        iteration_results.append({
            "model": "ANN",
            "r2": r2_score(test_y, predict_y),
            "mse": mean_squared_error(test_y, predict_y),
            "explainvar": explain_var(test_y, predict_y)
        })

        knn = KNeighborsRegressor(**BEST_PARAMS["KNN"])
        knn.fit(train_X, train_y)
        predict_y = knn.predict(test_X)
        iteration_results.append({
            "model": "KNN",
            "r2": r2_score(test_y, predict_y),
            "mse": mean_squared_error(test_y, predict_y),
            "explainvar": explain_var(test_y, predict_y)
        })

        svm = SVR(**BEST_PARAMS["SVM"])
        svm.fit(train_X, train_y)
        predict_y = svm.predict(test_X)
        iteration_results.append({
            "model": "SVM",
            "r2": r2_score(test_y, predict_y),
            "mse": mean_squared_error(test_y, predict_y),
            "explainvar": explain_var(test_y, predict_y)
        })

        for res in iteration_results:
            res.update({
                "season": season,
                "species": species,
                "iteration": i
            })

        all_results.extend(iteration_results)

    return pd.DataFrame(all_results)


def calculate_summary(results_df):
    summary = results_df.groupby(["season", "species", "model"]).agg({
        "r2": ["mean", "std"],
        "mse": ["mean", "std"],
        "explainvar": ["mean", "std"]
    }).reset_index()


    summary.columns = [
        "Season", "Species", "Model",
        "R2_Mean", "R2_SD",
        "MSE_Mean", "MSE_SD",
        "ExplainVar_Mean", "ExplainVar_SD"
    ]

    return summary


if __name__ == "__main__":
    all_results = []

    print("\nRunning experiments for Spring Loliginidae...")
    df_sp_loli = run_experiment(sp_env, sp_loli, "spring", "loli")
    all_results.append(df_sp_loli)

    print("\nRunning experiments for Spring Sepiolidae...")
    df_sp_sep = run_experiment(sp_env, sp_sep, "spring", "sep")
    all_results.append(df_sp_sep)

    print("\nRunning experiments for Autumn Loliginidae...")
    df_au_loli = run_experiment(au_env, au_loli, "autumn", "loli")
    all_results.append(df_au_loli)

    print("\nRunning experiments for Autumn Sepiolidae...")
    df_au_sep = run_experiment(au_env, au_sep, "autumn", "sep")
    all_results.append(df_au_sep)

    final_results = pd.concat(all_results, ignore_index=True)

    final_results.to_csv("all_results_detailed.csv", index=False)

    summary_df = calculate_summary(final_results)
    summary_df.to_csv("results_summary.csv", index=False)

    print("\nFinal Summary Results:")
    print(summary_df)

    print("\nEvaluation completed. Results saved to:")
    print("- all_results_detailed.csv (detailed results for all iterations)")
    print("- results_summary.csv (summary statistics)")