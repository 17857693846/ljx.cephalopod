# -*- coding: utf-8 -*-
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold, GridSearchCV
from tqdm import trange
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats
import json
import warnings

warnings.filterwarnings('ignore', category=UserWarning)  # 忽略特定警告

# 设置全局随机种子
BASE_SEED = 100
np.random.seed(BASE_SEED)

# 读取数据
two = pd.read_csv("two_species.csv", encoding="gbk")

def co_data():
    sp_env = two.loc[two["Month"] == 4].iloc[:, 1:10]
    sp_loli = two.loc[two["Month"] == 4].iloc[:, -2]
    sp_sep = two.loc[two["Month"] == 4].iloc[:, -1]
    au_env = two.loc[two["Month"] == 11].iloc[:, 1:10]
    au_loli = two.loc[two["Month"] == 11].iloc[:, -2]
    au_sep = two.loc[two["Month"] == 11].iloc[:, -1]
    combined_env = pd.concat([sp_env, au_env], axis=0)
    combined_target = pd.concat([
        sp_loli, au_loli
    ], axis=0)
    return combined_env, combined_target

def explain_var(a, b):
    residual = a - b
    return 1 - np.var(residual) / np.var(a)

def get_param_grid(model_name):
    if model_name == "RF":
        return {
            'n_estimators': [100, 200, 300, 400, 500],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_name == "XGB":
        return {
            'n_estimators': [100, 200, 300, 400, 500],
            'learning_rate': [0.01, 0.02, 0.05, 0.1],
            'max_depth': [3, 5, 7],
        }
    elif model_name == "GBDT":
        return {
            'n_estimators': [100, 200, 300, 400, 500],
            'learning_rate': [0.01, 0.02, 0.05, 0.1],
            'max_depth': [3, 5, 7],
        }
    elif model_name == "ANN":
        return {
            'hidden_layer_sizes': [(10,), (15,), (20,), (10, 10), (15, 15), (20, 20)],
            'alpha': [0.001, 0.01, 0.1]
        }
    elif model_name == "KNN":
        return {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }
    elif model_name == "SVM":
        return {
            'C': [0.1, 1, 10],
            'epsilon': [0.01, 0.1, 0.5],
            'kernel': ['linear', 'rbf']
        }
    return {}


def grid_search_cv(x, y, n_splits=5):
    results = []
    best_params = {}

    # 初始化模型字典
    models = {
        "RF": RandomForestRegressor(random_state=BASE_SEED),
        "XGB": XGBRegressor(random_state=BASE_SEED),
        "GBDT": GradientBoostingRegressor(random_state=BASE_SEED),
        "ANN": MLPRegressor(random_state=BASE_SEED, activation='logistic'),
        "KNN": KNeighborsRegressor(),
        "SVM": SVR()
    }

    # 创建交叉验证对象
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=BASE_SEED)

    print("Starting grid search with cross-validation...")

    # 对每个模型进行网格搜索
    for model_name, model in models.items():
        print(f"\nPerforming grid search for {model_name}...")
        start_time = time.time()

        # 获取参数网格
        param_grid = get_param_grid(model_name)

        # 设置GridSearchCV
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='r2',
            cv=kf,
            n_jobs=-1,
            verbose=1
        )

        # 执行网格搜索（对数值变换后的目标变量）
        grid_search.fit(x, np.log1p(y))

        # 记录最佳参数
        best_params[model_name] = grid_search.best_params_

        # 使用最佳参数训练最终模型
        best_model = grid_search.best_estimator_

        # 评估模型性能
        cv_results = grid_search.cv_results_
        best_index = grid_search.best_index_

        # 提取交叉验证结果
        mean_r2 = cv_results['mean_test_score'][best_index]
        std_r2 = cv_results['std_test_score'][best_index]

        # 记录结果
        results.append({
            "Model": model_name,
            "Best_Params": best_params[model_name],
            "Mean_R2": mean_r2,
            "Std_R2": std_r2,
            "Training_Time": time.time() - start_time
        })

        print(f"Completed {model_name} in {time.time() - start_time:.2f} seconds")
        print(f"Best parameters: {best_params[model_name]}")
        print(f"Score: {mean_r2:.2f} +- {std_r2:.2f}")

    # 创建结果DataFrame
    results_df = pd.DataFrame(results)

    # 保存最佳参数
    with open("best_parameters.json", "w") as f:
        json.dump(best_params, f, indent=2)

    return results_df, best_params


if __name__ == "__main__":
    # 合并数据
    env_data, target_data = co_data()
    print(f"Combined dataset size: {env_data.shape[0]} samples")

    # 执行网格搜索交叉验证
    results_df, best_params = grid_search_cv(env_data, target_data)

    # 打印并保存结果
    print("\nFinal Results:")
    print(results_df)
    results_df.to_csv("combined_species_grid_search_results.csv", index=False)

    # 输出最佳参数
    print("\nBest Parameters for Each Model:")
    for model, params in best_params.items():
        print(f"{model}: {params}")

    print("Grid search completed. Results saved to combined_species_grid_search_results.csv")