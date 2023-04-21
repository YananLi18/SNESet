# Essentials
import os
import numpy as np
import pandas as pd
import datetime
import random
import argparse
from math import sqrt
from time import time
from collections import Counter
# Plots
import seaborn as sns
import matplotlib.pyplot as plt
# Models
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# Stats
from scipy import stats
from scipy.stats import skew, norm
from scipy.special import boxcox1p, inv_boxcox
from scipy.stats import boxcox_normmax
# Misc
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.utils.fixes import loguniform
# pd.set_option('display.max_columns', None)
# pd.options.display.max_seq_items = 8000
# pd.options.display.max_rows = 8000
from pre_process import *
# Ignore useless warnings
import warnings
warnings.filterwarnings(action="ignore")


parser = argparse.ArgumentParser(description='ElasticNet  Regression')
# parser.add_argument('--data_scale', type=str, default='5k', help='[5k, 1w, 5w, ...]')
parser.add_argument('--n_iter', type=int, default=100, help='Number of parameter settings that are sampled.')
# parser.add_argument('--yeojohnson', action='store_true', default=False, help='whether transform label value')
parser.add_argument('--seed', type=int, default=4, help='Random state.')
parser.add_argument('--data_path', type=str, default='../../dataset/domain_specific/', help='the default data path')
# parser.add_argument('--specific', action='store_true', default=False, help='whether transform label value')
# parser.add_argument('--num_clusters', type=int, default=7, help='train epochs')
# parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
# parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
args = parser.parse_args()
seed = args.seed

df = pd.read_csv(args.data_path + 'domain_specific_isp_11domain_10prov_18.csv')
# s1, s2, s3 = 7952, 14309, 20397
s = len(df) // 3
train_df = df.iloc[:s, :].sample(n=8000, random_state=seed)
train_df.sort_index(ascending=True, inplace=True)
test_df1 = df.iloc[s:s*2, :].sample(n=8000, random_state=seed)
test_df1.sort_index(ascending=True, inplace=True)
test_df2 = df.iloc[s*2:s*3, :].sample(n=8000, random_state=seed)
test_df2.sort_index(ascending=True, inplace=True)

train_df_features, train_df_labels = feature_engineering(train_df, False)
test_df1_features, test_df1_labels = feature_engineering(test_df1, False)
test_df2_features, test_df2_labels = feature_engineering(test_df2, False)

df['id'] = df['node_name'] + df['domain_name']
for f in ['domain_name', 'prov', 'isp', 'node_name', 'id']:
    le = LabelEncoder()
    le.fit(df[f])
    dic1 = {x: le.transform([x])[0] for x in df[f].unique()}
    train_df_features[f] = train_df_features[f].map(dic1)
    test_df1_features[f] = test_df1_features[f].map(dic1)
    test_df2_features[f] = test_df2_features[f].map(dic1)

lg = LGBMRegressor(objective='mean_absolute_error',
                   boosting_type='dart',
                   # feature_name=df_features.columns.to_list(),
                   categorical_feature='2,3,4,5,143',
                   learning_rate=0.01,
                   bagging_seed=seed,
                   feature_fraction_seed=seed,
                   data_random_seed=seed,
                   n_jobs=5,
                   verbose=-1,
                   # device="gpu",
                   # gpu_platform_id=0,
                   # gpu_device_id=0,
                   # gpu_use_dp=False,
                   random_state=seed)
# cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv = KFold(n_splits=5, shuffle=False)
# RandomizedSearchCV for best parameters (skewed version)
n_iter_search = args.n_iter
"""
Parameter Tuning for LightGBM
1. https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html
2. https://lightgbm.readthedocs.io/en/latest/Parameters.html
3. https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
"""
param_dist = {
    "max_depth": [5, 10, 25, 50, 100],
    "max_bin": [255],
    # "num_leaves": [5, 10, 15, 20, 50],
    # "num_leaves": [50, 100, 200],
    "n_estimators": [10, 50, 100, 500, 1000, 2500, 3000],
    # "n_estimators": [10, 50, 100, 500, 1000],
    # "min_child_samples": [18, 19, 20, 21, 22],
    # "min_child_weight": [0.001, 0.002],
    # "feature_fraction": stats.uniform(0, 1),
    # "subsample": stats.uniform(0, 1),
    # "bagging_freq": [4, 8, 16],
    # "reg_alpha": [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5],
    # "reg_lambda": [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5],
    # "learning_rate": loguniform(1e-3, 0.5),
}
random_search = RandomizedSearchCV(
    lg, param_distributions=param_dist,
    n_iter=n_iter_search,
    scoring='neg_mean_absolute_error',
    cv=cv,
    refit=False,
    random_state=seed
)

start = time()
random_search.fit(train_df_features.values, train_df_labels.values)
print(
    "RandomizedSearchCV took %.2f seconds for %d candidates parameter settings."
    % ((time() - start), n_iter_search)
)
print("Best parameters: ", random_search.best_params_)
# CV on original data
lg_best = LGBMRegressor(objective='mean_absolute_error',
                        boosting_type='dart',
                        # feature_name=df_features.columns.to_list(),
                        # categorical_feature=['domain_name', 'node_name', 'prov', 'isp', 'id'],
                        # categorical_feature='name:domain_name, node_name, prov, isp, id',
                        categorical_feature='2,3,4,5,143',
                        max_depth=random_search.best_params_["max_depth"],
                        max_bin=random_search.best_params_["max_bin"],
                        # num_leaves=random_search.best_params_["num_leaves"],
                        n_estimators=random_search.best_params_["n_estimators"],
                        # min_child_samples=random_search.best_params_["min_child_samples"],
                        # min_child_weight=random_search.best_params_["min_child_weight"],
                        # feature_fraction=random_search.best_params_["feature_fraction"],
                        # subsample=random_search.best_params_["subsample"],
                        # bagging_freq=random_search.best_params_["bagging_freq"],
                        # reg_alpha=random_search.best_params_["reg_alpha"],
                        # reg_lambda=random_search.best_params_["reg_lambda"],
                        # learning_rate=random_search.best_params_["learning_rate"],
                        learning_rate=0.01,
                        bagging_seed=seed,
                        feature_fraction_seed=seed,
                        data_random_seed=seed,
                        verbose=-1,
                        random_state=seed)

start1 = time()
lg_best.fit(train_df_features.values, train_df_labels.values)
end1 = time()
train_pred = lg_best.predict(train_df_features.values).reshape(-1)
end2 = time()
test_pred1 = lg_best.predict(test_df1_features.values).reshape(-1)
end3 = time()
test_pred2 = lg_best.predict(test_df2_features.values).reshape(-1)
end4 = time()
print(f"Training time is {end1 - start1:.2f} s and inference time at train is {end2-end1:.2f}")
print(f"Train MAE: {mae(train_df_labels.values, train_pred):.4f} and SMAPE: {smape(train_df_labels.values, train_pred):.4f}")
print(f"Inference time at test1 is {end3 - end2:.2f} s")
print(f"Test1 MAE: {mae(test_df1_labels.values, test_pred1):.4f} and SMAPE: {smape(test_df1_labels.values, test_pred1):.4f}")
print(f"Inference time at test2 is {end4 - end3:.2f} s")
print(f"Test2 MAE: {mae(test_df2_labels.values, test_pred2):.4f} and SMAPE: {smape(test_df2_labels.values, test_pred2):.4f}")
train_d = len(train_df_features)
test1_d = len(test_df1_features)
test2_d = len(test_df2_features)
d_all = train_d + test1_d + test2_d
print(f"Average Inference time: {(end2-end1)* train_d / d_all + (end3-end2)* test1_d / d_all + (end4-end3) * test2_d / d_all:.2f} seconds")
