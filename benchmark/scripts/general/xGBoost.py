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
# import seaborn as sns
import matplotlib.pyplot as plt

# Models
# from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBRegressor

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


parser = argparse.ArgumentParser(description='XGBoost  Regression')
parser.add_argument('--data_scale', type=str, default='5k', help='[5k, 1w, 5w, ...]')
parser.add_argument('--n_iter', type=int, default=100, help='Number of parameter settings that are sampled.')
parser.add_argument('--yeojohnson', action='store_true', default=False, help='whether transform label value')
parser.add_argument('--seed', type=int, default=42, help='Random state.')
parser.add_argument('--data_path', type=str, default='../../dataset/domain_general/', help='the default data path')
parser.add_argument('--cv', action='store_true', default=False, help='whether transform label value')
parser.add_argument('--gpu', action='store_true', default=False, help='whether use gpu')
parser.add_argument('--gpu_id', type=int, default=1, help='GPU ID.')

parser.add_argument('--max_depth', type=int, default=5, help='GPU ID.')
parser.add_argument('--n_estimators', type=int, default=3000, help='GPU ID.')
parser.add_argument('--tree_method', type=str, default='hist', help='the default data path')
# parser.add_argument('--specific', action='store_true', default=False, help='whether transform label value')
# parser.add_argument('--num_clusters', type=int, default=7, help='train epochs')
# parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
# parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
args = parser.parse_args()
seed = args.seed
gpu_id = args.gpu_id
df = pd.read_csv(args.data_path + 'random_sample_' + args.data_scale + '.csv')
# if not args.specific:
#     df = pd.read_csv(args.data_path + 'random_sample_' + args.data_scale + '.csv')
# else:
#     df = pd.read_csv(args.data_path + 'domain_specific_node_278domain_10prov_29.csv')
df_features, df_labels = feature_engineering(df)

if args.yeojohnson:
    print("Perform Yeojohnson transformation for target label")
    label_values, lmbda = stats.yeojohnson(df_labels.values*100)
else:
    label_values = df_labels.values*100

# cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv = KFold(n_splits=5, shuffle=False)
# RandomizedSearchCV for best parameters (skewed version)
n_iter_search = args.n_iter

if args.cv:
    if args.gpu:
        xg = XGBRegressor(learning_rate=0.01,
                          # gamma=0.6,
                          objective='reg:absoluteerror',
                          # nthread=5,
                          seed=seed,
                          reg_alpha=1,
                          tree_method='gpu_hist',
                          gpu_id=gpu_id,
                          random_state=seed)
    else:
        xg = XGBRegressor(learning_rate=0.01,
                          # gamma=0.6,
                          objective='reg:absoluteerror',
                          # nthread=5,
                          seed=seed,
                          reg_alpha=1,
                          tree_method='gpu_hist',
                          # gpu_id=gpu_id,
                          random_state=seed)
    '''
    Parameter Tuning for XGBoost
    1. https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html
    2. https://xgboost.readthedocs.io/en/stable/parameter.html
    '''
    param_dist = {
        "max_depth": [5, 10, 25, 50, 100],
        "n_estimators": [10, 50, 100, 500, 1000, 2500, 3000],
        # "min_child_samples": [18, 19, 20, 21, 22],
        # "min_child_weight": [0.001, 0.002],
        # "subsample": stats.uniform(0, 1),
        # "colsample_bytree": stats.uniform(0, 1),
    }
    random_search = RandomizedSearchCV(
        xg, param_distributions=param_dist,
        n_iter=n_iter_search,
        scoring='neg_mean_absolute_error',
        cv=cv,
        refit=False,
        random_state=seed
    )

    start = time()
    random_search.fit(df_features.values, label_values)
    print(
        "RandomizedSearchCV took %.2f seconds for %d candidates parameter settings."
        % ((time() - start), n_iter_search)
    )
    print("Best parameters: ", random_search.best_params_)
    # CV on original data
    xg_best = XGBRegressor(learning_rate=0.01,
                           # gamma=0.6,
                           max_depth=random_search.best_params_["max_depth"],
                           n_estimators=random_search.best_params_["n_estimators"],
                           # subsample=random_search.best_params_["subsample"],
                           # colsample_bytree=random_search.best_params_["colsample_bytree"],
                           objective='reg:absoluteerror',
                           # nthread=5,
                           seed=seed,
                           reg_alpha=1,
                           tree_method='gpu_hist',
                           gpu_id=gpu_id,
                           random_state=seed)
else:
    print(f"No cross validation")
    if args.gpu:
        xg_best = XGBRegressor(learning_rate=0.01,
                               # gamma=0.6,
                               max_depth=args.max_depth,
                               n_estimators=args.n_estimators,
                               # subsample=random_search.best_params_["subsample"],
                               # colsample_bytree=random_search.best_params_["colsample_bytree"],
                               objective='reg:absoluteerror',
                               # nthread=5,
                               seed=seed,
                               reg_alpha=1,
                               tree_method='gpu_hist',
                               gpu_id=gpu_id,
                               random_state=seed)
    else:
        print(f"No GPU")
        xg_best = XGBRegressor(learning_rate=0.01,
                               # gamma=0.6,
                               max_depth=args.max_depth,
                               n_estimators=args.n_estimators,
                               objective='reg:absoluteerror',
                               # nthread=5,
                               seed=seed,
                               reg_alpha=1,
                               tree_method=args.tree_method,
                               # gpu_id=gpu_id,
                               random_state=seed)
df_labels_o = pd.DataFrame(df_labels, index=None, columns=['buffer_rate'])
df_labels_t = pd.DataFrame(label_values, index=None, columns=['buffer_rate'])

mae_lst = []
smape_lst = []
train_time = []
infer_time = []
for fold, (train_idx, val_idx) in enumerate(cv.split(df_features)):
    print(f"Fold {fold} start")
    X_train, X_val = df_features.iloc[train_idx, :].values, df_features.iloc[val_idx, :].values
    y_train, y_val = df_labels_t.iloc[train_idx].values, df_labels_o.iloc[val_idx].values
    start1 = time()
    xg_best.fit(X_train, y_train)
    end1 = time()
    train_time.append(end1 - start1)

    start2 = time()
    test_pred = xg_best.predict(X_val)
    end2 = time()
    infer_time.append(end2 - start2)
    print(f"Fold {fold} training time is {end1 - start1:.2f} s, testing time is {end2 - start2:.2f} s")
    if args.yeojohnson:
        test_array = inv_boxcox(test_pred.reshape(-1), lmbda) - 1
    else:
        test_array = test_pred.reshape(-1)
    test_array = test_array / 100
    mae_lst.append(mae(y_val, test_array))
    smape_lst.append(smape(y_val, test_array))
print(f"Average Training time: {np.mean(train_time):.2f} seconds")
print(f"Average Inference time: {np.mean(infer_time):.2f} seconds")
print(f"Mean MAE score: {np.mean(mae_lst)} (std: {np.std(mae_lst)})")
print(f"Mean SMAPE score: {np.mean(smape_lst)} (std: {np.std(smape_lst)})")
