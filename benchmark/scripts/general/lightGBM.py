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
parser.add_argument('--data_scale', type=str, default='5k', help='[5k, 1w, 5w, ...]')
parser.add_argument('--n_iter', type=int, default=100, help='Number of parameter settings that are sampled.')
parser.add_argument('--yeojohnson', action='store_true', default=False, help='whether transform label value')
parser.add_argument('--seed', type=int, default=42, help='Random state.')
parser.add_argument('--data_path', type=str, default='../../dataset/domain_general/', help='the default data path')
# parser.add_argument('--specific', action='store_true', default=False, help='whether transform label value')
parser.add_argument('--gpu', action='store_true', default=False, help='whether transform label value')
# parser.add_argument('--num_clusters', type=int, default=7, help='train epochs')
# parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
# parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
args = parser.parse_args()
seed = args.seed

df = pd.read_csv(args.data_path + 'random_sample_' + args.data_scale + '.csv')
# if not args.specific:
#     df = pd.read_csv(args.data_path + 'random_sample_' + args.data_scale + '.csv')
# else:
#     df = pd.read_csv(args.data_path + 'domain_specific_node_278domain_10prov_29.csv')
df_features, df_labels = feature_engineering(df)
# for f in ['domain_name', 'node_name', 'prov', 'isp', 'id']:
#     df_features[f] = df_features[f].astype('category')
if args.yeojohnson:
    print("Perform Yeojohnson transformation for target label")
    label_values, lmbda = stats.yeojohnson(df_labels.values*100)
else:
    label_values = df_labels.values*100

if args.gpu:
    lg = LGBMRegressor(objective='mean_absolute_error',
                       boosting_type='dart',
                       # feature_name=df_features.columns.to_list(),
                       # categorical_feature='2,3,4,5,143',
                       learning_rate=0.01,
                       bagging_seed=seed,
                       feature_fraction_seed=seed,
                       data_random_seed=seed,
                       # n_jobs=5,
                       verbose=-1,
                       device="gpu",
                       gpu_platform_id=0,
                       gpu_device_id=1,
                       random_state=seed)
else:
    lg = LGBMRegressor(objective='mean_absolute_error',
                       boosting_type='dart',
                       # feature_name=df_features.columns.to_list(),
                       # categorical_feature='2,3,4,5,143',
                       learning_rate=0.01,
                       bagging_seed=seed,
                       feature_fraction_seed=seed,
                       data_random_seed=seed,
                       # n_jobs=5,
                       verbose=-1,
                       # device="gpu",
                       # gpu_platform_id=0,
                       # gpu_device_id=1,
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
random_search.fit(df_features.values, label_values)
print(
    "RandomizedSearchCV took %.2f seconds for %d candidates parameter settings."
    % ((time() - start), n_iter_search)
)
print("Best parameters: ", random_search.best_params_)
# CV on original data
if args.gpu:
    lg_best = LGBMRegressor(objective='mean_absolute_error',
                            boosting_type='dart',
                            # feature_name=df_features.columns.to_list(),
                            # categorical_feature=['domain_name', 'node_name', 'prov', 'isp', 'id'],
                            # categorical_feature='name:domain_name, node_name, prov, isp, id',
                            # categorical_feature='2,3,4,5,143',
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
                            device="gpu",
                            gpu_platform_id=0,
                            gpu_device_id=1,
                            random_state=seed)
else:
    lg_best = LGBMRegressor(objective='mean_absolute_error',
                            boosting_type='dart',
                            # feature_name=df_features.columns.to_list(),
                            # categorical_feature=['domain_name', 'node_name', 'prov', 'isp', 'id'],
                            # categorical_feature='name:domain_name, node_name, prov, isp, id',
                            # categorical_feature='2,3,4,5,143',
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
                            # device="gpu",
                            # gpu_platform_id=0,
                            # gpu_device_id=1,
                            random_state=seed)
df_labels_o = pd.DataFrame(df_labels, index=None, columns=['buffer_rate'])
df_labels_t = pd.DataFrame(label_values, index=None, columns=['buffer_rate'])

mae_lst = []
smape_lst = []
train_time = []
infer_time = []
for fold, (train_idx, val_idx) in enumerate(cv.split(df_features)):
    X_train, X_val = df_features.iloc[train_idx, :].values, df_features.iloc[val_idx, :].values
    y_train, y_val = df_labels_t.iloc[train_idx].values, df_labels_o.iloc[val_idx].values
    start1 = time()
    lg_best.fit(X_train, y_train)
    end1 = time()
    train_time.append(end1 - start1)

    start2 = time()
    test_pred = lg_best.predict(X_val)
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
