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


parser = argparse.ArgumentParser(description='RandomForest Regression')
parser.add_argument('--data_scale', type=str, default='5k', help='[5k, 1w, 5w, ...]')
parser.add_argument('--n_iter', type=int, default=10, help='Number of parameter settings that are sampled.')
parser.add_argument('--seed', type=int, default=42, help='Random state.')
parser.add_argument('--yeojohnson', action='store_true', default=False, help='whether transform label value')
parser.add_argument('--data_path', type=str, default='../../dataset/domain_general/', help='the default data path')
parser.add_argument('--cv', action='store_true', default=False, help='whether transform label value')
parser.add_argument('--specific', action='store_true', default=False, help='whether transform label value')
# parser.add_argument('--num_clusters', type=int, default=7, help='train epochs')
# parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
# parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
args = parser.parse_args()
seed = args.seed

if not args.specific:
    df = pd.read_csv(args.data_path + 'random_sample_' + args.data_scale + '.csv')
else:
    df = pd.read_csv(args.data_path + 'domain_specific_node_278domain_10prov_29.csv')
df_features, df_labels = feature_engineering(df)
if args.yeojohnson:
    print("Perform Yeojohnson transformation for target label")
    label_values, lmbda = stats.yeojohnson(df_labels.values*100)
else:
    label_values = df_labels.values*100
cv = KFold(n_splits=5, shuffle=False)
if args.cv:
    rf = RandomForestRegressor(bootstrap=True, criterion="absolute_error",
                               random_state=seed)
    # cv = KFold(n_splits=5, shuffle=True, random_state=42)
    # RandomizedSearchCV for best parameters (skewed version)
    n_iter_search = args.n_iter
    param_dist = {
        # "n_estimators": [10, 50, 100, 500, 1000, 2500, 3000],
        "n_estimators": [5, 25, 50, 75, 100],
        "max_depth": [5, 10, 25, 50, 100],
        # "min_samples_split": stats.uniform(0, 1),
        # "min_samples_leaf": stats.uniform(0, 1),
        # "max_samples": stats.uniform(0, 1),
        # "max_features": stats.uniform(0, 1),
    }
    random_search = RandomizedSearchCV(
        rf, param_distributions=param_dist,
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
    rf_best = RandomForestRegressor(
                                    n_estimators=random_search.best_params_["n_estimators"],
                                    max_depth=random_search.best_params_["max_depth"],
                                    # min_samples_split=random_search.best_params_["min_samples_split"],
                                    # min_samples_leaf=random_search.best_params_["min_samples_leaf"],
                                    # max_samples=random_search.best_params_["max_samples"],
                                    # max_features=random_search.best_params_["max_features"],
                                    bootstrap=True,
                                    criterion="absolute_error",
                                    random_state=seed)
else:
    rf_best = RandomForestRegressor(n_estimators=100,
                                    max_depth=5,
                                    bootstrap=True,
                                    criterion="absolute_error",
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
    rf_best.fit(X_train, y_train)
    end1 = time()
    train_time.append(end1 - start1)

    start2 = time()
    test_pred = rf_best.predict(X_val)
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
