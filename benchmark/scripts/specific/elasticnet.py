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


parser = argparse.ArgumentParser(description='ElasticNet  Regression')
# parser.add_argument('--data_scale', type=str, default='5k', help='[5k, 1w, 5w, ...]')
parser.add_argument('--n_iter', type=int, default=20, help='Number of parameter settings that are sampled.')
parser.add_argument('--seed', type=int, default=1, help='Random state.')
# parser.add_argument('--yeojohnson', action='store_true', default=False, help='whether transform label value')
parser.add_argument('--data_path', type=str, default='../../dataset/domain_specific/', help='the default data path')
parser.add_argument('--cv', action='store_true', default=False, help='whether transform label value')
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

en = ElasticNet(random_state=seed)
# cv = KFold(n_splits=5, shuffle=True, random_state=seed)
cv = KFold(n_splits=5, shuffle=False)

if args.cv:
    # RandomizedSearchCV for best parameters (skewed version)
    n_iter_search = args.n_iter
    param_dist = {
        "l1_ratio": stats.uniform(0, 1),
        "alpha": [1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3],
    }

    random_search = RandomizedSearchCV(
        en, param_distributions=param_dist,
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
    en_best = ElasticNet(alpha=random_search.best_params_["alpha"],
                         l1_ratio=random_search.best_params_["l1_ratio"],
                         random_state=seed)
else:
    en_best = ElasticNet(alpha=0.001,
                         l1_ratio=0.046450412719997725,
                         random_state=seed)

start1 = time()
en_best.fit(train_df_features.values, train_df_labels.values)
end1 = time()
train_pred = en_best.predict(train_df_features.values).reshape(-1)
end2 = time()
test_pred1 = en_best.predict(test_df1_features.values).reshape(-1)
end3 = time()
test_pred2 = en_best.predict(test_df2_features.values).reshape(-1)
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
