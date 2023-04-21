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
from datetime import datetime
# Plots
import seaborn as sns
import matplotlib.pyplot as plt

# Models
# from lightgbm import LGBMRegressor
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
from utils import *
# torch
import torch
from torch import nn
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
# Ignore useless warnings
import warnings
warnings.filterwarnings(action="ignore")


parser = argparse.ArgumentParser(description='ElasticNet  Regression')
# parser.add_argument('--data_scale', type=str, default='5k', help='[5k, 1w, 5w, ...]')
# parser.add_argument('--n_iter', type=int, default=100, help='Number of parameter settings that are sampled.')
# parser.add_argument('--yeojohnson', action='store_true', default=False, help='whether transform label value')
parser.add_argument('--data_path', type=str, default='../../dataset/domain_specific/', help='the default data path')
# parser.add_argument('--specific', action='store_true', default=False, help='whether transform label value')

parser.add_argument('--seed', type=int, default=4, help='Random state.')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
parser.add_argument('--learning_rate', type=float, default=1, help='optimizer learning rate')
# parser.add_argument('--num_clusters', type=int, default=7, help='train epochs')
# parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
# parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
args = parser.parse_args()
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

df = pd.read_csv(args.data_path + 'domain_specific_isp_11domain_10prov_18.csv')
# s1, s2, s3 = 7952, 14309, 20397
s = len(df) // 3
train_df = df.iloc[:s, :]
train_df_1 = df.iloc[:s, :].sample(n=8000, random_state=seed)
train_df_1.sort_index(ascending=True, inplace=True)
test_df1 = df.iloc[s:s*2, :].sample(n=8000, random_state=seed)
test_df1.sort_index(ascending=True, inplace=True)
test_df2 = df.iloc[s*2:s*3, :].sample(n=8000, random_state=seed)
test_df2.sort_index(ascending=True, inplace=True)

train_df_features, train_df_labels = feature_engineering(train_df, False)
train_df_features_1, train_df_labels_1 = feature_engineering(train_df_1, False)
test_df1_features, test_df1_labels = feature_engineering(test_df1, False)
test_df2_features, test_df2_labels = feature_engineering(test_df2, False)
train_df_features.drop(labels=['domain_name', 'prov', 'isp', 'id'], axis=1, inplace=True)
train_df_features_1.drop(labels=['domain_name', 'prov', 'isp', 'id'], axis=1, inplace=True)
test_df1_features.drop(labels=['domain_name', 'prov', 'isp', 'id'], axis=1, inplace=True)
test_df2_features.drop(labels=['domain_name', 'prov', 'isp', 'id'], axis=1, inplace=True)
# df['id'] = df['node_name'] + df['domain_name']
idx_lst = train_df_features.columns.to_list()
dic_num, dic_idx = {}, {}
for f in ['node_name']:
    le = LabelEncoder()
    le.fit(df[f])
    dic1 = {x: le.transform([x])[0] for x in df[f].unique()}
    train_df_features[f] = train_df_features[f].map(dic1)
    train_df_features_1[f] = train_df_features_1[f].map(dic1)
    test_df1_features[f] = test_df1_features[f].map(dic1)
    test_df2_features[f] = test_df2_features[f].map(dic1)
    dic_num[f] = len(df[f].unique())
    dic_idx[f] = idx_lst.index(f)

cv = KFold(n_splits=5, shuffle=False)
device = torch.device('cuda:{}'.format(args.gpu))
learning_rate = args.learning_rate
num_epochs = args.train_epochs
batch_size = args.batch_size

# Feature Standardscaler
scaler = StandardScaler()
scaler.fit(train_df_features.values)
train_df_array = scaler.transform(train_df_features.values)
train_df_array_1 = scaler.transform(train_df_features_1.values)
test_df1_array = scaler.transform(test_df1_features.values)
test_df2_array = scaler.transform(test_df2_features.values)
for f in ['node_name']:
    idx = dic_idx[f]
    train_df_array[:, idx] = train_df_features[f].values
    train_df_array_1[:, idx] = train_df_features_1[f].values
    test_df1_array[:, idx] = test_df1_features[f].values
    test_df2_array[:, idx] = test_df2_features[f].values


train_dataset = CustomDataset(train_df_array, train_df_labels.values[:, np.newaxis])
train_dataset_1 = CustomDataset(train_df_array_1, train_df_labels_1.values[:, np.newaxis])
val_dataset_1 = CustomDataset(test_df1_array, test_df1_labels.values[:, np.newaxis])
val_dataset_2 = CustomDataset(test_df2_array, test_df2_labels.values[:, np.newaxis])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_loader_1 = DataLoader(train_dataset_1, batch_size=batch_size, shuffle=True)
val_loader_1 = DataLoader(val_dataset_1, batch_size=batch_size, shuffle=True)
val_loader_2 = DataLoader(val_dataset_2, batch_size=batch_size, shuffle=True)
dimension = train_df_features.shape[1]

model = FullNet(dic_num, dic_idx, dimension, device).to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.92)
path = './tmp/dnn_tc/' + datetime.now().strftime("%Y_%m_%d/%H_%M_%S/")
exp = Exp(model, criterion, optimizer, scheduler, path, device)

start1 = time()
results = exp.train(train_dataloader=train_loader, test_dataloader=val_loader_1, epochs=num_epochs)
end1 = time()
train_pred = exp.predict(train_loader_1).reshape(-1)
end2 = time()
test_pred1 = exp.predict(val_loader_1).reshape(-1)
end3 = time()
test_pred2 = exp.predict(val_loader_2).reshape(-1)
end4 = time()
print(f"Training time is {end1 - start1:.2f} s and inference time at train is {end2-end1:.2f}")
print(f"Train MAE: {mae(train_df_labels_1.values, train_pred):.4f} and SMAPE: {smape(train_df_labels_1.values, train_pred):.4f}")
print(f"Inference time at test1 is {end3 - end2:.2f} s")
print(f"Test1 MAE: {mae(test_df1_labels.values, test_pred1):.4f} and SMAPE: {smape(test_df1_labels.values, test_pred1):.4f}")
print(f"Inference time at test2 is {end4 - end3:.2f} s")
print(f"Test2 MAE: {mae(test_df2_labels.values, test_pred2):.4f} and SMAPE: {smape(test_df2_labels.values, test_pred2):.4f}")
train_d = len(train_df_features)
test1_d = len(test_df1_features)
test2_d = len(test_df2_features)
d_all = train_d + test1_d + test2_d
print(f"Average Inference time: {(end2-end1)* train_d / d_all + (end3-end2)* test1_d / d_all + (end4-end3) * test2_d / d_all:.2f} seconds")
