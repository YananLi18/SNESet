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
parser.add_argument('--data_scale', type=str, default='5k', help='[5k, 1w, 5w, ...]')
# parser.add_argument('--n_iter', type=int, default=100, help='Number of parameter settings that are sampled.')
parser.add_argument('--yeojohnson', action='store_true', default=False, help='whether transform label value')
parser.add_argument('--data_path', type=str, default='../../dataset/domain_general/', help='the default data path')
# parser.add_argument('--specific', action='store_true', default=False, help='whether transform label value')

parser.add_argument('--seed', type=int, default=42, help='Random state.')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
parser.add_argument('--learning_rate', type=float, default=0.01, help='optimizer learning rate')
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

df = pd.read_csv(args.data_path + 'random_sample_' + args.data_scale + '.csv')
# if not args.specific:
#     df = pd.read_csv(args.data_path + 'random_sample_' + args.data_scale + '.csv')
# else:
#     df = pd.read_csv(args.data_path + 'domain_specific_node_278domain_10prov_29.csv')
df_features, df_labels = feature_engineering(df, False)
# cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv = KFold(n_splits=5, shuffle=False)

# df_all = pd.read_csv("../../dataset/training_2nd_dataset.csv")
# df_all['id'] = df_all['node_name'] + df_all['domain_name']
idx_lst = df_features.columns.to_list()
dic_num, dic_idx = {}, {}
for f in ['domain_name', 'prov', 'isp', 'node_name', 'id']:
    le = LabelEncoder()
    le.fit(df_features[f])
    dic1 = {x: le.transform([x])[0] for x in df_features[f].unique()}
    df_features[f] = df_features[f].map(dic1)
    dic_num[f] = len(df_features[f].unique())
    dic_idx[f] = idx_lst.index(f)

if args.yeojohnson:
    print("Perform Yeojohnson transformation for target label")
    label_values, lmbda = stats.yeojohnson(df_labels.values*100)
else:
    label_values = df_labels.values*100


df_labels_o = pd.DataFrame(df_labels, index=None, columns=['buffer_rate'])
df_labels_t = pd.DataFrame(label_values, index=None, columns=['buffer_rate'])

mae_lst = []
smape_lst = []
train_time = []
infer_time = []
device = torch.device('cuda:{}'.format(args.gpu))
learning_rate = args.learning_rate
num_epochs = args.train_epochs
batch_size = args.batch_size
for fold, (train_idx, val_idx) in enumerate(cv.split(df_features)):
    x_train, x_val = df_features.iloc[train_idx, :], df_features.iloc[val_idx, :]
    y_train, y_val = df_labels_t.iloc[train_idx].values, df_labels_o.iloc[val_idx].values

    # Feature Standardscaler
    scaler = StandardScaler()
    scaler.fit(x_train.values)
    X_train = scaler.transform(x_train.values)
    X_val = scaler.transform(x_val.values)

    for f in ['domain_name', 'prov', 'isp', 'node_name', 'id']:
        idx = dic_idx[f]
        X_train[:, idx] = x_train[f].values
        X_val[:, idx] = x_val[f].values

    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    dimension = X_train.shape[1]

    model = FullNet(dic_num, dic_idx, dimension, device).to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    '''
    https://towardsdatascience.com/a-visual-guide-to-learning-rate-schedulers-in-pytorch-24bbb262c863
    '''
    # scheduler = CosineAnnealingWarmRestarts(optimizer,
    #                                         T_0=10,  # Number of iterations for the first restart
    #                                         T_mult=2,  # A factor increases TiTi​ after a restart
    #                                         eta_min=1e-5)  # Minimum learning rate
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.92)
    path = './tmp/dnn_tc_first/' + datetime.now().strftime("%Y_%m_%d/%H_%M_%S/")
    exp = Exp(model, criterion, optimizer, scheduler, path, device)

    start = time()
    # Setup training and save the results
    results = exp.train(train_dataloader=train_loader, test_dataloader=val_loader, epochs=num_epochs)
    # End the timer and print out how long it took
    train_time.append(time() - start)

    start = time()
    test_pred = exp.predict(val_loader)
    infer_time.append(time() - start)

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
