# Essentials
import numpy as np
import pandas as pd
from math import sqrt
from collections import Counter

from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings(action="ignore")


def logs(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(np.log(1.01+res[l])).values)
        res.columns.values[m] = l + '_log'
        m += 1
    return res


def exps(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(np.exp(res[l])).values)
        res.columns.values[m] = l + '_exp'
        m += 1
    return res


def squares(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(res[l]*res[l]).values)
        res.columns.values[m] = l + '_sq'
        m += 1
    return res


def calculate_nctt(rtt, loss_rate):
    return 8 * rtt + loss_rate * 8 * 2.5


def calculate_throughput(rtt, loss_rate):
    if (rtt-0.0) < 1e-6 or (loss_rate-0.0) < 1e-6:
        return 10
    return 1460.0 / rtt / 1024 * (1 / sqrt(loss_rate)) # unit MB/s


def calculate_throughput_improved(rtt, loss_rate):
    if (rtt-0.0) < 1e-6 or (loss_rate-0.0) < 1e-6:
        return 10
    return min(4194304.0/rtt/1024, 1/(rtt*sqrt(4/3*loss_rate) +
                                      min(1, 3*sqrt(0.75*loss_rate))*loss_rate*(1+32*loss_rate*loss_rate)))


def get_outliers(df, columns, weight=1.5):
    indices = []
    for col in columns:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        outlier_list = df[(df[col] < Q1 - weight * IQR) | (df[col] > Q3 + weight * IQR)].index
        indices.extend(outlier_list)
    indices = Counter(indices)
    result = []
    for i in indices:
        if indices[i] > 2:
            result.append(i)
    return result


def feature_engineering(df, cat_trans_flag=True):
    # 1. Remove Outliers
    '''
    @Source:
    https://www.kaggle.com/code/joonasyoon/dl-practice-on-regression#Outliers
    '''
    dff = df.copy()
    # out_num_lst = ['avg_fbt_time', 'tcp_conntime', 'icmp_rtt', 'inner_network_rtt']
    # outliers = get_outliers(dff, out_num_lst)
    # dff.drop(index=outliers, axis=0, inplace=True)
    # print('Remove {} outlier(s)'.format(len(outliers)))

    # 2. Rolling Window
    for f1 in ['icmp_lossrate', 'synack1_ratio', 'icmp_rtt', 'avg_fbt_time', 'reset_ratio']:
        for f2 in ['domain_name', 'node_name']:
            for i in range(1, 4):
                dff[f1 + '_' + f2 + '_shift_%d' % i] = dff.groupby(f2)[f1].shift(i)
            for i in range(1, 3):
                dff[f1 + '_' + f2 + '_shift_-%d' % i] = dff.groupby(f2)[f1].shift(-i)
            dff[f1 + '_' + f2 + '_mean_decay'] = dff[f1] * 0.5 + dff[f1 + '_' + f2 + '_shift_1'] * 0.3 + \
                                                dff[f1 + '_' + f2 + '_shift_2'] * 0.1 + \
                                                dff[f1 + '_' + f2 + '_shift_3'] * 0.1
        dff[f1 + '_rolling_mean_10'] = dff[f1].rolling(window=10, min_periods=1, center=True).mean()
        dff[f1 + '_rolling_mean_30'] = dff[f1].rolling(window=30, min_periods=1, center=True).mean()
    dff = dff.fillna(0)

    df_features = dff.copy()
    # df_features[['First', 'domain']] = df_features.domain_name.str.split("_", expand=True)
    # df_features.drop(labels=['First'], axis=1, inplace=True)
    # df_features[['First', 'node']] = df_features.node_name.str.split("_", expand=True)
    # df_features.drop(labels=['First'], axis=1, inplace=True)
    # df_features['domain'] = df_features['domain'].astype('int64')
    # df_features['node'] = df_features['node'].astype('int64')

    df_features['odd_synack1_ratio'] = 1 - df_features['synack1_ratio']
    df_features['odd_icmp_lossrate'] = 1 - df_features['icmp_lossrate']
    df_features['odd_ratio_499_5xx'] = 1 - df_features['ratio_499_5xx']
    df_features['odd_ng_traf_level'] = 1 - df_features['ng_traf_level']
    df_features['odd_inner_network_droprate'] = 1 - df_features['inner_network_droprate']

    df_features["nctt"] = df_features[["icmp_rtt", "icmp_lossrate"]].apply(lambda x: calculate_nctt(*x), axis=1)
    df_features["bw_up"] = df_features[["icmp_rtt", "icmp_lossrate"]].apply(lambda x: calculate_throughput(*x), axis=1)
    df_features["bw_im"] = df_features[["icmp_rtt", "icmp_lossrate"]].apply(lambda x: calculate_throughput_improved(*x),
                                                                            axis=1)
    # 3. Skew transform
    numeric = ['avg_fbt_time', 'tcp_conntime', 'icmp_rtt', 'synack1_ratio', 'reset_ratio',
               'tcp_conntime', 'icmp_lossrate',  'ratio_499_5xx',
               'inner_network_droprate', 'cpu_util', 'mem_util',
               'io_await_avg', 'io_await_max', 'io_util_avg', 'io_util_max','ng_traf_level']
    skew_features = df_features[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)
    high_skew = skew_features[skew_features > 0.5]
    skew_index = high_skew.index
    print("There are {} numerical features with Skew > 0.5 :".format(high_skew.shape[0]))
    for i in skew_index:
        df_features[i] = boxcox1p(df_features[i], boxcox_normmax(df_features[i] + 1))

    df_features['total_delay'] = (df_features['avg_fbt_time'] + df_features['tcp_conntime'] + df_features['icmp_rtt'])
    df_features['total_io_delay'] = df_features['io_await_avg'] + df_features['io_await_max']
    df_features['delta_io_delay'] = df_features['io_await_max'] - df_features['io_await_avg']
    df_features['delta_io_util'] = df_features['io_util_max'] - df_features['io_util_avg']
    df_features['avg_util'] = (0.5 * df_features['cpu_util']) + (0.5 * df_features['mem_util'])

    log_features = ['ng_traf_level', 'ratio_499_5xx', 'icmp_rtt', 'icmp_lossrate', 'io_util_avg',
                    'avg_fbt_time', 'synack1_ratio', 'tcp_conntime', 'inner_network_droprate', 'cpu_util',
                    'io_util_max',
                    'total_delay', 'delta_io_util', 'avg_util', 'nctt', 'bw_up', 'bw_im']
    df_features = logs(df_features, log_features)
    # exp_features = mi_reg_lst.remove('node')
    exp_features = ['ng_traf_level', 'ratio_499_5xx', 'icmp_lossrate', 'io_util_avg',
                    'synack1_ratio', 'inner_network_droprate', 'cpu_util', 'io_util_max',
                    'total_delay', 'delta_io_util', 'avg_util']
    df_features = exps(df_features, exp_features)
    squared_features = ['ng_traf_level', 'ratio_499_5xx', 'icmp_lossrate', 'io_util_avg',
                        'synack1_ratio', 'inner_network_droprate', 'cpu_util', 'io_util_max',
                        'total_delay', 'delta_io_util', 'avg_util']
    df_features = squares(df_features, squared_features)

    df_features.drop(labels=['inner_network_rtt'], axis=1, inplace=True)
    df_features['id'] = df_features['node_name'] + df_features['domain_name']
    df_features['no'] = df_features.groupby(['id'])['buffer_rate'].cumcount()
    df_features['no_max'] = df_features.groupby(['id'])['no'].transform('max')
    # Categorical
    if cat_trans_flag:
        for f in ['domain_name', 'node_name', 'prov', 'isp', 'id']:
            le = LabelEncoder()
            le.fit(df_features[f])
            dic = {x: le.transform([x])[0] for x in df_features[f].unique()}
            df_features[f] = df_features[f].map(dic)
    df_labels = df_features['buffer_rate'].reset_index(drop=True)
    df_features = df_features.drop(['buffer_rate'], axis=1)
    return df_features, df_labels*100


def mae(y, y_pred):
    y_c = y[(y_pred < 100) & (y_pred > 0)]
    y_pred_c = y_pred[(y_pred < 100) & (y_pred > 0)]
    return np.mean(np.abs(y_c-y_pred_c))


def smape(y, y_pred):
    y_c = y[(y_pred < 100) & (y_pred > 0)]
    y_pred_c = y_pred[(y_pred < 100) & (y_pred > 0)]

    numerator = np.abs(y_c-y_pred_c)
    denominator = 0.5*(np.abs(y_c) + np.abs(y_pred_c))
    return np.mean(numerator / denominator)

# total_df = pd.read_csv("../../dataset/training_2nd_dataset.csv")




