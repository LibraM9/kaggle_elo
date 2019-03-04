# -*- coding: utf-8 -*-
#@author: limeng
#@file: feature_trans2.py
#@time: 2019/1/2 15:13
"""
文件说明：阿希什版本
"""
import pandas as pd
import sys
sys.path.extend(['/home/dev/lm/utils_lm'])


data_dir = '/data/dev/lm/elo/data/'
out_dir = '/data/dev/lm/elo/feature/'
historical_transactions = pd.read_csv(data_dir+'historical_transactions.csv',parse_dates=['purchase_date'])
new_transactions = pd.read_csv(data_dir+'new_merchant_transactions.csv',parse_dates=['purchase_date'])
merchant =  pd.read_csv(data_dir+'merchants.csv')
train = pd.read_csv(data_dir+'train.csv')
test = pd.read_csv(data_dir+'test.csv')
sub = pd.read_csv(data_dir+'sample_submission.csv')

target = train['target']
del train['target']

def binarize(df):
    """将Y/N转为1/0
    :param df:
    :return:
    """
    for col in ['authorized_flag', 'category_1']:
        df[col] = df[col].map({'Y':1, 'N':0})
    return df

historical_transactions = binarize(historical_transactions)
new_transactions = binarize(new_transactions)

#one-hot，对种类2、种类3
historical_transactions = pd.get_dummies(historical_transactions, columns=['category_2', 'category_3'])
new_transactions = pd.get_dummies(new_transactions, columns=['category_2', 'category_3'])

from model_train.other_utils import reduce_mem_usage2
historical_transactions = reduce_mem_usage2(historical_transactions)
new_transactions = reduce_mem_usage2(new_transactions)
# from model_train.other_utils import reduce_mem_usage1
# historical_transactions,n1 = reduce_mem_usage1(historical_transactions)
# new_transactions,n2 = reduce_mem_usage1(new_transactions)

agg_fun = {'authorized_flag': ['sum','mean']}
auth_mean = historical_transactions.groupby(['card_id']).agg(agg_fun)
auth_mean.columns = ['_'.join(col).strip() for col in auth_mean.columns.values]
auth_mean.reset_index(inplace=True)

#按是否授权分开
authorized_transactions = historical_transactions[historical_transactions['authorized_flag'] == 1]
historical_transactions = historical_transactions[historical_transactions['authorized_flag'] == 0]

historical_transactions['purchase_month'] = historical_transactions['purchase_date'].dt.month
authorized_transactions['purchase_month'] = authorized_transactions['purchase_date'].dt.month
new_transactions['purchase_month'] = new_transactions['purchase_date'].dt.month

import numpy as np
#对id_card进行聚合，处理未授权的history和所有new
# history = historical_transactions.copy()
def aggregate_transactions(history):
    #日期转为float并作为索引
    history.loc[:, 'purchase_date'] = pd.DatetimeIndex(history['purchase_date']). \
                                          astype(np.int64) * 1e-9

    agg_func = {
        'category_1': ['sum', 'mean'],
        'category_2_1.0': ['mean'],
        'category_2_2.0': ['mean'],
        'category_2_3.0': ['mean'],
        'category_2_4.0': ['mean'],
        'category_2_5.0': ['mean'],
        'category_3_A': ['mean'],
        'category_3_B': ['mean'],
        'category_3_C': ['mean'],
        'merchant_id': ['nunique'],
        'merchant_category_id': ['nunique'],
        'state_id': ['nunique'],
        'city_id': ['nunique'],
        'subsector_id': ['nunique'],
        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
        'installments': ['sum', 'mean', 'max', 'min', 'std'],
        'purchase_month': ['mean', 'max', 'min', 'std'],
        'purchase_date': [np.ptp, 'min', 'max'],
        'month_lag': ['max', 'min'],
    }

    agg_history = history.groupby(['card_id']).agg(agg_func)
    agg_history.columns = ['_'.join(col).strip() for col in agg_history.columns.values]
    agg_history.reset_index(inplace=True)

    df = (history.groupby('card_id')
          .size()
          .reset_index(name='transactions_count'))

    agg_history = pd.merge(df, agg_history, on='card_id', how='left')

    return agg_history

history = aggregate_transactions(historical_transactions)
history.columns = ['hist_' + c if c != 'card_id' else c for c in history.columns]

authorized = aggregate_transactions(authorized_transactions)
authorized.columns = ['auth_' + c if c != 'card_id' else c for c in authorized.columns]

new = aggregate_transactions(new_transactions)
new.columns = ['new_' + c if c != 'card_id' else c for c in new.columns]

#对history中授权的数据做处理
def aggregate_per_month(history):
    grouped = history.groupby(['card_id', 'month_lag'])

    agg_func = {
        'purchase_amount': ['count', 'sum', 'mean', 'min', 'max', 'std'],
        'installments': ['count', 'sum', 'mean', 'min', 'max', 'std'],
    }

    intermediate_group = grouped.agg(agg_func)
    intermediate_group.columns = ['_'.join(col).strip() for col in intermediate_group.columns.values]
    intermediate_group.reset_index(inplace=True)

    final_group = intermediate_group.groupby('card_id').agg(['mean', 'std'])
    final_group.columns = ['_'.join(col).strip() for col in final_group.columns.values]
    final_group.reset_index(inplace=True)

    return final_group

final_group =  aggregate_per_month(authorized_transactions)

#聚合所有特征
train = pd.merge(train, history, on='card_id', how='left')
test = pd.merge(test, history, on='card_id', how='left')

train = pd.merge(train, authorized, on='card_id', how='left')
test = pd.merge(test, authorized, on='card_id', how='left')

train = pd.merge(train, new, on='card_id', how='left')
test = pd.merge(test, new, on='card_id', how='left')

train = pd.merge(train, final_group, on='card_id', how='left')
test = pd.merge(test, final_group, on='card_id', how='left')

train = pd.merge(train, auth_mean, on='card_id', how='left')
test = pd.merge(test, auth_mean, on='card_id', how='left')

train['target'] = target
train.to_csv(out_dir+'train_trans0111.csv',index=False)
test.to_csv(out_dir+'test_trans0111.csv',index=False)
