# -*- coding: utf-8 -*-
#@author: limeng
#@file: feature_trans3.py
#@time: 2019/1/28 8:57
"""
文件说明：阿什希版本加节假日特征
"""
import pandas as pd
import datetime
import gc

# #服务器路径
# import sys
# sys.path.extend(['/home/dev/lm/utils_lm'])
# data_dir = '/data/dev/lm/elo/data/'
# out_dir = '/data/dev/lm/elo/feature/'
# historical_transactions = pd.read_csv(data_dir+'historical_transactions.csv',parse_dates=['purchase_date'])
# new_transactions = pd.read_csv(data_dir+'new_merchant_transactions.csv',parse_dates=['purchase_date'])
# merchant =  pd.read_csv(data_dir+'merchants.csv')
# train = pd.read_csv(data_dir+'train.csv')
# test = pd.read_csv(data_dir+'test.csv')
# sub = pd.read_csv(data_dir+'sample_submission.csv')

#本机路径
data_dir = 'F:/数据集/1226kaggle_elo/'
out_dir = 'F:/数据集处理/elo/feature/'
historical_transactions = pd.read_csv(open(data_dir+'historical_transactions.csv',encoding='utf8'),parse_dates=['purchase_date'])
new_transactions = pd.read_csv(open(data_dir+'new_merchant_transactions.csv',encoding='utf8'),parse_dates=['purchase_date'])
merchant =  pd.read_csv(open(data_dir+'merchants.csv',encoding='utf8'))
train = pd.read_csv(open(data_dir+'train.csv',encoding='utf8'))
test = pd.read_csv(open(data_dir+'test.csv',encoding='utf8'))
sub = pd.read_csv(open(data_dir+'sample_submission.csv',encoding='utf8'))

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

##########################################
# 新增特征
#Feature Engineering - Adding new features inspired by Chau's first kernel
# history_transactions
historical_transactions['purchase_date'] = pd.to_datetime(historical_transactions['purchase_date'])
historical_transactions['year'] = historical_transactions['purchase_date'].dt.year
historical_transactions['weekofyear'] = historical_transactions['purchase_date'].dt.weekofyear
historical_transactions['month'] = historical_transactions['purchase_date'].dt.month
historical_transactions['dayofweek'] = historical_transactions['purchase_date'].dt.dayofweek
historical_transactions['weekend'] = (historical_transactions.purchase_date.dt.weekday >=5).astype(int)
historical_transactions['hour'] = historical_transactions['purchase_date'].dt.hour
historical_transactions['month_diff'] = ((datetime.datetime.today() - historical_transactions['purchase_date']).dt.days)//30
historical_transactions['month_diff'] += historical_transactions['month_lag']

#impute missing values - This is now excluded.
# historical_transactions['category_2'] = historical_transactions['category_2'].fillna(1.0,inplace=True)
# historical_transactions['category_3'] = historical_transactions['category_3'].fillna('A',inplace=True)
# historical_transactions['merchant_id'] = historical_transactions['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
gc.collect()

# New Features with Key Shopping times considered in the dataset. if the purchase has been made within 60 days, it is considered as an influence
#Christmas : December 25 2017
historical_transactions['Christmas_day_2017'] = (pd.to_datetime('2017-12-25') - historical_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
#Mothers Day: May 14 2017
#transactions['Mothers_Day_2017'] = (pd.to_datetime('2017-05-04') - transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
#fathers day: August 13 2017
historical_transactions['fathers_day_2017'] = (pd.to_datetime('2017-08-13') - historical_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
#Childrens day: October 12 2017
historical_transactions['Children_day_2017'] = (pd.to_datetime('2017-10-12') - historical_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
#Black Friday : 24th November 2017
historical_transactions['Black_Friday_2017'] = (pd.to_datetime('2017-11-24') - historical_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
#Valentines Day
historical_transactions['Valentine_day_2017'] = (pd.to_datetime('2017-06-12') - historical_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

#2018
#Mothers Day: May 13 2018
historical_transactions['Mothers_day_2018'] = (pd.to_datetime('2018-05-13') - historical_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
gc.collect()

# new_transactions
new_transactions['purchase_date'] = pd.to_datetime(new_transactions['purchase_date'])
new_transactions['year'] = new_transactions['purchase_date'].dt.year
new_transactions['weekofyear'] = new_transactions['purchase_date'].dt.weekofyear
new_transactions['month'] = new_transactions['purchase_date'].dt.month
new_transactions['dayofweek'] = new_transactions['purchase_date'].dt.dayofweek
new_transactions['weekend'] = (new_transactions.purchase_date.dt.weekday >=5).astype(int)
new_transactions['hour'] = new_transactions['purchase_date'].dt.hour
new_transactions['month_diff'] = ((datetime.datetime.today() - new_transactions['purchase_date']).dt.days)//30
new_transactions['month_diff'] += new_transactions['month_lag']

#impute missing values
# new_transactions['category_2'] = new_transactions['category_2'].fillna(1.0,inplace=True)
# new_transactions['category_3'] = new_transactions['category_3'].fillna('A',inplace=True)
# new_transactions['merchant_id'] = new_transactions['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)

# New Features with Key Shopping times considered in the dataset. if the purchase has been made within 60 days,
# it is considered as an influence

#Christmas : December 25 2017
new_transactions['Christmas_day_2017'] = (pd.to_datetime('2017-12-25') - new_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
#Mothers Day: May 14 2017 - Was not significant in Feature Importance
#new_transactions['Mothers_Day_2017'] = (pd.to_datetime('2017-06-04') - new_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
#fathers day: August 13 2017
new_transactions['fathers_day_2017'] = (pd.to_datetime('2017-08-13') - new_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
#Childrens day: October 12 2017
new_transactions['Children_day_2017'] = (pd.to_datetime('2017-10-12') - new_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
#Valentine's Day : 12th June, 2017
new_transactions['Valentine_day_2017'] = (pd.to_datetime('2017-06-12') - new_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
#Black Friday : 24th November 2017
new_transactions['Black_Friday_2017'] = (pd.to_datetime('2017-11-24') - new_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

#2018
#Mothers Day: May 13 2018
new_transactions['Mothers_day_2018'] = (pd.to_datetime('2018-05-13') - new_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
gc.collect()
###################################################
agg_fun = {'authorized_flag': ['sum','mean']}
auth_mean = historical_transactions.groupby(['card_id']).agg(agg_fun)
auth_mean.columns = ['_'.join(col).strip() for col in auth_mean.columns.values]
auth_mean.reset_index(inplace=True)

#按是否授权分开
authorized_transactions = historical_transactions[historical_transactions['authorized_flag'] == 1]
historical_transactions = historical_transactions[historical_transactions['authorized_flag'] == 0]

import numpy as np

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
        #'purchase_month': ['mean', 'max', 'min', 'std'],
        'purchase_date': [np.ptp, 'min', 'max'],
        'month_lag': ['max', 'min'],

        # 新增
        'weekend': ['sum', 'mean'],
        'month_diff': ['mean'],
        'month': ['nunique', 'mean', 'max', 'min', 'std'],
        'hour': ['nunique'],
        'weekofyear': ['nunique'],
        'dayofweek': ['nunique'],
        'year': ['nunique'],
        'Christmas_day_2017': ['mean'],
        'fathers_day_2017': ['mean'],
        'Children_day_2017': ['mean'],
        'Black_Friday_2017': ['mean'],
        'Valentine_day_2017': ['mean'],
        'Mothers_day_2018': ['mean']

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

## train & test 第一次访问时间
# Now extract the month, year, day, weekday
# train["first_active_month"] = pd.to_datetime(train["first_active_month"])
# train["month"] = train["first_active_month"].dt.month
# train["year"] = train["first_active_month"].dt.year
# train['week'] = train["first_active_month"].dt.weekofyear
# train['dayofweek'] = train['first_active_month'].dt.dayofweek
# train['days'] = (datetime.date(2018, 2, 1) - train['first_active_month'].dt.date).dt.days
# train['quarter'] = train['first_active_month'].dt.quarter
# train['is_month_start'] = train['first_active_month'].dt.is_month_start

# test["first_active_month"] = pd.to_datetime(test["first_active_month"])
# test["month"] = test["first_active_month"].dt.month
# test["year"] = test["first_active_month"].dt.year
# test['week'] = test["first_active_month"].dt.weekofyear
# test['dayofweek'] = test['first_active_month'].dt.dayofweek
# test['days'] = (datetime.date(2018, 2, 1) - test['first_active_month'].dt.date).dt.days
# test['quarter'] = test['first_active_month'].dt.quarter
# test['is_month_start'] = test['first_active_month'].dt.is_month_start

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
train.to_csv(out_dir+'train_trans0128.csv',index=False)
test.to_csv(out_dir+'test_trans0128.csv',index=False)