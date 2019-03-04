# -*- coding: utf-8 -*-
#@author: limeng
#@file: feature_transJie.py
#@time: 2019/2/1 20:54
"""
文件说明：杰少特征
"""
## 数据工具包
import numpy as np
np.random.seed(42)
import pandas as pd
from tqdm import tqdm,tqdm_notebook

## 字符串处理工具包
import string
import re
import gensim
from collections import Counter
import pickle
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from keras.preprocessing import text, sequence

import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
import lightgbm as lgb
from functools import partial

import os
import gc
import joblib#模型持久化 比pickle更高效
from scipy import stats
from scipy.sparse import vstack
import time
import datetime
import multiprocessing as mp
import seaborn as sns
tqdm.pandas()

#并行类########################################
import multiprocessing

def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))

#### 默认用20个核并行提取特征 ####
def parmap(f, X, nprocs= 20):
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()
    proc = [multiprocessing.Process(target=fun, args=(f, q_in, q_out))
                for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()
    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]
    [p.join() for p in proc]
    return [x for i, x in sorted(res)]

#降内存的类###############################
import numpy as np
import pandas as pd
from tqdm import tqdm

class _Data_Preprocess:
    def __init__(self):
        self.int8_max = np.iinfo(np.int8).max
        self.int8_min = np.iinfo(np.int8).min

        self.int16_max = np.iinfo(np.int16).max
        self.int16_min = np.iinfo(np.int16).min

        self.int32_max = np.iinfo(np.int32).max
        self.int32_min = np.iinfo(np.int32).min

        self.int64_max = np.iinfo(np.int64).max
        self.int64_min = np.iinfo(np.int64).min

        self.float16_max = np.finfo(np.float16).max
        self.float16_min = np.finfo(np.float16).min

        self.float32_max = np.finfo(np.float32).max
        self.float32_min = np.finfo(np.float32).min

        self.float64_max = np.finfo(np.float64).max
        self.float64_min = np.finfo(np.float64).min

    '''
    function: _get_type(self,min_val, max_val, types)
       get the correct types that our columns can trans to

    '''

    def _get_type(self, min_val, max_val, types):
        if types == 'int':
            if max_val <= self.int8_max and min_val >= self.int8_min:
                return np.int8
            elif max_val <= self.int16_max <= max_val and min_val >= self.int16_min:
                return np.int16
            elif max_val <= self.int32_max and min_val >= self.int32_min:
                return np.int32
            return None

        elif types == 'float':
            if max_val <= self.float16_max and min_val >= self.float16_min:
                return np.float16
            if max_val <= self.float32_max and min_val >= self.float32_min:
                return np.float32
            if max_val <= self.float64_max and min_val >= self.float64_min:
                return np.float64
            return None

    '''
    function: _memory_process(self,df) 
       column data types trans, to save more memory
    '''

    def _memory_process(self, df):
        init_memory = df.memory_usage().sum() / 1024 ** 2 / 1024
        print('Original data occupies {} GB memory.'.format(init_memory))
        df_cols = df.columns

        for col in tqdm_notebook(df_cols):
            try:
                if 'float' in str(df[col].dtypes):
                    max_val = df[col].max()
                    min_val = df[col].min()
                    trans_types = self._get_type(min_val, max_val, 'float')
                    if trans_types is not None:
                        df[col] = df[col].astype(trans_types)
                elif 'int' in str(df[col].dtypes):
                    max_val = df[col].max()
                    min_val = df[col].min()
                    trans_types = self._get_type(min_val, max_val, 'int')
                    if trans_types is not None:
                        df[col] = df[col].astype(trans_types)
            except:
                print(' Can not do any process for column, {}.'.format(col))
        afterprocess_memory = df.memory_usage().sum() / 1024 ** 2 / 1024
        print('After processing, the data occupies {} GB memory.'.format(afterprocess_memory))
        return df

##数据读取##############################
path = 'F:/数据集/1226kaggle_elo/'

train = pd.read_csv(open(path + 'train.csv',encoding='utf8'))
test  = pd.read_csv(open(path + 'test.csv',encoding='utf8'))
new_merchant_transactions = pd.read_csv(open(path + 'new_merchant_transactions.csv',encoding='utf8'))
historical_transactions   = pd.read_csv(open(path + 'historical_transactions.csv',encoding='utf8'))
sample_submission         = pd.read_csv(open(path + 'sample_submission.csv',encoding='utf8'))
merchants                 = pd.read_csv(open(path + 'merchants.csv',encoding='utf8'))

train_test = pd.concat([train[['card_id','first_active_month']], test[['card_id','first_active_month']] ], axis=0, ignore_index=True)

##特征工程#############################
memory_process = _Data_Preprocess()

 # 所有的特征函数我们用下面的形式进行命名,
 #    特征类型:
 #        (1).card_id 特征用 cardf
 #        (2).merchant_id 特征用 merchantf
 #    数据集(特征从哪个数据集进行构建)

def _get_cardf_train(df_):
    df = df_.copy()

    df['year'] = df['first_active_month'].fillna('0-0').apply(lambda x: int(str(x).split('-')[0]))
    #     df['month'] = df['first_active_month'].fillna('0-0').apply(lambda x:int(str(x).split('-')[-1]))
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['elapsed_time'] = (datetime.date(2018, 3, 1) - df['first_active_month'].dt.date).dt.days
    fea_cols = [col for col in df.columns if col != 'first_active_month']
    df = memory_process._memory_process(df[fea_cols])
    return df
cardf_train = _get_cardf_train(train)
cardf_test = _get_cardf_train(test)
#有很多卡早就激活了,消费记录里面没有
historical_transactions   = historical_transactions.merge(train_test[['card_id','first_active_month']], on=['card_id'], how='left')
new_merchant_transactions = new_merchant_transactions.merge(train_test[['card_id','first_active_month']], on=['card_id'], how='left')

#  时间特征处理扩充####################################
def month_trans(x):
    return x // 30

def week_trans(x):
    return x // 7

def _get_ht_expand_common(df_, authorized_flag=False):
    #  时间特征处理扩充，
    ## 0. 基础时间特征提取
    ## 1. 相邻两次消费的时间差
    ## 2. 距离激活的购买相对时间（天，周，月）
    ## 3. 距离最近一次购物的相对时间(以天为单位,以月为单位,以周为单位)
    ## 4.类别特征转换
    ## 5.基于时间的衰减销量特征

    # 0. 基础时间特征 #
    df = df_.copy()

    st = time.time()
    print('Start...')
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    #     df['purchase_date_to_180'+str(month)]  = (datetime.date(2018, month, 1) - df['purchase_date'].dt.date).dt.days
    df['purchase_hour'] = df['purchase_date'].dt.hour
    #     df['weekofyear']             =  df['purchase_date'].dt.weekofyear
    df['month'] = df['purchase_date'].dt.month
    df['dayofweek'] = df['purchase_date'].dt.dayofweek
    df['weekend'] = (df.purchase_date.dt.weekday >= 5).astype(int)
    df = df.sort_values(['card_id', 'purchase_date'])
    df['purchase_date_floorday'] = df['purchase_date'].dt.floor('d')  # 删除小于day的时间

    print('Basic feture trans spends {} seconds.'.format(time.time() - st))

    # 相邻两次消费的时间差(略有小问题) #
    ht_card_id_gp = df.groupby('card_id')

    #     df['purchase_date_diff']    =  ht_card_id_gp['purchase_date_floorday'].shift().values
    #     df['purchase_date_diff']    =  df['purchase_date_floorday'].values - df['purchase_date_diff'].values
    #     df['purchase_day_diff_day'] =  df['purchase_date_diff'].dt.days
    #     df['purchase_day_diff_day'] =  df['purchase_day_diff_day'].fillna(0)

    # 距离激活时间的相对时间,0, 1,2,3,...,max-act
    df['purchase_day_since_active_day'] = df['purchase_date_floorday'] - df[
        'first_active_month']  # ht_card_id_gp['purchase_date_floorday'].transform('min')
    df['purchase_day_since_active_day'] = df['purchase_day_since_active_day'].dt.days  # .astype('timedelta64[D]')
    df['purchase_month_since_active_day'] = df['purchase_day_since_active_day'].agg(month_trans).values
    df['purchase_week_since_active_day'] = df['purchase_day_since_active_day'].agg(week_trans).values

    # 距离最后一天时间的相对时间,0,1,2,3,...,max-act
    ht_card_id_gp = df.groupby('card_id')
    df['purchase_day_since_reference_day'] = ht_card_id_gp['purchase_date_floorday'].transform('max') - df[
        'purchase_date_floorday']
    df['purchase_day_since_reference_day'] = df['purchase_day_since_reference_day'].dt.days
    # 一个粗粒度的特征(距离最近购买过去了几周，几月)
    df['purchase_week_since_reference_day'] = df['purchase_day_since_reference_day'].agg(week_trans).values
    df['purchase_month_since_reference_day'] = df['purchase_day_since_reference_day'].agg(month_trans).values

    df['purchase_day_diff'] = df['purchase_date_floorday'].shift()
    df['purchase_day_diff'] = df['purchase_date_floorday'].values - df['purchase_day_diff'].values
    df['purchase_day_diff'] = df['purchase_day_diff'].dt.days
    df['purchase_week_diff'] = df['purchase_day_diff'].agg(week_trans).values
    df['purchase_month_diff'] = df['purchase_day_diff'].agg(month_trans).values

    print('Time trans spends {} seconds.'.format(time.time() - st))
    #     print(df.head())
    df['purchase_amount_ddgd_98'] = df['purchase_amount'].values * df['purchase_day_since_reference_day'].apply(
        lambda x: 0.98 ** x).values
    df['purchase_amount_ddgd_99'] = df['purchase_amount'].values * df['purchase_day_since_reference_day'].apply(
        lambda x: 0.99 ** x).values

    df['purchase_amount_wdgd_96'] = df['purchase_amount'].values * df['purchase_week_since_reference_day'].apply(
        lambda x: 0.96 ** x).values
    df['purchase_amount_wdgd_97'] = df['purchase_amount'].values * df['purchase_week_since_reference_day'].apply(
        lambda x: 0.97 ** x).values

    df['purchase_amount_mdgd_90'] = df['purchase_amount'].values * df['purchase_month_since_reference_day'].apply(
        lambda x: 0.9 ** x).values
    df['purchase_amount_mdgd_80'] = df['purchase_amount'].values * df['purchase_month_since_reference_day'].apply(
        lambda x: 0.8 ** x).values
    print('Time_based degrade trans spends {} seconds.'.format(time.time() - st))

    if authorized_flag == False:
        # 类别特征转换
        df['authorized_flag'] = df['authorized_flag'].map({'Y': 1, 'N': 0})
    df['category_1'] = df['category_1'].map({'Y': 1, 'N': 0})
    df['category_3'] = df['category_3'].map({'A': 0, 'B': 1, 'C': 2})

    df = memory_process._memory_process(df)
    print('Finish...')
    return df\
#historical_transactions数据集扩充1:基础特征
ht_expand_common = _get_ht_expand_common(historical_transactions,authorized_flag=False)

#特征工程1：全局################################################
def get_quantile(x, percentiles = [0.1, 0.25, 0.75, 0.9]):
    x_len = len(x)
    x = np.sort(x)
    sts_feas = []
    for per_ in percentiles:
        if per_ == 1:
            sts_feas.append(x[x_len - 1])
        else:
            sts_feas.append(x[int(x_len * per_)])
    return sts_feas


def _get_cardf_ht_parallel(indexes, month=3, is_au=False, suffix='_ht'):
    df_ = ht_expand_common.loc[ht_expand_common.card_id.isin(indexes)].copy()
    df = df_.copy()

    df['month_to_now'] = (datetime.date(2018, month, 1)) - df['purchase_date_floorday']

    df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days) // 30
    #     df['month_diff'] = df['month_diff'].astype(int)
    df['month_diff'] += df['month_lag']

    print('*' * 30, 'Part1, whole data', '*' * 30)
    cardid_features = pd.DataFrame()
    cardid_features['card_id'] = df['card_id'].unique()

    print('*' * 30, 'Traditional Features', '*' * 30)
    ht_card_id_gp = df.groupby('card_id')
    print(' card id : count')
    cardid_features['card_id_cnt'] = ht_card_id_gp['authorized_flag'].count().values

    if is_au == False:
        cardid_features['card_id_isau_mean'] = ht_card_id_gp['authorized_flag'].mean().values
        cardid_features['card_id_isau_sum'] = ht_card_id_gp['authorized_flag'].sum().values

    print('*' * 30, 'Traditional Features', '*' * 30)
    ht_card_id_gp = df.groupby('card_id')
    cardid_features['month_diff_mean'] = ht_card_id_gp['month_diff'].mean().values
    cardid_features['month_diff_median'] = ht_card_id_gp['month_diff'].median().values

    #  降分,过拟合了,上面的median已经够了
    #     cardid_features['month_diff_max'] = ht_card_id_gp['month_diff'].max().values
    #     cardid_features['month_diff_min'] = ht_card_id_gp['month_diff'].min().values
    #     cardid_features['month_diff_gap'] = cardid_features['month_diff_max'] - cardid_features['month_diff_min']

    #     del cardid_features['month_diff_max'],cardid_features['month_diff_min']

    print(' Eight time features, ')
    cardid_features['reference_day'] = ht_card_id_gp['purchase_date_floorday'].max().values
    cardid_features['first_day'] = ht_card_id_gp['purchase_date_floorday'].min().values
    cardid_features['activation_day'] = ht_card_id_gp['first_active_month'].max().values

    # first to activation day
    cardid_features['first_to_activation_day'] = cardid_features['first_day'] - cardid_features['activation_day']
    cardid_features['first_to_activation_day'] = cardid_features['first_to_activation_day'].dt.days

    # activation to reference day
    cardid_features['activation_to_reference_day'] = cardid_features['reference_day'] - cardid_features[
        'activation_day']
    cardid_features['activation_to_reference_day'] = cardid_features['activation_to_reference_day'].dt.days

    # first to last day
    cardid_features['first_to_reference_day'] = cardid_features['reference_day'] - cardid_features['first_day']
    cardid_features['first_to_reference_day'] = cardid_features['first_to_reference_day'].dt.days

    # reference day to now
    cardid_features['reference_day_to_now'] = (datetime.date(2018, month, 1)) - cardid_features['reference_day']
    cardid_features['reference_day_to_now'] = cardid_features['reference_day_to_now'].dt.days

    # first day to now
    cardid_features['first_day_to_now'] = (datetime.date(2018, month, 1)) - cardid_features['first_day']
    cardid_features['first_day_to_now'] = cardid_features['first_day_to_now'].dt.days

    print('card_id(month_lag, min to reference day):min')
    cardid_features['card_id_month_lag_min'] = ht_card_id_gp['month_lag'].agg('min').values

    #################

    # is_purchase_before_activation,first_to_reference_day_divide_activation_to_reference_day
    cardid_features['is_purchase_before_activation'] = cardid_features['first_to_activation_day'] < 0
    cardid_features['is_purchase_before_activation'] = cardid_features['is_purchase_before_activation'].astype(int)

    cardid_features['first_to_reference_day_divide_activation_to_reference_day'] = cardid_features[
                                                                                       'first_to_reference_day'] / (
                                                                                               cardid_features[
                                                                                                   'activation_to_reference_day'] + 0.01)
    cardid_features['days_per_count'] = cardid_features['first_to_reference_day'].values / cardid_features[
        'card_id_cnt'].values

    print('card id(city_id,installments,merchant_category_id,.......):nunique, cnt/nunique')  #
    for col in tqdm_notebook(
            ['category_1', 'category_2', 'category_3', 'state_id', 'city_id', 'installments', 'merchant_id',
             'merchant_category_id', 'subsector_id', 'month_lag', 'purchase_date_floorday']):
        cardid_features['card_id_%s_nunique' % col] = ht_card_id_gp[col].nunique().values
        cardid_features['card_id_cnt_divide_%s_nunique' % col] = cardid_features['card_id_cnt'].values / \
                                                                 cardid_features['card_id_%s_nunique' % col].values

    print('card_id(purchase_amount & degrade version ):mean,sum,std,median,quantile(10,25,75,90)')
    for col in tqdm_notebook(['installments', 'purchase_amount', 'purchase_amount_ddgd_98', 'purchase_amount_ddgd_99',
                              'purchase_amount_wdgd_96', 'purchase_amount_wdgd_97', 'purchase_amount_mdgd_90',
                              'purchase_amount_mdgd_80']):
        if col == 'purchase_amount':
            for opt in ['sum', 'mean', 'std', 'median', 'max', 'min']:
                cardid_features['card_id_' + col + '_' + opt] = ht_card_id_gp[col].agg(opt).values

            cardid_features['card_id_' + col + '_range'] = cardid_features['card_id_' + col + '_max'].values - \
                                                           cardid_features['card_id_' + col + '_min'].values
            percentiles = ht_card_id_gp[col].apply(lambda x: get_quantile(x, percentiles=[0.025, 0.25, 0.75, 0.975]))

            cardid_features[col + '_2.5_quantile'] = percentiles.map(lambda x: x[0]).values
            cardid_features[col + '_25_quantile'] = percentiles.map(lambda x: x[1]).values
            cardid_features[col + '_75_quantile'] = percentiles.map(lambda x: x[2]).values
            cardid_features[col + '_97.5_quantile'] = percentiles.map(lambda x: x[3]).values
            cardid_features['card_id_' + col + '_range2'] = cardid_features[col + '_97.5_quantile'].values - \
                                                            cardid_features[col + '_2.5_quantile'].values
            del cardid_features[col + '_2.5_quantile'], cardid_features[col + '_97.5_quantile']
            gc.collect()
        else:
            for opt in ['sum']:
                cardid_features['card_id_' + col + '_' + opt] = ht_card_id_gp[col].agg(opt).values

    print('*' * 30, 'Pivot Features', '*' * 30)
    print(
        'Count  Pivot')  # purchase_month_since_reference_day(可能和month_lag重复),百分比降分,暂时忽略 (dayofweek,merchant_cate,state_id)作用不大installments
    for pivot_col in tqdm_notebook(['category_1', 'category_2', 'category_3', 'month_lag', 'subsector_id', 'weekend',
                                    'merchant_category_id']):  # 'city_id',,

        tmp = df.groupby(['card_id', pivot_col])['merchant_id'].count().to_frame(pivot_col + '_count')
        tmp.reset_index(inplace=True)

        tmp_pivot = pd.pivot_table(data=tmp, index='card_id', columns=pivot_col, values=pivot_col + '_count',
                                   fill_value=0)
        tmp_pivot.columns = [tmp_pivot.columns.names[0] + '_cnt_pivot_' + str(col) for col in tmp_pivot.columns]
        tmp_pivot.reset_index(inplace=True)
        cardid_features = cardid_features.merge(tmp_pivot, on='card_id', how='left')

        #         tmp     = df.groupby(['card_id',pivot_col])['month_diff'].mean().to_frame(pivot_col + '_month_diff_mean')
        #         tmp.reset_index(inplace =True)

        #         tmp_pivot = pd.pivot_table(data=tmp,index = 'card_id',columns=pivot_col,values=pivot_col + '_month_diff_mean',fill_value=0)
        #         tmp_pivot.columns = [tmp_pivot.columns.names[0] + '_month_diff_mean_pivot_'+ str(col) for col in tmp_pivot.columns]
        #         tmp_pivot.reset_index(inplace = True)
        #         cardid_features = cardid_features.merge(tmp_pivot, on = 'card_id', how='left')

        if pivot_col != 'weekend' and pivot_col != 'installments':
            tmp = df.groupby(['card_id', pivot_col])['purchase_date_floorday'].nunique().to_frame(
                pivot_col + '_purchase_date_floorday_nunique')
            tmp1 = df.groupby(['card_id'])['purchase_date_floorday'].nunique().to_frame(
                'purchase_date_floorday_nunique')
            tmp.reset_index(inplace=True)
            tmp1.reset_index(inplace=True)
            tmp = tmp.merge(tmp1, on='card_id', how='left')
            tmp[pivot_col + '_day_nunique_pct'] = tmp[pivot_col + '_purchase_date_floorday_nunique'].values / tmp[
                'purchase_date_floorday_nunique'].values

            tmp_pivot = pd.pivot_table(data=tmp, index='card_id', columns=pivot_col,
                                       values=pivot_col + '_day_nunique_pct', fill_value=0)
            tmp_pivot.columns = [tmp_pivot.columns.names[0] + '_day_nunique_pct_' + str(col) for col in
                                 tmp_pivot.columns]
            tmp_pivot.reset_index(inplace=True)
            cardid_features = cardid_features.merge(tmp_pivot, on='card_id', how='left')

    ######## 在卡未激活之前就有过消费的记录  ##############

    print('*' * 30, 'Part2， data with time less than activation day', '*' * 30)
    df_part = df.loc[df.purchase_date < df.first_active_month]

    cardid_features_part = pd.DataFrame()
    cardid_features_part['card_id'] = df_part['card_id'].unique()
    ht_card_id_part_gp = df_part.groupby('card_id')
    cardid_features_part['card_id_part_cnt'] = ht_card_id_part_gp['authorized_flag'].count().values

    print('card_id(purchase_amount): sum')
    for col in tqdm_notebook(['purchase_amount']):
        for opt in ['sum', 'mean']:
            cardid_features_part['card_id_part_' + col + '_' + opt] = ht_card_id_part_gp[col].agg(opt).values

    cardid_features = cardid_features.merge(cardid_features_part, on='card_id', how='left')
    cardid_features['card_id_part_purchase_amount_sum_percent'] = cardid_features[
                                                                      'card_id_part_purchase_amount_sum'] / (
                                                                              cardid_features[
                                                                                  'card_id_purchase_amount_sum'] + 0.01)

    #     cardid_features = memory_process._memory_process(cardid_features)

    new_col_names = []
    for col in cardid_features.columns:
        if col == 'card_id':
            new_col_names.append(col)
        else:
            new_col_names.append(col + suffix)
    cardid_features.columns = new_col_names

    return cardid_features

ht_expand_common['installments'].replace(-1, np.nan,inplace = True)
ht_expand_common['installments'].replace(999,np.nan,inplace = True)
len_ = train_test.shape[0]
cut_len = 50000
indexs = [train_test['card_id'].values[i:i+cut_len] for i in range(0,len_,cut_len)]
res = parmap(_get_cardf_ht_parallel,indexs, nprocs=10)

cardf_ht = None
for i in tqdm_notebook(range(len(res))):
    if i == 0:
        cardf_ht = res[0]
    else:
        cardf_ht = pd.concat([cardf_ht,res[i]], axis=0, ignore_index=True)


def _get_cardf_htlast2_parallel(indexes, month=3, is_au=False, suffix='_htlast2'):
    df = ht_expand_common_last2month.loc[ht_expand_common_last2month.card_id.isin(indexes)].copy()
    df_ = df.copy()
    print('*' * 30, 'Part1, whole data', '*' * 30)
    cardid_features = pd.DataFrame()
    cardid_features['card_id'] = df['card_id'].unique()

    df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days) // 30
    df['month_diff'] = df['month_diff'].astype(int)
    df['month_diff'] += df['month_lag']

    print('*' * 30, 'Traditional Features', '*' * 30)
    ht_card_id_gp = df.groupby('card_id')
    print(' card id : count')
    cardid_features['card_id_cnt'] = ht_card_id_gp['authorized_flag'].count().values

    if is_au == False:
        cardid_features['card_id_isau_mean'] = ht_card_id_gp['authorized_flag'].mean().values
        cardid_features['card_id_isau_sum'] = ht_card_id_gp['authorized_flag'].sum().values

    cardid_features['month_diff_mean'] = ht_card_id_gp['month_diff'].mean().values
    #     cardid_features['month_diff_std']    = ht_card_id_gp['month_diff'].std().values
    #     cardid_features['month_diff_median'] = ht_card_id_gp['month_diff'].median().values

    print('card id(city_id,installments,merchant_category_id,.......):nunique, cnt/nunique')
    for col in tqdm_notebook(
            ['state_id', 'city_id', 'installments', 'merchant_id', 'merchant_category_id', 'purchase_date_floorday']):
        cardid_features['card_id_%s_nunique' % col] = ht_card_id_gp[col].nunique().values
        cardid_features['card_id_cnt_divide_%s_nunique' % col] = cardid_features['card_id_cnt'].values / \
                                                                 cardid_features['card_id_%s_nunique' % col].values

    #     print('card_id(purchase_amount & degrade version ):mean,sum,std,median,quantile(10,25,75,90)')
    for col in tqdm_notebook(
            ['purchase_amount', 'purchase_amount_ddgd_98', 'purchase_amount_wdgd_96', 'purchase_amount_mdgd_90',
             'purchase_amount_mdgd_80']):  # ,'purchase_amount_ddgd_98','purchase_amount_ddgd_99','purchase_amount_wdgd_96','purchase_amount_wdgd_97','purchase_amount_mdgd_90','purchase_amount_mdgd_80']):
        if col == 'purchase_amount':
            for opt in ['sum', 'mean', 'std', 'median']:
                cardid_features['card_id_' + col + '_' + opt] = ht_card_id_gp[col].agg(opt).values
        else:
            for opt in ['sum']:
                cardid_features['card_id_' + col + '_' + opt] = ht_card_id_gp[col].agg(opt).values

    print('*' * 30, 'Pivot Features', '*' * 30)
    print(
        'Count  Pivot')  # purchase_month_since_reference_day(可能和month_lag重复),百分比降分,暂时忽略 (dayofweek,merchant_cate,state_id)作用不大

    for pivot_col in tqdm_notebook(['category_1', 'category_2', 'category_3', 'month_lag', 'subsector_id', 'weekend',
                                    'merchant_category_id']):  # 'city_id',

        tmp = df.groupby(['card_id', pivot_col])['merchant_id'].count().to_frame(pivot_col + '_count')
        tmp.reset_index(inplace=True)

        tmp_pivot = pd.pivot_table(data=tmp, index='card_id', columns=pivot_col, values=pivot_col + '_count',
                                   fill_value=0)
        tmp_pivot.columns = [tmp_pivot.columns.names[0] + '_cnt_pivot_' + str(col) for col in tmp_pivot.columns]
        tmp_pivot.reset_index(inplace=True)
        cardid_features = cardid_features.merge(tmp_pivot, on='card_id', how='left')

        if pivot_col != 'weekend' and pivot_col != 'installments':
            tmp = df.groupby(['card_id', pivot_col])['purchase_date_floorday'].nunique().to_frame(
                pivot_col + '_purchase_date_floorday_nunique')
            tmp1 = df.groupby(['card_id'])['purchase_date_floorday'].nunique().to_frame(
                'purchase_date_floorday_nunique')
            tmp.reset_index(inplace=True)
            tmp1.reset_index(inplace=True)
            tmp = tmp.merge(tmp1, on='card_id', how='left')
            tmp[pivot_col + '_day_nunique_pct'] = tmp[pivot_col + '_purchase_date_floorday_nunique'].values / tmp[
                'purchase_date_floorday_nunique'].values

            tmp_pivot = pd.pivot_table(data=tmp, index='card_id', columns=pivot_col,
                                       values=pivot_col + '_day_nunique_pct', fill_value=0)
            tmp_pivot.columns = [tmp_pivot.columns.names[0] + '_day_nunique_pct_' + str(col) for col in
                                 tmp_pivot.columns]
            tmp_pivot.reset_index(inplace=True)
            cardid_features = cardid_features.merge(tmp_pivot, on='card_id', how='left')

    ######## 在卡未激活之前就有过消费的记录  ##############
    cardid_features = memory_process._memory_process(cardid_features)

    new_col_names = []
    for col in cardid_features.columns:
        if col == 'card_id':
            new_col_names.append(col)
        else:
            new_col_names.append(col + suffix)
    cardid_features.columns = new_col_names

    return cardid_features

ht_expand_common_last2month = ht_expand_common.loc[ht_expand_common.month_lag >= -2].copy()
res = parmap(_get_cardf_htlast2_parallel,indexs, nprocs=10)

cardf_htlast2 = None
for i in tqdm_notebook(range(len(res))):
    if i == 0:
        cardf_htlast2 = res[0]
    else:
        cardf_htlast2 = pd.concat([cardf_htlast2,res[i]], axis=0, ignore_index=True)

#Card_id特征+historical_transactions(au=1)表格+全局特征
authorized_transactions = historical_transactions.loc[historical_transactions['authorized_flag'] == 'Y']
ht_au_expand_common     = _get_ht_expand_common(authorized_transactions,   authorized_flag = True)