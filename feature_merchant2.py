# -*- coding: utf-8 -*-
#@author: limeng
#@file: feature_merchant2.py
#@time: 2019/1/22 9:09
"""
文件说明：对merchant做简单的特征
"""
import pandas as pd
import numpy as np
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

#画图测试
# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.figure(num=1)
# plt.hist(merchant['subsector_id'])
# plt.show()
# plt.tight_layout()

#one-hot编码
merchant_onehot= pd.get_dummies(merchant,
                         columns=['category_1', 'most_recent_sales_range'
                             ,'most_recent_purchases_range','active_months_lag3'
                             ,'active_months_lag6','active_months_lag12','category_4'
                             ,'category_2'])
from model_train.other_utils import reduce_mem_usage2
merchant_onehot = reduce_mem_usage2(merchant_onehot)

#和tras拼接
historical_transactions = historical_transactions[['card_id','merchant_id']]
new_transactions = new_transactions[['card_id','merchant_id']]

historical_trans_merchant = historical_transactions.merge(merchant_onehot,how='left',on='merchant_id')
new_trans_merchant = new_transactions.merge(merchant_onehot,how='left',on='merchant_id')
del historical_transactions
del new_transactions
import gc
print(gc.collect())

agg_func = {
'merchant_group_id':['nunique'],
'merchant_category_id':['nunique'],
'subsector_id':['nunique'],
'numerical_1':['sum','max','min','std','mean'],
'numerical_2':['sum','max','min','std','mean'],
'avg_sales_lag3':['sum','max','min','std','mean'],
'avg_purchases_lag3':['sum','max','min','std','mean'],
'avg_sales_lag6':['sum','max','min','std','mean'],
'avg_purchases_lag6':['sum','max','min','std','mean'],
'avg_sales_lag12':['sum','max','min','std','mean'],
'avg_purchases_lag12':['sum','max','min','std','mean'],
'city_id':['nunique'],
'state_id':['nunique'],
'category_1_N':['sum','mean'],
'category_1_Y':['sum','mean'],
'most_recent_sales_range_A':['sum','mean'],
'most_recent_sales_range_B':['sum','mean'],
'most_recent_sales_range_C':['sum','mean'],
'most_recent_sales_range_D':['sum','mean'],
'most_recent_sales_range_E':['sum','mean'],
'most_recent_purchases_range_A':['sum','mean'],
'most_recent_purchases_range_B':['sum','mean'],
'most_recent_purchases_range_C':['sum','mean'],
'most_recent_purchases_range_D':['sum','mean'],
'most_recent_purchases_range_E':['sum','mean'],
'active_months_lag3_1':['sum','mean'],
'active_months_lag3_2':['sum','mean'],
'active_months_lag3_3':['sum','mean'],
'active_months_lag6_1':['sum','mean'],
'active_months_lag6_2':['sum','mean'],
'active_months_lag6_3':['sum','mean'],
'active_months_lag6_4':['sum','mean'],
'active_months_lag6_5':['sum','mean'],
'active_months_lag6_6':['sum','mean'],
'active_months_lag12_1':['sum','mean'],
'active_months_lag12_2':['sum','mean'],
'active_months_lag12_3':['sum','mean'],
'active_months_lag12_4':['sum','mean'],
'active_months_lag12_5':['sum','mean'],
'active_months_lag12_6':['sum','mean'],
'active_months_lag12_7':['sum','mean'],
'active_months_lag12_8':['sum','mean'],
'active_months_lag12_9':['sum','mean'],
'active_months_lag12_10':['sum','mean'],
'active_months_lag12_11':['sum','mean'],
'active_months_lag12_12':['sum','mean'],
'category_4_N':['sum','mean'],
'category_4_Y':['sum','mean'],
'category_2_1.0':['sum','mean'],
'category_2_2.0':['sum','mean'],
'category_2_3.0':['sum','mean'],
'category_2_4.0':['sum','mean'],
'category_2_5.0':['sum','mean']
    }

#以card_id为主键聚合
del historical_trans_merchant['merchant_id']
his_merchant = historical_trans_merchant.groupby(['card_id']).agg(agg_func)
his_merchant.columns = ['hismer_'+'_'.join(col).strip() for col in his_merchant.columns.values]
his_merchant = his_merchant.reset_index()
del new_trans_merchant['merchant_id']
new_merchant = new_trans_merchant.groupby(['card_id']).agg(agg_func)
new_merchant.columns = ['newmer_'+'_'.join(col).strip() for col in new_merchant.columns.values]
new_merchant = new_merchant.reset_index()

#拼接train/test
train = train[['card_id']]
test = test[['card_id']]

train = pd.merge(train, his_merchant, on='card_id', how='left')
test = pd.merge(test, his_merchant, on='card_id', how='left')

train = pd.merge(train, new_merchant, on='card_id', how='left')
test = pd.merge(test, new_merchant, on='card_id', how='left')

train.to_csv(out_dir+'train_merchant0122.csv',index=False)
test.to_csv(out_dir+'test_merchant0122.csv',index=False)