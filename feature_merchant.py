# -*- coding: utf-8 -*-
#@author: limeng
#@file: feature_merchant.py
#@time: 2018/12/28 9:29
"""
文件说明：特征构造 merchant，在服务器运行
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

#探查商家
merchant_echo = merchant.merchant_id.value_counts().reset_index()
merchant_echo2 = merchant[merchant.merchant_id.isin(merchant_echo[merchant_echo.merchant_id>1]['index'])]
#删除重复商家
merchant = merchant.drop(merchant_echo2.index.values)
merchant = merchant.reset_index(drop=True)

#one-hot编码
# merchant['category_2'] = merchant['category_2'].apply(lambda x:int(x) if np.isnan(x)== False else x)
merchant_onehot= pd.get_dummies(merchant,
                         columns=['category_1', 'most_recent_sales_range'
                             ,'most_recent_purchases_range','category_4'
                             ,'category_2'])

#对id类进行映射处理
def object_map(df,feature_list):
    agg_func1 = {
        'merchant_id':['count'],
        'numerical_1':['sum','mean'],
        'numerical_2':['sum','mean'],
        'most_recent_sales_range_D':['sum','mean'],
        'most_recent_sales_range_E':['sum','mean'],
        'most_recent_purchases_range_D':['sum','mean'],
        'most_recent_purchases_range_E':['sum','mean'],
    }
    agg_func2 = {
        'merchant_id': ['count'],
        'numerical_1': ['sum', 'mean', 'std'],
        'numerical_2': ['sum', 'mean', 'std'],
        'category_1_N':['sum','mean'],
        'most_recent_sales_range_D': ['sum','mean'],
        'most_recent_sales_range_E': ['sum','mean'],
        'most_recent_purchases_range_D': ['sum','mean'],
        'most_recent_purchases_range_E': ['sum','mean'],
        'avg_sales_lag3':['sum', 'mean', 'std'],
        'avg_purchases_lag3':['sum', 'mean', 'std'],
        'active_months_lag3':['sum', 'mean'],
        'avg_sales_lag6':['sum', 'mean', 'std'],
        'avg_purchases_lag6':['sum', 'mean', 'std'],
        'active_months_lag6':['sum', 'mean'],
        'avg_sales_lag12':['sum', 'mean', 'std'],
        'avg_purchases_lag12':['sum', 'mean', 'std'],
        'active_months_lag12':['sum', 'mean'],
        'category_4_N':['sum','mean'],
        'category_2_1.0':['sum','mean']
    }
    for i in feature_list:
        print(i)
        if i == 'merchant_group_id':
            temp = df.groupby([i]).agg(agg_func1)
        else:
            temp = df.groupby([i]).agg(agg_func2)
        temp.columns = [i+'_'+'_'.join(col).strip() for col in temp.columns.values]
        temp = temp.reset_index()
        df = df.merge(temp, how='left', on=i)
        del df[i]
    return df

feature_list = ['merchant_group_id','merchant_category_id','subsector_id','city_id','state_id']
merchant_final = object_map(merchant_onehot,feature_list)

#减少内存
# from model_train.other_utils import reduce_mem_usage1
# merchant_final,nanlist = reduce_mem_usage1(merchant_final)

from model_train.other_utils import reduce_mem_usage2
merchant_final = reduce_mem_usage2(merchant_final)

#与trans表合并
historical_transactions = historical_transactions[['card_id','merchant_id']]
new_transactions = new_transactions[['card_id','merchant_id']]

historical_transactions1 = historical_transactions.iloc[:5000000,:]
historical_transactions2 = historical_transactions.iloc[5000000:10000000,:]
historical_transactions3 = historical_transactions.iloc[10000000:15000000,:]
historical_transactions4 = historical_transactions.iloc[15000000:20000000,:]
historical_transactions5 = historical_transactions.iloc[20000000:25000000,:]
historical_transactions6 = historical_transactions.iloc[25000000:,:]

print('trans和merchant拼接')
historical_transactions1= historical_transactions1.merge(merchant_final,how='left',on='merchant_id')
historical_transactions2= historical_transactions2.merge(merchant_final,how='left',on='merchant_id')
historical_transactions = pd.concat([historical_transactions1,historical_transactions2])
del historical_transactions1
del historical_transactions2
historical_transactions3= historical_transactions3.merge(merchant_final,how='left',on='merchant_id')
historical_transactions = pd.concat([historical_transactions,historical_transactions3])
del historical_transactions3
historical_transactions4= historical_transactions4.merge(merchant_final,how='left',on='merchant_id')
historical_transactions = pd.concat([historical_transactions,historical_transactions4])
del historical_transactions4
historical_transactions5= historical_transactions5.merge(merchant_final,how='left',on='merchant_id')
historical_transactions = pd.concat([historical_transactions,historical_transactions5])
del historical_transactions5
historical_transactions6= historical_transactions6.merge(merchant_final,how='left',on='merchant_id')
historical_transactions = pd.concat([historical_transactions,historical_transactions6])
del historical_transactions6
import gc
print(gc.collect())

new_transactions= new_transactions.merge(merchant_final,how='left',on='merchant_id')

del historical_transactions['merchant_id']
his_merchant = historical_transactions.groupby(['card_id']).agg(['sum','mean'])
his_merchant.columns = ['hist_'+'_'.join(col).strip() for col in his_merchant.columns.values]
his_merchant = his_merchant.reset_index()
del historical_transactions

del new_transactions['merchant_id']
new_merchant = new_transactions.groupby(['card_id']).agg(['sum','mean'])
new_merchant.columns = ['new_'+'_'.join(col).strip() for col in new_merchant.columns.values]
new_merchant = new_merchant.reset_index()
del new_transactions

#拼接train/test
train = train[['card_id']]
test = test[['card_id']]

train = pd.merge(train, his_merchant, on='card_id', how='left')
test = pd.merge(test, his_merchant, on='card_id', how='left')

train = pd.merge(train, new_merchant, on='card_id', how='left')
test = pd.merge(test, new_merchant, on='card_id', how='left')

train.to_csv(out_dir+'train_merchant1229.csv',index=False)
test.to_csv(out_dir+'test_merchant1229.csv',index=False)