# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 10:36:46 2019

@author: rate9
杰少+节日+cookly 全特征 3.65441
"""
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
#from keras.preprocessing import text, sequence 

import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
import lightgbm as lgb
from functools import partial

import matplotlib.pyplot as plt
import seaborn as sns
import os 
import gc
import joblib
from scipy import stats 
from scipy.sparse import vstack  
import time
import datetime
import multiprocessing as mp
import seaborn as sns 
tqdm.pandas() 

from sklearn.preprocessing import MinMaxScaler 
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor

dir2 = 'F:/数据集处理/elo/3.676_features/'
input_dir = 'F:/数据集处理/elo/feature/'
input_dir2 = 'F:/数据集处理/elo/Cookly特征/'
out_dir = 'F:/数据集处理/elo/'

# cardf_test = joblib.load(dir2+'cardf_test.jl.z')
# cardf_train= joblib.load(dir2+'cardf_train.jl.z')
# cardf_ht = joblib.load(dir2+'cardf_ht.jl.z')
# cardf_htlast2 = joblib.load(dir2+'cardf_htlast2.jl.z')
# cardf_htdays = joblib.load(dir2+'cardf_htdays.jl.z')
# cardf_htau = joblib.load(dir2+'cardf_htau.jl.z')
# cardf_nmth = joblib.load(dir2+'cardf_nmth.jl.z')
#
# train=cardf_train.merge(cardf_ht,on='card_id',how='left')
# train=train.merge(cardf_htlast2,on='card_id',how='left')
# train=train.merge(cardf_htdays,on='card_id',how='left')
# train=train.merge(cardf_htau,on='card_id',how='left')
# train=train.merge(cardf_nmth,on='card_id',how='left')
#
# test=cardf_test.merge(cardf_ht,on='card_id',how='left')
# test=test.merge(cardf_htlast2,on='card_id',how='left')
# test=test.merge(cardf_htdays,on='card_id',how='left')
# test=test.merge(cardf_htau,on='card_id',how='left')
# test=test.merge(cardf_nmth,on='card_id',how='left')
#
# print (train[['card_id']].head(5))
# print (train[['reference_day_ht', 'first_day_ht', 'activation_day_ht']].head(5))

train = pd.read_csv(open(input_dir+'train_trans_jie.csv',encoding='utf8'))
test = pd.read_csv(open(input_dir+'test_trans_jie.csv',encoding='utf8'))

train_0 = pd.read_csv(open(input_dir2+'train_stacking_base0.csv',encoding='utf8'))
train_1 = pd.read_csv(open(input_dir2+'train_stacking_base1.csv',encoding='utf8'))
train_2 = pd.read_csv(open(input_dir2+'train_stacking_base2.csv',encoding='utf8'))
test_0 = pd.read_csv(open(input_dir2+'test_stacking_base0.csv',encoding='utf8'))
test_1 = pd.read_csv(open(input_dir2+'test_stacking_base1.csv',encoding='utf8'))
test_2 = pd.read_csv(open(input_dir2+'test_stacking_base2.csv',encoding='utf8'))

train = train.merge(train_0,how='left',left_on='card_id',right_on='listing_id')
train = train.merge(train_1,how='left',left_on='card_id',right_on='listing_id')
train = train.merge(train_2,how='left',left_on='card_id',right_on='listing_id')
del train['listing_id'],train['listing_id_x'],train['listing_id_y']
del train['level'],train['level_x'],train['level_y']

test = test.merge(test_0,how='left',left_on='card_id',right_on='listing_id')
test = test.merge(test_1,how='left',left_on='card_id',right_on='listing_id')
test = test.merge(test_2,how='left',left_on='card_id',right_on='listing_id')
del test['listing_id'],test['listing_id_x'],test['listing_id_y']

train.to_csv('F:/数据集处理/elo/feature/train_trans0218.csv',index=False)
test.to_csv('F:/数据集处理/elo/feature/test_trans0218.csv', index=False)

features = [c for c in tqdm(train.columns) if c not in ['card_id', 'target','reference_day_ht',
'first_day_ht', 
'activation_day_ht', 
'reference_day_htau', 
'first_day_htau', 
'activation_day_htau', 
'activation_day_newht', 
'first_day_newht', 
'last_day_newht', 
'reference_day_newht']]
#features = [f for f in features if f not in unimportant_features]
target = train['target']
categorical_feats = [c for c in features if 'feature_' in c]

from sklearn.model_selection import StratifiedKFold, KFold

param ={       'task': 'train',
                'boosting': 'goss',
                'objective': 'regression',
                'metric': 'rmse',
                'learning_rate': 0.01,
                'subsample': 0.9855232997390695,
                'max_depth': 7,
                'top_rate': 0.9064148448434349,
                'num_leaves': 63,
                'min_child_weight': 41.9612869171337,
                'other_rate': 0.0721768246018207,
                'reg_alpha': 9.677537745007898,
                'colsample_bytree': 0.5665320670155495,
                'min_split_gain': 9.820197773625843,
                'reg_lambda': 8.2532317400459,
                'min_data_in_leaf': 21,
                'verbose': -1,
                'seed':100,
                'bagging_seed':150,
                'drop_seed':200
                }

folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
start = time.time()
feature_importance_df = pd.DataFrame()

from sklearn.metrics import log_loss, mean_squared_error
start = time.time()
for fold_, (trn_idx, val_idx) in enumerate(tqdm(folds.split(train.values, target.values))):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=target.iloc[trn_idx], categorical_feature=categorical_feats)
    val_data = lgb.Dataset(train.iloc[val_idx][features], label=target.iloc[val_idx], categorical_feature=categorical_feats)

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 200)
    oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits

#print("CV score: {:<8.5f}".format(log_loss(target, oof)))
print("CV score: {:<8.5f}".format(mean_squared_error(oof, target)**0.5))
end = time.time()
print (end-start)

df_outlier_prob = pd.DataFrame({"card_id":test["card_id"].values})
df_outlier_prob["target"] = predictions
df_outlier_prob.head()
gc.collect()

print (feature_importance_df.head(5))
cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index) #[:60]

len(cols)

best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(14,25))
sns.barplot(x="importance",
            y="feature",
            data=best_features.sort_values(by="importance",
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.show()

import_feat=feature_importance_df.groupby('feature')['importance'].agg([('import_feat','mean')]).reset_index()
import_feat = import_feat.sort_values('import_feat',ascending=False)
import_feat['sort_num']=import_feat['import_feat'].rank(ascending=0,method='dense')

print('round1')
top_features = import_feat[import_feat['sort_num']<=250]
top_features = top_features['feature']
top_features = top_features.tolist()
#X_train=(train[top_features])
lgb_train = lgb.Dataset(train[top_features], train['target'])
#lgb_eval = lgb.Dataset(test[top_features], reference=lgb_train)
gbm = lgb.train(param,
lgb_train,
num_boost_round=3000,
#valid_sets=lgb_eval,
verbose_eval=100)
predictions = gbm.predict(test[top_features])
test['pred_250'] = predictions

print('round2')
top_features = import_feat[import_feat['sort_num']<=300]
top_features = top_features['feature']
top_features = top_features.tolist()
lgb_train = lgb.Dataset(train[top_features], train['target'])
gbm = lgb.train(param,
lgb_train,
num_boost_round=3000,
verbose_eval=100)
predictions = gbm.predict(test[top_features])
test['pred_300'] = predictions

print('round3')
top_features = import_feat[import_feat['sort_num']<=350]
top_features = top_features['feature']
top_features = top_features.tolist()
lgb_train = lgb.Dataset(train[top_features], train['target'])
gbm = lgb.train(param,
lgb_train,
num_boost_round=3000,
verbose_eval=100)
predictions = gbm.predict(test[top_features])
test['pred_350'] = predictions

print('round4')
top_features = import_feat[import_feat['sort_num']<=400]
top_features = top_features['feature']
top_features = top_features.tolist()
lgb_train = lgb.Dataset(train[top_features], train['target'])
gbm = lgb.train(param,
lgb_train,
num_boost_round=3000,
verbose_eval=100)
predictions = gbm.predict(test[top_features])
test['pred_400'] = predictions

test_pred=test[['card_id','pred_250','pred_300','pred_350','pred_400']]
test_pred=test_pred.merge(df_outlier_prob,on='card_id',how='left')
test_pred.rename(columns={'target':'pred'},inplace=True)
# zr_out = test_pred[['card_id','pred']].rename(columns={'pred':'target'})
# zr_out.to_csv(out_dir+"zr_all.csv", index=False)
# zr_out = test_pred[['card_id','pred_250']].rename(columns={'pred_250':'target'})
# zr_out.to_csv(out_dir+"zr_250.csv", index=False)
# zr_out = test_pred[['card_id','pred_300']].rename(columns={'pred_300':'target'})
# zr_out.to_csv(out_dir+"zr_300.csv", index=False)
# zr_out = test_pred[['card_id','pred_350']].rename(columns={'pred_350':'target'})
# zr_out.to_csv(out_dir+"zr_350.csv", index=False)
# zr_out = test_pred[['card_id','pred_400']].rename(columns={'pred_400':'target'})
# zr_out.to_csv(out_dir+"zr_400.csv", index=False)
test_pred['target']=test_pred['pred']*0.2+test_pred['pred_250']*0.2+test_pred['pred_300']*0.2+test_pred['pred_350']*0.2+test_pred['pred_400']*0.2
#test_pred['target']=test_pred['pred_250']*0.25+test_pred['pred_300']*0.25+test_pred['pred_350']*0.25+test_pred['pred_400']*0.25
testpred=test_pred[['card_id','target']]
testpred.to_csv(out_dir+"zr0218.csv", index=False)

data1 = pd.read_csv(open(out_dir+'fuji3691.csv',encoding='utf8'))
data1.rename(columns={'target':'targetx'},inplace=True)
testpred.rename(columns={'target':'targety'},inplace=True)

data12=data1.merge(testpred,how='left',on='card_id')
data12['target']=data12['targetx']*0.4+data12['targety']*0.6

del data12['targetx']
del data12['targety']

data12.to_csv(out_dir+"zr_stack46.csv", index=False)