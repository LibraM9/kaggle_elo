# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 10:36:46 2019

@author: rate9
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

cardf_test = joblib.load('E:\\Data\\elo\\3.676_features\\cardf_test.jl.z')
cardf_train = joblib.load('E:\\Data\\elo\\3.676_features\\cardf_train.jl.z')
cardf_ht = joblib.load('E:\\Data\\elo\\3.676_features\\cardf_ht.jl.z')
cardf_htlast2 = joblib.load('E:\\Data\\elo\\3.676_features\\cardf_htlast2.jl.z')
cardf_htdays = joblib.load('E:\\Data\\elo\\3.676_features\\cardf_htdays.jl.z')
cardf_htau = joblib.load('E:\\Data\\elo\\3.676_features\\cardf_htau.jl.z')
cardf_nmth = joblib.load('E:\\Data\\elo\\3.676_features\\cardf_nmth.jl.z')

train=cardf_train.merge(cardf_ht,on='card_id',how='left')
train=train.merge(cardf_htlast2,on='card_id',how='left')
train=train.merge(cardf_htdays,on='card_id',how='left')
train=train.merge(cardf_htau,on='card_id',how='left')
train=train.merge(cardf_nmth,on='card_id',how='left')

test=cardf_test.merge(cardf_ht,on='card_id',how='left')
test=test.merge(cardf_htlast2,on='card_id',how='left')
test=test.merge(cardf_htdays,on='card_id',how='left')
test=test.merge(cardf_htau,on='card_id',how='left')
test=test.merge(cardf_nmth,on='card_id',how='left')

print (train[['card_id']].head(5))
print (train[['reference_day_ht', 'first_day_ht', 'activation_day_ht']].head(5))

features = [c for c in train.columns if c not in ['card_id', 'target','reference_day_ht', 
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

from sklearn.metrics import log_loss
start = time.time()
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
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
from sklearn.metrics import mean_squared_error
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
plt.tight_layout()
plt.savefig('E:/Data/elo/lgb_importances_zj.png')

import_feat=feature_importance_df.groupby('feature')['importance'].agg([('import_feat','mean')]).reset_index()
import_feat = import_feat.sort_values('import_feat',ascending=False)
import_feat['sort_num']=import_feat['import_feat'].rank(ascending=0,method='dense')


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
test_pred['target']=test_pred['pred']*0.2+test_pred['pred_250']*0.2+test_pred['pred_300']*0.2+test_pred['pred_350']*0.2+test_pred['pred_400']*0.2
#test_pred['target']=test_pred['pred_250']*0.25+test_pred['pred_300']*0.25+test_pred['pred_350']*0.25+test_pred['pred_400']*0.25
testpred=test_pred[['card_id','target']]
testpred.to_csv("E:/Data/elo/testpred.csv", index=False)

data1 = pd.read_csv('E:/Data/elo/fuji3691.csv')
data1.rename(columns={'target':'targetx'},inplace=True)
testpred.rename(columns={'target':'targety'},inplace=True)

data12=data1.merge(testpred,how='left',on='card_id')
data12['target']=data12['targetx']*0.4+data12['targety']*0.6
del data12['targetx']
del data12['targety']

data12.to_csv("E:/Data/elo/data_1testpred.csv", index=False)

df_outlier_prob = pd.read_csv('E:/Data/elo/df_outlier_prob.csv')
df_outlier_prob = df_outlier_prob.sort_values(by='target',ascending = False)

outlier_id = pd.DataFrame(df_outlier_prob.sort_values(by='target',ascending = False).head(25000))
best_submission = data12
most_likely_liers = best_submission.merge(outlier_id,how='right',on='card_id')
most_likely_liers = most_likely_liers.sort_values(by='target_y',ascending = False)

def modify(x): #对行构造规则 -15 0.2 -=5 3.689；-13 0.16 -=5 3.685;-13 0.16 -=10 3.684
    if x['target_x'] < -12 and x['target_y'] > 0.15:
        x['target_x'] -= 10
    if x['target_x']<=-33.21928095:
        x['target_x'] =-33.21928095
    return x['target_x']

most_likely_liers['target_x']= most_likely_liers.apply(modify,axis=1)#对结果应用规则

def Correction(x): #对行构造规则 -15 0.2 -=5 3.689；-13 0.16 -=5 3.685;-13 0.16 -=10 3.684
    if x['target_x']<=-33.21928095:
        x['target_x'] =-33.21928095
    return x['target_x']

most_likely_liers['target_x']= most_likely_liers.apply(Correction,axis=1)
#把最有可能为-33的id替换为原最优值，其余使用无-33的数据所训练的结果
#model_without_outliers中选择特定id，将值替换为most_likely_liers中的值

model_without_outliers = pd.read_csv("E:/Data/elo/model_without_outliers.csv")

for card_id in most_likely_liers['card_id']:
    model_without_outliers.loc[model_without_outliers['card_id']==card_id,'target'] \
        = most_likely_liers.loc[most_likely_liers['card_id']==card_id,'target_x'].values

model_without_outliers.to_csv("E:/Data/elo/combining_data_1testpred.csv", index=False)