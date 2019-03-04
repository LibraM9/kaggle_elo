# -*- coding: utf-8 -*-
#@author: limeng
#@file: train0202.py
#@time: 2019/2/2 23:58
"""
文件说明：阿希什+杰少组合特征训练
杰少特征：
cardf_train训练集处理
cardf_test测试集处理
cardf_ht history表全局基础特征
cardf_htlast2 局部特征，对基础特征的扩展
cardf_htau 对au=1的数据做全局基础特征
cardf_level12_ht 补充二阶特征 groupby（card_id,特征1）[特征2]
ht_expand_grpdays 对history构造基于天的特征 相邻两次购物的时间差（以天为单位）
cardf_htdays 对基于天的特征ht_expand_grpdays聚合
cardf_nmth 对new构造基于天的特征并聚合
"""

import pandas as pd
import joblib

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_rows', 1000)

dir1 = 'F:/数据集处理/elo/feature/'
dir2 = 'F:/数据集处理/elo/3.676_features/'
input_dir = 'F:/数据集处理/elo/feature/'
out_dir = 'F:/数据集处理/elo/'
#
# train1 = pd.read_csv(open(dir1+'train_trans0128.csv', encoding='utf8'))
# test1 = pd.read_csv(open(dir1+'test_trans0128.csv', encoding='utf8'))
#
# cardf_test = joblib.load(dir2+'cardf_test.jl.z')
# # cardf_train= joblib.load(dir2+'cardf_train.jl.z')
# # cardf_ht = joblib.load(dir2+'cardf_ht.jl.z')
# # cardf_htlast2 = joblib.load(dir2+'cardf_htlast2.jl.z')
# # cardf_htdays = joblib.load(dir2+'cardf_htdays.jl.z')
# # cardf_htau = joblib.load(dir2+'cardf_htau.jl.z')
# # cardf_nmth = joblib.load(dir2+'cardf_nmth.jl.z')
#
# train2 = cardf_train.merge(cardf_ht, on='card_id', how='left')
# train2 = train2.merge(cardf_htlast2, on='card_id', how='left')
# train2 = train2.merge(cardf_htdays, on='card_id', how='left')
# train2 = train2.merge(cardf_htau, on='card_id', how='left')
# train2 = train2.merge(cardf_nmth, on='card_id', how='left')
#
# test2 = cardf_test.merge(cardf_ht, on='card_id', how='left')
# test2 = test2.merge(cardf_htlast2, on='card_id', how='left')
# test2 = test2.merge(cardf_htdays, on='card_id', how='left')
# test2 = test2.merge(cardf_htau, on='card_id', how='left')
# test2 = test2.merge(cardf_nmth, on='card_id', how='left')
#
# holidays = ['hist_Christmas_day_2017_mean','hist_fathers_day_2017_mean','hist_Children_day_2017_mean',
#             'hist_Black_Friday_2017_mean','hist_Valentine_day_2017_mean','hist_Mothers_day_2018_mean',
#             'auth_Christmas_day_2017_mean','auth_fathers_day_2017_mean','auth_Children_day_2017_mean',
#             'auth_Black_Friday_2017_mean','auth_Valentine_day_2017_mean','auth_Mothers_day_2018_mean']
# train = pd.concat([train2,train1[holidays]],axis=1)
# test = pd.concat([test2,test1[holidays]],axis=1)
#
# del_cols = ['reference_day_ht','first_day_ht','activation_day_ht',
#             'reference_day_htau','first_day_htau','activation_day_htau',
#             'activation_day_newht','first_day_newht','last_day_newht','reference_day_newht']
#
# for i in del_cols:
#     try:
#         del train[i]
#         del test[i]
#     except:
#         continue
#
# train.to_csv('F:/数据集处理/elo/feature/train_trans_jie.csv',index=False)
# test.to_csv('F:/数据集处理/elo/feature/test_trans_jie.csv', index=False)

##开始训练########################################
train = pd.read_csv(open(input_dir+'train_trans_jie.csv',encoding='utf8'))
test = pd.read_csv(open(input_dir+'test_trans_jie.csv',encoding='utf8'))

target = train['target']
del train['target']

import lightgbm as lgb
import time
from sklearn.model_selection import RepeatedKFold,KFold
import numpy as np
np.random.seed(42)
from sklearn import metrics

features = [c for c in train.columns if c not in ['card_id', 'first_active_month']]
categorical_feats = [c for c in features if 'feature_' in c]

param = {'num_leaves': 31,
            'boosting_type': 'gbdt',
             'min_data_in_leaf': 30,
             'objective':'regression',
             'max_depth': -1,
             'learning_rate': 0.01,
         # "min_child_samples": 20,
             "feature_fraction": 0.9,
             "bagging_freq": 1,
             "bagging_fraction": 0.9 ,
             "bagging_seed": 11,
             "metric": 'rmse',
             "lambda_l1": 0.1,
             "verbosity": -1,
           "nthread": 4,
         "random_state": 4590}

folds = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4520)
# folds = KFold(n_splits=5, shuffle=True, random_state=15)

oof_lgb = np.zeros(len(train))
predictions_lgb = np.zeros(len(test))
start = time.time()
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=target.iloc[trn_idx],
                           categorical_feature=categorical_feats)
    val_data = lgb.Dataset(train.iloc[val_idx][features], label=target.iloc[val_idx],
                           categorical_feature=categorical_feats)

    num_round = 11000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=100,
                    early_stopping_rounds=100)
    oof_lgb[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    predictions_lgb += clf.predict(test[features], num_iteration=clf.best_iteration) / (5 * 2)

print("CV score: {:<8.5f}".format(metrics.mean_squared_error(oof_lgb, target) ** 0.5))

#特征重要度
import matplotlib.pyplot as plt
import seaborn as sns

cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:50].index)

best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(14,25))
sns.barplot(x="importance",
            y="feature",
            data=best_features.sort_values(by="importance",
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.show()

sub_df1 = pd.DataFrame({"card_id":test["card_id"].values})
sub_df1["target"] = predictions_lgb
# sub_df1.to_csv(out_dir+"submit_gbdt0204.csv", index=False)

###对特征进行筛选并训练###############################################
import_feat=feature_importance_df.groupby('feature')['importance'].agg([('import_feat','mean')]).reset_index()
import_feat = import_feat.sort_values('import_feat',ascending=False)
import_feat['sort_num']=import_feat['import_feat'].rank(ascending=0,method='dense')

print('round1')
top_features = import_feat[import_feat['sort_num']<=250]
top_features = top_features['feature']
top_features = top_features.tolist()
#X_train=(train[top_features])
lgb_train = lgb.Dataset(train[top_features], target)
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
lgb_train = lgb.Dataset(train[top_features], target)
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
lgb_train = lgb.Dataset(train[top_features], target)
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
lgb_train = lgb.Dataset(train[top_features], target)
gbm = lgb.train(param,
lgb_train,
num_boost_round=3000,
verbose_eval=100)
predictions = gbm.predict(test[top_features])
test['pred_400'] = predictions

test_pred=test[['card_id','pred_250','pred_300','pred_350','pred_400']]
test_pred=test_pred.merge(sub_df1,on='card_id',how='left')
test_pred.rename(columns={'target':'pred'},inplace=True)
test_pred['target']=test_pred['pred']*0.2+test_pred['pred_250']*0.2+test_pred['pred_300']*0.2+test_pred['pred_350']*0.2+test_pred['pred_400']*0.2
#test_pred['target']=test_pred['pred_250']*0.25+test_pred['pred_300']*0.25+test_pred['pred_350']*0.25+test_pred['pred_400']*0.25
testpred=test_pred[['card_id','target']]
testpred.to_csv("F:/数据集处理/elo/submit_gbdt_fs0204.csv", index=False)
#stacking##############################
sub_df1 = pd.read_csv(open('F:/数据集处理/elo/submit_gbdt3687.csv',encoding='utf8'))
data1 = pd.read_csv(open('F:/数据集处理/elo/fuji3691.csv',encoding='utf8'))
data1.rename(columns={'target':'targetx'},inplace=True)
sub_df1.rename(columns={'target':'targety'},inplace=True)

data12=data1.merge(sub_df1,how='left',on='card_id')
data12['target']=data12['targetx']*0.4+data12['targety']*0.6
del data12['targetx']
del data12['targety']

data12.to_csv("F:/数据集处理/elo/submit_stack.csv", index=False)


