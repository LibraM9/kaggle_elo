# -*- coding: utf-8 -*-
#@author: limeng
#@file: train_outlier_zero.py
#@time: 2019/1/31 9:32
"""
文件说明：以0作为outlier进行训练
训练集共有1630个0，测试集为训练集的61.03%，预计有994个0
"""
import numpy as np
import pandas as pd
import time
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss

feature_dir = '/data/dev/lm/elo/feature/'
out_dir = '/data/dev/lm/elo/out/'

# df_train = pd.read_csv(feature_dir+'train_trans0111.csv') #mse 1.55739 log_loss 0.04663
# df_test = pd.read_csv(feature_dir+'test_trans0111.csv')
# df_train = pd.read_csv(feature_dir+'train_trans0128.csv') #mse 1.55646 log_loss 0.04585
# df_test = pd.read_csv(feature_dir+'test_trans0128.csv')
# df_train = pd.read_csv(feature_dir+'train_trans0129.csv') #mse 1.55909 log_loss 0.04621
# df_test = pd.read_csv(feature_dir+'test_trans0129.csv')
df_train = pd.read_csv(open('F:/数据集处理/elo/feature/train_trans_jie.csv',
                            encoding='utf8'))  # mse 1.55274 log_loss 0.04593 skf0.4576 单独训练3687
df_test = pd.read_csv(open('F:/数据集处理/elo/feature/test_trans_jie.csv', encoding='utf8'))
df_train['outliers'] = df_train['target'].apply(lambda x:1 if x==0 else 0)

target = df_train['outliers']
del df_train['outliers']
del df_train['target']

features = [c for c in df_train.columns if c not in ['card_id', 'first_active_month']]
categorical_feats = [c for c in features if 'feature_' in c]

param = {'num_leaves': 31,
         'min_data_in_leaf': 30,
         'objective':'binary',
         'max_depth': 6,
         'learning_rate': 0.01,
         "boosting": "rf",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'binary_logloss',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "random_state": 2333,
         'is_unbalance':True}

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)
oof = np.zeros(len(df_train))
predictions = np.zeros(len(df_test))
feature_importance_df = pd.DataFrame()

start = time.time()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train.values, target.values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(df_train.iloc[trn_idx][features], label=target.iloc[trn_idx],
                           categorical_feature=categorical_feats)
    val_data = lgb.Dataset(df_train.iloc[val_idx][features], label=target.iloc[val_idx],
                           categorical_feature=categorical_feats)

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=100,
                    early_stopping_rounds=200)
    oof[val_idx] = clf.predict(df_train.iloc[val_idx][features], num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    predictions += clf.predict(df_test[features], num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(log_loss(target, oof)))
ans2 = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)

df_outlier_prob = pd.DataFrame({"card_id":df_test["card_id"].values})
df_outlier_prob["target"] = predictions

#覆盖度 0.7 221,0.6 389，0.5 661，0.4 667，0.3 948
#0.6 微弱上升 稳,0.5下降一个点
def modify(x):
    if x['target_y'] > 0.6 and (x['target_x']<1 and x['target_x']>-1):
        x['target_x'] =0
    return x['target_x']

outlier_id = pd.DataFrame(df_outlier_prob.sort_values(by='target',ascending = False))
# best_submission = pd.read_csv(feature_dir+'zr3.695.csv')
# best_submission = pd.read_csv(feature_dir+'submit_gbdt3692.csv')
# best_submission = pd.read_csv(feature_dir+'fuji3691.csv')
best_submission = pd.read_csv(open('F:/数据集处理/elo/combining_submission3664.csv',encoding="utf8"))
most_likely_liers = best_submission.merge(outlier_id,how='right',on='card_id')
most_likely_liers = most_likely_liers.sort_values(by='target_y',ascending = False)
most_likely_liers['target_x'] = most_likely_liers.apply(modify,axis=1)#对结果应用规则

df_out = best_submission.merge(most_likely_liers,how='left',on='card_id')
df_out = df_out[['card_id','target_x']]
df_out.columns=['card_id','target']
df_out.to_csv("F:/数据集处理/elo/3664_modify_zero06.csv", index=False)