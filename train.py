# -*- coding: utf-8 -*-
#@author: limeng
#@file: train.py
#@time: 2018/12/28 11:39
"""
文件说明：训练及验证，在服务器运行
1229是3.715特征
0111是阿希什特征
0125 加入节假日特征
"""
import pandas as pd
import numpy as np
import datetime

feature_dir = '/data/dev/lm/elo/feature/'
out_dir = '/data/dev/lm/elo/out/'

def read_data(input_file):
    df = pd.read_csv(input_file)
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['elapsed_time'] = (datetime.date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days
    return df

# train = read_data(feature_dir+'train_trans0111.csv') #老版本特征
# test = read_data(feature_dir+'test_trans0111.csv')
train = read_data(feature_dir+'train_trans0128.csv') #加入节假日的特征
test = read_data(feature_dir+'test_trans0128.csv')

#加入商家特征
# train_merchant = pd.read_csv(feature_dir+'train_merchant0122.csv')
# test_merchant = pd.read_csv(feature_dir+'test_merchant0122.csv')
# train = train.merge(train_merchant,how='left',on='card_id')
# test = test.merge(test_merchant,how='left',on='card_id')
#加入outlier###
# outlier_train = pd.read_csv(feature_dir+'outlier_train.csv')
# # outlier_test = pd.read_csv(feature_dir+'outlier_test.csv')
# # train['outlier'] = outlier_train
# # test['outlier'] = outlier_test
###############
target = train['target']
del train['target']

####lgb-gbdt################################
import lightgbm as lgb
import time
from sklearn.model_selection import KFold
features = [c for c in train.columns if c not in ['card_id', 'first_active_month']]
categorical_feats = [c for c in features if 'feature_' in c]

param = {'num_leaves': 31,
         'min_data_in_leaf': 32,
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.01,
         # "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "nthread": 4,
         "verbosity": -1}

folds = KFold(n_splits=5, random_state=15, shuffle=True)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
start = time.time()
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(train.iloc[trn_idx][features],
                           label=target.iloc[trn_idx],
                           categorical_feature=categorical_feats
                           )
    val_data = lgb.Dataset(train.iloc[val_idx][features],
                           label=target.iloc[val_idx],
                           categorical_feature=categorical_feats
                           )

    num_round = 10000
    clf = lgb.train(param,
                    trn_data,
                    num_round,
                    valid_sets=[trn_data, val_data],#将训练集和验证集共同作为验证集
                    verbose_eval=100,
                    early_stopping_rounds=200)

    oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits

from sklearn import metrics
print("CV score: {:<8.5f}".format(metrics.mean_squared_error(oof, target) ** 0.5))

#贝叶斯调参
# from model_train.a3_LGBM_Opt import bayes_opt_lgb_regress
# yy = pd.cut(target, 10, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# best_param = bayes_opt_lgb_regress(train[features], yy,n_estimators=500,n_iter = 100,categorical_feature=categorical_feats)

#特征重要度
import matplotlib.pyplot as plt
import seaborn as sns

cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:100].index)

best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(14,25))
sns.barplot(x="importance",
            y="feature",
            data=best_features.sort_values(by="importance",
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.show()
plt.tight_layout()
plt.savefig('lgbm_importances.png')

sub_df = pd.DataFrame({"card_id":test["card_id"].values})
sub_df["target"] = predictions
sub_df.to_csv(out_dir+"submit_gbdt0128.csv", index=False)
####lgb-随机森林###########################################
rfparam = {'num_leaves': 31,
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
from sklearn.model_selection import RepeatedKFold

folds = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4520)

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
    clf = lgb.train(rfparam, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=100,
                    early_stopping_rounds=100)
    oof_lgb[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    predictions_lgb += clf.predict(test[features], num_iteration=clf.best_iteration) / (5 * 2)

print("CV score: {:<8.5f}".format(metrics.mean_squared_error(oof_lgb, target) ** 0.5))

sub_df1 = pd.DataFrame({"card_id":test["card_id"].values})
sub_df1["target"] = predictions_lgb
sub_df1.to_csv(out_dir+"submit_gbdt0128.csv", index=False)
#模型集成#########################################################
from sklearn.linear_model import BayesianRidge
train_stack = np.vstack([oof,oof_lgb]).transpose()
test_stack = np.vstack([predictions,predictions_lgb]).transpose()

folds = RepeatedKFold(n_splits=5, n_repeats=1, random_state=4520)
oof_stack = np.zeros(train_stack.shape[0])
predictions_stack = np.zeros(test_stack.shape[0])

#将两次训练结果作为输入，用贝叶斯再训练
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_stack, target)):
    print("fold n°{}".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], target.iloc[trn_idx].values
    val_data, val_y = train_stack[val_idx], target.iloc[val_idx].values

    print("-" * 10 + "Stacking " + str(fold_) + "-" * 10)
    #     cb_model = CatBoostRegressor(iterations=3000, learning_rate=0.1, depth=8, l2_leaf_reg=20, bootstrap_type='Bernoulli',  eval_metric='RMSE', metric_period=50, od_type='Iter', od_wait=45, random_seed=17, allow_writing_files=False)
    #     cb_model.fit(trn_data, trn_y, eval_set=(val_data, val_y), cat_features=[], use_best_model=True, verbose=True)
    clf = BayesianRidge()
    clf.fit(trn_data, trn_y)

    oof_stack[val_idx] = clf.predict(val_data)
    predictions_stack += clf.predict(test_stack) / 5

print(np.sqrt(metrics.mean_squared_error(target.values, oof_stack)))

sub_df_stack = pd.DataFrame({"card_id":test["card_id"].values})
sub_df_stack["target"] = predictions_stack
sub_df_stack.to_csv(out_dir+"submit_stack_unknown0125.csv", index=False)