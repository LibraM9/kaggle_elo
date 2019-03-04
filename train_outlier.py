# -*- coding: utf-8 -*-
# @author: limeng
# @file: train_outlier.py
# @time: 2019/1/11 14:28
"""
文件说明：对于-33.21928095的训练, 将所有-33标为1，共2207个，占比1.09%
测试集有数据123623,预计-33个数为1347个
1630个0
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
# df_train = pd.read_csv(feature_dir+'train_trans0128.csv') #mse 1.55646 log_loss 0.04585 单独训练3692
# df_test = pd.read_csv(feature_dir+'test_trans0128.csv')
# df_train = pd.read_csv(feature_dir+'train_trans0129.csv') #mse 1.55909 log_loss 0.04621 单独训练3693
# df_test = pd.read_csv(feature_dir+'test_trans0129.csv')
df_train = pd.read_csv(open('F:/数据集处理/elo/feature/train_trans_jie.csv',
                            encoding='utf8'))  # mse 1.55274 log_loss 0.04593 skf0.4576 单独训练3687
df_test = pd.read_csv(open('F:/数据集处理/elo/feature/test_trans_jie.csv', encoding='utf8'))
# df_train = pd.read_csv(open('F:/数据集处理/elo/feature/train_trans0218.csv',encoding='utf8')) #mse 1.55473 log_loss skf0.4623 单独训练3685
# df_test = pd.read_csv(open('F:/数据集处理/elo/feature/test_trans0218.csv',encoding='utf8'))
df_train['outliers'] = df_train['target'].apply(lambda x: 1 if x < -33 else 0)

# 增加特征merchant特征#####
# train_merchant = pd.read_csv(feature_dir+'train_merchant0122.csv') #mse 1.55749 log_loss 0.04683
# test_merchant = pd.read_csv(feature_dir+'test_merchant0122.csv')
# df_train = df_train.merge(train_merchant,how='left',on='card_id')
# df_test = df_test.merge(test_merchant,how='left',on='card_id')
##########################
# 对target不为-33的数据进行训练
df_train = df_train[df_train['outliers'] == 0]
target = df_train['target']
del df_train['target']
features = [c for c in df_train.columns if c not in ['card_id', 'first_active_month', 'outliers']]
categorical_feats = [c for c in features if 'feature_' in c]

param = {'objective': 'regression',
         'num_leaves': 31,
         'min_data_in_leaf': 25,
         'max_depth': 7,  # 7
         'learning_rate': 0.01,
         'lambda_l1': 0.13,
         "boosting": "gbdt",
         "feature_fraction": 0.85,
         'bagging_freq': 8,
         "bagging_fraction": 0.9,
         "metric": 'rmse',
         "verbosity": -1,
         "random_state": 2333}

# folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2333)
folds = KFold(n_splits=5, shuffle=True, random_state=2333)
oof = np.zeros(len(df_train))
predictions = np.zeros(len(df_test))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train, df_train['outliers'].values)):
    print("fold {}".format(fold_))
    trn_data = lgb.Dataset(df_train.iloc[trn_idx][features],
                           label=target.iloc[trn_idx])  # , categorical_feature=categorical_feats)
    val_data = lgb.Dataset(df_train.iloc[val_idx][features],
                           label=target.iloc[val_idx])  # , categorical_feature=categorical_feats)

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=100,
                    early_stopping_rounds=200)
    oof[val_idx] = clf.predict(df_train.iloc[val_idx][features], num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    predictions += clf.predict(df_test[features], num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(mean_squared_error(oof, target) ** 0.5))

model_without_outliers = pd.DataFrame({"card_id": df_test["card_id"].values})
model_without_outliers["target"] = predictions
# model_without_outliers["target1"] = predictions
# model_without_outliers["target2"] = predictions
# model_without_outliers["target"] = 0.5*model_without_outliers["target1"]+0.5* model_without_outliers["target2"]
# del model_without_outliers["target1"]
# del model_without_outliers["target2"]

ans1 = feature_importance_df[["Feature", "importance"]].groupby("Feature").mean().sort_values(by="importance",
                                                                                              ascending=False)
# 以outliers为目标值进行训练
# df_train = pd.read_csv(feature_dir+'train_trans0111.csv')
# df_test = pd.read_csv(feature_dir+'test_trans0111.csv')
# df_train = pd.read_csv(feature_dir+'train_trans0128.csv')
# df_test = pd.read_csv(feature_dir+'test_trans0128.csv')
# df_train = pd.read_csv(feature_dir+'train_trans0129.csv')
# df_test = pd.read_csv(feature_dir+'test_trans0129.csv')
df_train = pd.read_csv(open('F:/数据集处理/elo/feature/train_trans_jie.csv', encoding='utf8'))
df_test = pd.read_csv(open('F:/数据集处理/elo/feature/test_trans_jie.csv', encoding='utf8'))
# df_train = pd.read_csv(open('F:/数据集处理/elo/feature/train_trans0218.csv',encoding='utf8'))
# df_test = pd.read_csv(open('F:/数据集处理/elo/feature/test_trans0218.csv',encoding='utf8'))
df_train['outliers'] = df_train['target'].apply(lambda x: 1 if x < -33 else 0)

# 增加特征merchant特征######
# train_merchant = pd.read_csv(feature_dir+'train_merchant0122.csv')
# test_merchant = pd.read_csv(feature_dir+'test_merchant0122.csv')
# df_train = df_train.merge(train_merchant,how='left',on='card_id')
# df_test = df_test.merge(test_merchant,how='left',on='card_id')
############################
target = df_train['outliers']
del df_train['outliers']
del df_train['target']

features = [c for c in df_train.columns if c not in ['card_id', 'first_active_month']]
categorical_feats = [c for c in features if 'feature_' in c]

param = {'num_leaves': 31,
         'min_data_in_leaf': 30,
         'objective': 'binary',
         'max_depth': 6,
         'learning_rate': 0.01,
         "boosting": "rf",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": 'binary_logloss',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "random_state": 2333,
         'is_unbalance': True}

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)
# folds = KFold(n_splits=5, shuffle=True, random_state=15)
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
ans2 = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",
                                                                                              ascending=False)

# 把outlier存入CSV
# pd.DataFrame(oof,columns=['outlier_train']).to_csv("/data/dev/lm/elo/feature/outlier_train.csv",index=None)
# pd.DataFrame(predictions,columns=['outlier_test']).to_csv("/data/dev/lm/elo/feature/outlier_test.csv",index=None)
#####################
df_outlier_prob = pd.DataFrame({"card_id": df_test["card_id"].values})
df_outlier_prob["target"] = predictions


# df_outlier_prob["target1"] = predictions
# df_outlier_prob["target2"] = predictions
# df_outlier_prob["target"] = 0.5*df_outlier_prob["target1"]+0.5*df_outlier_prob["target2"]

# 无-33、-33分类与最优结果的合并
# 训练集中包含1.06%的-33，测试集中预计包含1310个-33
# 对行3.695构造规则 -15 0.2 -=5 3.689；-13 0.16 -=5 3.685;-13 0.16 -=10 3.684
# -13 0.16 -=11 3.680
# def modify(x):
#     if x['target_x'] < -13 and x['target_y'] > 0.16:
#         x['target_x'] -= 11
#     if x['target_x']<=-33.21928095:
#         x['target_x'] =-33.21928095
#     return x['target_x']

# 覆盖度 -16 18个，-15 24个，-14 47个，-13 77个，-12 114个， -11 160, -10 217
# 0128特征未加权 -16 0.2 -=5 3.686，-14 0.2 -=5 3.685, -13 0.16 -=5 3.683，-12 0.16 -=5 3.682
# -11 0.16 -=5 3.680
# 加权后 -13 0.15 -=10 3.675；-=9 3.673，-12 0.15 -=10 3.673，-=11 3.673；-11 0.15 -=10 3.674
# jie版本+zr3681 -12 0.15 -=10 3664；-11.5 0.15 3.662；-11 0.15 -=10 3.664;
# jie版本+zr3677 -14 0.15 -=10 3.666;-13 0.15 -=10 3663; -12 0.15 -=10 3.666
# jie版本+zr3679 -12 0.15 -=10 3.664
def modify(x):
    if x['target_x'] < -11.5 and x['target_y'] > 0.15:
        x['target_x'] -= 10
    if x['target_x'] <= -33.21928095:
        x['target_x'] = -33.21928095
    return x['target_x']


# most_likely_liers[(most_likely_liers['target_x']<-11.5)&(most_likely_liers['target_y']>0.15)].shape
#覆盖度 -13 0.15 50，-12 0.15 89，-12 0.13 90，-11 0.15 133，-11 0.13 135
# -13 0.29 3.680多一点，-13 0.25 3.684，
# 最小的20个均设为-33, 3.686；-26 0.3 3.673多一点；-25 0.3 3.672；-24 0.3 3.674
def modify2(x):  # 对行构造规则；
    if x['target_x'] < -27 and x['target_y'] > 0.3:
        x['target_x'] = -33.21928095
    return x['target_x']


outlier_id = pd.DataFrame(df_outlier_prob.sort_values(by='target', ascending=False).head(25000))
# best_submission = pd.read_csv(feature_dir+'zr3.695.csv')
# best_submission = pd.read_csv(feature_dir+'submit_gbdt3692.csv')
# best_submission = pd.read_csv(feature_dir+'fuji3691.csv')
# best_submission = pd.read_csv(feature_dir+'submit_stack3687.csv')
best_submission = pd.read_csv(open('F:/数据集处理/elo/'+'zr_stack3681.csv',encoding='utf8'))#我的3687&3691融合版本
# best_submission = pd.read_csv(open('F:/数据集处理/elo/'+'zr_stack3677.csv',encoding='utf8'))#卓然3686&3691融合版本
# best_submission = pd.read_csv(open('F:/数据集处理/elo/' + 'zr_stack3679.csv', encoding='utf8'))  # 我的3685&3691融合版本
most_likely_liers = best_submission.merge(outlier_id, how='right', on='card_id')
most_likely_liers = most_likely_liers.sort_values(by='target_y', ascending=False)
most_likely_liers['target_x'] = most_likely_liers.apply(modify, axis=1)  # 对结果应用规则
# most_likely_liers['target_x'] = most_likely_liers.apply(modify2,axis=1)#对结果应用规则

# 把最有可能为-33的id替换为原最优值，其余使用无-33的数据所训练的结果
# model_without_outliers中选择特定id，将值替换为most_likely_liers中的值
for card_id in most_likely_liers['card_id']:
    model_without_outliers.loc[model_without_outliers['card_id'] == card_id, 'target'] \
        = most_likely_liers.loc[most_likely_liers['card_id'] == card_id, 'target_x'].values

# model_without_outliers.to_csv(out_dir+"combining_submission_unknown0131.csv", index=False)
model_without_outliers.to_csv('F:/数据集处理/elo/' + "combining_submission_rule0225.csv", index=False)
