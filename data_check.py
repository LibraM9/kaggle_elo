# -*- coding: utf-8 -*-
#@author: limeng
#@file: data_check.py
#@time: 2018/12/27 16:21
"""
文件说明：数据探查
R(Recency)
F(Frequency)
M (Monetary)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import sys
sys.path.extend(['/home/dev/lm/utils_lm'])

data_dir = '/data/dev/lm/elo/data/'
out_dir = '/data/dev/lm/elo/feature/'
historical_transactions = pd.read_csv(data_dir+'historical_transactions.csv',parse_dates=['purchase_date'])
new_transactions = pd.read_csv(data_dir+'new_merchant_transactions.csv',parse_dates=['purchase_date'])
merchant =  pd.read_csv(data_dir+'merchants.csv')
train = pd.read_csv(data_dir+'train.csv',parse_dates=["first_active_month"])
test = pd.read_csv(data_dir+'test.csv',parse_dates=["first_active_month"])
sub = pd.read_csv(data_dir+'sample_submission.csv')

test = historical_transactions.iloc[:100,:]

plt.figure(figsize=(14,25))
sns.barplot(x='purchase_amount',y='category_1',data=test)
plt.title('LightGBM Features (avg over folds)')
plt.show()
plt.tight_layout()
plt.savefig('lgbm_importances.png')

main_train = pd.read_csv(open("F:/数据集/1226kaggle_elo/train.csv"),encoding='utf8')
plt.subplots(figsize=(18,8))
g = sns.FacetGrid(main_train, hue="feature_3", col="feature_2", margin_titles=True,
                  palette={1:"blue", 0:"red"} )
g=g.map(plt.scatter, "first_active_month", "target",edgecolor="w").add_legend()
ans = main_train[main_train["target"]<-33]
# plt.show()
