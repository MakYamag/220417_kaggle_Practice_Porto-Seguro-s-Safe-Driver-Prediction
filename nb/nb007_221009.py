#!/usr/bin/env python
# coding: utf-8

# # Overview
# - nb005ベースに、パラメータを調節しやすいように書き換えた。
# - XGBoostは、二値分類モデル（binary:logistic）を使用。

# In[101]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import gini_coefficient

pd.set_option('display.max_columns', 100)


# ## 1) データ読み込み

# In[102]:


# trainデータ
# -----------
data_train = pd.read_csv('../data/train_nb003.csv', index_col=0)
data_train.tail()


# In[103]:


# testデータ
# ----------
data_test = pd.read_csv('../data/test_nb003.csv', index_col=0)
data_test.tail()


# In[104]:


# testデータ(id)
# --------------
id_test = pd.read_csv('../data/id_test_nb003.csv', index_col=0)
id_test.tail()


# In[105]:


X = data_train.drop('target', axis=1)
y = data_train['target']
X_test = data_test


# ## 2) 解析

# In[106]:


# 訓練用、チェック用にデータ分割
# -------------------------------

from sklearn.model_selection import train_test_split

X_train, X_check, y_train, y_check = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)   # 訓練:テスト = 80:20

print('Label counts in y_train: [0 1] =', np.bincount(y_train.astype(np.int64)))
print('Label counts in y_check: [0 1] =', np.bincount(y_check.astype(np.int64)))


# In[112]:


# ===============
# Model: mdl_xgb
# XGBoost / 分類
# ===============

import xgboost as xgb

### データ加工
xgb_train = xgb.DMatrix(X_train.values, label=y_train)
xgb_check = xgb.DMatrix(X_check.values, label=y_check)
xgb_test = xgb.DMatrix(X_test.values)

### モデルパラメータ設定
params = {'objective':'binary:logistic', 'eta':0.01, 'random_state':21}
watchlist=[(xgb_train, 'train'), (xgb_check, 'check')]
# "binary:logistic"は、二値分類、ロジスティック回帰の意味
# "eta"はLearning Rateに相当するもの

### モデル構築
mdl_xgb = xgb.train(params=params, dtrain=xgb_train,
                    num_boost_round=10000, early_stopping_rounds=100,
                    custom_metric=gini_coefficient.gini_xgb, maximize=True,
                    evals=watchlist, verbose_eval=100
                   )
# 評価指標は"gini_xgb"で、"maximixe=True"で大きいほうがよいと設定

### ベスト時点の結果出力
print('Normalized Gini Coef (Train):', gini_coefficient.gini_xgb(mdl_xgb.predict(xgb_train, ntree_limit=mdl_xgb.best_ntree_limit), xgb_train))
print('Normalized Gini Coef (Check):', gini_coefficient.gini_xgb(mdl_xgb.predict(xgb_check, ntree_limit=mdl_xgb.best_ntree_limit), xgb_check))


# In[113]:


### testデータで予測
xgb_pred = mdl_xgb.predict(xgb_test)
# Classifierだが".predict"で確率を予測出力する

submission_proba = pd.DataFrame({"id": id_test['id'], "target": xgb_pred})
# 書き込みには予測出力の2列目(target=1の確率)を使用する

# csvに出力
#submission_proba.to_csv("submission_nb005_xgb.csv", index = False)

submission_proba

