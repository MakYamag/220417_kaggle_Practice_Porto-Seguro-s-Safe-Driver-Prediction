#!/usr/bin/env python
# coding: utf-8

# # Overview
# - nb004までは、訓練時の評価指標はtarget（0 or 1）に対する予測の正解率だったが、コンペの評価指標である「標準化gini係数」を評価指標として訓練するようにした。
# - 解析モデルとしてXGBoostを使用。ハイパーパラメータはネットで見かけたものの受け売りで、回帰モデルを使用。
# - 参考：Stratified KFold+XGBoost+EDA Tutorial(0.281), https://www.kaggle.com/code/sudosudoohio/stratified-kfold-xgboost-eda-tutorial-0-281/notebook

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common import gini_coefficient

pd.set_option('display.max_columns', 100)


# ## 1) データ読み込み

# In[2]:


# trainデータ
# -----------
data_train = pd.read_csv('../data/train_nb003.csv', index_col=0)
data_train.tail()


# In[3]:


# testデータ
# ----------
data_test = pd.read_csv('../data/test_nb003.csv', index_col=0)
data_test.tail()


# In[4]:


# testデータ(id)
# --------------
id_test = pd.read_csv('../data/id_test_nb003.csv', index_col=0)
id_test.tail()


# In[5]:


X = data_train.drop('target', axis=1)
y = data_train['target']
X_test = data_test


# ## 2) 解析

# In[6]:


# 訓練用、チェック用にデータ分割
# -------------------------------
X_train, X_check, y_train, y_check = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)   # 訓練:テスト = 80:20

print('Label counts in y_train: [0 1] =', np.bincount(y_train.astype(np.int64)))
print('Label counts in y_check: [0 1] =', np.bincount(y_check.astype(np.int64)))


# In[9]:


# ================
# Model: mdl_xgb
# XGBoost / 回帰
# ================

import xgboost as xgb

### データ加工
xgb_train = xgb.DMatrix(X_train.values, label=y_train)
xgb_check = xgb.DMatrix(X_check.values, label=y_check)
xgb_test = xgb.DMatrix(X_test.values)

### パラメータ設定
params = {'objective':'reg:squarederror', 'random_state':21}
watchlist = [(xgb_train, 'train'), (xgb_check, 'check')]
# "reg:squarederror"は、回帰問題(regression)、二乗誤差の意味

### モデル構築
mdl_xgb = xgb.train(params, xgb_train, num_boost_round=500, early_stopping_rounds=100,
                    evals=watchlist, custom_metric=gini_coefficient.gini_xgb, maximize=True,
                    verbose_eval=20)
# 評価指標は"gini_xgb"で、"maximixe=True"で大きいほうがよいと設定


# In[10]:


# testデータで予測
# =================

xgb_pred = mdl_xgb.predict(xgb_test)
# ".predict"で確率を予測出力する

submission_proba = pd.DataFrame({"id": id_test['id'], "target": xgb_pred})

# csvに出力
#submission_proba.to_csv("submission_nb005_xgb.csv", index = False)

submission_proba

