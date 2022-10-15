#!/usr/bin/env python
# coding: utf-8

# # Overview
# - nb005ベースに、パラメータを調節しやすいように書き換えた。
# - XGBoostは、回帰モデル（reg:squarederror）を使用。

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

from sklearn.model_selection import train_test_split

X_train, X_check, y_train, y_check = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)   # 訓練:テスト = 80:20

print('Label counts in y_train: [0 1] =', np.bincount(y_train.astype(np.int64)))
print('Label counts in y_check: [0 1] =', np.bincount(y_check.astype(np.int64)))


# In[15]:


# ===============
# Model: mdl_xgb
# XGBoost / 回帰
# ===============

import xgboost as xgb

### データ加工
xgb_train = xgb.DMatrix(X_train.values, label=y_train, feature_names=X.columns)
xgb_check = xgb.DMatrix(X_check.values, label=y_check, feature_names=X.columns)
xgb_test = xgb.DMatrix(X_test.values, feature_names=X.columns)

### モデルパラメータ設定
params = {'objective':'reg:squarederror', 'eta':0.01, 'random_state':21}
# "reg:squarederror"は、回帰問題(regression)、二乗誤差の意味
# "eta"はLearning Rateに相当するもの

watchlist=[(xgb_train, 'train'), (xgb_check, 'check')]
evals_result ={}
# "evals_result": 解析過程保存用

### モデル構築
mdl_xgb = xgb.train(params=params, dtrain=xgb_train,
                    num_boost_round=10000, early_stopping_rounds=100,
                    custom_metric=gini_coefficient.gini_xgb, maximize=True,
                    evals=watchlist, evals_result=evals_result, verbose_eval=50
                   )
# 評価指標は"gini_xgb"で、"maximize=True"で大きいほうがよいと設定

### ベスト時点の結果出力
print('Normalized Gini Coef (Train):', gini_coefficient.gini_xgb(mdl_xgb.predict(xgb_train, ntree_limit=mdl_xgb.best_ntree_limit), xgb_train))
print('Normalized Gini Coef (Check):', gini_coefficient.gini_xgb(mdl_xgb.predict(xgb_check, ntree_limit=mdl_xgb.best_ntree_limit), xgb_check))


# In[52]:


### testデータで予測
xgb_pred = mdl_xgb.predict(xgb_test)
# Regressionなので".predict"で確率らしい数値を予測出力する

submission_proba = pd.DataFrame({"id": id_test['id'], "target": xgb_pred})

# csvに出力
#submission_proba.to_csv("submission_nb005_xgb.csv", index = False)

submission_proba


# In[16]:


# 結果の可視化
# =============

### 解析過程グラフ出力
fig1 = plt.figure(figsize=(4.8, 3.2), dpi=80)

plt.plot(evals_result['train']['gini_xgb'], color='blue', markersize=5, label='Training')
plt.plot(evals_result['check']['gini_xgb'], color='green', markersize=5, label='Validation')
plt.grid()
plt.title('Normalized Gini Coefficient - Regression')
plt.xlabel('Number of Boost Round')
plt.ylabel('Normalized Gini Coefficient')
plt.legend()

plt.savefig('../image/nb006_norm_gini_reg.png')

### パラメータ重要度出力
xgb.plot_importance(mdl_xgb, importance_type='weight', max_num_features=10,
                    title='Feature Importance - Regression', xlabel='weight', ylabel='Features')

plt.savefig('../image/nb006_feat_importance_reg.png')

