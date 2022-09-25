#!/usr/bin/env python
# coding: utf-8

# # Overview
# - nb004までは、訓練時の評価指標はtarget（0 or 1）に対する予測の正解率だったが、コンペの評価指標である「標準化gini係数」を評価指標として訓練するようにした。
# - 解析モデルとしてXGBoostを使用。ハイパーパラメータはネットで見かけたものの受け売り。
#  参考：Stratified KFold+XGBoost+EDA Tutorial(0.281), https://www.kaggle.com/code/sudosudoohio/stratified-kfold-xgboost-eda-tutorial-0-281/notebook

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common import gini_coefficient

pd.set_option('display.max_columns', 100)


# ## 1) データ読み込み

# In[3]:


# trainデータ
# -----------
data_train = pd.read_csv('../data/train_nb003.csv', index_col=0)
data_train.tail()


# In[4]:


# testデータ
# ----------
data_test = pd.read_csv('../data/test_nb003.csv', index_col=0)
data_test.tail()


# In[5]:


# testデータ(id)
# --------------
id_test = pd.read_csv('../data/id_test_nb003.csv', index_col=0)
id_test.tail()


# In[6]:


X = data_train.drop('target', axis=1)
y = data_train['target']
X_test = data_test


# ## 2) 解析

# In[7]:


# 訓練用、チェック用にデータ分割
# -------------------------------
X_train, X_check, y_train, y_check = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)   # 訓練:テスト = 80:20

print('Label counts in y_train: [0 1] =', np.bincount(y_train.astype(np.int64)))
print('Label counts in y_check: [0 1] =', np.bincount(y_check.astype(np.int64)))


# In[6]:


# =================================
# Pipeline: pl_svc
# SVC / k分割交差検証 / グリッドサーチ
# =================================

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

# SVCパイプライン作成
pl_svc = make_pipeline(StandardScaler(), SVC(random_state=21, max_iter=100, probability=True))

# 標準化gini係数を評価指標とする
normalized_gini = make_scorer(gini_coefficient.gini_norm, greater_is_better=True)
# 評価指標は引数に(y_true(正解データ), y_pred(予測データ))を呼ぶ必要がある

### Grid Search試行用 ###
#svc_param_range = [1.0]
#svc_param_grid = [{'svc__C': svc_param_range, 'svc__kernel': ['rbf'], 'svc__gamma': svc_param_range}]

### Grid Search本番用 1 ###
svc_param_range = [0.01, 0.1, 1.0, 10.0]
svc_param_grid = [{'svc__C': svc_param_range, 'svc__kernel': ['rbf'], 'svc__gamma': svc_param_range}]

### Grid Search本番用 2 ###
#svc_param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
#svc_param_grid = [{'svc__C': svc_param_range, 'svc__kernel': ['linear']},
#                  {'svc__C': svc_param_range, 'svc__kernel': ['poly', 'rbf', 'sigmoid'], 'svc__gamma': svc_param_range}]


svc_gs = GridSearchCV(estimator=pl_svc, param_grid=svc_param_grid, scoring=normalized_gini, cv=5, refit=True, n_jobs=-1)
svc_gs.fit(X_train, y_train)

print('Grid Search best score:', svc_gs.best_score_)
print('Best parameters:', svc_gs.best_params_)
svc_bestclf = svc_gs.best_estimator_


# In[7]:


pd.DataFrame(svc_gs.cv_results_)


# In[8]:


# testデータで予測
# =================

#svc_pred = svc_bestclf.predict(X_test)
#submission = pd.DataFrame({"id": id_test['id'], "target": svc_pred})

svc_pred_proba = svc_bestclf.predict_proba(X_test)
submission_proba = pd.DataFrame({"id": id_test['id'], "target": svc_pred_proba[:,1]})

# csvに出力
#submission_proba.to_csv("submission_nb005_svc.csv", index = False)

submission_proba


# In[9]:


max(submission_proba['target'])


# In[10]:


# =============================================
# Pipeline: pl_randf
# ランダムフォレスト / k分割交差検証 / グリッドサーチ
# =============================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

# Random Forestパイプライン作成
pl_randf = make_pipeline(RandomForestClassifier(random_state=21, n_jobs=-1))

# 標準化gini係数を評価指標とする
normalized_gini = make_scorer(gini_coefficient.gini_norm, greater_is_better=True)
# 評価指標は引数に(y_true(正解データ), y_pred(予測データ))を呼ぶ必要がある

from sklearn.model_selection import GridSearchCV

### Grid Search試行用 ###
rf_param_estimators_range = [100, 200]
rf_param_depth_range = [5, 10]
rf_param_split_range = [5, 10]

### Grid Search本番用 ###
#rf_param_estimators_range = [100, 200, 300, 400, 500]
#rf_param_depth_range = [5, 10, 15, 20, 25, 30]
#rf_param_split_range = [5, 10, 15, 20, 25, 30]

rf_param_grid = [{'randomforestclassifier__criterion': ['gini', 'entropy'],
                  'randomforestclassifier__n_estimators': rf_param_estimators_range,
                  'randomforestclassifier__max_depth': rf_param_depth_range,
                  'randomforestclassifier__min_samples_split': rf_param_split_range}]
rf_gs = GridSearchCV(estimator=pl_randf, param_grid=rf_param_grid, scoring=normalized_gini, cv=5, refit=True, n_jobs=-1)
rf_gs.fit(X_train, y_train)

print('Grid Search best score:', rf_gs.best_score_)
print(rf_gs.best_params_)
rf_bestclf = rf_gs.best_estimator_


# In[11]:


pd.DataFrame(rf_gs.cv_results_)


# In[8]:


# =============================================
# Model: mdl_xgb
# =============================================

import xgboost as xgb

# データ加工
xgb_train = xgb.DMatrix(X_train.values, label=y_train)
xgb_check = xgb.DMatrix(X_check.values, label=y_check)
xgb_test = xgb.DMatrix(X_test.values)

# パラメータ設定
params = {'objective': 'reg:squarederror', 'silent':1, 'random_state':21}
num_round = 500
watchlist = [(xgb_train, 'train'), (xgb_check, 'check')]

# モデル構築
mdl_xgb = xgb.train(params, xgb_train, num_round, early_stopping_rounds=100,
                    evals=watchlist, feval=gini_coefficient.gini_xgb, maximize=True,
                    verbose_eval=100)


# In[9]:


# testデータで予測
# =================

xgb_pred = mdl_xgb.predict(xgb_test)
submission_proba = pd.DataFrame({"id": id_test['id'], "target": xgb_pred})

# csvに出力
#submission_proba.to_csv("submission_nb005_xgb.csv", index = False)

submission_proba

