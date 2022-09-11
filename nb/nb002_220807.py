#!/usr/bin/env python
# coding: utf-8

# # Overview
# - nb001で出力したデータをSVC、ランダムフォレストのグリッドサーチで解析。

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', 100)

import seaborn as sns
import string
from sklearn.utils import shuffle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


# ## 1) データ読み込み

# In[6]:


data_train = pd.read_csv('../data/train_nb001.csv', index_col=0)
data_train.tail()


# In[7]:


X = data_train.drop('target', axis=1)
y = data_train['target']


# ## 2) 解析

# In[8]:


# 訓練用、CV用にデータ分割
# -------------------------
X_train, X_check, y_train, y_check = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)   # 訓練:テスト = 80:20

print('Label counts in y_train: [0 1] =', np.bincount(y_train.astype(np.int64)))
print('Label counts in y_check: [0 1] =', np.bincount(y_check.astype(np.int64)))


# In[26]:


# =================================
# Pipeline: pl_svc
# SVC / k分割交差検証 / グリッドサーチ
# =================================

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

### max_iter ###
pl_svc = make_pipeline(StandardScaler(), SVC(random_state=21, max_iter=100))

from sklearn.model_selection import GridSearchCV

### Grid Search試行用 ###
#svc_param_range = [0.01, 0.1]
#svc_param_grid = [{'svc__C': svc_param_range, 'svc__kernel': ['linear']},
#                  {'svc__C': svc_param_range, 'svc__kernel': ['poly', 'rbf', 'sigmoid'], 'svc__gamma': svc_param_range}]

### Grid Search本番用 ###
svc_param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
svc_param_grid = [{'svc__C': svc_param_range, 'svc__kernel': ['linear']},
                  {'svc__C': svc_param_range, 'svc__kernel': ['poly', 'rbf', 'sigmoid'], 'svc__gamma': svc_param_range}]

svc_gs = GridSearchCV(estimator=pl_svc, param_grid=svc_param_grid, scoring='accuracy', cv=10, refit=True, n_jobs=-1)
svc_gs.fit(X_train, y_train)

print('CV best accuracy:', svc_gs.best_score_)
print('Best parameters:', svc_gs.best_params_)
svc_bestclf = svc_gs.best_estimator_
print('Test accuracy: %f' % svc_bestclf.score(X_check, y_check))

#svc_pred = svc_bestclf.predict(X_test)


# In[9]:


# =============================================
# Pipeline: pl_randf
# ランダムフォレスト / k分割交差検証 / グリッドサーチ
# =============================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

pl_randf = make_pipeline(RandomForestClassifier(random_state=21, n_jobs=-1))
34
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
rf_gs = GridSearchCV(estimator=pl_randf, param_grid=rf_param_grid, scoring='accuracy', cv=10, refit=True, n_jobs=-1)
rf_gs.fit(X_train, y_train)

print('CV best accuracy:', rf_gs.best_score_)
print(rf_gs.best_params_)
rf_bestclf = rf_gs.best_estimator_
print('Test accuracy: %f' % rf_bestclf.score(X_check, y_check))

#rf_pred = rf_bestclf.predict(X_test)

