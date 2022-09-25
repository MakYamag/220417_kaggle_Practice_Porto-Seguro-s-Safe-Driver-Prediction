#!/usr/bin/env python
# coding: utf-8

# # Overview
# - gini係数および標準化gini係数を定義。
# - 参考：https://www.kaggle.com/code/batzner/gini-coefficient-an-intuitive-explanation/notebook

# In[1]:


import numpy as np


# In[4]:


# gini_coef(target, pred)：gini係数
# ----------------------------------
#   target：正解（0 or 1）
#   pred：targetが1となる予測確率
# ==================================

def gini_coef(target, pred):
    
    # targetとpredの長さが同じでないとエラーになる
    assert (len(target) == len(pred))
    
    # 左から[target, pred, 通し番号]となるように結合
    all = np.asarray(np.c_[target, pred, np.arange(len(target))], dtype=np.float)
    
    # predが大きい順、通し番号の小さい順の優先順位でソート
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses
    giniSum -= (len(target) + 1) / 2.
    
    return giniSum / len(target)



# gini_norm(target, pred)：標準化gini係数
# ----------------------------------
#   target：正解（0 or 1）
#   pred：targetが1となる予測確率
# ==================================

def gini_norm(target, pred):
    
    return gini_coef(target, pred) / gini_coef(target, target)


# In[2]:


# gini_xgb(pred, xgb_train)：XGBoost評価用gini係数
# ------------------------------------------------
#   pred：予測確率
#   xgb_train：XGBoost用に加工されたtrainデータ
# ==============================================

def gini_xgb(pred, xgb_train):
    target = xgb_train.get_label().astype(int)
    gini_score = gini_norm(target, pred)
    
    return 'gini_xgb', gini_score

