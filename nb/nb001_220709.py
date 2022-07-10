#!/usr/bin/env python
# coding: utf-8

# # Overview
# - *Name*を用いて同じ家族を同定し、生存率を計算した新たな列*Family_SurvRate*を作成する。
# - *Ticket*を用いて上6桁が同じチケット番号のグループ分けし、生存率を計算した新たな列*Ticket_SurvRate*を作成。

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string


# In[49]:


#df_train = pd.read_csv('/content/drive/My Drive/Colab Notebooks/data/train.csv')   # Google Colabの場合はこちら
data_train = pd.read_csv('C:/Users/ultra/Documents/GitHub/data/220417_kaggle_Practice_Porto-Seguro-s-Safe-Driver-Prediction/train.csv')   # ローカルの場合はこちら
data_train.head()


# In[50]:


#df_train = pd.read_csv('/content/drive/My Drive/Colab Notebooks/data/test.csv')   # Google Colabの場合はこちら
data_test = pd.read_csv('C:/Users/ultra/Documents/GitHub/data/220417_kaggle_Practice_Porto-Seguro-s-Safe-Driver-Prediction/test.csv')   # ローカルの場合はこちら
data_test.head()


# ## データ可視化

# In[52]:


print('train data: ', np.shape(data_train))
print('test data: ', np.shape(data_test))
print('')
print(data_train.info())


# 
# ### データをメタ化する

# In[84]:


# 'meta'というDataFrameに、各columnについて以下カテゴリの分類結果を入れる
# 'role': target, id, input
# 'level': binary, nominal, interval, ordinal
# 'keep': True except 'id'
# 'dtype': int, float, str


data = []

for f in data_train.columns:
    # 'role'分類
    if f == 'target':
        role = 'target'
    elif f == 'id':
        role = 'id'
    else:
        role = 'input'
    
    # 'level'分類
    if 'bin' in f or f == 'target':
        level = 'binary'
    elif 'cat' in f or f == 'id':
        level = 'nominal'
    elif data_train[f].dtype == float:
        level = 'interval'
    #elif data_train[f].dtype == int:
    else:
        level = 'ordinal'
    
    # 'keep'分類
    keep = True
    if f == 'id':
        keep = False
    
    # 'dtype'分類
    dtype = data_train[f].dtype
    
    # Dict形式にする
    f_dict = {
        'varname': f, 'role': role, 'level': level, 'keep': keep, 'dtype': dtype
    }
    data.append(f_dict)
    
meta = pd.DataFrame(data, columns=['varname', 'role', 'level', 'keep', 'dtype'])
meta.set_index('varname', inplace=True)
meta


# In[90]:


# 各levelに該当する説明変数の数をカウント

pd.DataFrame({'count': meta.groupby(['role','level'])['role'].count()}).reset_index()


# ### データ全体の描写

# In[96]:


# 'interval'変数を詳しく見る
# ---------------------------

v = meta[(meta['level'] == 'interval') & (meta['keep'] == True)].index
data_train[v].describe()

# --------------------------
# 'reg'
# ・ps_reg_03のみ欠損値(-1)あり
# ・変数の範囲に違いがあるため、スケーリングを検討
#
# 'car'
# ・ps_car_12とps_car_14に欠損値(-1)あり
# ・変数の範囲に違いがあるため、スケーリングを検討
#
# 'calc'
# ・欠損値なし
# ・変数範囲は0～0.9で一致
# ・平均値、標準偏差がほぼ等しく、似た分布と思われる
# --------------------------


# In[97]:


# 'ordinal'変数を詳しく見る
# --------------------------

v = meta[(meta['level'] == 'ordinal') & (meta['keep'] == True)].index
data_train[v].describe()

# --------------------------
# 'ind'
# ・欠損値なし
# ・変数の範囲に違いがあるため、スケーリングを検討
#
# 'car'
# ・唯一の変数であるps_car_11に欠損値(-1)あり
#
# 'calc'
# ・欠損値なし
# ・変数の範囲に違いがあるため、スケーリングを検討
# --------------------------


# In[100]:


# 'binary'変数を詳しく見る
# --------------------------

v = meta[(meta['level'] == 'binary') & (meta['keep'] == True)].index
data_train[v].describe()

# --------------------------
# 'target'
# ・欠損値なし
# ・平均0.0364のため、0に比べて1が極端に少ない(3.64%)
#
# 'ind'
# ・欠損値なし
# ・ps_ind_10、ps_ind_11、ps_ind_12、ps_ind_13は0に比べて1が極端に少ない
#
# 'calc'
# ・欠損値なし
# --------------------------


# ### 欠損データの確認

# In[109]:


vars_missing = []

for f in data_train.columns:
    n_missing = data_train[data_train[f] == -1][f].count()
    
    if n_missing > 0:
        vars_missing.append(f)
        perc_missing = n_missing / data_train.shape[0]
        print('Variable {} has {} records ({:.2%})'.format(f, n_missing, perc_missing)
              + ' with missing values')
        
print('In total, there are {} missing variables'.format(len(vars_missing))
      + ' with missing values')

# ----------------------------------
# ・ps_car_03_cat、ps_car_05_catは特に欠損値の割合が大きい => 変数自体を取り除く
# ・ps_reg_03は18％が欠損値 => 'interval'なのでmeanで置き換える
# ----------------------------------


# ### nominal変数の特異点数の確認

# In[113]:


v = meta[(meta['level'] == 'nominal') & (meta['keep'] == True)].index

for f in v:
    dist_values = data_train[f].value_counts().shape[0]
    print('Varaiables {} has {} distinct values'.format(f, dist_values))
    
# ----------------------------------
# ・ps_car_11_catのみ極端に特異点数が多い
# ----------------------------------


# ### target変数の不均衡対策

# ### 欠損値処理
