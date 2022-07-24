#!/usr/bin/env python
# coding: utf-8

# # Overview
# - *Name*を用いて同じ家族を同定し、生存率を計算した新たな列*Family_SurvRate*を作成する。
# - *Ticket*を用いて上6桁が同じチケット番号のグループ分けし、生存率を計算した新たな列*Ticket_SurvRate*を作成。

# In[116]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
from sklearn.utils import shuffle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures


# In[2]:


#df_train = pd.read_csv('/content/drive/My Drive/Colab Notebooks/data/train.csv')   # Google Colabの場合はこちら
data_train = pd.read_csv('C:/Users/ultra/Documents/GitHub/data/220417_kaggle_Practice_Porto-Seguro-s-Safe-Driver-Prediction/train.csv')   # ローカルの場合はこちら
data_train.head()


# In[3]:


#df_train = pd.read_csv('/content/drive/My Drive/Colab Notebooks/data/test.csv')   # Google Colabの場合はこちら
data_test = pd.read_csv('C:/Users/ultra/Documents/GitHub/data/220417_kaggle_Practice_Porto-Seguro-s-Safe-Driver-Prediction/test.csv')   # ローカルの場合はこちら
data_test.head()


# ## 1) データ可視化

# In[4]:


print('train data: ', np.shape(data_train))
print('test data: ', np.shape(data_test))
print('')
print(data_train.info())


# 
# ### 1.1) データをメタ化する

# In[5]:


# 'meta'というDataFrameに、各columnについて以下カテゴリの分類結果を入れる
# 'role': target, id, input
# 'level': binary, categorical, continuous, ordinal
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
        level = 'binary'   # バイナリ変数（0か1）のデータ
    elif 'cat' in f or f == 'id':
        level = 'categorical'   # カテゴリ変数のデータ
    elif data_train[f].dtype == float:
        level = 'continuous'   # 浮動小数点のノーマルデータ
    #elif data_train[f].dtype == int:
    else:
        level = 'ordinal'   # 整数のノーマルデータ
    
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


# In[6]:


# 各levelに該当する説明変数の数をカウント

pd.DataFrame({'count': meta.groupby(['role','level'])['role'].count()}).reset_index()


# ### 1.2) データ全体の描写

# In[7]:


# 'continuous'変数を詳しく見る
# ---------------------------

v = meta[(meta['level'] == 'continuous') & (meta['keep'] == True)].index
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


# In[8]:


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


# In[9]:


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


# ### 1.3) 欠損データの確認

# In[10]:


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
# ・ps_reg_03は18％が欠損値 => 'continuous'なのでmeanで置き換える
# ・その他欠損値も'continuous'はmean、'ordinal'はmodeで置き換える
# ----------------------------------


# ### 1.4) categorical変数の一意なデータ数の確認

# In[11]:


v = meta[(meta['level'] == 'categorical') & (meta['keep'] == True)].index

for f in v:
    dist_values = data_train[f].value_counts().shape[0]
    print('Varaiables {} has {} distinct values'.format(f, dist_values))
    
# ----------------------------------
# ・ps_car_11_catのみ極端に一意なデータ数が多い
# ----------------------------------


# ## 2) データ可視化

# ### 2.1) categoricalデータの可視化

# In[97]:


# 各categoricalデータにおいて、値ごとの
# 「'target'の平均」=「'target'=1である確率」をグラフ化
# ------------------------------------------------------

v = meta[(meta['level'] == 'categorical') & (meta['keep'] == True)].index

vs = v.size
row = int(np.ceil(vs / 2))

i = 1
fig = plt.figure(figsize=(9.6, 21.6), dpi=100)

for f in v:
    ax = fig.add_subplot(row, 2, i, xlabel=f, ylabel='% target')
    
    # 各データでの'target'平均値を計算
    cart_perc = data_train[[f, 'target']].groupby([f], as_index=False).mean()
    cart_perc.sort_values(by='target', ascending=False, inplace=True)
    
    # グラフに出力
    X = cart_perc[f]
    Y = cart_perc['target']
    color = [('lightpink' if i==-1 else 'lightblue') for i in X]
    ax.bar(X, Y, color=color)   # 欠損値のみピンクで出力する

    i += 1

    
# ------------------------------------------------
#
# 欠損値が'target'=1に寄与している変数が多いことがわかる
#
# ------------------------------------------------


# ### 3.2) continuousデータの可視化

# In[98]:


# 各continuousデータの相関を可視化
# ---------------------------------

def corr_heatmap(v):
    correlations = data_train[v].corr()
    
    # カラーマップの作成
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    fig = plt.figure(figsize=(9.6, 9.6), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    sns.heatmap(correlations, cmap=cmap, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={'shrink': .75})

    
v = meta[(meta['level'] == 'continuous') & (meta['keep'] == True)].index
corr_heatmap(v)


# -------------------------------------
# ・'ps_reg_02'と'ps_reg_03'の相関0.7
# ・'ps_car_!2'と'ps_car_13'の相関0.67
# ・'ps_car_12'と'ps_car_14'の相関0.58
# ・'ps_car_13'と'ps_car_15'の相関0.53
# -------------------------------------


# In[100]:


# 高速化のため10%のデータをランダムに抽出
s = data_train.sample(frac=0.1)


# 'ps_reg_02'と'ps_reg_03'をプロット
# -----------------------------------

sns.lmplot(x='ps_reg_02', y='ps_reg_03', data=s, hue='target', palette='Set1', 
           scatter_kws={'alpha': 0.2})


# In[101]:


# 'ps_car_12'と'ps_car_13'をプロット
# -----------------------------------

sns.lmplot(x='ps_car_12', y='ps_car_13', data=s, hue='target', palette='Set1', 
           scatter_kws={'alpha': 0.2})


# In[102]:


# 'ps_car_12'と'ps_car_14'をプロット
# -----------------------------------

sns.lmplot(x='ps_car_12', y='ps_car_14', data=s, hue='target', palette='Set1', 
           scatter_kws={'alpha': 0.2})


# In[103]:


# 'ps_car_13'と'ps_car_15'をプロット
# -----------------------------------

sns.lmplot(x='ps_car_13', y='ps_car_15', data=s, hue='target', palette='Set1', 
           scatter_kws={'alpha': 0.2})


# ### 3.3) ordinalデータの可視化

# In[96]:


# 各ordinalデータの相関を可視化
# -------------------------------

v = meta[(meta['level'] == 'ordinal') & (meta['keep'] == True)].index
corr_heatmap(v)


# -----------------------------------------
# 'ordinal'データは互いに相関はほとんどない
# -----------------------------------------


# ## 3) データ加工

# ### 3.1) target変数の不均衡対策

# In[105]:


# 1.2)で見たように'target'内で1の割合が極端に少ない
# 対策として0をundersamplingして相対的に0と1の割合差を小さくする
# --------------------------------------------------------------

desired_apriori = 0.10   # 全体に対して1がこの割合になるよう0をundersampleする

# 0と1のindexを取得
idx_0 = data_train[data_train['target'] == 0].index
idx_1 = data_train[data_train['target'] == 1].index

nb_0 = len(idx_0)
nb_1 = len(idx_1)

# undersampling rateを計算
undersampling_rate = ((1 - desired_apriori) * nb_1) / ((desired_apriori) * nb_0)
undersampled_nb_0 = int(undersampling_rate * nb_0)

print('Rate to undersample records with target=0: {}'.format(undersampling_rate))
print('Number of records with target=0 after undrsampling: {}'.format(undersampled_nb_0))

# 'target'=0のデータを必要な数だけランダムに選択する
undersampled_idx = shuffle(idx_0, random_state=21, n_samples=undersampled_nb_0)

# 'target'=0および1のリストを作成する
idx_list = list(undersampled_idx) + list(idx_1)

# undersampleしたデータを取り出す
train = data_train.loc[idx_list].reset_index(drop=True)
train_w_nan = train.copy()   # 欠損値ありのものをコピーしておく
train


# ### 3.2) 欠損値処理

# In[109]:


# 欠損値の割合が大きい ps_car_03_cat、ps_car_05_cat を取り除く
# -----------------------------------------------------------

vars_to_drop = ['ps_car_03_cat', 'ps_car_05_cat']
if np.size(train, axis=1) == 59:
    train.drop(vars_to_drop, inplace=True, axis=1)

# 'meta'をアップデートしておく
meta.loc[(vars_to_drop), 'keep'] = False


# 欠損値にmean またはmodeを代入する
# ----------------------------------

mean_imp = SimpleImputer(missing_values=-1, strategy='mean')
mode_imp = SimpleImputer(missing_values=-1, strategy='most_frequent')

vars_to_mean = ['ps_reg_03', 'ps_car_12', 'ps_car_14']
mean_imp.fit(train[(vars_to_mean)])
train[(vars_to_mean)] = mean_imp.transform(train[(vars_to_mean)])

vars_to_mode = ['ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_car_01_cat',
                'ps_car_02_cat', 'ps_car_07_cat', 'ps_car_09_cat', 'ps_car_11']
mode_imp.fit(train[(vars_to_mode)])
train[(vars_to_mode)] = mode_imp.transform(train[(vars_to_mode)])


# 欠損値がなくなったか確認
# -------------------------

vars_missing = []

for f in train.columns:
    n_missing = train[train[f] == -1][f].count()
    
    if n_missing > 0:
        vars_missing.append(f)
        perc_missing = n_missing / train.shape[0]
        print('Variable {} has {} records ({:.2%})'.format(f, n_missing, perc_missing)
              + ' with missing values')
        
print('In total, there are {} missing variables'.format(len(vars_missing))
      + ' with missing values')


# ### 3.3) categorical変数のダミー化

# In[111]:


v = meta[(meta['level'] == 'categorical') & (meta['keep'] == True)].index
print('Before dummification, we have {} variables in train data'.format(train.shape[1]))

train = pd.get_dummies(train, columns=v, drop_first=True)
print('After dummification, we have {} variables in train data'.format(train.shape[1]))


# ### 3.4) continuous変数のべき乗および交互作用の特徴量を作成

# In[126]:


v = meta[(meta['level'] == 'continuous') & meta['keep'] == True].index
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
interactions = pd.DataFrame(data=poly.fit_transform(train[v]),
                            columns=poly.get_feature_names(v))

# オリジナルの特徴量は重複となるので削除
interactions.drop(v, axis=1, inplace=True)

print('Before creating interactions, we have {} variables in train data'.format(train.shape[1]))

train = pd.concat([train, interactions], axis=1)
print('After creating interactions, we have {} variables in train data'.format(train.shape[1]))

interactions


# In[ ]:




