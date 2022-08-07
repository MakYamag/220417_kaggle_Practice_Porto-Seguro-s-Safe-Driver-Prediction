# 220417_kaggle_Practice_Porto-Seguro-s-Safe-Driver-Prediction
Porto Seguroの自動車保険請求予測コンペ練習用レポジトリ。

## Log
### 220731
- trainデータの可視化と、データ加工を実施。

#### [nb001]
- kaggleのNotebook(BERT CARREMANS氏)を参考に、1)データ読込、2)可視化、3)加工を実施。
- 3)のデータ加工は、(1)目的変数の不均衡対策としてのUndersampling、(2)欠損値補完、(3)ダミー変数作成、(4)交互作用特徴量の作成、(5)Random Forest重要特徴量による特徴量選択、を実施。
- 参考: Data Preparation & Exploration, BERT CARREMANS, https://www.kaggle.com/code/bertcarremans/data-preparation-exploration/notebook
