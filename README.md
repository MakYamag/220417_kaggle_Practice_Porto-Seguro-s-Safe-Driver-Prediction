# 220417_kaggle_Practice_Porto-Seguro-s-Safe-Driver-Prediction
Porto Seguroの自動車保険請求予測コンペ練習用レポジトリ。

## Log
### 220731
- trainデータの可視化と、データ加工を実施。

#### [nb001]
- kaggleのNotebook(BERT CARREMANS氏)を参考に、1)データ読込、2)可視化、3)加工を実施。
- 3)のデータ加工は、(1)目的変数の不均衡対策としてのUndersampling、(2)欠損値補完、(3)ダミー変数作成、(4)交互作用特徴量の作成、(5)Random Forest重要特徴量による特徴量選択、を実施。
- 参考: Data Preparation & Exploration, BERT CARREMANS, https://www.kaggle.com/code/bertcarremans/data-preparation-exploration/notebook


### 220807
- nb001で加工・出力したデータを解析。

#### [nb002]
- nb001で出力したデータをSVC、ランダムフォレストのグリッドサーチで解析。
- SVCは全154通りのグリッドサーチで1～2時間程度、ランダムフォレストは全8通りのグリッドサーチで1時間程度。
- 結果は、SVC：Test accuracy=0.895755、ランダムフォレスト：Test accuracy=0.899995と、わずかながらランダムフォレストに分がある。


### 220911
- nb003にて、nb001を、trainおよびtestデータの両方を加工できるように変更。
- kaggle_nb001にて、nb003の加工データを基にnb002のSVCで解析。

#### [nb003]
- nb001をtestデータの加工、出力にも対応するように変更した。
- idの列は番号が歯抜けになっており、Submissionの際に必要なため、testデータのidは別ファイルで保存する仕様。

#### [kaggle_nb001]
- nb002のSVC部分を使って解析。
- 結果は、Public：-0.00031、Private：-0.00730と、低い結果となった（トップランカーで0.29程度）。出力結果であるtargetを0か1で出しているが、1にどれだけ近いかの確率で出したほうが良さそう？
