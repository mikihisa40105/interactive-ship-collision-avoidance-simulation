# Ship Collision Avoidance Target Model

[English](README.md) | **日本語**

## 概要

本リポジトリでは、船舶の衝突回避時における他船の行動を確率的に表現する**インタラクティブ他船行動モデル（Interactive Target Ship Behavior Model）** の学習コード、学習済みモデル、および同モデルを用いたシミュレーションコードを公開しています。

本モデルは、船舶間の相対的な位置関係や運動状態から、相手船が自船の左右どちら側を、どの程度の距離で航過するかを確率的に推定します。同一の遭遇条件でも複数の行動が生じ得ることを表現でき、両船の行動の相互作用や不確実性を考慮した反復シミュレーションによる安全性評価に利用できます。

公開コードは研究手法の理解と再利用を目的として整理したものです。**研究に用いた実データは公開していませんが、学習済みモデルは公開しています。**

## 公開内容

- **モデル学習コード**：航過方向の分類モデルと、航過位置の確率分布モデルの学習
- **学習済みモデル**：分類モデル1個と、右舷側・左舷側それぞれの対数正規分布モデル
- **シミュレーションコード**：統合モデルを用いた遭遇シミュレーションと反復試行
- **スクリプト・補助コード**：データ生成、学習、シミュレーションのコマンドライン実行

## モデルの概要

モデルは、衝突予測点（Predicted Point of Collision: PPC）に基づく遭遇幾何と船舶の運動状態を入力として、将来の航過位置に関する次の2つを予測します。

1. **航過方向（Passing Side）**：右舷側・左舷側それぞれの確率を分類モデルで推定します。
2. **航過位置（Passing Position）**：各側の将来の衝突予測点の X 座標を、対数正規分布モデルで表現します。モデルが扱う座標および符号の定義は、実装とデータ形式の説明を参照してください。

3つのモデルを `ModelIntegrator` で統合して、シミュレーションで利用します。

## Main Workflow（基本的な処理の流れ）

1. 元データから `data/input_data.npy` と `data/output_data.npy` を作成します。
2. 航過方向を予測する分類モデル `Classification.h5` を学習します。
3. 右舷側・左舷側の航過位置を対数正規分布で表すモデル `LogNormStb.h5` と `LogNormPrt.h5` を学習します。
4. `ModelIntegrator` で3つの学習済みモデルを統合します。
5. 統合モデルを他船行動モデルとして使用し、`AgentSimulation11_ins1` 相当の代表的な避航シミュレーションを実行します。必要に応じて試行を繰り返し、接近距離などの指標を集計します。

## Repository Layout（ディレクトリ構成）

```text
src/ship_collision_model/
  target_model.py       # 分類モデルと対数正規分布モデルの統合
  simulation.py         # 遭遇シミュレーションと反復試行
  ships.py              # 船舶クラスと操船判断ロジック
  geometry.py           # 航法・CPA・衝突予測点に関する処理
  losses.py             # KerasのNLL損失関数・評価指標
  evaluation.py         # Gamma分布と対数正規分布の比較補助
  sample_data.py        # 配列形状に適合する合成データ生成

scripts/
  generate_dummy_data.py
  train_classifier.py
  train_passage_distribution.py
  run_simulation_demo.py

data/
  README.md

models/
  README.md

outputs/
  README.md

```

## Required Data and Models（必要なデータとモデル）

研究で用いた実データは、利用権限上の制約により含まれていません。想定する主なファイルは以下のとおりです。

| ファイル | 内容 |
| --- | --- |
| `data/input_data.npy` | 入力データ。形状 `(N, 6)` |
| `data/output_data.npy` | 教師データ。形状 `(N, 2)` |
| `models/Classification.h5` | 航過方向の学習済み分類モデル |
| `models/LogNormStb.h5` | 右舷側の学習済み分布モデル |
| `models/LogNormPrt.h5` | 左舷側の学習済み分布モデル |

ファイル形式と配置の詳細は [data/README.md](data/README.md) および [models/README.md](models/README.md) を参照してください。実データが非公開のため、本研究と同一の学習条件を再現することはできません。一方、学習済みモデルを利用した推論とシミュレーションを想定しています。

## Quick Smoke Test（簡易動作確認）

実データを使用せず、データの形状と基本的なコード実行を確認するために、合成ダミーデータを生成できます。

```bash
pip install -e .
python scripts/generate_dummy_data.py
```

**注意：** 合成ダミーデータは研究結果の再現やモデル精度の評価には使用できません。上記コマンドは学習済みモデルによるシミュレーションの実行を意味するものではありません。

## 実行手順

`data/input_data.npy` と `data/output_data.npy` を配置した後、以下でモデルを学習します。

```bash
python scripts/train_classifier.py
python scripts/train_passage_distribution.py
```

`data/input_data.npy` と `models/` 内の3つの学習済みモデルを配置した後、以下で代表シミュレーションを実行します。

```bash
python scripts/run_simulation_demo.py
```

元の notebook は履歴資料として `legacy/` に残しており、サポート対象の実行手順には含めません。

## Limitations and Disclaimer（利用上の注意）

本リポジトリは研究・教育目的で公開しています。実船での衝突回避性能や安全性を保証するものではありません。学習済みモデルの予測特性は学習データに依存するため、データに十分含まれない遭遇条件への適用には注意が必要です。

## License

本リポジトリで公開しているソースコードおよび学習済みモデルの重みは、MIT License の下で公開しています。

学習に使用した元データは第三者から提供されたものであり、本リポジトリには含まれていません。また、当該データは本リポジトリのライセンス対象外です。

詳細については、LICENSE ファイルを参照してください。
