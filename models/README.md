# Models Directory

`Classification.h5`, `LogNormStb.h5`, and `LogNormPrt.h5` are versioned here
and are loaded by the default `ModelIntegrator(model_dir="models")` setting.

学習済みモデルはこのリポジトリで管理します。

シミュレーション実行には以下の Keras モデルを想定しています。

- `Classification.h5`: 将来の collision point が右舷側か左舷側かを推定する分類モデル。
- `LogNormStb.h5`: 右舷側の将来 collision point X 座標の対数正規分布パラメータを推定するモデル。
- `LogNormPrt.h5`: 左舷側の将来 collision point X 座標の対数正規分布パラメータを推定するモデル。

`ship_collision_model.target_model.ModelIntegrator` は、これらのモデルと `data/input_data.npy` から作成した標準化器を使って、将来 collision point の分布計算とサンプリングを行います。
