# Data Directory

研究に利用した学習データは権利上の問題により公開版には含めていません。

このリポジトリのコードは主に以下の NumPy 配列を想定しています。

- `all_X.npy`: 前処理前または前処理中の入力候補データ。
- `all_Y.npy`: 前処理前または前処理中の出力候補データ。
- `input_data.npy`: 学習用入力データ。shape は `(N, 6)`。
- `output_data.npy`: 学習用出力データ。shape は `(N, 2)`。

`input_data.npy` の列は以下を想定しています。

1. `x_cp_now`
2. `y_cp_now`
3. `tgt_length`
4. `own_length`
5. `own_spd`
6. `tgt_cog`

`output_data.npy` は、将来の collision point の X 座標と、通過または避航タイミングに関する値を格納する想定です。

