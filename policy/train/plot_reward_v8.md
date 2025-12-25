# plot_reward_v8.py の使い方

このスクリプトは、学習ログ（CSV）から `reward_mean` と `v8` の推移を可視化し、PNG 形式で保存します。以下の手順で利用してください。

## 前提条件
- Python 3 系がインストールされていること
- 依存ライブラリ `matplotlib` が利用できること（`pip install matplotlib` など）

## 実行コマンド
```
python policy/train/plot_reward_v8.py <ログCSVまたはディレクトリ> [--output-dir 出力先ディレクトリ]
```

### 引数の説明
- `<ログCSVまたはディレクトリ>`: 対象となる学習ログ CSV ファイル、もしくは CSV がまとまったディレクトリを指定します。
- `--output-dir`: 生成した PNG ファイルの保存先を指定します。省略した場合、CSV が単体ならその親ディレクトリ、ディレクトリ指定なら同じ場所に保存します。

## 使用例
### 指定ディレクトリ内のログをまとめてプロットする場合
```
python policy/train/plot_reward_v8.py policy/train/sweep_results/logs --output-dir policy/train/sweep_results/plots
```
上記コマンドでは、`policy/train/sweep_results/logs` 内の全 CSV を読み込み、`reward_mean` と `v8` を同一グラフに描画した PNG を `policy/train/sweep_results/plots` に保存します。

### 単一の CSV からプロットする場合
```
python policy/train/plot_reward_v8.py policy/train/sweep_results/logs/train_log_map_5x4_agents_5.csv
```
この場合、出力ファイルは CSV と同じディレクトリに保存されます。

## 出力ファイル
- 出力ファイル名: `<CSVファイル名>_reward_v8.png`
- グラフの構成:
  - 左軸（青線）: `reward_mean`
  - 右軸（赤線）: `v8`
  - 両指標を同じ図で比較できるようツイン軸で描画します。

## エラー処理
- CSV から有効な `reward_mean` または `v8` が読み取れない場合、該当ファイルはスキップされます。
- プロットが一枚も生成されないと、スクリプトはエラーメッセージを出力して終了します。
