## continue_train_by_gpu.py で学習曲線を保存する方法

- 単体実行: `python policy/train/continue_train_by_gpu.py --log-csv policy/train/train_log_gpu.csv`  
  - `--plot-png` を省略すると、`<log-csv と同じ場所>/<log名>_learning_curve.png` に学習曲線（平均報酬の推移）が自動保存されます。  
  - 任意の保存先にしたい場合は `--plot-png policy/train/plots/custom.png` のように明示指定します。

- スイープ実行: `python policy/train/continue_train_by_gpu.py --sweep --iterations 40 --population 16 --log-csv policy/train/train_log_gpu.csv`  
  - 各マップ/エージェント数ごとのログに対して、自動で `<log名>_learning_curve.png` を生成します。例えば `sweep_results/logs/train_log_map_shibuya_agents_10.csv` に対しては `sweep_results/logs/train_log_map_shibuya_agents_10_learning_curve.png` が出力され、ファイル名から対象マップとエージェント数が分かります。  
  - `--plot-png` を明示指定すると、全タスクで同じパスを使うので、スイープ時はデフォルトの自動命名を推奨します。

- 依存関係: 内部で `matplotlib` を使用するため、事前にインストール済みであることを確認してください。
