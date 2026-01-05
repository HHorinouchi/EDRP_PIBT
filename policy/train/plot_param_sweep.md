# plot_param_sweep.py 説明

`policy/train/plot_param_sweep.py` は、`policy/train/sweep_results/logs` にある
学習ログ（`train_log_<map>_agents_<n>.csv`）と
`policy/train/sweep_results/priority_params_<map>_agents_<n>.json` を読み取り、
各環境ごとに JSON の最善パラメータを基準値に設定し、
`my_policy.py` で用いられている各パラメータを 1 つずつ
`±2.5`（0.1 刻み）動かしたときの報酬平均（100 エピソード）を計算します。

計算は `ProcessPoolExecutor` を用いた並列処理で行われ、
環境ごとに以下を出力します:

- PNG プロット: `policy/train/sweep_results/param_sweep/param_sweep_<map>_agents_<n>_<param>.png`
- CSV: `policy/train/sweep_results/param_sweep/param_sweep_<map>_agents_<n>.csv`
- JSON: `policy/train/sweep_results/param_sweep/param_sweep_<map>_agents_<n>_base.json`
  （最善パラメータと best_reward_mean の保存）

## 実行内容の概要

1. `policy/train/sweep_results/logs` のログ CSV を列挙
2. `priority_params_<map>_agents_<n>.json` を最善パラメータとして取得
3. 各パラメータを 1 つずつ `±2.5`（0.1 刻み）でスイープ
4. 各点について 100 エピソードの平均報酬を算出
5. パラメータごとの報酬曲線をプロットし、PNG/CSV/JSON を保存

## 使い方

例:

```bash
python policy/train/plot_param_sweep.py --episodes 100 --workers 8
```

### 主なオプション

- `--logs-dir`  
  ログディレクトリ（デフォルト: `policy/train/sweep_results/logs`）

- `--output-dir`  
  出力先ディレクトリ（デフォルト: `policy/train/sweep_results/param_sweep`）

- `--episodes`  
  各点で回すエピソード数（デフォルト: `100`）

- `--workers`  
  並列ワーカー数（デフォルト: `CPU コア数`）

- `--max-steps`  
  各エピソードの最大ステップ数（`0` で未指定、環境の time_limit に従う）

- `--seed`  
  乱数シード（デフォルト: `0`）

## パラメータ対応

ログ列 `v0..v8` は次の順序で解釈します（best_reward_mean 記録用）:

1. `goal_weight`
2. `pick_weight`
3. `drop_weight`
4. `idle_bias`
5. `idle_penalty`
6. `assign_pick_weight`
7. `assign_drop_weight`
8. `congestion_weight`
9. `step_tolerance`

スイープ対象は `my_policy.py` のパラメータ一式で、
`assign_idle_bias` も含めています（ログに含まれないため、
基準値は `my_policy.py` の現在値を使用）。
