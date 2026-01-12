# continue_train_by_gpu.py ガイド

`policy/train/continue_train_by_gpu.py` に実装されている GPU 対応進化戦略 (Evolution Strategy; ES) トレーナーの使い方と、内部処理の流れをまとめたドキュメントです。

## 概要

このスクリプトは `policy/my_policy.py` が利用する PIBT の優先度パラメータを最適化します。`DrpEnv` 上で候補パラメータを評価し、CUDA もしくは Apple MPS (利用可能な場合) で ES の更新計算を行います。環境のロールアウト処理は CPU 側で実行されます。

1 イテレーションあたりの主な処理:

1. 現在のパラメータベクトル周辺にガウスノイズを付加して候補を生成。
2. 各候補パラメータを 1 本以上のエピソードで評価 (必要に応じてドメインランダム化を実施)。
3. 報酬を正規化し、GPU 上で ES の勾配推定と更新を計算。
4. 最良スコアのパラメータを記録し、必要に応じて JSON へ書き出し。
5. 各イテレーションの統計情報を CSV に追記。

既定では `time_limit=100000` とし、この上限に達するかすべてのタスクが消化された時点でエピソードが終了して次の試行へ移ります。
エピソード報酬は学習のために以下の式で再計算されます。

- `r_goal × 完了タスク数 + r_move × ステップ数 + (衝突時は r_coll × speed)`
- `reward_list` の値 (`r_goal`, `r_move`, `r_coll`) を参照し、衝突が発生した時点でもエピソードを即終了します。

## 基本的な使い方

リポジトリ直下で以下を実行します:

```bash
python policy/train/continue_train_by_gpu.py \
  --map-name map_5x4 \
  --agent-num 5 \
  --iterations 300 \
  --population 16 \
  --episodes-per-candidate 8 \
  --eval-episodes 16 \
  --candidate-workers 16 \
  --sweep-plot-dir policy/train/sweep_results/plots \
  --save-params-json sweep_results/priority_params_map_5x4_agents_5.json \
  --log-csv /sweep_results/logs/train_log_map_5x4_agents_5.csv\
  --resume
```

主なオプション:

| オプション | 説明 |
| --- | --- |
| `--iterations` | ES の更新ステップ数。
| `--population` | イテレーション毎に評価する候補パラメータ数。
| `--sigma` | 探索ノイズの標準偏差。
| `--lr` | ES 更新に適用する学習率。
| `--episodes-per-candidate` | 候補パラメータ 1 つ当たりのロールアウト回数。
| `--eval-episodes` | 最終スコア算出時の評価エピソード数。
| `--map-name` / `--agent-num` | 単体学習時のマップ・エージェント数上書き。
| `--task-density`, `--speed`, `--time-limit`, `--collision` | 追加の環境設定上書き。
| `--domain-randomize` | ロールアウト毎にマップやパラメータをランダム化。
| `--collision-penalty` | 衝突発生時の報酬を数値で上書き (`none` で従来通り)。
| `--log-csv` | イテレーション毎のメトリクス出力先。
| `--plot-png` | 学習曲線を保存する PNG の出力パス。未指定なら `--log-csv` と同じフォルダに `<stem>_learning_curve.png` を作成。 |
| `--save-params-json` | 最良パラメータの JSON 保存先 (`PriorityParams` のフィールド名で保存)。
| `--clip-step-norm` | ES 更新ベクトルのノルム制限 (任意)。
| `--workers` | CPU ロールアウトの並列プロセス数。
| `--candidate-workers` | ES の候補パラメータを並列評価するプロセス数。`-1` で自動的に CPU コア数に設定。 |
| `--sweep-workers` | スイープ時に同時実行する学習ジョブ数 (デフォルトは `--gpu-devices` の数か 1)。 |
| `--sweep-plot-dir` | スイープ実行時に各ランの学習曲線 PNG を保存するディレクトリ。指定しない場合は生成しない。 |
| `--gpu-devices` | 利用するデバイスをカンマ区切りで指定 (`"0,1"`, `"cuda:1,cpu"` など)。スイープ時は順番に割り当て。 |

## ランダムエージェント数で学習する場合

以下のスクリプトは、各エピソードで `agent_num` を
`2〜(ノード数の0.5倍)` の範囲でランダム化して学習します。

```bash
python policy/train/continue_train_by_gpu_random_agent_num.py --map-name map_5x4 --workers 8
```

`--random-map` を付けると MAP_POOL からマップもランダムに選びます。

同じ設定でランダムエージェント数版を使う場合の例:

```bash
python policy/train/continue_train_by_gpu_random_agent_num.py \
  --sweep-maps \
  --iterations 150 \
  --population 32 \
  --episodes-per-candidate 20 \
  --eval-episodes 20 \
  --candidate-workers 16 \
  --workers 0 \
  --plot-png policy/train/sweep_results/plots/random_agent_{map_name}.png \
  --log-csv policy/train/sweep_results/logs/train_log_random_agent_{map_name}.csv \
  --resume-from-log
```

## スイープモード

あらかじめ定義されたマップとエージェント数 (各マップでノード数の 1/4・1/2・3/4 を丸めた値、最低 3 体) を順番に学習させるには、以下を実行します:
総エピソード数： 

```bash
python policy/train/continue_train_by_gpu.py \
  --sweep \
  --iterations 150 \
  --population 32 \
  --episodes-per-candidate 20 \
  --eval-episodes 20 \
  --candidate-workers 16 \
  --workers 0 \
  --sweep-plot-dir policy/train/sweep_results/plots \
  --sweep-output-dir policy/train/sweep_results \
  --resume
```

出力構成:

- `policy/train/sweep_results/priority_params_<map>_agents_<n>.json` — 各学習の最良パラメータ。
- `policy/train/sweep_results/logs/train_log_<map>_agents_<n>.csv` — イテレーションログ。
- `policy/train/sweep_results/sweep_summary.json` — マップ・エージェント数・スコア・シード等のメタデータ。

`--sweep` を付けると `--map-name` / `--agent-num` は無視され、指定がある場合は警告を表示します。

## リソース最大活用のヒント

- `--candidate-workers -1` で ES の候補評価を CPU コア数いっぱいに並列化できます。`--workers` も同時に使う場合は、総プロセス数が過剰にならないよう両者の値を調整してください。
- スイープ時は `--sweep-workers <並列ジョブ数>` と `--gpu-devices 0,1,...` を組み合わせると、各ジョブを異なる GPU に割り当てつつ同時実行できます (例: `--gpu-devices 0,1 --sweep-workers 2`)。
- `--sweep-plot-dir` を設定すると、各マップ×エージェント数ごとの学習曲線 PNG が指定ディレクトリに自動保存されます。容量が増えるため必要な条件に絞って有効化してください。
- GPU での計算は ES 更新部のみですが、複数ジョブを並列化することで全 GPU を稼働させられます。CPU 側の環境ロールアウトがボトルネックになるため、必要に応じて `--candidate-workers` や `--workers` を増やしメモリ使用量と相談してください。
- フルパラレル実行例:

  ```bash
  python policy/train/continue_train_by_gpu.py \
    --sweep \
    --iterations 40 \
    --population 16 \
    --episodes-per-candidate 8 \
    --eval-episodes 20 \
    --candidate-workers -1 \
    --workers 2 \
    --sweep-workers 4 \
    --gpu-devices 0,1,2,3 \
    --sweep-output-dir policy/train/sweep_results_fast
  ```

  上記では 4 ジョブを同時に走らせ、各ジョブ内で候補評価も CPU 全体を用いて並列化します。メモリ使用量が増えるため、並列度は計算資源に合わせて調整してください。

## 出力と保存物

- `save_priority_params` 経由で最良パラメータを既定のチェックポイント位置に保存し、必要に応じて JSON へも書き出します。
- CSV ログにはイテレーション番号、報酬統計、ベスト候補のエピソード指標（`best_goal_rate` / `best_collision_rate` / `best_timeup_rate` / `best_avg_steps` / `best_avg_task_completion`）、そしてパラメータベクトルが記録されます。
- `--plot-png` を指定すると Matplotlib (ヘッドレス対応バックエンド) で報酬推移グラフを保存します。

### ログの可視化

`policy/train/plot_training_metrics.py` を使うと、上記 CSV から以下の図を自動生成できます。

```bash
python policy/train/plot_training_metrics.py \
  --log-csv policy/train/train_log_gpu.csv \
  --output-dir policy/train/plots
```

生成される主な図:

- `reward_curve.png`: 平均報酬と標準偏差、最大報酬の推移。
- `event_rates.png`: ベスト候補におけるゴール/衝突/タイムアップ率。
- `steps_and_tasks.png`: 平均ステップ数とタスク完了数。
- `parameter_heatmap.png`: 各パラメータのイテレーションごとの変化をヒートマップ表示。

## 実務上のヒント

- 未知のマップでは `iterations` や `population` を控えめに設定し、実行時間と収束傾向を確認してから段階的に増やすのが安全です。
- マルチコア CPU では `--workers` を併用してロールアウトを並列化すると高速化できます。GPU は ES 計算のみを高速化します。
- 衝突でエピソードが打ち切られる場合は、衝突ペナルティを強めるか `bounceback` モードを試すと安定しやすくなります。
- 再現性が必要な比較では `--seed` を揃えて実行してください。
- スイープ実行後は `sweep_summary.json` を確認し、追加チューニングすべきマップ・エージェント組を見極めましょう。
