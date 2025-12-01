# train.py の使い方

このディレクトリの `train.py` は、MAPD × PIBT 系の方策で用いる「エージェント行動優先度」と「タスク振り分け優先度」の重みを、進化戦略（OpenAI-ES 風）で学習します。学習済みパラメータは `policy/priority_params.json` に保存され、`policy/my_policy.py` が自動で読み込みます。

## 学習対象パラメータ

`policy/my_policy.py` 内の `PriorityParams` が対象です。

- 行動優先度（detect_actions 用）
  - `goal_weight`（ドロップへ向かうときの距離重み）
  - `pick_weight`（未ピック時のピックまでの距離重み）
  - `drop_weight`（未ピック時のピック→ドロップの距離重み）
  - `idle_bias`（タスク未割当時の加算バイアス）
  - `idle_penalty`（タスク未割当エージェントの基礎スコア）
- タスク振り分け（assign_task 用）
  - `assign_pick_weight`（エージェント→ピックの距離重み）
  - `assign_drop_weight`（ピック→ドロップの距離重み）
  - `assign_idle_bias`（割当スコアのバイアス）
- グローバル指標（全体状況に基づく重み）
  - `congestion_weight`（各エージェント周辺の混雑度ペナルティ重み）
  - `load_balance_weight`（未割当タスク比率に基づく負荷分散バイアス）

保存先: `policy/priority_params.json`（読み込みは `my_policy` が import 時に自動実施）

---

## 実行方法

事前にリポジトリ直下（プロジェクトルート）で実行してください。

### 1. 依存関係

必要に応じて（任意）

```bash
# プロジェクトルートにて
python -m pip install -r requirements.txt
```

注: 実行時に Gym の非推奨警告が出ますが、現状のスクリプトは実行可能です。長期的には Gymnasium 移行を検討してください。

### 2. 基本的な学習実行

```bash
python policy/train/train.py \
  --iterations 40 \
  --population 8 \
  --sigma 0.2 \
  --lr 0.05 \
  --episodes-per-candidate 2 \
  --eval-episodes 5 \
  --seed 0
```

- 学習の過程で、各候補パラメータを用いたエピソード報酬の平均で最適化します。
- 学習が終了すると、最良パラメータが `policy/priority_params.json` に保存されます。

### 3. 短時間のスモークテスト（動作確認用）

```bash
python policy/train/train.py \
  --iterations 5 \
  --population 6 \
  --episodes-per-candidate 1 \
  --eval-episodes 3 \
  --seed 0
```

### 4. 学習をスキップして評価のみ

保存済みパラメータ（`priority_params.json`）で評価だけを実行します。

```bash
python policy/train/train.py --eval-only --eval-episodes 5
```

---

## 単一マップ固定の実行例

- 学習（map_3x3, agent 3, time_limit 300 などを固定）
```bash
python policy/train/train.py \
  --iterations 40 \
  --population 8 \
  --episodes-per-candidate 2 \
  --eval-episodes 5 \
  --seed 0 \
  --map-name map_3x3 \
  --agent-num 3 \
  --speed 1.0 \
  --time-limit 300 \
  --collision bounceback \
  --task-density 1.0
```

- 評価のみ（同条件）
```bash
python policy/train/train.py \
  --eval-only \
  --eval-episodes 5 \
  --map-name map_3x3 \
  --agent-num 3 \
  --speed 1.0 \
  --time-limit 300 \
  --collision bounceback \
  --task-density 1.0
```

## 推奨設定と総エピソード量

総エピソード数の目安は、ES の総試行回数 T（iterations）× 反復ごとの候補数 K（population）× 1候補あたりのロールアウト数 N（episodes-per-candidate）で概算できます。
- 目安式: 総エピソード ≒ T × K × N
- 備考: `--eval-episodes` は最終評価の回数であり、上式には含めません。

以下は単一マップ（map_3x3）での固定条件のコマンド例です。必要に応じて `--map-name` などはプロジェクト要件に合わせて変更してください。

1) 軽量（20分前後）
- 設定: T=40, K=8, N=2（総エピソード ≒ 40 × 8 × 2 = 640）
```bash
python policy/train/train.py \
  --iterations 40 \
  --population 8 \
  --episodes-per-candidate 2 \
  --eval-episodes 5 \
  --seed 0 \
  --map-name map_3x3 \
  --agent-num 3 \
  --speed 1.0 \
  --time-limit 300 \
  --collision bounceback \
  --task-density 1.0
```

2) バランス（精度とコストのバランス）
- 設定: T=60, K=12, N=3（総エピソード ≒ 60 × 12 × 3 = 2160）
```bash
python policy/train/train.py \
  --iterations 60 \
  --population 12 \
  --episodes-per-candidate 3 \
  --eval-episodes 5 \
  --seed 0 \
  --map-name map_3x3 \
  --agent-num 3 \
  --speed 1.0 \
  --time-limit 300 \
  --collision bounceback \
  --task-density 1.0
```

3) 高信頼（論文化・評価図向け）
- 設定: T=80, K=16, N=5（総エピソード ≒ 80 × 16 × 5 = 6400）
```bash
python policy/train/train.py \
  --iterations 80 \
  --population 16 \
  --episodes-per-candidate 5 \
  --eval-episodes 5 \
  --seed 0 \
  --map-name map_3x3 \
  --agent-num 3 \
  --speed 1.0 \
  --time-limit 300 \
  --collision bounceback \
  --task-density 1.0
```

## 学習安定化とロギング

以下のオプションで、学習の安定性向上と再現性・可視化を強化できます。

- 衝突時の即時打ち切り＋強制ペナルティ
  - `--collision-penalty -1000` などを指定すると、衝突検出時にそのエピソードの総報酬を即座にこの値に置き換えて終了（進化方向に強い抑制をかける）。
- ログ CSV 出力
  - `--log-csv policy/train/train_log.csv` を指定すると、各イテレーションごとの `reward_mean/reward_std/reward_max` と現在のパラメータベクトルが CSV に追記保存される。
- 収束曲線の PNG 出力（非対話環境向け）
  - `--plot-png policy/train/reward_curve.png` を指定すると、`reward_mean` の推移グラフを PNG に保存する（matplotlib のバックエンドは自動で Agg に切替）。
- 進化方向のノルムクリップ
  - `--clip-step-norm 1.0` のように L2 ノルムの上限を与えると、ステップが過大になって発散するリスクを抑えられる。

例（安定化を意識したバランス設定）
```bash
python policy/train/train.py \
  --iterations 60 \
  --population 12 \
  --episodes-per-candidate 5 \
  --sigma 0.1 \
  --lr 0.05 \
  --collision-penalty -1000 \
  --log-csv policy/train/train_log.csv \
  --plot-png policy/train/reward_curve.png \
  --clip-step-norm 1.0 \
  --eval-episodes 5 \
  --seed 0 \
  --map-name map_3x3 \
  --agent-num 3 \
  --speed 1.0 \
  --time-limit 300 \
  --collision bounceback \
  --task-density 1.0
```

episodes_per_candidate は 5〜10 を推奨。報酬ばらつきが大きい場合は更に増やしてください。

## 引数一覧

- `--iterations`（デフォルト: 40）: ES の反復回数
- `--population`（デフォルト: 8）: 各反復でサンプリングする候補数
- `--sigma`（デフォルト: 0.2）: ノイズスケール
- `--lr`（デフォルト: 0.05）: 学習率
- `--episodes-per-candidate`（デフォルト: 2）: 1 候補あたりのロールアウト回数（平均を取ります）
- `--eval-episodes`（デフォルト: 5）: 最終評価のエピソード数
- `--seed`（デフォルト: 0）: 乱数シード
- `--eval-only`（フラグ）: 学習をスキップして評価のみ実行
- `--domain-randomize`（フラグ）: 複数環境からの学習を有効化（デフォルト無効、単一マップ特化で不要）
- `--map-name`（例: `map_3x3`）: 単一マップ固定用
- `--agent-num`（整数）: エージェント数を固定
- `--speed`（float）: エージェントの移動速度
- `--time-limit`（int）: エピソードの最大ステップ
- `--collision`（`bounceback` | `terminated`）: 衝突時の挙動
- `--task-density`（float）: タスク生成密度（ポアソン平均）
- `--collision-penalty`（float, 既定: -1000.0）: 衝突発生時に当該エピソードの総報酬として即時適用する値（エピソードは打ち切り）
- `--log-csv`（str, 既定: policy/train/train_log.csv）: 各イテレーションの統計値とパラメータベクトルを追記保存する CSV のパス
- `--plot-png`（str, 既定: None）: 学習後に `reward_mean` 推移の図を PNG 保存するパス（指定時のみ）
- `--clip-step-norm`（float, 既定: 0.0）: >0 であれば ES の更新ステップの L2 ノルムを上限クリップ
 
 ---

## 環境設定とドメインランダム化

既定では単一マップ特化（ドメインランダム化は無効）で学習・評価します。複数環境から学習したい場合のみ `--domain-randomize` を付与してください。単一マップ固定の条件は CLI の上書き引数（例: `--map-name`, `--agent-num`, `--speed`, `--time-limit`, `--collision`, `--task-density`）で指定できます。

- 変動させるもの
  - マップ（`drp_env/EE_map.py` の `UNREAL_MAP` にあるものからサンプル）
  - エージェント数（マップのノード数に応じて 2〜5 の範囲）
  - 速度 `speed`、`time_limit`
  - 衝突モード（`bounceback` または `terminated`）
  - タスク密度 `task_density`（ポアソン分布の平均として使用）
- 固定条件で評価したい場合
  - `ENV_CONFIG` を編集して希望の条件に固定し、`--eval-only` で評価してください。

---

## 出力

- `policy/priority_params.json`
  - 学習により得られた最良パラメータを保存。
  - `policy/my_policy.py` は import 時にこのファイルを読み込み、推論（行動決定・タスク割当）に適用します。
- 標準出力ログ
  - 各反復ごとの平均報酬・最大報酬などが表示されます。
  - 最終的なパラメータと平均エピソード報酬を出力します。

---

## 学習結果の利用

評価・運用時に `my_policy` を用いれば、保存済みの優先度が自動で反映されます。

```python
from drp_env.drp_env import DrpEnv
from policy.my_policy import policy  # priority_params.json を自動ロード

env = DrpEnv(
    agent_num=3, speed=1.0,
    start_ori_array=[], goal_array=[],
    visu_delay=0.0, state_repre_flag="onehot",
    time_limit=300, collision="bounceback",
    task_flag=True, map_name="map_3x3", task_density=1.0,
)

obs = env.reset()
done = [False] * env.agent_num
while not all(done):
    actions, task_assign = policy(obs, env)
    obs, rewards, done, info = env.step({"pass": actions, "task": task_assign})
env.close()
```

---

## 参考: エッジ重みペナルティ（衝突回避の促進）

別機構として、衝突発生時に当該エッジのコストを動的に増加させ、次回以降そのエッジが選ばれにくくなる仕組みを追加済みです。

- `DrpEnv` から取得できる情報
  - `env.get_node_info()` / `env.get_edge_info()` / `env.get_adjacency_list()`
- 手動でペナルティ適用
  - `env.add_edge_penalty([(u, v)], penalty=10.0)`
- デモ
  - `python scripts/edge_penalty_demo.py`

学習 (`train.py`) と併用することで、衝突の再発を抑制しつつ、優先度パラメータの最適化が可能です。

---

## トラブルシューティング

- `ModuleNotFoundError: No module named 'drp_env'`
  - プロジェクトルートから実行してください（`python policy/train/train.py`）。もしくは `PYTHONPATH` にプロジェクトルートを追加。
- Gym の非推奨警告
  - 実行は可能です。将来的には Gymnasium への移行（`import gymnasium as gym`）をご検討ください。
- 実行速度が遅い / 不安定
  - `--iterations`、`--population`、`--episodes-per-candidate` を小さくしてスモークテストを先に実施し、徐々に増やしてください。
- パラメータの過学習が疑われる
  - ドメインランダム化の範囲（マップ/エージェント数/タスク密度など）を見直すか、評価時は固定条件で比較してください。

---

## ライセンス・注意

- 本学習で得られた重みは、シミュレーションの設定・報酬設計に依存します。実運用に適用する場合は、評価基準（衝突回数、完了タスク数、待機時間など）を運用要件に合わせて見直してください。
