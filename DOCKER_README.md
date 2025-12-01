# Docker での学習手順

このプロジェクトを Docker 上で実行するまでの流れをまとめています。研究室サーバーやクラウド環境で同じセットアップを再現したい場合に利用してください。

---

## 1. 事前準備

- Docker 24.x 以降がインストール済みであることを確認します。
  ```bash
  docker --version
  ```
- GPU を使用する場合は NVIDIA ドライバと `nvidia-container-toolkit` が設定済みである必要があります。
  ```bash
  nvidia-smi
  docker info | grep -i nvidia
  ```

---

## 2. イメージのビルド

リポジトリのルートディレクトリで以下を実行してコンテナイメージを作成します。

```bash
docker build -t edrp-train .
```

- `-t edrp-train` は生成するイメージ名です。任意に変更可能です。
- Dockerfile では `requirements.txt` と追加の `torch`（CPU 版）をインストールしています。GPU 版 PyTorch が必要な場合は Dockerfile の該当行を `pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121` のように書き換えてください。

---

## 3. 学習ジョブの実行

### CPU で実行する場合
```bash
docker run --rm -v "$(pwd)":/workspace edrp-train
```
- `-v "$(pwd)":/workspace` でホストの作業ディレクトリをコンテナ内 `/workspace` にマウントします。
- Dockerfile の `CMD` によりデフォルトで `python policy/train/train.py` が実行されます。

### GPU を使用する場合
```bash
docker run --rm --gpus all -v "$(pwd)":/workspace edrp-train
```
- NVIDIA Docker 対応環境を想定しています。

### 学習パラメータを変更したい場合
`docker run` の末尾にコマンドを指定すると `CMD` を上書きできます。
```bash
docker run --rm -v "$(pwd)":/workspace edrp-train \
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

---

## 4. ログと成果物の保存

- ボリュームマウントしているため、学習結果やチェックポイントはホスト側のファイルシステムに直接保存されます。
- 長時間ジョブをバックグラウンドで動かしたい場合は `-d` を付けて起動し、`docker logs -f <container_id>` で進捗を確認してください。

---

## 5. クリーンアップ

不要になったコンテナやイメージを削除する場合:

```bash
# 停止中コンテナの一覧表示
docker ps -a

# コンテナ削除 (例: <container_id>)
docker rm <container_id>

# イメージ削除 (例: edrp-train)
docker rmi edrp-train
```

---

## 6. トラブルシューティング

| 症状 | 対処例 |
| ---- | ------ |
| `ModuleNotFoundError` | `requirements.txt` に依存ライブラリが揃っているか確認し、再ビルドする。 |
| `CUDA driver version is insufficient` | NVIDIA ドライバをホストに再インストール／更新する。 |
| `permission denied` | 自身が `docker` グループに所属しているかを管理者に確認する。 |

---

## 7. 参考

- Dockerfile に手を加える場合は `COPY requirements.txt` → `RUN pip install ...` の順序を保つことでビルドキャッシュを効率的に利用できます。
- GPU 環境で PyTorch の CUDA 版を使う場合は、ベースイメージを `nvidia/cuda:12.2.0-runtime-ubuntu22.04` 等に変更するとセットアップが簡単です。
