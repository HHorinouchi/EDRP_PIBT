# syntax=docker/dockerfile:1

# ベースイメージには軽量な Python 3.10 を使用
FROM python:3.10-slim

# 必要な OS パッケージをインストール
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libffi-dev \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを作成
WORKDIR /workspace

# 依存ライブラリを先にコピーしてインストール
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu

# プロジェクト全体をコピー
COPY . .

# デフォルトのエントリーポイント。引数で上書き可能。
CMD ["python", "policy/train/train.py"]
