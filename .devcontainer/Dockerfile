# Microsoft の devcontainer (Debian 12/bookworm, Python 3.11, arm64)
FROM mcr.microsoft.com/devcontainers/python:3.11-bookworm

# 非対話
ENV DEBIAN_FRONTEND=noninteractive

# bash をシェルに
SHELL ["/bin/bash", "-lc"]

# ---- 基本パッケージ & GUI/X11 依存（OpenCV/Qt/FreeCAD向け） ----
# 重要ポイント：
# - freecad（0.20.x, Qt5/PySide2 系。Python モジュールは /usr/lib/freecad/lib）
# - xcb 系プラグイン、OpenGL ソフトウェアレンダ（LIBGL_ALWAYS_SOFTWARE=1）
# - 日本語ロケール（ja_JP.UTF-8）と C.UTF-8 の両方を準備
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # 基本
    ca-certificates \
    curl \
    git \
    build-essential \
    pkg-config \
    # X11/Qt (xcb) 依存
    libgl1 \
    libglu1-mesa \
    libglx0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxkbcommon-x11-0 \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xinput0 \
    # FreeCAD
    freecad \
    # 念のため PySide2 明示（FreeCAD 0.20系は Qt5/PySide2）
    python3-pyside2.qtcore \
    python3-pyside2.qtgui \
    python3-pyside2.qtwidgets \
    # ロケール/フォント
    locales \
    fontconfig \
    fonts-noto-cjk \
    fonts-noto-cjk-extra \
    fonts-noto-color-emoji \
    # ユーティリティ
    vim less unzip && \
    # 日本語ロケール生成
    sed -i 's/^# *ja_JP.UTF-8 UTF-8/ja_JP.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen ja_JP.UTF-8 && \
    update-locale LANG=ja_JP.UTF-8 LC_CTYPE=ja_JP.UTF-8 && \
    # クリーンアップ
    rm -rf /var/lib/apt/lists/*

# ---- 環境変数 ----
# UTF-8 ロケールを既定（アプリ側が C.UTF-8 を期待しても困らないよう両対応）
ENV LANG=ja_JP.UTF-8 \
    LC_ALL=ja_JP.UTF-8

# FreeCAD の Python モジュールパスを通す
# Debian/Ubuntu 系は /usr/lib/freecad/lib に FreeCAD.so などが入る
ENV PYTHONPATH="/usr/lib/freecad/lib:${PYTHONPATH}"

# 既定はヘッドレス（GUI 使うときはコンテナ起動後に QT_QPA_PLATFORM=xcb を export）
ENV QT_QPA_PLATFORM=offscreen
# GPU なしでも OpenGL を使えるように
ENV LIBGL_ALWAYS_SOFTWARE=1

# ---- vscode ユーザーに切替 ----
USER vscode

# ---- 作業ディレクトリ ----
WORKDIR /workspaces/FreeCAD_API_with_doc

# ---- Python 依存 ----
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade --no-cache-dir pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- 既定はシェル起動（Dev Container 前提）----
CMD ["/bin/bash"]
