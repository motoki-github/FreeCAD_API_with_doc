# How to Use

---

## Environment（環境構築）

### Create v-env

```bash
base ❯ conda create -n py3.11 python=3.11
base ❯ conda activate py3.11
```

＊FreeCADは3.11で動作（3.12だとうまく動かない）

### Libralies

本体アプリ関係

```bash
py3.11 ❯ conda install conda-forge::pyside2
py3.11 ❯ conda install conda-forge::openai
py3.11 ❯ conda install conda-forge::pivy
```

RAG関係

```bash
py3.11 ❯ conda install conda-forge::beautifulsoup4
py3.11 ❯ conda install conda-forge::html2text
py3.11 ❯ conda install conda-forge::sentence-transformers
py3.11 ❯ conda install conda-forge::faiss-cpu
```

pyarrow=21.0.0では動作しないため、バージョンダウン

```bash
py3.11 ❯ conda install -c conda-forge "pyarrow<15" --force-reinstall
```


### Python Path

FreeCAD の Python モジュールが入ったディレクトリをPYTHONPATH に追加

- macOS/Linux

```bash
py3.11 ❯ export PYTHONPATH=/Applications/FreeCAD.app/Contents/Resources/lib:${PYTHONPATH}
```

- Windows

```bash
py3.11 ❯ C:\Program Files\FreeCAD XX\bin
```

### AddonManager を無効化

```text
py3.11 ❯ /applications/FreeCAD.app/Contents/Resources/Mod/AddonManager
```

上記ディレクトリをデスクトップなどの別フォルダにドラッグして退避

### APIキーを環境変数に設定

conda環境であればconda環境内で設定する

OpneAIを使う場合

```bash
py3.11 ❯ export OPENAI_API_KEY="xxxx"
```

DeepSeekを使う場合

```bash
py3.11 ❯ export DEEPSEEK_API_KEY="xxxx"
```

（アンセットスクリプト）

```bash
py3.11 ❯ unset OPENAI_API_KEY
py3.11 ❯ unset DEEPSEEK_API_KEY
```

---

## KB構築

クロール → クリーニング → Markdown化 → チャンク化 → 埋め込み → FAISS作成

```bash

py3.11 ❯ cd /Users/mo/Projects/FreeCAD_with_LLM/FreeCAD_API/Knowledge_Base
# 外部KB（URL）をインデックス化（external_*.{jsonl,faiss,json}を出力）
py3.11 ❯ python kb_builder.py build --urls external_kb.md --version 0.20 --out kb_out

# 内部KB（Markdown）をインデックス化（internal_*.{jsonl,faiss,json}を出力）
py3.11 ❯ python kb_builder.py build --text internal_kb.md --version internal --out kb_out
```

### 生成物

- 出力ファイルは入力種別でプレフィックスが付きます。
  - 外部KB: `external_chunks.jsonl`, `external_index.faiss`, `external_meta.json`
  - 内部KB: `internal_chunks.jsonl`, `internal_index.faiss`, `internal_meta.json`

### 検索テスト（RAGの前段で動作確認）

```bash
# 外部KBでの検索例
python kb_builder.py query \
  --index kb_out/external_index.faiss \
  --store kb_out/external_chunks.jsonl \
  --q "Part Box makeBox"

# 内部KBでの検索例
python kb_builder.py query \
  --index kb_out/internal_index.faiss \
  --store kb_out/internal_chunks.jsonl \
  --q "設計規約 フィレット 面取り 順序"
```

---

## Run

```bash
base ❯ conda activate py3.11
py3.11 ❯ cd /Users/mo/Projects/FreeCAD_with_LLM/FreeCAD_API
py3.11 ❯ export PYTHONPATH=/Applications/FreeCAD.app/Contents/Resources/lib:${PYTHONPATH}
py3.11 ❯ export OPEN_API_KEY="xxxx"
py3.11 ❯ export DEEPSEEK_API_KEY="xxxx"
py3.11 ❯ python app.py
```

---

## REFERENCES

### コードの解説

#### app.py

pythonコードでアプリを起動。テキスト入力欄にFreeCAD用のモデル生成コードを入力（ジオメトリ部分のみでO.K.）。右側にFreeCAD_GUIを埋め込み表示。

- Populating font family aliases took 90 ms. Replace uses of missing font family "Courier" with one that exists to avoid this cost. >>> フォントの呼び出しエラー。無視してO.K.
- 21:09:47  Wizard shaft module cannot be loaded >>> ギアやシャフト設計用ウィザード）が、対応するモジュール／ワークベンチを見つけられずロードに失敗している。無視してO.K.
- Unknown command 'Std_AddonMgr' >>> 無視してO.K.

### OpenAI SDK を使った DeepSeek API の呼び出し例

DeepSeek は OpenAI 互換のエンドポイントを持つので、openai ライブラリをそのまま流用できる。

```python
from openai import OpenAI

# クライアントの初期化
client = OpenAI(
    api_key="<YOUR_DEEPSEEK_API_KEY>",
    base_url="https://api.deepseek.com"  # または "https://api.deepseek.com/v1"
)

# チャット補完リクエスト
response = client.chat.completions.create(
    model="deepseek-chat",           # deepseek-chat (V3) or deepseek-reasoner (R1)
    messages=[
        {"role": "system",  "content": "You are a helpful assistant."},
        {"role": "user",    "content": "こんにちは、調子はどう？"}
    ],
    stream=False                     # True にするとストリーミング応答
)

# 結果の表示
print(response.choices[0].message.content)
```

### KB想定動作

#### 検索テスト（RAGの前段で動作確認）

```bash
# 外部KBでの検索例
python kb_builder.py query \
  --index kb_out/external_index.faiss \
  --store kb_out/external_chunks.jsonl \
  --q "Part Box makeBox"

# 内部KBでの検索例
python kb_builder.py query \
  --index kb_out/internal_index.faiss \
  --store kb_out/internal_chunks.jsonl \
  --q "設計規約 フィレット 面取り 順序"
```

1. **保存済みインデックスの読み込み**

   - `./kb_out/internal_index.faiss` … 事前に構築したベクトル検索用インデックス（FAISS形式）
   - `./kb_out/internal_chunks.jsonl` … インデックスに対応するテキストチャンク（Markdown化したドキュメントの断片＋URLやモジュール名などのメタ情報）

   → この2つをペアでロードします。

2. **クエリのベクトル化**

   - 引数 `--q "xxxxx"` のテキストを埋め込みモデルでベクトルに変換。xxxxに関する情報を探す準備。
    （デフォルト: `sentence-transformers/all-MiniLM-L6-v2`）

3. **FAISSで類似検索**

   - クエリベクトルと各チャンクベクトルを比較して、コサイン類似度が高いものを上位 `k` 件（デフォルト5件）返す。
   - 「`xxxx` の説明が書かれた公式Wikiの断片」などが上位にヒットする。

4. **検索結果の出力**

   - JSON形式で結果を表示。
   - 含まれる情報：

     - `score`: 類似度スコア
     - `id`: チャンクID
     - `url`: 元のドキュメントURL
     - `title`: ページタイトル
     - `module`: 推定モジュール名（例: Part）
     - `version`: FreeCADバージョンラベル（例: 0.20）
     - `chunk_index`: チャンク番号
     - `preview`: ヒットした本文の先頭400文字（確認用の抜粋）

   出力イメージ：

   ```json
   [
     {
       "score": 0.87,
       "id": "abc123def456...",
       "url": "https://wiki.freecad.org/Part_Box",
       "title": "Part Box",
       "module": "Part",
       "version": "0.20",
       "chunk_index": 0,
       "preview": "The Part Box command creates a cuboid solid. It can also be created with Python using ... makeBox(length, width, height) ..."
     },
     ...
   ]
   ```

このクエリは「**生成AIにコードを書かせる前に、関連する公式情報を取り出す**」ための検索部分。

- 人間が直接調べたいときにも使える
- RAGパイプラインでは、この検索結果の本文を LLM に渡して「根拠付きコード生成」をさせる

#### REFERENCES

- model に "deepseek-chat"（V3）（会話向き）か "deepseek-reasoner"（R1）（推論・コーディング向き）を指定する。`stream=True` とするとリアルタイム出力を受け取れる。

- Model Princing
  - [OpenAI Model Pricing](https://platform.openai.com/docs/pricing)
  - [DeepSeek Modek Pricing](https://api-docs.deepseek.com/quick_start/pricing)

- FreeCAD Docs
  - [FreeCAD Part Workbech](https://wiki.freecad.org/Part_Workbench)
