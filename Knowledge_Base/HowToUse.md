# How to Use

---

## 作業ディレクトリ

以下、Knowledge_Baseディレクトリにて実行


## 依存インストール

Conda仮想環境内で

```bash
conda install conda-forge::beautifulsoup4
conda install conda-forge::html2text
conda install conda-forge::sentence-transformers
conda install conda-forge::faiss-cpu
```

pyarrow=21.0.0では動作しないため、バージョンダウン

```bash
conda install -c conda-forge "pyarrow<15" --force-reinstall
```

## URLリストを編集（external KB）

`Knowledge_Base/external_kb.md` に FreeCAD公式Wiki/APIのURLをMarkdown形式で列挙。
例:

```md
- https://wiki.freecad.org/Part_Box
- https://wiki.freecad.org/Part_Cylinder
```

## KBを構築

クロール → クリーニング → Markdown化 → チャンク化 → 埋め込み → FAISS作成

```bash
# 外部KB（URL）をインデックス化（external_*.{jsonl,faiss,json}を出力）
python kb_builder.py build --urls external_kb.md --version 0.20 --out kb_out

# 内部KB（Markdown）をインデックス化（internal_*.{jsonl,faiss,json}を出力）
python kb_builder.py build --text internal_kb.md --version internal --out kb_out
```

## 生成物

- 出力ファイルは入力種別でプレフィックスが付きます。
  - 外部KB: `external_chunks.jsonl`, `external_index.faiss`, `external_meta.json`
  - 内部KB: `internal_chunks.jsonl`, `internal_index.faiss`, `internal_meta.json`

---

## 検索テスト（RAGの前段で動作確認）

```bash
# 外部KBでの検索例
python kb_builder.py query \
  --index Knowledge_Base/kb_out/external_index.faiss \
  --store Knowledge_Base/kb_out/external_chunks.jsonl \
  --q "Part Box makeBox"

# 内部KBでの検索例
python kb_builder.py query \
  --index Knowledge_Base/kb_out/internal_index.faiss \
  --store Knowledge_Base/kb_out/internal_chunks.jsonl \
  --q "設計規約 フィレット 面取り 順序"
```

---

## 想定動作フロー

1. **保存済みインデックスの読み込み**

   - `./kb_out/index.faiss` … 事前に構築した FreeCAD KB のベクトル検索用インデックス（FAISS形式）
   - `./kb_out/chunks.jsonl` … インデックスに対応するテキストチャンク（Markdown化したFreeCADドキュメントの断片＋URLやモジュール名などのメタ情報）

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

   例イメージ：

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
