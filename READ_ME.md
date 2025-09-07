# How to Use

---

## Run

```bash
base ❯ conda activate py3.11
py3.11 ❯ cd /Users/mo/Projects/FreeCAD_with_LLM/FreeCAD_API
py3.11 ❯ export PYTHONPATH=/Applications/FreeCAD.app/Contents/Resources/lib:${PYTHONPATH}
py3.11 ❯ export OPEN_API_KEY="xxxx"
py3.11 ❯ export DEEPSEEK_API_KEY="xxxx"
```

---

## Environment

### Create v-env

```bash
conda create -n py3.11 python=3.11
```

＊FreeCADは3.11で動作（3.12だとうまく動かない）

### Libralies

```bash
conda install conda-forge::pyside2
conda install conda-forge::openai
conda install conda-forge::pivy
```

### Python Path

FreeCAD の Python モジュールが入ったディレクトリをPYTHONPATH に追加

- macOS/Linux

```bash
export PYTHONPATH=/Applications/FreeCAD.app/Contents/Resources/lib:${PYTHONPATH}
```

- Windows

```text
C:\Program Files\FreeCAD XX\bin
```

### AddonManager を無効化

```text
/applications/FreeCAD.app/Contents/Resources/Mod/AddonManager
```

上記ディレクトリをデスクトップなどの別フォルダにドラッグして退避

### APIキーを環境変数に設定

conda環境であればconda環境内で設定する

OpneAIを使う場合

```bash
export OPENAI_API_KEY="xxxx"
```

DeepSeekを使う場合

```bash
export DEEPSEEK_API_KEY="xxxx"
```

アンセットスクリプト

```bash
unset OPENAI_API_KEY
unset DEEPSEEK_API_KEY
```

---

### コードの解説

#### app.py

pythonコードでアプリを起動。テキスト入力欄にFreeCAD用のモデル生成コードを入力（ジオメトリ部分のみでO.K.）。右側にFreeCAD_GUIを埋め込み表示。

- Populating font family aliases took 90 ms. Replace uses of missing font family "Courier" with one that exists to avoid this cost. >>> フォントの呼び出しエラー。無視してO.K.
- 21:09:47  Wizard shaft module cannot be loaded >>> ギアやシャフト設計用ウィザード）が、対応するモジュール／ワークベンチを見つけられずロードに失敗している。無視してO.K.
- Unknown command 'Std_AddonMgr' >>> 無視してO.K.

---

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

---

## REFERENCES

- model に "deepseek-chat"（V3）（会話向き）か "deepseek-reasoner"（R1）（推論・コーディング向き）を指定する。`stream=True` とするとリアルタイム出力を受け取れる。

- Model Princing
  - [OpenAI Model Pricing](https://platform.openai.com/docs/pricing)
  - [DeepSeek Modek Pricing](https://api-docs.deepseek.com/quick_start/pricing)

- FreeCAD Docs
  - [FreeCAD Part Workbech](https://wiki.freecad.org/Part_Workbench)
