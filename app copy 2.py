#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import re
import base64
import mimetypes
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from PySide2 import QtWidgets, QtCore
from PySide2.QtGui import QPixmap
import FreeCAD as App
import FreeCADGui as Gui

# ---------------------------
# 設定ファイルのロード
# ---------------------------
def load_config() -> Dict[str, Any]:
    cfg_path = Path(__file__).resolve().parent / "config.json"
    defaults: Dict[str, Any] = {
        "provider_defaults": {"provider": "ChatGPT", "model": "gpt-4o-mini"},
        "api_base_urls": {
            "ChatGPT": "https://api.openai.com/v1",
            "DeepSeek": "https://api.deepseek.com/v1",
        },
        "model_catalog": {
            "ChatGPT": ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4o", "gpt-4o-mini"],
            "DeepSeek": ["deepseek-chat", "deepseek-reasoner"],
        },
        "system_prompt_path": "system_msg.md",
        "window_size": {"width": 1400, "height": 800},
        "freecad_doc_name": "EmbeddedDoc",
        "hidden_panels": ["Model", "Python console", "Report view", "Tasks"],
        "initial_code_template_path": None,
        "retry": {"enabled": True, "max_attempts": 3},
        "vision": {"supported_models": ["gpt-4o", "gpt-4o-mini"]},
        "images": {"file_filter": "画像ファイル (*.png *.jpg *.jpeg *.gif *.webp *.bmp);;すべてのファイル (*)"},
        "rag": {
            "kb_base_path": str(Path(__file__).resolve().parent / "Knowledge_Base" / "kb_out"),
            "top_k": 6,
            "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
            "assistant_preamble": (
                "以下はFreeCADに関する参考資料の抜粋です。必要なAPI/用法を優先的に参照して正確なコードを作成してください。"
                "必ず参照元の意図に従い、不明点は仮定せず安全なデフォルトを選んでください。"
            ),
            "keyword_fallback": {"min_token_len": 3},
            "sources": {"internal": True, "external": True}
        },
        "view": {"isometric": True, "fit_all": True},
        "logging": {"level": "INFO"},
        "secrets": {"env_keys": {"ChatGPT": "OPENAI_API_KEY", "DeepSeek": "DEEPSEEK_API_KEY"}},
    }
    try:
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            # shallow merge
            def merge(a, b):
                for k, v in b.items():
                    if isinstance(v, dict) and isinstance(a.get(k), dict):
                        merge(a[k], v)
                    else:
                        a[k] = v
                return a
            return merge(defaults, user_cfg)
    except Exception as e:
        print(f"[Config] Failed to load config.json: {e}")
    return defaults

CONFIG = load_config()

# Optional RAG deps (handled gracefully if missing)
try:
    import faiss  # type: ignore
except Exception:
    faiss = None
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None

# OpenAI/DeepSeekクライアント取得関数
# provider: 'DeepSeek' または 'ChatGPT'
def get_openai_client(provider):
    env_map = CONFIG.get("secrets", {}).get("env_keys", {})
    api_env = env_map.get(provider, "OPENAI_API_KEY" if provider == "ChatGPT" else "DEEPSEEK_API_KEY")
    base_url_map = CONFIG.get("api_base_urls", {})
    base_url = base_url_map.get(provider, "https://api.openai.com/v1")
    if provider == "DeepSeek":
        prompt_title = "Input DeepSeek API Key"
        prompt_label = "Please Input Your DeepSeek API Key.:"
    else:
        prompt_title = "Input OpenAI API Key"
        prompt_label = "Please Input Your OpenAI API Key:"

    api_key = os.getenv(api_env)

    if not api_key:
        key, ok = QtWidgets.QInputDialog.getText(
            None, prompt_title, prompt_label, QtWidgets.QLineEdit.Password
        )
        if not ok or not key.strip():
            QtWidgets.QMessageBox.critical(None, "Error", "API key is not set. Exiting application.")
            sys.exit(1)
        api_key = key.strip()

    # OpenAIクライアントにbase_urlを正しく渡す
    return OpenAI(api_key=api_key, base_url=base_url)

class ProviderModelDialog(QtWidgets.QDialog):
    def __init__(self, combos, parent=None, default_text: str = None):
        super().__init__(parent)
        self.setWindowTitle("Select Provider + Model")
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)
        self.list = QtWidgets.QListWidget(self)
        for p, m in combos:
            self.list.addItem(f"{p} : {m}")
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        if default_text:
            matches = self.list.findItems(default_text, QtCore.Qt.MatchExactly)
            if matches:
                self.list.setCurrentItem(matches[0])
            else:
                self.list.setCurrentRow(0)
        else:
            self.list.setCurrentRow(0)
        self.list.itemDoubleClicked.connect(self.accept)
        layout.addWidget(self.list)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal,
            self,
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def selected(self):
        item = self.list.currentItem()
        if not item:
            return None, None
        text = item.text()
        provider_part, model_part = [s.strip() for s in text.split(":", 1)]
        return provider_part, model_part


class FreeCADEmbedApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # Provider + Model を一括選択（一覧から選択）
        model_catalog = CONFIG.get("model_catalog", {})
        combos = [(prov, model) for prov, models in model_catalog.items() for model in models]
        defaults = CONFIG.get("provider_defaults", {})
        default_text = f"{defaults.get('provider','ChatGPT')} : {defaults.get('model','gpt-4o-mini')}"
        dlg = ProviderModelDialog(combos, self, default_text=default_text)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            sys.exit(0)
        provider_part, model_part = dlg.selected()
        if not provider_part or not model_part:
            sys.exit(0)
        self.provider = provider_part
        self.model_name = model_part
        self.client = get_openai_client(self.provider)

        # system prompt from file or defaults
        sys_msg_path = CONFIG.get("system_prompt_path")
        system_msg = None
        if sys_msg_path:
            try:
                p = Path(__file__).resolve().parent / sys_msg_path if not Path(sys_msg_path).is_absolute() else Path(sys_msg_path)
                if p.exists():
                    system_msg = p.read_text(encoding="utf-8").strip()
            except Exception as e:
                print(f"[Config] Failed to read system prompt: {e}")
        if not system_msg:
            system_msg = (
                "あなたは FreeCAD+PySide2 アプリのエキスパートです。"
                "・出力は必ず コードの説明　からはじまり、その後　python のコードブロックで終わること"
                "・関数シグネチャは def create(doc): のみ"
                "・実行前に既存オブジェクトをすべて削除する処理を組み込む"
                "・PEP8 準拠、余計な説明テキストは一切含めない"
                "・最後に return doc を追加すること"
            )
        self.messages = [{"role": "system", "content": system_msg}]

        self.setWindowTitle(f"FreeCAD Embedded App ({self.provider} - {self.model_name})")
        ws = CONFIG.get("window_size", {"width": 1400, "height": 800})
        self.resize(int(ws.get("width", 1400)), int(ws.get("height", 800)))

        # エラーリトライカウンタ
        self.error_count = 0
        self.retry_enabled = bool(CONFIG.get("retry", {}).get("enabled", True))
        self.retry_max_attempts = int(CONFIG.get("retry", {}).get("max_attempts", 3))

        # ---------------------------
        # RAG: KB のロード（あれば）
        # ---------------------------
        kb_base_cfg = CONFIG.get("rag", {}).get("kb_base_path")
        if kb_base_cfg:
            kb_base_path = Path(kb_base_cfg)
            if not kb_base_path.is_absolute():
                kb_base_path = Path(__file__).resolve().parent / kb_base_path
        else:
            kb_base_path = Path(__file__).resolve().parent / "Knowledge_Base" / "kb_out"
        self.kb_base: Path = kb_base_path
        self.kb_loaded = False
        self.kb_internal_rows: List[Dict[str, Any]] = []
        self.kb_external_rows: List[Dict[str, Any]] = []
        self.kb_internal_index = None
        self.kb_external_index = None
        self.kb_embedder = None
        self._init_kb()

        # RAG 設定キャッシュ
        self.rag_top_k = int(CONFIG.get("rag", {}).get("top_k", 6))
        self.rag_assistant_preamble = CONFIG.get("rag", {}).get("assistant_preamble", "")
        self.rag_sources_internal = bool(CONFIG.get("rag", {}).get("sources", {}).get("internal", True))
        self.rag_sources_external = bool(CONFIG.get("rag", {}).get("sources", {}).get("external", True))
        self.keyword_min_len = int(CONFIG.get("rag", {}).get("keyword_fallback", {}).get("min_token_len", 3))
        self.view_isometric = bool(CONFIG.get("view", {}).get("isometric", True))
        self.view_fit_all = bool(CONFIG.get("view", {}).get("fit_all", True))

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        h_layout = QtWidgets.QHBoxLayout(central)
        h_layout.setContentsMargins(5,5,5,5)
        h_layout.setSpacing(5)

        # 左ペイン: 問い合わせ
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0,0,0,0)
        left_layout.setSpacing(5)
        self.query_edit = QtWidgets.QPlainTextEdit()
        self.query_edit.setPlaceholderText("問い合わせ内容を入力してください...")
        left_layout.addWidget(self.query_edit, 1)
        self.query_btn = QtWidgets.QPushButton("問い合わせ")
        left_layout.addWidget(self.query_btn)
        # 画像入力UI
        img_controls = QtWidgets.QHBoxLayout()
        self.add_image_btn = QtWidgets.QPushButton("画像を追加")
        self.remove_image_btn = QtWidgets.QPushButton("選択画像を削除")
        img_controls.addWidget(self.add_image_btn)
        img_controls.addWidget(self.remove_image_btn)
        left_layout.addLayout(img_controls)
        self.image_list = QtWidgets.QListWidget()
        self.image_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.image_list.setFixedHeight(100)
        left_layout.addWidget(self.image_list)
        # 画像プレビュー
        self.image_preview = QtWidgets.QLabel()
        self.image_preview.setMinimumHeight(160)
        self.image_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.image_preview.setStyleSheet("QLabel { border: 1px solid #aaa; background: #fafafa; }")
        self.image_preview.setText("画像プレビューなし")
        left_layout.addWidget(self.image_preview)
        self.response_edit = QtWidgets.QPlainTextEdit()
        self.response_edit.setReadOnly(True)
        left_layout.addWidget(self.response_edit, 1)
        h_layout.addWidget(left, 1)

        # 中央ペイン: コード
        center = QtWidgets.QWidget()
        center_layout = QtWidgets.QVBoxLayout(center)
        center_layout.setContentsMargins(0,0,0,0)
        self.code_edit = QtWidgets.QPlainTextEdit()
        # 初期コードテンプレート
        tmpl_path = CONFIG.get("initial_code_template_path")
        code_tmpl = None
        if tmpl_path:
            tp = Path(__file__).resolve().parent / tmpl_path if not Path(tmpl_path).is_absolute() else Path(tmpl_path)
            if tp.exists():
                try:
                    code_tmpl = tp.read_text(encoding="utf-8")
                except Exception as e:
                    print(f"[Config] Failed to read template: {e}")
        if not code_tmpl:
            code_tmpl = (
                "# FreeCAD 用 Python コード\n"
                "def create(doc):\n"
                "    box = doc.addObject('Part::Box','Box')\n"
                "    box.Length = 10\n"
                "    box.Width = 20\n"
                "    box.Height = 30\n"
            )
        self.code_edit.setPlainText(code_tmpl)
        center_layout.addWidget(self.code_edit)
        self.model_btn = QtWidgets.QPushButton("モデル生成")
        center_layout.addWidget(self.model_btn)
        h_layout.addWidget(center, 2)

        # 右ペイン: FreeCADビュー
        Gui.showMainWindow()
        self.doc = App.newDocument(CONFIG.get("freecad_doc_name", "EmbeddedDoc"))
        fc_widget = Gui.getMainWindow()
        h_layout.addWidget(fc_widget, 3)

        # 不要パネルを非表示
        self._hide_freecad_panels()

        # シグナル
        self.query_btn.clicked.connect(self.on_query)
        self.model_btn.clicked.connect(self.on_generate)
        self.add_image_btn.clicked.connect(self.on_add_image)
        self.remove_image_btn.clicked.connect(self.on_remove_image)
        self.image_list.itemSelectionChanged.connect(self.on_image_selection_changed)

        QtWidgets.QApplication.instance().aboutToQuit.connect(self._cleanup_freecad)

    # ---------------------------
    # RAG: KB 読み込み/検索
    # ---------------------------
    def _init_kb(self):
        try:
            if not self.kb_base.exists():
                print(f"[RAG] KB base not found: {self.kb_base}")
                return

            def _read_jsonl(p: Path) -> List[Dict[str, Any]]:
                rows = []
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            rows.append(json.loads(line))
                return rows

            internal_chunks = self.kb_base / "internal_chunks.jsonl"
            external_chunks = self.kb_base / "external_chunks.jsonl"
            internal_index = self.kb_base / "internal_index.faiss"
            external_index = self.kb_base / "external_index.faiss"

            if internal_chunks.exists():
                self.kb_internal_rows = _read_jsonl(internal_chunks)
            if external_chunks.exists():
                self.kb_external_rows = _read_jsonl(external_chunks)

            # Try to load FAISS + embedder if available
            if faiss is not None and SentenceTransformer is not None:
                if internal_index.exists() and self.kb_internal_rows:
                    self.kb_internal_index = faiss.read_index(str(internal_index))
                if external_index.exists() and self.kb_external_rows:
                    self.kb_external_index = faiss.read_index(str(external_index))
                # Use the same model as builder default
                try:
                    embed_name = CONFIG.get("rag", {}).get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")
                    self.kb_embedder = SentenceTransformer(embed_name)
                except Exception as e:
                    print(f"[RAG] Failed to init embedder: {e}")

            total = len(self.kb_internal_rows) + len(self.kb_external_rows)
            if total:
                self.kb_loaded = True
                print(f"[RAG] KB loaded: {total} chunks (internal={len(self.kb_internal_rows)}, external={len(self.kb_external_rows)})")
            else:
                print("[RAG] No KB chunks found.")
        except Exception as e:
            print(f"[RAG] Init failed: {e}")

    def _rag_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.kb_loaded:
            return []

        # If FAISS + embedder available, use vector search for both sources
        hits: List[Tuple[float, Dict[str, Any]]] = []
        try:
            import numpy as np  # lazy import to avoid hard dependency
            if self.kb_embedder is not None and (self.kb_internal_index is not None or self.kb_external_index is not None):
                q_emb = self.kb_embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

                def _search(index, rows, label: str):
                    if index is None or not rows:
                        return
                    if label == "internal" and not self.rag_sources_internal:
                        return
                    if label == "external" and not self.rag_sources_external:
                        return
                    D, I = index.search(np.array([q_emb]), min(top_k, len(rows)))
                    for score, idx in zip(D[0], I[0]):
                        r = rows[int(idx)]
                        rec = {"score": float(score), "source": label, **r}
                        hits.append((float(score), rec))

                _search(self.kb_internal_index, self.kb_internal_rows, "internal")
                _search(self.kb_external_index, self.kb_external_rows, "external")

                hits.sort(key=lambda x: x[0], reverse=True)
                return [h[1] for h in hits[:top_k]]
        except Exception as e:
            print(f"[RAG] Vector search failed, fallback to keyword: {e}")

        # Fallback: simple keyword scoring
        def _score(text: str, terms: List[str]) -> int:
            s = 0
            for t in terms:
                s += text.lower().count(t)
            return s

        terms = [w for w in re.findall(r"[\w-]+", query.lower()) if len(w) >= self.keyword_min_len]
        all_rows = []
        if self.rag_sources_internal:
            all_rows.extend([("internal", r) for r in self.kb_internal_rows])
        if self.rag_sources_external:
            all_rows.extend([("external", r) for r in self.kb_external_rows])
        scored = []
        for label, r in all_rows:
            sc = _score(r.get("text", ""), terms)
            if sc > 0:
                scored.append((sc, {"score": float(sc), "source": label, **r}))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[:top_k]]

    def _format_rag_context(self, hits: List[Dict[str, Any]]) -> str:
        if not hits:
            return ""
        blocks = []
        for i, h in enumerate(hits, 1):
            title = h.get("title", "")
            url = h.get("url", "")
            module = h.get("module", "")
            ver = h.get("version", "")
            chunk_idx = h.get("chunk_index", 0)
            text = h.get("text", "")
            preview = text.strip()
            blocks.append(
                f"[#{i}] title: {title} | module: {module} | ver: {ver} | chunk: {chunk_idx} | url: {url}\n{text}"
            )
        return "\n\n".join(blocks)

    def _hide_freecad_panels(self):
        mw = Gui.getMainWindow()
        if not mw:
            return
        targets = CONFIG.get("hidden_panels", [
            "Model",        # Model/Tasksを含む
            "Python console",
            "Report view",
            "Tasks",
        ])
        for dock in mw.findChildren(QtWidgets.QDockWidget):
            title = dock.windowTitle()
            name = dock.objectName()
            if any(t.lower() == title.lower() or t.lower() in title.lower() or t.lower() == name.lower() for t in targets):
                dock.hide()

    def on_query(self):
        prompt = self.query_edit.toPlainText().strip()
        if not prompt:
            return
        print(f"[Query] {prompt}")
        # RAG 検索（可能なら）
        rag_hits = self._rag_search(prompt, top_k=self.rag_top_k if hasattr(self, 'rag_top_k') else 6)
        rag_ctx = self._format_rag_context(rag_hits)
        # 画像を含めたユーザーメッセージを構築
        image_paths = []
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            path = item.data(QtCore.Qt.UserRole)
            if path:
                image_paths.append(path)

        user_msg = None
        if image_paths and self._model_supports_vision(self.provider, self.model_name):
            content_parts = [{"type": "text", "text": prompt}]
            for p in image_paths:
                try:
                    data_url = self._image_file_to_data_url(p)
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": data_url}
                    })
                except Exception as ie:
                    print(f"[ImageSkip] {p}: {ie}")
            user_msg = {"role": "user", "content": content_parts}
        else:
            if image_paths:
                QtWidgets.QMessageBox.information(
                    self,
                    "画像は無視されます",
                    "現在のプロバイダ/モデルでは画像入力に未対応のため、テキストのみ送信します。"
                )
            user_msg = {"role": "user", "content": prompt}

        # RAG コンテキストを先頭に差し込む（あれば）
        if rag_ctx:
            preamble = getattr(self, 'rag_assistant_preamble', '')
            preamble = preamble.strip() if isinstance(preamble, str) else ''
            assist_ctx = f"{preamble}\n\n{rag_ctx}" if preamble else rag_ctx
            # 直前に system 情報として加えるより、assistantの補助発話として添付
            self.messages.append({"role": "assistant", "content": assist_ctx})

        self.messages.append(user_msg)
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name, messages=self.messages
            )
            text = resp.choices[0].message.content.strip()
            print(f"[Response] {text}")
            self.messages.append({"role": "assistant", "content": text})
            self.response_edit.setPlainText(text)

            # コード転送＆生成試行
            code = self.extract_code(text)
            self.code_edit.setPlainText(code)
            self.error_count = 0
            self.perform_generate_with_retry(code)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "エラー", f"{type(e).__name__}: {e}")

    def extract_code(self, text):
        matches = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
        return "\n".join(matches) if matches else text

    def perform_generate_with_retry(self, code):
        try:
            self._generate_model(code)
            # 成功時はエラーカウントリセット
            self.error_count = 0
        except Exception as e:
            if not getattr(self, 'retry_enabled', True):
                raise
            self.error_count += 1
            limit = getattr(self, 'retry_max_attempts', 3)
            if self.error_count < limit:
                # エラー発生を問い合わせ
                err_msg = f"モデル生成中にエラーが発生しました: {type(e).__name__}: {e}"
                print(f"[Error] {err_msg}")
                self.messages.append({"role": "user", "content": err_msg})
                resp = self.client.chat.completions.create(
                    model=self.model_name, messages=self.messages
                )
                fix_code = resp.choices[0].message.content.strip()
                print(f"[FixCode] {fix_code}")
                self.messages.append({"role": "assistant", "content": fix_code})
                # 修正コードを再試行
                code_to_try = self.extract_code(fix_code)
                self.code_edit.setPlainText(code_to_try)
                self.perform_generate_with_retry(code_to_try)
            else:
                # 3回失敗時は通知
                QtWidgets.QMessageBox.information(
                    self, "連続エラー", f"モデル生成が{limit}回連続で失敗しました。手動でご確認ください。"
                )

    def _model_supports_vision(self, provider, model):
        # 代表的な視覚モデルのみ許可
        supported = set(CONFIG.get("vision", {}).get("supported_models", ["gpt-4o", "gpt-4o-mini"]))
        if provider == "ChatGPT" and model in supported:
            return True
        return False

    def _image_file_to_data_url(self, path):
        mt, _ = mimetypes.guess_type(path)
        if not mt or not mt.startswith("image/"):
            # 既知拡張子でなければPNGとして送る
            mt = "image/png"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mt};base64,{b64}"

    def on_add_image(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "画像ファイルを選択",
            "",
            CONFIG.get("images", {}).get("file_filter", "画像ファイル (*.png *.jpg *.jpeg *.gif *.webp *.bmp);;すべてのファイル (*)"),
        )
        if not files:
            return
        # 重複を避けて追加
        existing = set()
        for i in range(self.image_list.count()):
            p = self.image_list.item(i).data(QtCore.Qt.UserRole)
            if p:
                existing.add(p)
        for path in files:
            if path in existing:
                continue
            item = QtWidgets.QListWidgetItem(os.path.basename(path))
            item.setToolTip(path)
            item.setData(QtCore.Qt.UserRole, path)
            self.image_list.addItem(item)
        # 初回追加時など、選択がなければ先頭を選択
        if self.image_list.selectedItems() == [] and self.image_list.count() > 0:
            self.image_list.setCurrentRow(0)
        self._update_image_preview()

    def on_remove_image(self):
        for item in list(self.image_list.selectedItems()):
            self.image_list.takeItem(self.image_list.row(item))
        # 選択の調整とプレビュー更新
        if self.image_list.count() > 0 and not self.image_list.selectedItems():
            self.image_list.setCurrentRow(0)
        self._update_image_preview()

    def on_image_selection_changed(self):
        self._update_image_preview()

    def _update_image_preview(self):
        # 先頭の選択アイテムをプレビュー
        sel = self.image_list.selectedItems()
        path = None
        if sel:
            path = sel[0].data(QtCore.Qt.UserRole)
        if not path:
            self.image_preview.setText("画像プレビューなし")
            self.image_preview.setPixmap(QPixmap())
            return
        pm = QPixmap(path)
        if pm.isNull():
            self.image_preview.setText("画像を読み込めません")
            self.image_preview.setPixmap(QPixmap())
            return
        # ラベルに収まるようにスケーリング
        target_size = self.image_preview.size() - QtCore.QSize(6, 6)
        if target_size.width() <= 0 or target_size.height() <= 0:
            self.image_preview.setPixmap(pm)
        else:
            spm = pm.scaled(target_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.image_preview.setPixmap(spm)

    def _generate_model(self, code):
        # FreeCADオブジェクトクリア
        for obj in list(self.doc.Objects):
            self.doc.removeObject(obj.Name)
        namespace = {"App":App, "Gui":Gui, "doc":self.doc}
        exec(code, namespace)
        fn = namespace.get("create")
        if not callable(fn):
            raise RuntimeError("create(doc) が定義されていません")
        fn(self.doc)
        self.doc.recompute()
        v = Gui.ActiveDocument.ActiveView
        if getattr(self, 'view_isometric', True):
            v.viewIsometric()
        if getattr(self, 'view_fit_all', True):
            v.fitAll()

    def on_generate(self):
        code = self.code_edit.toPlainText()
        self.error_count = 0
        try:
            self.perform_generate_with_retry(code)
        except Exception as e:
            # ここは通常通らない
            QtWidgets.QMessageBox.critical(self, "致命的エラー", str(e))

    def _cleanup_freecad(self):
        try: App.closeDocument(self.doc.Name)
        except: pass
        try: Gui.getMainWindow().close()
        except: pass

    def closeEvent(self, event):
        self._cleanup_freecad()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = FreeCADEmbedApp()
    win.show()
    sys.exit(app.exec_())
