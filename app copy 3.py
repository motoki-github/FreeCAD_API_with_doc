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
from PySide2 import QtWidgets, QtCore
from PySide2.QtGui import QPixmap
import FreeCAD as App
import FreeCADGui as Gui
from config_loader import CONFIG
from llm_client import get_openai_client
from dialogs import ProviderModelDialog
from rag_helper import RagHelper
from image_utils import image_file_to_data_url


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
        self.client = get_openai_client(self.provider, CONFIG)

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

        # RAG helper + view settings
        self.rag = RagHelper(CONFIG)
        self.rag_top_k = self.rag.top_k
        self.rag_assistant_preamble = self.rag.assistant_preamble
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
        # プロンプトファイル読込UI
        file_row = QtWidgets.QHBoxLayout()
        self.load_prompt_btn = QtWidgets.QPushButton("ファイルから読込")
        self.prompt_file_label = QtWidgets.QLabel("未選択")
        self.prompt_file_label.setStyleSheet("color: #555;")
        self.prompt_file_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        file_row.addWidget(self.load_prompt_btn)
        file_row.addWidget(self.prompt_file_label, 1)
        left_layout.addLayout(file_row)
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
        self.load_prompt_btn.clicked.connect(self.on_load_prompt_file)

        QtWidgets.QApplication.instance().aboutToQuit.connect(self._cleanup_freecad)

    # RAG 関連は rag_helper に分離

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
        rag_hits = self.rag.search(prompt, top_k=self.rag_top_k)
        rag_ctx = RagHelper.format_context(rag_hits)
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
                    data_url = image_file_to_data_url(p)
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
        supported = set(CONFIG.get("vision", {}).get("supported_models", ["gpt-4o", "gpt-4o-mini"]))
        return bool(provider == "ChatGPT" and model in supported)

    # 画像のDataURL変換は image_utils.image_file_to_data_url を使用

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

    def on_load_prompt_file(self):
        # テキスト/Markdown/JSON/YAML を対象に読込
        filt = (
            CONFIG.get("prompt_file_filter",
                       "テキスト (*.txt *.md *.markdown *.json *.yaml *.yml);;すべてのファイル (*)")
        )
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "プロンプトファイルを選択",
            "",
            filt,
        )
        if not path:
            return
        try:
            # まずはそのままテキストとして読み込み
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            # JSON 形式で messages を含む場合の簡易対応
            # 例: [{"role":"user","content":"..."}, ...]
            prompt_text = None
            try:
                data = json.loads(content)
                if isinstance(data, list) and all(
                    isinstance(m, dict) and "role" in m and "content" in m for m in data
                ):
                    # user メッセージの content を連結してテキスト化
                    user_parts = [str(m.get("content", "")) for m in data if m.get("role") == "user"]
                    prompt_text = "\n\n".join(p.strip() for p in user_parts if p)
            except Exception:
                pass
            if prompt_text is None:
                prompt_text = content

            self.query_edit.setPlainText(prompt_text.strip())
            self.prompt_file_label.setText(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "読込エラー", f"{type(e).__name__}: {e}")

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
