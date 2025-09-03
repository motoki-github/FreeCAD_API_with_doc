#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import re
from openai import OpenAI
from PySide6 import QtWidgets, QtCore
import FreeCAD as App
import FreeCADGui as Gui

# OpenAI/DeepSeekクライアント取得関数
# provider: 'DeepSeek' または 'ChatGPT'
def get_openai_client(provider):
    if provider == "DeepSeek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = "https://api.deepseek.com/v1"
        prompt_title = "DeepSeek API Key入力"
        prompt_label = "DeepSeek API Keyを入力してください:"
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = "https://api.openai.com/v1"
        prompt_title = "OpenAI API Key入力"
        prompt_label = "OpenAI API Keyを入力してください:"

    if not api_key:
        key, ok = QtWidgets.QInputDialog.getText(
            None, prompt_title, prompt_label, QtWidgets.QLineEdit.Password
        )
        if not ok or not key.strip():
            QtWidgets.QMessageBox.critical(None, "エラー", "APIキーが設定されていません。アプリを終了します。")
            sys.exit(1)
        api_key = key.strip()

    # OpenAIクライアントにbase_urlを正しく渡す
    return OpenAI(api_key=api_key, base_url=base_url)

class FreeCADEmbedApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        provider, ok = QtWidgets.QInputDialog.getItem(
            None, "サービス選択", "使用するサービスを選択してください:",
            ["DeepSeek", "ChatGPT"], 0, False
        )
        if not ok:
            sys.exit(0)
        self.provider = provider
        self.model_name = "deepseek-reasoner" if provider == "DeepSeek" else "gpt-4o-mini"
        self.client = get_openai_client(self.provider)

        system_msg = (
            "あなたは FreeCAD+PySide2 アプリのエキスパートです。"
            "・出力は必ず コードの説明　からはじまり、その後　python のコードブロックで終わること"
            "・関数シグネチャは def create(doc): のみ"
            "・実行前に既存オブジェクトをすべて削除する処理を組み込む"
            "・PEP8 準拠、余計な説明テキストは一切含めない"
            "・最後に return doc を追加すること"
        )
        self.messages = [{"role": "system", "content": system_msg}]

        self.setWindowTitle(f"FreeCAD Embedded App ({self.provider})")
        self.resize(1400, 800)

        # エラーリトライカウンタ
        self.error_count = 0

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
        self.response_edit = QtWidgets.QPlainTextEdit()
        self.response_edit.setReadOnly(True)
        left_layout.addWidget(self.response_edit, 1)
        h_layout.addWidget(left, 1)

        # 中央ペイン: コード
        center = QtWidgets.QWidget()
        center_layout = QtWidgets.QVBoxLayout(center)
        center_layout.setContentsMargins(0,0,0,0)
        self.code_edit = QtWidgets.QPlainTextEdit()
        self.code_edit.setPlainText(
            "# FreeCAD 用 Python コード\n"
            "def create(doc):\n"
            "    box = doc.addObject('Part::Box','Box')\n"
            "    box.Length = 10\n"
            "    box.Width = 20\n"
            "    box.Height = 30\n"
        )
        center_layout.addWidget(self.code_edit)
        self.model_btn = QtWidgets.QPushButton("モデル生成")
        center_layout.addWidget(self.model_btn)
        h_layout.addWidget(center, 2)

        # 右ペイン: FreeCADビュー
        Gui.showMainWindow()
        self.doc = App.newDocument("EmbeddedDoc")
        fc_widget = Gui.getMainWindow()
        h_layout.addWidget(fc_widget, 3)

        # シグナル
        self.query_btn.clicked.connect(self.on_query)
        self.model_btn.clicked.connect(self.on_generate)

        QtWidgets.QApplication.instance().aboutToQuit.connect(self._cleanup_freecad)

    def on_query(self):
        prompt = self.query_edit.toPlainText().strip()
        if not prompt:
            return
        print(f"[Query] {prompt}")
        self.messages.append({"role": "user", "content": prompt})
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
            self.error_count += 1
            if self.error_count < 3:
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
                    self, "連続エラー", "モデル生成が3回連続で失敗しました。手動でご確認ください。"
                )

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
        v.viewIsometric(); v.fitAll()

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
