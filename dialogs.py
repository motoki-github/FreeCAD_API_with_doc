#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PySide2 import QtWidgets, QtCore


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

