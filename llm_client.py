#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Dict, Any
from openai import OpenAI
from PySide2 import QtWidgets


def get_openai_client(provider: str, config: Dict[str, Any]) -> OpenAI:
    env_map = config.get("secrets", {}).get("env_keys", {})
    api_env = env_map.get(provider, "OPENAI_API_KEY" if provider == "ChatGPT" else "DEEPSEEK_API_KEY")
    base_url_map = config.get("api_base_urls", {})
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
            raise SystemExit(1)
        api_key = key.strip()

    return OpenAI(api_key=api_key, base_url=base_url)

