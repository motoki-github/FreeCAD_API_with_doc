#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Dict, Any


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
            "sources": {"internal": True, "external": True},
        },
        "view": {"isometric": True, "fit_all": True},
        "logging": {"level": "INFO"},
        "secrets": {"env_keys": {"ChatGPT": "OPENAI_API_KEY", "DeepSeek": "DEEPSEEK_API_KEY"}},
    }
    try:
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as f:
                user_cfg = json.load(f)

            def merge(a, b):
                for k, v in b.items():
                    if isinstance(v, dict) and isinstance(a.get(k), dict):
                        a[k] = merge(a[k], v)
                    else:
                        a[k] = v
                return a

            return merge(defaults, user_cfg)
    except Exception as e:
        print(f"[Config] Failed to load config.json: {e}")
    return defaults


CONFIG: Dict[str, Any] = load_config()

