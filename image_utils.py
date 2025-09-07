#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64
import mimetypes


def image_file_to_data_url(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    if not mt or not mt.startswith("image/"):
        mt = "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mt};base64,{b64}"

