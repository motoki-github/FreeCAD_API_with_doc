#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Optional heavy deps
try:
    import faiss  # type: ignore
except Exception:
    faiss = None
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None


class RagHelper:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        rb = config.get("rag", {}).get("kb_base_path")
        if rb:
            p = Path(rb)
            if not p.is_absolute():
                p = Path(__file__).resolve().parent / p
        else:
            p = Path(__file__).resolve().parent / "Knowledge_Base" / "kb_out"
        self.base: Path = p
        self.loaded = False

        self.rows_internal: List[Dict[str, Any]] = []
        self.rows_external: List[Dict[str, Any]] = []
        self.index_internal = None
        self.index_external = None
        self.embedder = None

        # knobs
        rcfg = config.get("rag", {})
        self.top_k = int(rcfg.get("top_k", 6))
        self.embed_model = rcfg.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.assistant_preamble = rcfg.get("assistant_preamble", "")
        self.use_internal = bool(rcfg.get("sources", {}).get("internal", True))
        self.use_external = bool(rcfg.get("sources", {}).get("external", True))
        self.keyword_min_len = int(rcfg.get("keyword_fallback", {}).get("min_token_len", 3))

        self._init()

    def _read_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    def _init(self):
        try:
            if not self.base.exists():
                print(f"[RAG] KB base not found: {self.base}")
                return

            internal_chunks = self.base / "internal_chunks.jsonl"
            external_chunks = self.base / "external_chunks.jsonl"
            internal_index = self.base / "internal_index.faiss"
            external_index = self.base / "external_index.faiss"

            if internal_chunks.exists():
                self.rows_internal = self._read_jsonl(internal_chunks)
            if external_chunks.exists():
                self.rows_external = self._read_jsonl(external_chunks)

            if faiss is not None and SentenceTransformer is not None:
                if internal_index.exists() and self.rows_internal:
                    self.index_internal = faiss.read_index(str(internal_index))
                if external_index.exists() and self.rows_external:
                    self.index_external = faiss.read_index(str(external_index))
                try:
                    self.embedder = SentenceTransformer(self.embed_model)
                except Exception as e:
                    print(f"[RAG] Failed to init embedder: {e}")

            total = len(self.rows_internal) + len(self.rows_external)
            if total:
                self.loaded = True
                print(f"[RAG] KB loaded: {total} chunks (internal={len(self.rows_internal)}, external={len(self.rows_external)})")
            else:
                print("[RAG] No KB chunks found.")
        except Exception as e:
            print(f"[RAG] Init failed: {e}")

    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        if not self.loaded:
            return []
        k = int(top_k or self.top_k)

        hits: List[Tuple[float, Dict[str, Any]]] = []
        try:
            import numpy as np  # lazy
            if self.embedder is not None and (self.index_internal is not None or self.index_external is not None):
                q_emb = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

                def _search(index, rows, label: str):
                    if index is None or not rows:
                        return
                    if label == "internal" and not self.use_internal:
                        return
                    if label == "external" and not self.use_external:
                        return
                    D, I = index.search(np.array([q_emb]), min(k, len(rows)))
                    for score, idx in zip(D[0], I[0]):
                        r = rows[int(idx)]
                        rec = {"score": float(score), "source": label, **r}
                        hits.append((float(score), rec))

                _search(self.index_internal, self.rows_internal, "internal")
                _search(self.index_external, self.rows_external, "external")

                hits.sort(key=lambda x: x[0], reverse=True)
                return [h[1] for h in hits[:k]]
        except Exception as e:
            print(f"[RAG] Vector search failed, fallback to keyword: {e}")

        # keyword fallback
        def _score(text: str, terms: List[str]) -> int:
            s = 0
            for t in terms:
                s += text.lower().count(t)
            return s

        terms = [w for w in re.findall(r"[\w-]+", query.lower()) if len(w) >= self.keyword_min_len]
        all_rows = []
        if self.use_internal:
            all_rows.extend([("internal", r) for r in self.rows_internal])
        if self.use_external:
            all_rows.extend([("external", r) for r in self.rows_external])
        scored = []
        for label, r in all_rows:
            sc = _score(r.get("text", ""), terms)
            if sc > 0:
                scored.append((sc, {"score": float(sc), "source": label, **r}))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[:k]]

    @staticmethod
    def format_context(hits: List[Dict[str, Any]]) -> str:
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
            blocks.append(
                f"[#{i}] title: {title} | module: {module} | ver: {ver} | chunk: {chunk_idx} | url: {url}\n{text}"
            )
        return "\n\n".join(blocks)

