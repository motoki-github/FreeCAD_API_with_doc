#!/usr/bin/env python3
"""
FreeCAD KB Builder (Markdown -> chunks -> embeddings -> FAISS)

Features:
- Crawl/ingest FreeCAD Wiki/API pages (MediaWiki)
- Clean DOM (remove nav/toc/footers)
- Convert HTML -> Markdown (via html2text)
- Chunk with metadata (URL, title, module, version)
- Embed (sentence-transformers) and index with FAISS
- Simple CLI to add/update pages and query

Usage examples:
  # 1) Build KB from a list of URLs (supports .txt or Markdown .md)
  python kb_builder.py build --urls external_kb.md --version 0.20 --out ./kb_out

  # 2) Query the KB
  python kb_builder.py query --index ./kb_out/index.faiss --store ./kb_out/chunks.jsonl --q "Part Box makeBox"

Notes:
- Internet access is required to fetch the pages when building.
- For strictly offline builds, download HTML beforehand and point --html-dir.
"""
import argparse
import concurrent.futures as futures
import dataclasses
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

# Optional heavy deps are imported lazily where needed
import requests
from bs4 import BeautifulSoup

try:
    import html2text  # lightweight HTML->Markdown
except Exception:
    html2text = None

# Embeddings / Index
try:
    import faiss  # type: ignore
except Exception:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None


# ------------------------------
# Utilities
# ------------------------------

SAFE_CHUNK_CHARS = 4800  # ~1200 tokens (approx 4 chars/token)
OVERLAP_CHARS = 400      # ~100 tokens overlap

MEDIAWIKI_STRIP_IDS = {
    "mw-navigation", "mw-panel", "footer", "siteSub", "mw-head-base", "toc"
}
MEDIAWIKI_STRIP_CLASSES = {
    "vector-menu", "navbox", "catlinks", "printfooter", "metadata"
}


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def hash_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def guess_module_from_url(url: str) -> str:
    # Example: https://wiki.freecad.org/Part_Box -> module "Part"
    m = re.search(r"/([A-Za-z]+)_", url)
    return m.group(1) if m else "Unknown"


def clean_mediawiki_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Remove typical MediaWiki chrome
    for id_ in MEDIAWIKI_STRIP_IDS:
        el = soup.find(id=id_)
        if el:
            el.decompose()
    for cls in MEDIAWIKI_STRIP_CLASSES:
        for el in soup.select(f".{cls}"):
            el.decompose()

    # Keep main content if present
    content = soup.find(id="content") or soup
    return str(content)


def html_to_markdown(html: str) -> str:
    if html2text is None:
        # Fallback: very simple text extraction if html2text not available
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text("\n", strip=True)
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = True
    h.body_width = 0
    return h.handle(html)


def extract_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # MediaWiki pages often have an H1 with id="firstHeading"
    h1 = soup.find(id="firstHeading") or soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    # fallback: <title>
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return "Untitled"


def chunk_text(text: str, chunk_chars: int = SAFE_CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> List[str]:
    if len(text) <= chunk_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap  # overlap
        if start < 0:
            start = 0
    return chunks


@dataclass
class KBChunk:
    id: str
    url: str
    title: str
    module: str
    version: str
    chunk_index: int
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# ------------------------------
# Build pipeline
# ------------------------------

def fetch_url(url: str, timeout: int = 20) -> str:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "FreeCAD-KB/1.0"})
    r.raise_for_status()
    return r.text


def process_page(url: str, version: str) -> List[KBChunk]:
    raw = fetch_url(url)
    title = extract_title(raw)
    module = guess_module_from_url(url)
    cleaned = clean_mediawiki_html(raw)
    md = html_to_markdown(cleaned)

    # Normalize whitespace a bit
    md = re.sub(r"\n{3,}", "\n\n", md).strip()

    chunks = []
    for i, ch in enumerate(chunk_text(md)):
        cid = hash_id(f"{url}::{i}")
        chunks.append(KBChunk(
            id=cid,
            url=url,
            title=title,
            module=module,
            version=version,
            chunk_index=i,
            text=ch,
        ))
    return chunks


def load_urls(urls_path: Path) -> List[str]:
    """
    Load URLs from a text or Markdown file.
    - Supports one-URL-per-line .txt
    - Supports Markdown bullets or links in .md (e.g., "- https://..." or "[Title](https://...)" )
    """
    urls: List[str] = []
    seen = set()
    content = urls_path.read_text(encoding="utf-8")
    for line in content.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        # Extract Markdown links: [text](https://example)
        for u in re.findall(r"\((https?://[^)\s]+)\)", s):
            if u not in seen:
                urls.append(u); seen.add(u)
        # Extract bare URLs (optionally after a bullet)
        m = re.search(r"https?://[^)\s]+", s)
        if m:
            u = m.group(0)
            if u not in seen:
                urls.append(u); seen.add(u)
    return urls


def save_chunks(chunks: List[KBChunk], out_jsonl: Path):
    with out_jsonl.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch.to_dict(), ensure_ascii=False) + "\n")
    log(f"Saved chunks: {out_jsonl} ({len(chunks)} records)")


# ------------------------------
# Embeddings & FAISS
# ------------------------------

def embed_texts(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed. Please `pip install sentence-transformers`.")
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return embs


def build_faiss(embs):
    if faiss is None:
        raise RuntimeError("faiss-cpu is not installed. Please `pip install faiss-cpu`.")
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine if normalized
    index.add(embs)
    return index


def save_faiss(index, path: Path):
    if faiss is None:
        raise RuntimeError("faiss-cpu is not installed.")
    faiss.write_index(index, str(path))


def load_faiss(path: Path):
    if faiss is None:
        raise RuntimeError("faiss-cpu is not installed.")
    return faiss.read_index(str(path))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


# ------------------------------
# CLI Commands
# ------------------------------

def cmd_build(args):
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # input source exclusivity
    specified = [bool(args.urls), bool(args.html_dir), bool(args.text)]
    if sum(specified) != 1:
        raise SystemExit("Specify exactly one of --urls, --html-dir, or --text")

    chunks: List[KBChunk] = []

    if args.urls:
        urls = load_urls(Path(args.urls))
        log(f"Loaded {len(urls)} URLs")
        with futures.ThreadPoolExecutor(max_workers=min(16, len(urls) or 1)) as ex:
            futs = [ex.submit(process_page, u, args.version) for u in urls]
            for fu in futures.as_completed(futs):
                try:
                    chunks.extend(fu.result())
                except Exception as e:
                    log(f"Error processing page: {e}")
    elif args.html_dir:
        # Offline mode: process local HTML files
        html_dir = Path(args.html_dir)
        html_files = list(html_dir.glob("*.html")) + list(html_dir.glob("*.htm"))
        log(f"Loaded {len(html_files)} local HTML files")
        for fp in html_files:
            raw = fp.read_text(encoding="utf-8", errors="ignore")
            # fabricate URL-like id
            url = f"file://{fp.name}"
            title = extract_title(raw) or fp.stem
            module = guess_module_from_url(fp.stem)
            cleaned = clean_mediawiki_html(raw)
            md = html_to_markdown(cleaned)
            md = re.sub(r"\n{3,}", "\n\n", md).strip()
            for i, ch in enumerate(chunk_text(md)):
                cid = hash_id(f"{url}::{i}")
                chunks.append(KBChunk(
                    id=cid, url=url, title=title, module=module,
                    version=args.version, chunk_index=i, text=ch
                ))
    else:
        # Plain text file mode (e.g., internal knowledge base)
        text_path = Path(args.text)
        if not text_path.exists():
            raise SystemExit(f"Text file not found: {text_path}")
        raw = text_path.read_text(encoding="utf-8", errors="ignore").strip()
        # Title from first Markdown H1 if present, else filename
        m = re.search(r"^#\s+(.+)$", raw, re.M)
        title = m.group(1).strip() if m else text_path.stem
        url = f"file://{text_path.name}"
        module = "Internal"
        md = raw
        md = re.sub(r"\n{3,}", "\n\n", md).strip()
        for i, ch in enumerate(chunk_text(md)):
            cid = hash_id(f"{url}::{i}")
            chunks.append(KBChunk(
                id=cid, url=url, title=title, module=module,
                version=args.version, chunk_index=i, text=ch
            ))

    # Decide file prefix by source
    if args.urls:
        file_prefix = "external_"
    elif args.text:
        file_prefix = "internal_"
    else:
        file_prefix = ""

    # Save chunks jsonl
    store_path = out_dir / f"{file_prefix}chunks.jsonl"
    save_chunks(chunks, store_path)

    # Build embeddings + faiss
    texts = [c.text for c in chunks]
    embs = embed_texts(texts, model_name=args.embed_model)
    index = build_faiss(embs)
    index_path = out_dir / f"{file_prefix}index.faiss"
    save_faiss(index, index_path)

    # Map index -> metadata
    meta_path = out_dir / f"{file_prefix}meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in chunks], f, ensure_ascii=False, indent=2)

    log(f"Done. Index: {index_path}, Store: {store_path}, Meta: {meta_path}")


def cmd_query(args):
    index_path = Path(args.index).resolve()
    store_path = Path(args.store).resolve()
    meta_path = Path(args.meta).resolve() if args.meta else None

    index = load_faiss(index_path)

    rows = read_jsonl(store_path)
    texts = [r["text"] for r in rows]

    # embed query
    q_emb = embed_texts([args.q], model_name=args.embed_model)[0]

    # search
    k = args.k
    import numpy as np
    D, I = index.search(np.array([q_emb]), k)
    hits = []
    for score, idx in zip(D[0], I[0]):
        rec = rows[idx]
        item = {
            "score": float(score),
            "id": rec["id"],
            "url": rec["url"],
            "title": rec.get("title", ""),
            "module": rec.get("module", ""),
            "version": rec.get("version", ""),
            "chunk_index": rec.get("chunk_index", 0),
            "preview": rec["text"][:400].replace("\n", " ") + ("..." if len(rec["text"]) > 400 else ""),
        }
        hits.append(item)

    print(json.dumps(hits, ensure_ascii=False, indent=2))
    if meta_path and meta_path.exists():
        print("\n# meta path:", meta_path)


def main():
    p = argparse.ArgumentParser(description="FreeCAD Knowledge Base Builder")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="Build KB and FAISS index")
    pb.add_argument("--urls", type=str, help="Path to a text file containing URLs (one per line)")
    pb.add_argument("--html-dir", type=str, help="Directory of local HTML files (offline mode)")
    pb.add_argument("--text", type=str, help="Path to a plain text/Markdown file to ingest for RAG")
    pb.add_argument("--version", type=str, default="0.20", help="FreeCAD version label to attach")
    pb.add_argument("--embed-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    pb.add_argument("--out", type=str, default="./kb_out", help="Output directory")
    pb.set_defaults(func=cmd_build)

    pq = sub.add_parser("query", help="Query an existing FAISS index")
    pq.add_argument("--index", type=str, required=True, help="Path to index.faiss")
    pq.add_argument("--store", type=str, required=True, help="Path to chunks.jsonl")
    pq.add_argument("--meta", type=str, help="Optional path to meta.json")
    pq.add_argument("--embed-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    pq.add_argument("--q", type=str, required=True, help="Query text")
    pq.add_argument("--k", type=int, default=5, help="Top-k results")
    pq.set_defaults(func=cmd_query)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
