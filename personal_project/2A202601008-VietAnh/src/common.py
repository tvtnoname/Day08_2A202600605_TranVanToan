"""
Common helpers dùng chung cho Day08 RAG pipeline.

Gồm:
    - Cấu hình từ .env (Jina, Weaviate).
    - jina_embed / jina_embed_query: gọi Jina Embeddings API (giống lab07).
    - get_weaviate_client: kết nối Weaviate Cloud.
    - COLLECTION_NAME: tên collection dùng xuyên suốt các task.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# --- Jina ---
JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_EMBEDDING_MODEL = os.getenv("JINA_EMBEDDING_MODEL", "jina-embeddings-v5-text-small")
JINA_RERANKER_MODEL = os.getenv("JINA_RERANKER_MODEL", "jina-reranker-v2-base-multilingual")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
JINA_EMBED_URL = "https://api.jina.ai/v1/embeddings"

# --- Weaviate ---
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")
COLLECTION_NAME = "DrugLawDocs"


def _jina_post(payload: dict) -> dict:
    """POST tới Jina Embeddings API bằng urllib (không cần thêm dependency)."""
    request = urllib.request.Request(
        JINA_EMBED_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {JINA_API_KEY}",
            # Cloudflare (error 1010) chặn User-Agent mặc định của urllib.
            "User-Agent": "curl/8.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def jina_embed(texts: list[str], task: str = "retrieval.passage", batch_size: int = 64) -> list[list[float]]:
    """
    Embed danh sách văn bản qua Jina API. Tự chia batch để tránh giới hạn request.

    task: "retrieval.passage" cho document, "retrieval.query" cho câu truy vấn
          (asymmetric embedding — tăng chất lượng retrieval).
    """
    if not JINA_API_KEY:
        raise RuntimeError("Thiếu JINA_API_KEY trong .env")
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        data = _jina_post({
            "model": JINA_EMBEDDING_MODEL,
            "input": batch,
            "task": task,
            "normalized": True,
        })
        out.extend([d["embedding"] for d in data["data"]])
    return out


def jina_embed_query(text: str) -> list[float]:
    """Embed 1 câu query (task=retrieval.query)."""
    return jina_embed([text], task="retrieval.query")[0]


def get_weaviate_client():
    """Kết nối Weaviate Cloud (đọc URL + API key từ .env)."""
    import weaviate
    from weaviate.classes.init import Auth

    if not WEAVIATE_URL or not WEAVIATE_API_KEY:
        raise RuntimeError("Thiếu WEAVIATE_URL / WEAVIATE_API_KEY trong .env")
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
    )
