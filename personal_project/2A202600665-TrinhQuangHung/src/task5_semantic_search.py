"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
        from dotenv import load_dotenv
except ImportError:
        load_dotenv = None

if load_dotenv is not None:
        load_dotenv(Path(__file__).parent.parent / ".env")

from src.task4_chunking_indexing import (
        CF_GATEWAY_URL,
        CLOUDFLARE_EMBEDDING_MODEL,
        CloudflareEmbedder,
        WEAVIATE_COLLECTION,
)


def _embed_query(query: str) -> list[float]:
        embedder = CloudflareEmbedder(model_name=CLOUDFLARE_EMBEDDING_MODEL, gateway_url=CF_GATEWAY_URL)
        return embedder(query)


def _query_weaviate(query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        import requests

        graphql_url = "http://localhost:8080/v1/graphql"
        query_payload = {
                "query": f"""
                {{
                    Get {{
                        {WEAVIATE_COLLECTION}(
                            nearVector: {{vector: {query_embedding!r}}}
                            limit: {top_k}
                        ) {{
                            content
                            source
                            path
                            doc_type
                            chunk_index
                            _additional {{
                                distance
                            }}
                        }}
                    }}
                }}
                """
        }

        response = requests.post(graphql_url, json=query_payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("data", {}).get("Get", {}).get(WEAVIATE_COLLECTION, []) or []


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    if not query.strip():
        return []

    try:
        query_embedding = _embed_query(query)
    except Exception:
        return []

    try:
        raw_results = _query_weaviate(query_embedding, top_k=top_k)
    except Exception:
        return []

    results: list[dict[str, Any]] = []
    for item in raw_results:
        additional = item.get("_additional", {}) if isinstance(item, dict) else {}
        distance = additional.get("distance")
        score = 1.0 - float(distance) if distance is not None else 0.0
        results.append(
            {
                "content": item.get("content", ""),
                "score": score,
                "metadata": {
                    "source": item.get("source", ""),
                    "path": item.get("path", ""),
                    "doc_type": item.get("doc_type", ""),
                    "chunk_index": item.get("chunk_index", 0),
                },
            }
        )

    results.sort(key=lambda result: result["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
