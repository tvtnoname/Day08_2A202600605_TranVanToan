"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

from __future__ import annotations

import re
import os
import tempfile
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(Path(__file__).parent.parent / ".env")

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
_UPLOADED_DOC_IDS: list[str] = []


def _load_documents() -> list[dict]:
    documents: list[dict] = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if not md_file.is_file():
            continue

        documents.append(
            {
                "path": md_file,
                "content": md_file.read_text(encoding="utf-8"),
                "metadata": {
                    "filename": md_file.name,
                    "path": str(md_file.relative_to(STANDARDIZED_DIR)),
                    "type": "legal" if "legal" in md_file.parts else "news",
                },
            }
        )
    return documents


def _simple_score(query: str, content: str) -> float:
    query_tokens = set(re.findall(r"\w+", query.lower(), flags=re.UNICODE))
    content_tokens = re.findall(r"\w+", content.lower(), flags=re.UNICODE)
    if not query_tokens or not content_tokens:
        return 0.0

    overlap = sum(1 for token in content_tokens if token in query_tokens)
    return overlap / max(len(content_tokens), 1)


def _fallback_search(query: str, top_k: int) -> list[dict]:
    documents = _load_documents()
    scored = []
    for document in documents:
        score = _simple_score(query, document["content"])
        if score > 0:
            scored.append(
                {
                    "content": document["content"],
                    "score": float(score),
                    "metadata": document["metadata"],
                    "source": "pageindex",
                }
            )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def _get_client():
    from pageindex import PageIndexClient

    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _parse_retrieval_payload(payload: dict) -> list[dict]:
    candidates: list[dict] = []

    if not isinstance(payload, dict):
        return candidates

    possible_lists = []
    for key in ("results", "data", "chunks", "retrievals", "nodes"):
        value = payload.get(key)
        if isinstance(value, list):
            possible_lists.append(value)

    for values in possible_lists:
        for item in values:
            if not isinstance(item, dict):
                continue
            text = item.get("text") or item.get("content") or item.get("snippet") or item.get("answer")
            if not text:
                continue

            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            score = item.get("score") or item.get("relevance_score") or item.get("rank_score") or 0.0
            candidates.append(
                {
                    "content": text,
                    "score": float(score),
                    "metadata": metadata,
                    "source": "pageindex",
                }
            )

    if candidates:
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates

    text = payload.get("text") or payload.get("content") or payload.get("answer")
    if isinstance(text, str) and text.strip():
        return [
            {
                "content": text.strip(),
                "score": float(payload.get("score", payload.get("relevance_score", 0.0))),
                "metadata": payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {},
                "source": "pageindex",
            }
        ]

    return candidates


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    global _UPLOADED_DOC_IDS

    if not PAGEINDEX_API_KEY:
        return []

    try:
        client = _get_client()
    except Exception:
        return []

    uploaded_doc_ids: list[str] = []
    documents = _load_documents()

    for document in documents:
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as temp_file:
            temp_file.write(document["content"])
            temp_path = temp_file.name

        try:
            response = client.submit_document(temp_path)
            doc_id = response.get("doc_id")
            if doc_id:
                uploaded_doc_ids.append(doc_id)
        except Exception:
            continue
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    _UPLOADED_DOC_IDS = uploaded_doc_ids
    return uploaded_doc_ids


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if not query.strip():
        return []

    if not PAGEINDEX_API_KEY:
        return _fallback_search(query, top_k)

    if not _UPLOADED_DOC_IDS:
        uploaded_doc_ids = upload_documents()
    else:
        uploaded_doc_ids = _UPLOADED_DOC_IDS

    try:
        client = _get_client()
    except Exception:
        return _fallback_search(query, top_k)

    results: list[dict] = []
    try:
        for doc_id in uploaded_doc_ids or [None]:
            if not doc_id:
                continue

            try:
                retrieval = client.submit_query(doc_id=doc_id, query=query)
                retrieval_id = retrieval.get("retrieval_id")
                if not retrieval_id:
                    continue

                payload = client.get_retrieval(retrieval_id)
                parsed_results = _parse_retrieval_payload(payload)
                results.extend(parsed_results)
            except Exception:
                continue
    except Exception:
        return _fallback_search(query, top_k)

    if not results:
        return _fallback_search(query, top_k)

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
