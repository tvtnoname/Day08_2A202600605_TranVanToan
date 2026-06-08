"""
Task 8 — PageIndex Vectorless RAG (https://pageindex.ai).

PageIndex KHÔNG dùng vector/embedding. Nó build một "cây" cấu trúc tài liệu (mục lục,
section) bằng LLM, rồi khi truy vấn sẽ cho LLM "đi" trên cây để chọn node liên quan
(reasoning-based retrieval). Phù hợp tài liệu có cấu trúc rõ như văn bản luật.

Luồng:
    1. submit_document(pdf) -> doc_id   (build tree, async)
    2. is_retrieval_ready(doc_id)       (chờ xử lý xong)
    3. submit_query(doc_id, query) -> retrieval_id  (async)
    4. get_retrieval(retrieval_id) -> retrieved_nodes (poll tới status=completed)

Dùng làm FALLBACK ở Task 9 khi hybrid search không đủ tốt.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
DOC_IDS_FILE = Path(__file__).parent.parent / "data" / "pageindex_docs.json"


def _client():
    from pageindex import PageIndexClient
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("Thiếu PAGEINDEX_API_KEY trong .env")
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def upload_documents() -> dict:
    """
    Upload các PDF luật lên PageIndex và lưu mapping {filename: doc_id} vào file.
    Idempotent: bỏ qua file đã upload.
    """
    client = _client()
    mapping = {}
    if DOC_IDS_FILE.exists():
        mapping = json.loads(DOC_IDS_FILE.read_text(encoding="utf-8"))

    for pdf in sorted(LEGAL_DIR.glob("*.pdf")):
        if pdf.name in mapping:
            print(f"  • Đã có: {pdf.name} -> {mapping[pdf.name]}")
            continue
        print(f"  Uploading: {pdf.name}")
        try:
            resp = client.submit_document(str(pdf))
            mapping[pdf.name] = resp["doc_id"]
            print(f"    ✓ doc_id={resp['doc_id']}")
        except Exception as e:
            print(f"    ✗ Lỗi: {e}")

    DOC_IDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    DOC_IDS_FILE.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    # Chờ tất cả docs sẵn sàng.
    for name, doc_id in mapping.items():
        for _ in range(40):
            try:
                if client.is_retrieval_ready(doc_id):
                    break
            except Exception:
                pass
            time.sleep(5)
    print(f"✓ {len(mapping)} documents sẵn sàng trên PageIndex")
    return mapping


def _query_one(client, doc_id: str, query: str, timeout: int = 90) -> list[dict]:
    """submit_query + poll get_retrieval cho 1 doc, trả về retrieved_nodes."""
    rid = client.submit_query(doc_id, query)["retrieval_id"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get_retrieval(rid)
        if r.get("status") == "completed":
            return r.get("retrieved_nodes", [])
        if r.get("status") == "failed":
            return []
        time.sleep(4)
    return []


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval qua PageIndex (fallback).

    Returns:
        List of {'content', 'score', 'metadata', 'source': 'pageindex'}.
    """
    if not DOC_IDS_FILE.exists():
        return []
    mapping = json.loads(DOC_IDS_FILE.read_text(encoding="utf-8"))
    if not mapping:
        return []

    client = _client()
    items = []
    for name, doc_id in mapping.items():
        try:
            nodes = _query_one(client, doc_id, query)
        except Exception:
            nodes = []
        for node in nodes:
            md = node.get("metadata", [])
            source = md[1] if len(md) > 1 and md[1] else name
            title = node.get("title", "")
            for group in node.get("relevant_contents", []):
                for piece in group:
                    content = piece.get("relevant_content", "").strip()
                    if not content:
                        continue
                    items.append({
                        "content": content,
                        "metadata": {"source": source, "type": "legal", "section": title},
                        "source": "pageindex",
                    })

    # PageIndex không cho score số → gán score giảm dần theo thứ hạng.
    for rank, it in enumerate(items):
        it["score"] = round(1.0 - rank * 0.05, 4)
    return items[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong .env (đăng ký tại https://pageindex.ai/)")
    else:
        print("Uploading documents...")
        upload_documents()
        print("\nTest query:")
        for r in pageindex_search("Hình phạt tội tàng trữ trái phép chất ma túy?", top_k=3):
            print(f"[{r['score']}] ({r['metadata']['section'][:40]}) {r['content'][:90]}...")
