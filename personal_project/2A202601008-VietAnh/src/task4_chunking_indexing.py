"""
Task 4 — Chunking & Indexing vào Vector Store (Weaviate Cloud).

Pipeline: load markdown → chunk → embed (Jina) → index (Weaviate).

LỰA CHỌN & LÝ DO
================
Chunking — RecursiveCharacterTextSplitter (size=500, overlap=50):
    - size=500 ký tự: đủ chứa trọn 1 "Điều" luật ngắn hoặc 1 đoạn báo, mà vẫn nhỏ để
      embedding bắt đúng ngữ nghĩa cục bộ (chunk quá to → vector bị "loãng").
    - overlap=50 (10%): giữ ngữ cảnh ở ranh giới, tránh cắt ngang ý/câu.
    - Recursive: tách theo \\n\\n → \\n → ". " → " " → ký tự, ưu tiên ranh giới tự nhiên;
      an toàn cho cả văn bản luật (có heading "Điều") lẫn bài báo.
    - Đo bằng số ký tự (len) nên dễ kiểm soát kích thước (khớp test ≤ size*1.1).

Embedding — Jina jina-embeddings-v5-text-small (1024-dim, API):
    - Đa ngữ, mạnh tiếng Việt; dùng asymmetric embedding (passage vs query).
    - Đồng bộ với lab07 và với reranker Jina (Task 7) — cùng 1 API key.
    - Không cần tải model nặng về máy.

Vector store — Weaviate Cloud:
    - Hỗ trợ hybrid search built-in (dense near_vector + sparse BM25) → phục vụ Task 5/6/9.
    - vectorizer=none: ta tự cung cấp vector từ Jina (bring-your-own-vector).
"""

from pathlib import Path

from .common import (
    COLLECTION_NAME,
    EMBEDDING_DIM,
    JINA_EMBEDDING_MODEL,
    get_weaviate_client,
    jina_embed,
)

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# =============================================================================
# CONFIGURATION
# =============================================================================
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = JINA_EMBEDDING_MODEL
# EMBEDDING_DIM imported from common (1024)

VECTOR_STORE = "weaviate"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        if not content.strip():
            continue
        doc_type = "legal" if "legal" in md_file.parts else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type},
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents bằng RecursiveCharacterTextSplitter.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []
    for doc in documents:
        for i, chunk_text in enumerate(splitter.split_text(doc["content"])):
            if not chunk_text.strip():
                continue
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Embed toàn bộ chunks bằng Jina; thêm key 'embedding' vào mỗi chunk."""
    texts = [c["content"] for c in chunks]
    embeddings = jina_embed(texts, task="retrieval.passage")
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """Tạo collection (nếu chưa có) và insert chunks + vector vào Weaviate."""
    from weaviate.classes.config import Configure, DataType, Property

    client = get_weaviate_client()
    try:
        # Reset collection để index lại sạch.
        if client.collections.exists(COLLECTION_NAME):
            client.collections.delete(COLLECTION_NAME)

        client.collections.create(
            name=COLLECTION_NAME,
            # Ta tự cung cấp vector (Jina) → self_provided (server tự chọn index, vd hfresh).
            vector_config=Configure.Vectors.self_provided(),
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="doc_type", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
            ],
        )
        collection = client.collections.get(COLLECTION_NAME)

        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                m = chunk["metadata"]
                batch.add_object(
                    properties={
                        "content": chunk["content"],
                        "source": m.get("source", ""),
                        "doc_type": m.get("type", ""),
                        "chunk_index": int(m.get("chunk_index", 0)),
                    },
                    vector=chunk["embedding"],
                )

        failed = collection.batch.failed_objects
        if failed:
            print(f"  ⚠ {len(failed)} object insert lỗi. Ví dụ: {failed[0]}")
        count = collection.aggregate.over_all(total_count=True).total_count
        print(f"  ✓ Collection '{COLLECTION_NAME}' hiện có {count} objects")
    finally:
        client.close()


def run_pipeline():
    """Chạy toàn bộ: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks (dim={len(chunks[0]['embedding']) if chunks else 0})")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
