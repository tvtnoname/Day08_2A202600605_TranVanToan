"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(Path(__file__).parent.parent / ".env")

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# RecursiveCharacterTextSplitter giữ câu đoạn tương đối tự nhiên,
# phù hợp khi data đã gồm legal text dài và news article ngắn; 500 ký tự
# đủ nhỏ để truy hồi chính xác nhưng vẫn giữ ngữ cảnh, overlap 80 để giảm
# đứt ý giữa các chunk liên tiếp.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# API embeddings qua Cloudflare Gateway.
CLOUDFLARE_EMBEDDING_MODEL = "bge-m3"
EMBEDDING_MODEL = CLOUDFLARE_EMBEDDING_MODEL
EMBEDDING_DIM = 1024

# Weaviate local (Docker) hỗ trợ lưu vector và hybrid search sau này.
VECTOR_STORE = "weaviate"  # "weaviate" | "chromadb" | "faiss"
WEAVIATE_COLLECTION = "DrugLawDocs"
CF_GATEWAY_URL = os.getenv("CF_GATEWAY_URL", "http://127.0.0.1:8787")


def _md5_digest(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _as_float_list(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values]


def _extract_embedding_data(response_data: dict) -> list:
    if "result" in response_data and isinstance(response_data["result"], dict) and "data" in response_data["result"]:
        return response_data["result"]["data"]
    if "data" in response_data:
        return response_data["data"]
    raise ValueError(f"Could not find embedding data in response: {response_data}")


def _normalize_embedding_response(response_data: dict, expected_count: int) -> list[list[float]]:
    raw_data = _extract_embedding_data(response_data)

    if not isinstance(raw_data, list) or not raw_data:
        raise ValueError(f"Invalid embedding shape: {raw_data}")

    if expected_count == 1:
        if isinstance(raw_data[0], list):
            return [_as_float_list(raw_data[0])]
        return [_as_float_list(raw_data)]

    if isinstance(raw_data[0], list):
        vectors = [_as_float_list(row) for row in raw_data]
        if len(vectors) != expected_count:
            raise ValueError(
                f"Expected {expected_count} embeddings, received {len(vectors)}"
            )
        return vectors

    raise ValueError(
        "Expected a list of embeddings for batch input, "
        f"but received a single vector: {raw_data}"
    )


class CloudflareEmbedder:
    """Cloudflare Workers AI Gateway-backed embedder with local disk caching."""

    def __init__(
        self,
        model_name: str = CLOUDFLARE_EMBEDDING_MODEL,
        gateway_url: str | None = None,
        cache_path: str | None = None,
        request_timeout: float = 30.0,
        max_workers: int = 10,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self._backend_name = f"cloudflare ({model_name})"
        self.cf_gateway_url = (gateway_url or CF_GATEWAY_URL).strip()
        self.cache_path = cache_path or os.getenv("EMBEDDINGS_CACHE_PATH", "embeddings_cache.json")
        self.request_timeout = request_timeout
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.cache: dict[str, list[float]] = {}
        self._load_cache()

    def _cache_key(self, text: str) -> str:
        return f"{self.model_name}:{_md5_digest(text)}"

    def _load_cache(self) -> None:
        if not os.path.exists(self.cache_path):
            return

        try:
            with open(self.cache_path, "r", encoding="utf-8") as file:
                raw_cache = json.load(file)
        except Exception as error:
            print(f"Failed to load embeddings cache: {error}")
            return

        if isinstance(raw_cache, dict):
            self.cache = {
                str(key): _as_float_list(value)
                for key, value in raw_cache.items()
                if isinstance(value, list)
            }

    def _save_cache(self) -> None:
        try:
            with open(self.cache_path, "w", encoding="utf-8") as file:
                json.dump(self.cache, file, ensure_ascii=False)
        except Exception as error:
            print(f"Failed to save embeddings cache: {error}")

    def _post_embeddings(self, payload_text: str | list[str]) -> list[list[float]]:
        import requests

        payload = {
            "model": self.model_name,
            "text": payload_text,
        }
        response = requests.post(self.cf_gateway_url, json=payload, timeout=self.request_timeout)
        response.raise_for_status()
        expected_count = 1 if isinstance(payload_text, str) else len(payload_text)
        return _normalize_embedding_response(response.json(), expected_count=expected_count)

    def __call__(self, text: str) -> list[float]:
        cache_key = self._cache_key(text)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        vectors = self._post_embeddings(text)
        if not vectors:
            raise ValueError("Cloudflare embedding API returned no vectors")

        vector = vectors[0]
        self.cache[cache_key] = vector
        self._save_cache()
        return vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        missing_entries: list[tuple[int, str]] = []

        for index, text in enumerate(texts):
            cached = self.cache.get(self._cache_key(text))
            if cached is not None:
                results[index] = cached
            else:
                missing_entries.append((index, text))

        if not missing_entries:
            return [result for result in results if result is not None]

        batches: list[list[tuple[int, str]]] = [
            missing_entries[start : start + self.batch_size]
            for start in range(0, len(missing_entries), self.batch_size)
        ]

        def fetch_batch(batch_entries: list[tuple[int, str]]) -> list[tuple[int, list[float]]]:
            batch_texts = [text for _, text in batch_entries]
            try:
                vectors = self._post_embeddings(batch_texts)
                if len(vectors) != len(batch_entries):
                    raise ValueError(
                        f"Expected {len(batch_entries)} embeddings, received {len(vectors)}"
                    )
                return [
                    (index, vector)
                    for (index, _), vector in zip(batch_entries, vectors)
                ]
            except Exception as error:
                print(f"Cloudflare batch embedding failed: {error}")
                raise

        batch_results: list[list[tuple[int, list[float]]]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(fetch_batch, batch) for batch in batches]
            for future in as_completed(futures):
                batch_results.append(future.result())

        has_new_values = False
        for batch in batch_results:
            for index, vector in batch:
                text = texts[index]
                cache_key = self._cache_key(text)
                if cache_key not in self.cache:
                    self.cache[cache_key] = vector
                    has_new_values = True
                results[index] = vector

        if has_new_values:
            self._save_cache()
        return [result for result in results if result is not None]


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents: list[dict[str, Any]] = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if not md_file.is_file():
            continue

        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in md_file.parts else "news"
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "path": str(md_file.relative_to(STANDARDIZED_DIR)),
                    "type": doc_type,
                },
            }
        )
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise ImportError(
            "langchain-text-splitters is required for Task 4 chunking"
        ) from exc

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict[str, Any]] = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for chunk_index, chunk_text in enumerate(splits):
            cleaned = chunk_text.strip()
            if not cleaned:
                continue
            chunks.append(
                {
                    "content": cleaned,
                    "metadata": {**doc["metadata"], "chunk_index": chunk_index},
                }
            )
    return chunks


def _hash_embedding(text: str) -> list[float]:
    """Fallback embedding để pipeline vẫn chạy nếu model transformer chưa sẵn sàng."""
    vector = [0.0] * EMBEDDING_DIM
    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIM
        vector[index] += 1.0

    norm = sum(value * value for value in vector) ** 0.5
    if norm:
        vector = [value / norm for value in vector]
    return vector


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    texts = [chunk["content"] for chunk in chunks]
    try:
        embedder = CloudflareEmbedder()
        print(f"  Embedding backend: cloudflare ({EMBEDDING_MODEL})")
        embeddings = embedder.embed_batch(texts)
    except Exception as error:
        raise RuntimeError(f"Cloudflare embedding failed: {error}") from error

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    if VECTOR_STORE != "weaviate":
        raise NotImplementedError("Task 4 hiện được cấu hình cho Weaviate local")

    try:
        import weaviate
        from weaviate.classes.config import Configure, DataType, Property
    except ImportError as exc:
        raise ImportError("weaviate-client is required for Task 4 indexing") from exc

    try:
        client = weaviate.connect_to_local(skip_init_checks=True)
    except Exception:
        _index_to_weaviate_rest(chunks)
        return

    try:
        try:
            client.collections.delete(WEAVIATE_COLLECTION)
        except Exception:
            pass

        client.collections.create(
            name=WEAVIATE_COLLECTION,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="path", data_type=DataType.TEXT),
                Property(name="doc_type", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
            ],
        )

        collection = client.collections.get(WEAVIATE_COLLECTION)
        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                metadata = chunk["metadata"]
                batch.add_object(
                    properties={
                        "content": chunk["content"],
                        "source": metadata.get("source", ""),
                        "path": metadata.get("path", ""),
                        "doc_type": metadata.get("type", ""),
                        "chunk_index": int(metadata.get("chunk_index", 0)),
                    },
                    vector=chunk["embedding"],
                )
    except Exception:
        _index_to_weaviate_rest(chunks)
    finally:
        client.close()


def _index_to_weaviate_rest(chunks: list[dict]) -> None:
    """Fallback REST indexing for older local Weaviate Docker versions."""
    import requests

    base_url = "http://localhost:8080/v1"
    class_name = WEAVIATE_COLLECTION

    schema_response = requests.get(f"{base_url}/schema", timeout=30)
    schema_response.raise_for_status()
    existing_classes = schema_response.json().get("classes", [])
    if any(cls.get("class") == class_name for cls in existing_classes if isinstance(cls, dict)):
        requests.delete(f"{base_url}/schema/{class_name}", timeout=30)

    schema_payload = {
        "class": class_name,
        "vectorizer": "none",
        "properties": [
            {"name": "content", "dataType": ["text"]},
            {"name": "source", "dataType": ["text"]},
            {"name": "path", "dataType": ["text"]},
            {"name": "doc_type", "dataType": ["text"]},
            {"name": "chunk_index", "dataType": ["int"]},
        ],
    }
    create_response = requests.post(f"{base_url}/schema", json=schema_payload, timeout=30)
    create_response.raise_for_status()

    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        objects = []
        for chunk in batch:
            metadata = chunk["metadata"]
            objects.append(
                {
                    "class": class_name,
                    "id": str(uuid4()),
                    "properties": {
                        "content": chunk["content"],
                        "source": metadata.get("source", ""),
                        "path": metadata.get("path", ""),
                        "doc_type": metadata.get("type", ""),
                        "chunk_index": int(metadata.get("chunk_index", 0)),
                    },
                    "vector": chunk["embedding"],
                }
            )

        batch_response = requests.post(
            f"{base_url}/batch/objects",
            json={"objects": objects},
            timeout=120,
        )
        batch_response.raise_for_status()


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
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
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
