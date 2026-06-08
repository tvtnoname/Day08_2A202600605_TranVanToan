"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


# NumPy 2.0 compatibility monkey-patch for ChromaDB
import numpy as np
if not hasattr(np, "float_"):
    np.float_ = np.float64


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
    import chromadb
    from sentence_transformers import SentenceTransformer
    from pathlib import Path

    project_dir = Path(__file__).parent.parent
    chroma_path = project_dir / "data" / "chromadb"
    
    client = chromadb.PersistentClient(path=str(chroma_path))
    
    try:
        collection = client.get_collection(name="drug_law_docs")
    except Exception:
        # Nếu chưa index, trả về danh sách rỗng thay vì báo lỗi
        return []
        
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = model.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    if not results or not results["documents"] or not results["documents"][0]:
        return []
        
    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    
    formatted_results = []
    for doc, dist, meta in zip(documents, distances, metadatas):
        # Cosine distance -> Cosine similarity
        score = 1.0 - float(dist)
        formatted_results.append({
            "content": doc,
            "score": score,
            "metadata": meta
        })
        
    formatted_results.sort(key=lambda x: x["score"], reverse=True)
    return formatted_results


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
