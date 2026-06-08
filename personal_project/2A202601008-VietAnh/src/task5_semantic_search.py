"""
Task 5 — Semantic Search Module (dense retrieval).

Embed query bằng Jina (task=retrieval.query) rồi near_vector trên Weaviate.
Score = 1 - cosine_distance (cao hơn = liên quan hơn), sort giảm dần.
"""

from .common import COLLECTION_NAME, get_weaviate_client, jina_embed_query


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa bằng vector similarity.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}, sorted desc.
    """
    from weaviate.classes.query import MetadataQuery

    query_vec = jina_embed_query(query)

    client = get_weaviate_client()
    try:
        collection = client.collections.get(COLLECTION_NAME)
        res = collection.query.near_vector(
            near_vector=query_vec,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )
        results = []
        for obj in res.objects:
            p = obj.properties
            dist = obj.metadata.distance
            score = 1.0 - dist if dist is not None else 0.0
            results.append({
                "content": p.get("content", ""),
                "score": float(score),
                "metadata": {
                    "source": p.get("source", ""),
                    "type": p.get("doc_type", ""),
                    "chunk_index": p.get("chunk_index", 0),
                },
            })
    finally:
        client.close()

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


if __name__ == "__main__":
    for r in semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5):
        print(f"[{r['score']:.3f}] ({r['metadata']['type']}) {r['content'][:90]}...")
