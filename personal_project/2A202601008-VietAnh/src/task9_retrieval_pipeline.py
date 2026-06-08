"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

    Query
      ├→ Semantic Search (Task 5) ─┐
      │                            ├→ RRF merge (source=hybrid) → Rerank (Task 7)
      ├→ Lexical/BM25 (Task 6) ────┘                                   │
      │                                                                ▼
      └→ Nếu best_score < threshold → Fallback PageIndex (Task 8, vectorless)
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search

# =============================================================================
# CONFIGURATION
# =============================================================================
SCORE_THRESHOLD = 0.3          # best score (sau rerank) < ngưỡng → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback.

    Returns:
        List of {'content', 'score', 'metadata', 'source' ∈ {'hybrid','pageindex'}}.
    """
    # Step 1: dense + sparse (lấy rộng top_k*2 mỗi nguồn).
    try:
        dense = semantic_search(query, top_k=top_k * 2)
    except Exception as e:
        print(f"  ⚠ semantic_search lỗi: {e}")
        dense = []
    try:
        sparse = lexical_search(query, top_k=top_k * 2)
    except Exception as e:
        print(f"  ⚠ lexical_search lỗi: {e}")
        sparse = []

    # Step 2: merge bằng RRF (gộp theo thứ hạng, không cần cùng scale điểm).
    merged = rerank_rrf([dense, sparse], top_k=top_k * 2)
    for item in merged:
        item["source"] = "hybrid"

    # Step 3: rerank cross-encoder để chấm lại độ liên quan thật.
    if use_reranking and merged:
        final = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        for item in final:
            item["source"] = "hybrid"
    else:
        final = merged[:top_k]

    # Step 4: kiểm tra ngưỡng → fallback PageIndex.
    best = final[0]["score"] if final else 0.0
    if not final or best < score_threshold:
        print(f"  ⚠ Hybrid best score = {best:.3f} < threshold {score_threshold} → fallback PageIndex")
        try:
            fallback = pageindex_search(query, top_k=top_k)
        except Exception as e:
            print(f"  ⚠ PageIndex fallback lỗi: {e}")
            fallback = []
        if fallback:
            return fallback[:top_k]

    return final[:top_k]


if __name__ == "__main__":
    for q in [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý",
    ]:
        print(f"\nQuery: {q}\n" + "-" * 60)
        for i, r in enumerate(retrieve(q, top_k=3), 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:75]}...")
