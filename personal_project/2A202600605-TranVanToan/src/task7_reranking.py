"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.
"""

from typing import Optional


_cross_encoder_model = None

def get_cross_encoder():
    global _cross_encoder_model
    if _cross_encoder_model is None:
        try:
            from sentence_transformers import CrossEncoder
            # Model cực nhẹ (~80MB), chất lượng tốt, chạy cục bộ nhanh
            _cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            print(f"Warning: Failed to load CrossEncoder ({e}). Reranking will fallback to cosine similarity.")
    return _cross_encoder_model


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    if not candidates:
        return []
        
    model = get_cross_encoder()
    if model is not None:
        try:
            pairs = [[query, c["content"]] for c in candidates]
            scores = model.predict(pairs)
            res = []
            for c, score in zip(candidates, scores):
                item = c.copy()
                item["score"] = float(score)
                res.append(item)
            res.sort(key=lambda x: x["score"], reverse=True)
            return res[:top_k]
        except Exception as e:
            print(f"Reranking error: {e}. Falling back to cosine similarity scoring.")
            
    # Fallback: tính độ tương đồng cosine thông qua all-MiniLM-L6-v2
    try:
        from sentence_transformers import SentenceTransformer
        import math
        
        transformer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        q_emb = transformer.encode(query)
        doc_embs = transformer.encode([c["content"] for c in candidates])
        
        def cosine_sim(v1, v2):
            dot = sum(a*b for a, b in zip(v1, v2))
            n1 = math.sqrt(sum(a*a for a in v1))
            n2 = math.sqrt(sum(b*b for b in v2))
            return dot / (n1 * n2) if n1 > 0 and n2 > 0 else 0.0
            
        res = []
        for c, emb in zip(candidates, doc_embs):
            item = c.copy()
            item["score"] = float(cosine_sim(q_emb, emb))
            res.append(item)
        res.sort(key=lambda x: x["score"], reverse=True)
        return res[:top_k]
    except Exception as e:
        print(f"Fallback scoring error: {e}. Returning original candidates sorted by score.")
        res = [c.copy() for c in candidates]
        res.sort(key=lambda x: x.get("score", 0), reverse=True)
        return res[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    if not candidates:
        return []
        
    import math
    def cosine_sim(vec1: list[float], vec2: list[float]) -> float:
        dot_prod = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_prod / (norm1 * norm2)

    # Tính toán embedding bị thiếu (nếu có)
    missing_embs = [i for i, c in enumerate(candidates) if "embedding" not in c]
    if missing_embs:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            texts_to_embed = [candidates[i]["content"] for i in missing_embs]
            embs = model.encode(texts_to_embed)
            for i, emb in zip(missing_embs, embs):
                candidates[i]["embedding"] = emb.tolist()
        except Exception as e:
            print(f"Error computing missing embeddings for MMR: {e}")
            return [c.copy() for c in candidates[:top_k]]

    selected = []
    remaining = list(range(len(candidates)))
    
    # Tính toán trước độ tương đồng với query
    query_similarities = {}
    for idx in remaining:
        query_similarities[idx] = cosine_sim(query_embedding, candidates[idx]["embedding"])

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float('-inf')
        
        for idx in remaining:
            relevance = query_similarities[idx]
            
            if not selected:
                mmr_score = relevance
            else:
                max_sim_to_selected = max(
                    cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
                    for sel_idx in selected
                )
                mmr_score = lambda_param * relevance - (1.0 - lambda_param) * max_sim_to_selected
                
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
                
        if best_idx is None:
            break
            
        selected.append(best_idx)
        remaining.remove(best_idx)
        
    res = []
    for i in selected:
        item = candidates[i].copy()
        item["score"] = query_similarities[i]
        res.append(item)
    return res


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores = {}  # content -> score
    content_map = {}  # content -> full dict
    
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (k + rank)
            if key not in content_map:
                content_map[key] = item.copy()
            else:
                if "metadata" in item:
                    content_map[key]["metadata"].update(item.get("metadata", {}))
                    
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = score
        results.append(item)
        
    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        query_embedding = model.encode(query).tolist()
        return rerank_mmr(query_embedding, candidates, top_k)
    elif method == "rrf":
        if candidates and isinstance(candidates[0], list):
            return rerank_rrf(candidates, top_k)
        else:
            return rerank_rrf([candidates], top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
