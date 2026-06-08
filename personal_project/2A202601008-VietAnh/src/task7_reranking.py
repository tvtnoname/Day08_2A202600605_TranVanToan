"""
Task 7 — Reranking Module.

Chính: Cross-encoder reranker qua Jina Reranker API (jina-reranker-v2-base-multilingual).
    - Bi-encoder (Task 5) embed query & doc RIÊNG → nhanh nhưng thô.
    - Cross-encoder ghép (query, doc) thành 1 input, đọc cùng lúc → chấm liên quan
      chính xác hơn nhiều, dùng để lọc lại top 3–5 trước khi đưa vào LLM.

Bổ sung (tự implement, dùng ở Task 9): RRF gộp nhiều ranked list, và MMR chống trùng lặp.
"""

import json
import urllib.request

from .common import JINA_API_KEY, JINA_RERANKER_MODEL

JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Rerank candidates bằng Jina cross-encoder API.

    Returns:
        top_k candidates, re-scored (score = relevance_score) và sort giảm dần.
    """
    if not candidates:
        return []
    if not JINA_API_KEY:
        # Fallback an toàn: giữ thứ tự score cũ.
        return sorted(candidates, key=lambda c: c.get("score", 0), reverse=True)[:top_k]

    documents = [c["content"] for c in candidates]
    payload = {
        "model": JINA_RERANKER_MODEL,
        "query": query,
        "documents": documents,
        "top_n": top_k,
    }
    request = urllib.request.Request(
        JINA_RERANK_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {JINA_API_KEY}",
            "User-Agent": "curl/8.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    results = []
    for r in data["results"]:
        idx = r["index"]
        item = dict(candidates[idx])
        item["score"] = float(r["relevance_score"])
        results.append(item)
    return results


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5 or 1.0
    nb = sum(y * y for y in b) ** 0.5 or 1.0
    return num / (na * nb)


def rerank_mmr(query_embedding, candidates: list[dict], top_k: int = 5, lambda_param: float = 0.7) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.
    MMR = λ·sim(query, doc) - (1-λ)·max(sim(doc, selected)).
    Yêu cầu mỗi candidate có key 'embedding'.
    """
    selected, remaining = [], list(range(len(candidates)))
    while remaining and len(selected) < top_k:
        best_idx, best_score = None, float("-inf")
        for idx in remaining:
            emb = candidates[idx]["embedding"]
            relevance = _cosine(query_embedding, emb)
            max_sim = max((_cosine(emb, candidates[s]["embedding"]) for s in selected), default=0.0)
            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
            if mmr > best_score:
                best_score, best_idx = mmr, idx
        selected.append(best_idx)
        remaining.remove(best_idx)
    return [candidates[i] for i in selected]


def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp nhiều ranked list bằng thứ hạng (bỏ qua scale điểm).
    RRF(d) = Σ 1/(k + rank_r(d)). Dùng để merge dense + sparse ở Task 9.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item
    ordered = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for content, score in ordered[:top_k]:
        item = dict(content_map[content])
        item["score"] = score
        results.append(item)
    return results


def rerank(query: str, candidates: list[dict], top_k: int = 5, method: str = "cross_encoder") -> list[dict]:
    """Unified reranking interface (mặc định cross-encoder Jina)."""
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hướng dẫn nấu phở bò Hà Nội", "score": 0.6, "metadata": {}},
    ]
    for r in rerank("hình phạt tàng trữ ma tuý", dummy, top_k=2):
        print(f"[{r['score']:.3f}] {r['content']}")
