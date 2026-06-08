"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

from __future__ import annotations

import re
from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Load corpus từ data/standardized/
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}
_BM25_INDEX = None


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    return [token for token in tokens if token.strip()]


def _load_corpus() -> list[dict]:
    corpus: list[dict] = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if not md_file.is_file():
            continue

        corpus.append(
            {
                "content": md_file.read_text(encoding="utf-8"),
                "metadata": {
                    "source": md_file.name,
                    "path": str(md_file.relative_to(STANDARDIZED_DIR)),
                    "type": "legal" if "legal" in md_file.parts else "news",
                },
            }
        )
    return corpus


def _ensure_corpus_loaded() -> list[dict]:
    global CORPUS
    if not CORPUS:
        CORPUS = _load_corpus()
    return CORPUS


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        class BM25Okapi:  # type: ignore[no-redef]
            def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
                self.tokenized_corpus = tokenized_corpus
                self.k1 = k1
                self.b = b
                self.doc_lengths = [len(doc) for doc in tokenized_corpus]
                self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
                self.doc_freqs: dict[str, int] = {}
                for doc in tokenized_corpus:
                    for token in set(doc):
                        self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

            def get_scores(self, query_tokens: list[str]) -> list[float]:
                total_docs = len(self.tokenized_corpus) or 1
                scores: list[float] = []
                for doc, doc_len in zip(self.tokenized_corpus, self.doc_lengths):
                    score = 0.0
                    for token in query_tokens:
                        tf = doc.count(token)
                        if tf == 0:
                            continue
                        df = self.doc_freqs.get(token, 0)
                        idf = max(0.0, ((total_docs - df + 0.5) / (df + 0.5)))
                        denom = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl if self.avgdl else 0.0))
                        score += idf * (tf * (self.k1 + 1)) / denom
                    scores.append(score)
                return scores

    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    corpus = _ensure_corpus_loaded()
    if not corpus or not query.strip():
        return []

    global _BM25_INDEX
    if _BM25_INDEX is None:
        _BM25_INDEX = build_bm25_index(corpus)

    tokenized_query = _tokenize(query)
    if not tokenized_query:
        return []

    scores = _BM25_INDEX.get_scores(tokenized_query)
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]

    results: list[dict] = []
    for index, score in ranked:
        if score <= 0:
            continue
        results.append(
            {
                "content": corpus[index]["content"],
                "score": float(score),
                "metadata": corpus[index]["metadata"],
            }
        )

    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
