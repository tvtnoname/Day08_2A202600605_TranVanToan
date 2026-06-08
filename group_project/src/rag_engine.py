from __future__ import annotations

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Đảm bảo đã import config để load sys.path
from . import config

# Import các module từ personal_project thông qua personal_src
from personal_src.task5_semantic_search import semantic_search
from personal_src.task6_lexical_search import lexical_search
from personal_src.task7_reranking import rerank as personal_rerank, rerank_rrf
from personal_src.task8_pageindex_vectorless import pageindex_search
from personal_src.task10_generation import reorder_for_llm, format_context, SYSTEM_PROMPT

load_dotenv()

def condense_question(query: str, chat_history: list[dict]) -> str:
    """
    Sử dụng LLM để viết lại câu hỏi mới nhất kết hợp với lịch sử trò chuyện
    thành một câu hỏi độc lập (standalone question) để tìm kiếm chính xác hơn.
    """
    if not chat_history:
        return query
        
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-xxx"):
        return query # Fallback về câu hỏi gốc nếu không có API Key
        
    client = OpenAI(api_key=api_key)
    
    # Format lịch sử trò chuyện thành chuỗi văn bản
    history_str = ""
    for msg in chat_history[-5:]: # Chỉ lấy tối đa 5 tin nhắn gần nhất để tránh quá tải
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "")
        history_str += f"{role}: {content}\n"
        
    prompt = f"""Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question in Vietnamese. The standalone question should contain all necessary context from the history so it can be used for search/retrieval. Do not add any conversational filler, just return the standalone question.

Chat History:
{history_str}
Follow-up Question: {query}

Standalone Question:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150
        )
        standalone_query = response.choices[0].message.content.strip()
        print(f"Condensed Query: '{query}' -> '{standalone_query}'")
        return standalone_query
    except Exception as e:
        print(f"Error condensing question: {e}")
        return query

def retrieve_dynamic(
    query: str,
    top_k: int = config.DEFAULT_TOP_K,
    score_threshold: float = config.DEFAULT_SCORE_THRESHOLD,
    rerank_method: str = config.DEFAULT_RERANK_METHOD
) -> list[dict]:
    """
    Tải tài liệu động từ dense + sparse search, gộp kết quả qua RRF,
    Rerank theo phương pháp được chỉ định, và fallback qua PageIndex nếu điểm số dưới ngưỡng.
    """
    # 1. Semantic & Lexical search
    dense_results = semantic_search(query, top_k=top_k * 2)
    sparse_results = lexical_search(query, top_k=top_k * 2)
    
    # 2. Merge (RRF)
    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
    for item in merged:
        item["source"] = "hybrid"
        
    # 3. Rerank
    if rerank_method != "none" and merged:
        final_results = personal_rerank(query, merged, top_k=top_k, method=rerank_method)
    else:
        final_results = merged[:top_k]
        
    # 4. Fallback PageIndex
    best_score = final_results[0]["score"] if final_results else 0.0
    if not final_results or best_score < score_threshold:
        print(f"  ⚠ Hybrid score ({best_score:.3f}) < threshold ({score_threshold}). Fallback → PageIndex")
        try:
            fallback = pageindex_search(query, top_k=top_k)
            return fallback
        except Exception as e:
            print(f"  PageIndex fallback failed: {e}. Returning hybrid results.")
            
    return final_results[:top_k]

def generate_with_citation_dynamic(
    query: str,
    chat_history: list[dict] = None,
    top_k: int = config.DEFAULT_TOP_K,
    score_threshold: float = config.DEFAULT_SCORE_THRESHOLD,
    rerank_method: str = config.DEFAULT_RERANK_METHOD
) -> dict:
    """
    Hàm sinh câu trả lời RAG có Citation và hội thoại hoàn chỉnh.
    """
    # 1. Condense question
    search_query = condense_question(query, chat_history) if chat_history else query
    
    # 2. Retrieve chunks
    chunks = retrieve_dynamic(search_query, top_k=top_k, score_threshold=score_threshold, rerank_method=rerank_method)
    
    # 3. Reorder chunks tránh "Lost in the middle"
    reordered = reorder_for_llm(chunks)
    
    # 4. Format context
    context = format_context(reordered)
    
    # 5. Build prompt
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"
    
    # 6. Call OpenAI API
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-xxx"):
        raise ValueError("OPENAI_API_KEY is not set or is set to placeholder in .env")
        
    client = OpenAI(api_key=api_key)
    
    # Tạo messages bao gồm cả lịch sử trò chuyện để LLM có ngữ cảnh trò chuyện tốt hơn
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        for msg in chat_history[-5:]: # Gửi kèm lịch sử cuộc trò chuyện
            messages.append({"role": msg["role"], "content": msg["content"]})
            
    messages.append({"role": "user", "content": user_message})
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3,
        top_p=0.9,
    )
    
    answer = response.choices[0].message.content
    
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none"
    }

"""RAG Core Engine cho bài tập nhóm.

File này do Thành viên 1 phụ trách.

Chức năng chính:
- Load markdown documents từ data/standardized của các thành viên.
- Chunking tài liệu.
- Semantic search local bằng TF-IDF cosine similarity.
- Lexical search bằng BM25.
- Merge kết quả bằng Reciprocal Rank Fusion.
- Rerank kết quả.
- Sinh câu trả lời có citation.
- Trả về sources để UI và evaluation dùng chung.

Các thành viên khác chỉ cần gọi:

    from src.rag_engine import generate_with_citation, get_source_summary

"""

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_PATH,
    DEFAULT_CONFIG_NAME,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_TOP_K,
    GROUP_INDEX_DIR,
    MAX_CONTEXT_CHUNKS,
    MAX_SENTENCES_PER_ANSWER,
    RETRIEVAL_CONFIGS,
    STANDARDIZED_DIRS,
)


_TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)

_STOPWORDS = {
    "và",
    "là",
    "của",
    "có",
    "cho",
    "trong",
    "với",
    "theo",
    "được",
    "các",
    "một",
    "những",
    "này",
    "đó",
    "về",
    "tại",
    "từ",
    "khi",
    "để",
    "hoặc",
    "thì",
    "mà",
    "như",
    "đã",
    "bị",
    "các",
    "nhiều",
    "năm",
}


def tokenize(text: str) -> list[str]:
    """Tokenize tiếng Việt đơn giản cho search local."""
    tokens = [token.lower() for token in _TOKEN_RE.findall(text or "")]
    return [token for token in tokens if len(token) > 1 and token not in _STOPWORDS]


def load_markdown_documents() -> list[dict[str, Any]]:
    """
    Đọc toàn bộ file markdown trong các thư mục data/standardized.

    Returns:
        [
            {
                "content": "...",
                "metadata": {
                    "source": "...",
                    "title": "...",
                    "path": "...",
                    "doc_type": "legal/news/unknown",
                    "standardized_root": "..."
                }
            }
        ]
    """
    documents: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    for standardized_dir in STANDARDIZED_DIRS:
        if not standardized_dir.exists():
            continue

        for md_file in sorted(standardized_dir.rglob("*.md")):
            if not md_file.is_file() or md_file.name.startswith("."):
                continue

            file_key = md_file.resolve().as_posix()
            if file_key in seen_paths:
                continue

            seen_paths.add(file_key)

            try:
                content = md_file.read_text(encoding="utf-8", errors="ignore").strip()
            except OSError:
                continue

            if not content:
                continue

            try:
                rel_path = md_file.relative_to(standardized_dir).as_posix()
            except ValueError:
                rel_path = md_file.name

            if rel_path.startswith("legal/"):
                doc_type = "legal"
            elif rel_path.startswith("news/"):
                doc_type = "news"
            else:
                doc_type = "unknown"

            title = md_file.stem
            for line in content.splitlines():
                line = line.strip()
                if line:
                    title = line.strip("# ").strip()
                    break

            documents.append(
                {
                    "content": content,
                    "metadata": {
                        "source": md_file.name,
                        "title": title[:180],
                        "path": md_file.as_posix(),
                        "doc_type": doc_type,
                        "standardized_root": standardized_dir.as_posix(),
                    },
                }
            )

    return documents


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Recursive-like text splitter bằng Python thuần.

    Lý do chọn:
    - An toàn với văn bản pháp luật dài.
    - Không cần thư viện ngoài.
    - Overlap giúp giữ ngữ cảnh ở ranh giới giữa 2 chunks.
    """
    text = re.sub(r"\n{3,}", "\n\n", text or "").strip()

    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    parts: list[str] = []

    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if len(paragraph) <= chunk_size:
            parts.append(paragraph)
            continue

        sentences = re.split(r"(?<=[.!?])\s+|\n", paragraph)

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            if len(sentence) <= chunk_size:
                parts.append(sentence)
            else:
                step = max(1, chunk_size - overlap)

                for start in range(0, len(sentence), step):
                    piece = sentence[start : start + chunk_size].strip()

                    if piece:
                        parts.append(piece)

    chunks: list[str] = []
    current = ""

    for part in parts:
        candidate = f"{current}\n\n{part}".strip() if current else part

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)

        tail = current[-overlap:] if overlap > 0 else ""
        current = f"{tail}\n{part}".strip() if tail else part

        if len(current) > chunk_size:
            step = max(1, chunk_size - overlap)

            for start in range(0, len(current), step):
                piece = current[start : start + chunk_size].strip()

                if piece:
                    chunks.append(piece)

            current = ""

    if current:
        chunks.append(current)

    return chunks


def build_chunks(force_rebuild: bool = False) -> list[dict[str, Any]]:
    """
    Tạo hoặc đọc group index từ group_project/index/chunks.json.

    Đây là index dùng chung cho:
    - Streamlit chatbot.
    - Evaluation pipeline.
    - Demo nhóm.
    """
    if CHUNKS_PATH.exists() and not force_rebuild:
        try:
            return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    documents = load_markdown_documents()
    chunks: list[dict[str, Any]] = []

    for doc_index, document in enumerate(documents):
        doc_chunks = split_text(document["content"])

        for chunk_index, chunk_text in enumerate(doc_chunks):
            metadata = dict(document["metadata"])
            metadata.update(
                {
                    "doc_index": doc_index,
                    "chunk_index": chunk_index,
                    "chunk_total": len(doc_chunks),
                }
            )

            chunks.append(
                {
                    "id": f"doc{doc_index}_chunk{chunk_index}",
                    "content": chunk_text,
                    "metadata": metadata,
                }
            )

    GROUP_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return chunks


def _compute_idf(tokenized_docs: list[list[str]]) -> dict[str, float]:
    """Tính IDF cho TF-IDF local."""
    n_docs = max(1, len(tokenized_docs))
    df: Counter[str] = Counter()

    for tokens in tokenized_docs:
        df.update(set(tokens))

    return {
        term: math.log((1 + n_docs) / (1 + freq)) + 1.0
        for term, freq in df.items()
    }


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    """Tạo sparse TF-IDF vector."""
    counts = Counter(tokens)

    if not counts:
        return {}

    max_tf = max(counts.values())

    return {
        term: (freq / max_tf) * idf.get(term, 1.0)
        for term, freq in counts.items()
    }


def _cosine_similarity(
    vec_a: dict[str, float],
    vec_b: dict[str, float],
) -> float:
    """Cosine similarity giữa 2 sparse vectors."""
    if not vec_a or not vec_b:
        return 0.0

    common_terms = set(vec_a) & set(vec_b)
    dot = sum(vec_a[term] * vec_b[term] for term in common_terms)

    norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
    norm_b = math.sqrt(sum(value * value for value in vec_b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def semantic_search(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
    """
    Semantic search local bằng TF-IDF cosine similarity.

    Đây là dense-like retrieval local, dùng được khi không có API key/embedding model.
    """
    chunks = build_chunks()

    if not chunks:
        return []

    tokenized_docs = [tokenize(chunk["content"]) for chunk in chunks]
    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    idf = _compute_idf(tokenized_docs + [query_tokens])
    query_vector = _tfidf_vector(query_tokens, idf)

    results: list[dict[str, Any]] = []

    for chunk, tokens in zip(chunks, tokenized_docs):
        doc_vector = _tfidf_vector(tokens, idf)
        score = _cosine_similarity(query_vector, doc_vector)

        if score > 0:
            results.append(
                {
                    "content": chunk["content"],
                    "score": float(score),
                    "metadata": chunk["metadata"],
                    "retriever": "semantic",
                }
            )

    results.sort(key=lambda item: item["score"], reverse=True)

    return results[:top_k]


def lexical_search(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
    """
    Lexical search bằng BM25 Python thuần.

    Không cần rank-bm25 nên dễ chạy trên máy nhóm.
    """
    chunks = build_chunks()

    if not chunks:
        return []

    tokenized_docs = [tokenize(chunk["content"]) for chunk in chunks]
    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    n_docs = len(tokenized_docs)
    avgdl = sum(len(tokens) for tokens in tokenized_docs) / n_docs if n_docs else 0.0

    df: Counter[str] = Counter()

    for tokens in tokenized_docs:
        df.update(set(tokens))

    idf = {
        term: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
        for term, freq in df.items()
    }

    k1 = 1.5
    b = 0.75

    results: list[dict[str, Any]] = []

    for chunk, doc_tokens in zip(chunks, tokenized_docs):
        counts = Counter(doc_tokens)
        dl = len(doc_tokens)
        score = 0.0

        for term in query_tokens:
            tf = counts.get(term, 0)

            if tf == 0:
                continue

            denom = tf + k1 * (1 - b + b * dl / (avgdl or 1.0))
            score += idf.get(term, 0.0) * (tf * (k1 + 1)) / denom

        if score > 0:
            results.append(
                {
                    "content": chunk["content"],
                    "score": float(score),
                    "metadata": chunk["metadata"],
                    "retriever": "lexical",
                }
            )

    results.sort(key=lambda item: item["score"], reverse=True)

    return results[:top_k]


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    top_k: int = DEFAULT_TOP_K,
    k: int = 60,
) -> list[dict[str, Any]]:
    """
    Gộp kết quả từ semantic và lexical bằng Reciprocal Rank Fusion.

    Công thức:
        RRF(d) = sum(1 / (k + rank_i(d)))
    """
    scores: dict[str, float] = {}
    best_items: dict[str, dict[str, Any]] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            metadata = item.get("metadata", {})

            key = (
                f"{metadata.get('path', '')}"
                f"::{metadata.get('doc_index', '')}"
                f"::{metadata.get('chunk_index', '')}"
            )

            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

            if key not in best_items or item["score"] > best_items[key]["score"]:
                best_items[key] = dict(item)

    merged: list[dict[str, Any]] = []

    for key, rrf_score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        item = dict(best_items[key])
        item["score"] = float(rrf_score)
        item["retriever"] = "hybrid"
        merged.append(item)

        if len(merged) >= top_k:
            break

    return merged


def _overlap_score(query: str, content: str) -> float:
    """Tính độ phủ từ khóa giữa query và chunk."""
    query_terms = set(tokenize(query))
    doc_terms = set(tokenize(content))

    if not query_terms or not doc_terms:
        return 0.0

    recall = len(query_terms & doc_terms) / len(query_terms)
    precision = len(query_terms & doc_terms) / len(doc_terms)

    return 0.8 * recall + 0.2 * precision


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """
    Rerank local bằng:
    - retrieval score ban đầu
    - overlap score với query
    - domain boost cho câu hỏi pháp luật ma túy
    """
    if not candidates:
        return []

    max_score = max(float(candidate.get("score", 0)) for candidate in candidates) or 1.0

    reranked: list[dict[str, Any]] = []

    for candidate in candidates:
        item = dict(candidate)

        original_score = float(candidate.get("score", 0)) / max_score
        overlap = _overlap_score(query, candidate["content"])

        final_score = 0.55 * original_score + 0.45 * overlap

        content_lower = candidate["content"].lower()
        query_lower = query.lower()

        if "hình phạt" in query_lower and "phạt tù" in content_lower:
            final_score += 0.2

        if "mức phạt" in query_lower and "phạt" in content_lower:
            final_score += 0.15

        if "tàng trữ" in query_lower and "tàng trữ" in content_lower:
            final_score += 0.1

        if "mua bán" in query_lower and "mua bán" in content_lower:
            final_score += 0.1

        if "vận chuyển" in query_lower and "vận chuyển" in content_lower:
            final_score += 0.1

        if "cai nghiện" in query_lower and "cai nghiện" in content_lower:
            final_score += 0.1

        item["score"] = float(final_score)
        item["reranked"] = True

        reranked.append(item)

    reranked.sort(key=lambda item: item["score"], reverse=True)

    return reranked[:top_k]


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    config_name: str = DEFAULT_CONFIG_NAME,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> list[dict[str, Any]]:
    """
    Retrieval pipeline chính cho chatbot và evaluation.

    Các config:
    - hybrid_rerank
    - hybrid_no_rerank
    - dense_only
    - lexical_only
    """
    config = RETRIEVAL_CONFIGS.get(config_name, RETRIEVAL_CONFIGS[DEFAULT_CONFIG_NAME])

    candidate_lists: list[list[dict[str, Any]]] = []

    if config["use_semantic"]:
        candidate_lists.append(semantic_search(query, top_k=top_k * 2))

    if config["use_lexical"]:
        candidate_lists.append(lexical_search(query, top_k=top_k * 2))

    if not candidate_lists:
        return []

    if len(candidate_lists) == 1:
        candidates = candidate_lists[0][: top_k * 2]
    else:
        candidates = reciprocal_rank_fusion(candidate_lists, top_k=top_k * 2)

    if config["use_rerank"]:
        results = rerank(query, candidates, top_k=top_k)
    else:
        results = candidates[:top_k]

    if not results:
        return []

    if float(results[0].get("score", 0)) < score_threshold:
        fallback = lexical_search(query, top_k=top_k)

        if fallback:
            for item in fallback:
                item["retriever"] = "fallback_lexical"

            return fallback[:top_k]

    return results[:top_k]


def build_standalone_query(
    question: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> str:
    """
    Ghép history ngắn để hỗ trợ follow-up questions.

    Ví dụ:
    User hỏi: "Vậy mức phạt là bao nhiêu?"
    Hệ thống sẽ ghép với ngữ cảnh hội thoại trước đó.
    """
    if not conversation_history:
        return question

    recent_turns = conversation_history[-4:]
    history_lines: list[str] = []

    for turn in recent_turns:
        role = turn.get("role", "")
        content = turn.get("content", "")

        if role and content:
            history_lines.append(f"{role}: {content}")

    if not history_lines:
        return question

    return "\n".join(history_lines) + f"\nCâu hỏi hiện tại: {question}"


def reorder_for_llm(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Reorder context để giảm lost-in-the-middle.

    Pattern:
    [1, 2, 3, 4, 5] -> [1, 3, 5, 4, 2]
    """
    if len(chunks) <= 2:
        return chunks

    reordered: list[dict[str, Any]] = []

    for index in range(0, len(chunks), 2):
        reordered.append(chunks[index])

    last_odd = len(chunks) - 1 if (len(chunks) - 1) % 2 == 1 else len(chunks) - 2

    for index in range(last_odd, 0, -2):
        reordered.append(chunks[index])

    return reordered


def _citation_label(source: dict[str, Any], index: int) -> str:
    """Tạo nhãn citation từ metadata."""
    metadata = source.get("metadata", {})

    title = metadata.get("title") or metadata.get("source") or f"Nguồn {index}"
    doc_type = metadata.get("doc_type", "unknown")

    title = str(title).strip().replace("\n", " ")
    title = title[:80]

    if doc_type and doc_type != "unknown":
        return f"{title}, {doc_type}"

    return title


def _split_sentences(text: str) -> list[str]:
    """Tách câu đơn giản cho extractive generation."""
    text = re.sub(r"\s+", " ", text or "").strip()

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    output: list[str] = []

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        if len(sentence) <= 450:
            output.append(sentence)
        else:
            for start in range(0, len(sentence), 360):
                piece = sentence[start : start + 360].strip()

                if piece:
                    output.append(piece)

    return output


def generate_answer_from_context(
    query: str,
    sources: list[dict[str, Any]],
) -> str:
    """
    Sinh câu trả lời dạng extractive có citation.

    Ưu điểm:
    - Không cần API key.
    - Không hallucinate.
    - Mỗi ý đều có citation.
    """
    if not sources:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    query_terms = set(tokenize(query))
    evidence: list[tuple[float, str, str]] = []

    ordered_sources = reorder_for_llm(sources[:MAX_CONTEXT_CHUNKS])

    for index, source in enumerate(ordered_sources, start=1):
        citation = _citation_label(source, index)
        sentences = _split_sentences(source["content"])

        for sentence in sentences[:8]:
            sentence_terms = set(tokenize(sentence))
            overlap = len(query_terms & sentence_terms) / max(1, len(query_terms))
            score = overlap + float(source.get("score", 0)) * 0.1

            if overlap > 0 or len(evidence) < 2:
                evidence.append((score, sentence, citation))

    evidence.sort(key=lambda item: item[0], reverse=True)

    selected: list[tuple[float, str, str]] = []
    seen_sentences: set[str] = set()

    for item in evidence:
        _, sentence, _ = item
        normalized = sentence.lower()

        if normalized in seen_sentences:
            continue

        selected.append(item)
        seen_sentences.add(normalized)

        if len(selected) >= MAX_SENTENCES_PER_ANSWER:
            break

    if not selected:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    answer_lines = ["Dựa trên các nguồn đã truy xuất, câu trả lời là:"]

    for _, sentence, citation in selected:
        answer_lines.append(f"- {sentence.strip()} [{citation}]")

    return "\n".join(answer_lines)


def generate_with_citation(
    question: str,
    conversation_history: list[dict[str, str]] | None = None,
    top_k: int = DEFAULT_TOP_K,
    config_name: str = DEFAULT_CONFIG_NAME,
) -> dict[str, Any]:
    """
    Hàm chính cho UI và evaluation gọi.

    Args:
        question: Câu hỏi người dùng.
        conversation_history: Lịch sử hội thoại để hỗ trợ follow-up.
        top_k: Số nguồn muốn lấy.
        config_name: Config retrieval để chạy A/B testing.

    Returns:
        {
            "answer": str,
            "sources": list[dict],
            "standalone_query": str,
            "config_name": str
        }
    """
    standalone_query = build_standalone_query(question, conversation_history)

    sources = retrieve(
        standalone_query,
        top_k=top_k,
        config_name=config_name,
    )

    answer = generate_answer_from_context(question, sources)

    return {
        "answer": answer,
        "sources": sources,
        "standalone_query": standalone_query,
        "config_name": config_name,
    }


def get_source_summary(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Chuẩn hóa source để UI hiển thị ở sidebar/accordion.
    """
    summary: list[dict[str, Any]] = []

    for index, source in enumerate(sources, start=1):
        metadata = source.get("metadata", {})

        summary.append(
            {
                "rank": index,
                "title": metadata.get("title", metadata.get("source", "Unknown")),
                "source": metadata.get("source", "Unknown"),
                "doc_type": metadata.get("doc_type", "unknown"),
                "score": round(float(source.get("score", 0)), 4),
                "retriever": source.get("retriever", "unknown"),
                "preview": source.get("content", "")[:300],
            }
        )

    return summary


def demo() -> None:
    """Chạy thử core engine độc lập."""
    chunks = build_chunks(force_rebuild=True)

    print(f"Đã index {len(chunks)} chunks.")
    print(f"Tìm thấy {len(STANDARDIZED_DIRS)} thư mục standardized.")

    questions = [
        "Hình phạt cho tội tàng trữ trái phép chất ma túy là gì?",
        "Luật phòng chống ma túy quy định gì về cai nghiện?",
        "Có bài báo nào nói về nghệ sĩ liên quan tới ma túy không?",
    ]

    for question in questions:
        print("=" * 80)
        print("QUESTION:", question)

        result = generate_with_citation(question)

        print(result["answer"])
        print("\nSOURCES:")

        for source in get_source_summary(result["sources"]):
            print(
                f"- {source['rank']}. {source['title']} "
                f"({source['doc_type']}) "
                f"score={source['score']} "
                f"retriever={source['retriever']}"
            )

        print()


if __name__ == "__main__":
    demo()