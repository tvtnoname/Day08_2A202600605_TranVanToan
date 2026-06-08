"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"
"""

import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from .task9_retrieval_pipeline import retrieve
except ImportError:
    from src.task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: factual RAG cần câu trả lời ổn định, nhưng vẫn đủ mở để model
# diễn đạt tự nhiên khi trích dẫn nhiều nguồn khác nhau.
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return list(chunks)

    # Đưa các chunk quan trọng nhất lên đầu, giữ chunk ít quan trọng hơn
    # ở giữa, và đẩy một phần xuống cuối để giảm "lost in the middle".
    # Ví dụ: [1, 2, 3, 4, 5] -> [1, 3, 5, 4, 2]
    front = chunks[::2]
    back = chunks[1::2][::-1]
    return list(front) + list(back)


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    def _source_label(chunk: dict[str, Any], fallback_index: int) -> str:
        metadata = chunk.get("metadata", {}) if isinstance(chunk.get("metadata"), dict) else {}
        source = str(metadata.get("source") or metadata.get("filename") or f"Source {fallback_index}")
        source_name = Path(source).stem

        year = metadata.get("year") or metadata.get("date") or metadata.get("published_year")
        if isinstance(year, str):
            year_match = re.search(r"(19|20)\d{2}", year)
            year = year_match.group(0) if year_match else year.strip()
        elif year is None:
            combined = " ".join(
                str(value)
                for value in (
                    metadata.get("path"),
                    metadata.get("source"),
                    metadata.get("filename"),
                )
                if value
            )
            match = re.search(r"(19|20)\d{2}", combined)
            year = match.group(0) if match else "n.d."
        else:
            year = str(year)

        return f"[{source_name}, {year}]"

    context_parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {}) if isinstance(chunk.get("metadata"), dict) else {}
        source = metadata.get("source") or metadata.get("filename") or f"Source {i}"
        doc_type = metadata.get("type", "unknown")
        score = chunk.get("score", 0.0)
        citation = _source_label(chunk, i)
        content = str(chunk.get("content", "")).strip()
        if not content:
            continue
        context_parts.append(
            f"[Chunk {i} | Citation: {citation} | Source: {source} | Type: {doc_type} | Score: {score:.3f}]\n"
            f"{content}"
        )

    return "\n\n---\n\n".join(context_parts)


def _get_openai_client():
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None

    if not api_key:
        return None

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    return OpenAI(**kwargs)


def _has_citation(answer: str) -> bool:
    return "[" in answer and "]" in answer


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(
    query: str,
    context_chunks: list[dict] | None = None,
    top_k: int = TOP_K,
) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    chunks = context_chunks if context_chunks is not None else retrieve(query, top_k=top_k)

    if not chunks:
        return {
            "answer": "I cannot verify this information",
            "sources": [],
            "retrieval_source": "none",
        }

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    if not context.strip():
        return {
            "answer": "I cannot verify this information",
            "sources": chunks[:top_k],
            "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
        }

    user_message = (
        "Context:\n"
        f"{context}\n\n"
        "---\n\n"
        f"Question: {query}\n\n"
        "Use only the context above. Every factual statement must include a citation."
    )

    client = _get_openai_client()
    if client is None:
        return {
            "answer": "I cannot verify this information",
            "sources": chunks[:top_k],
            "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
        }

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
            stream=False,
        )
        answer = (response.choices[0].message.content or "").strip()
    except Exception:
        answer = ""

    if not answer or (answer != "I cannot verify this information" and not _has_citation(answer)):
        answer = "I cannot verify this information"

    return {
        "answer": answer,
        "sources": chunks[:top_k],
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
