"""
Task 10 — Generation Có Citation.

R → A → G: sau khi retrieve (Task 9), ta ĐÓNG GÓI context (reorder chống lost-in-the-middle
+ gắn nhãn nguồn) rồi yêu cầu LLM sinh câu trả lời GROUNDED, có citation, biết abstain.

LLM: dùng router OpenAI-compatible (cấu hình trong .env: OPENAI_BASE_URL/KEY/LLM_MODEL).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from .task9_retrieval_pipeline import retrieve

# =============================================================================
# CONFIGURATION — giải thích lựa chọn
# =============================================================================
# top_k=5: đủ chứng cứ mà không làm context quá dài (giữ headroom cho prompt/output,
#          và giảm nguy cơ lost-in-the-middle).
TOP_K = 5
# top_p=0.9: nucleus sampling đủ tự nhiên nhưng không quá ngẫu nhiên.
TOP_P = 0.9
# temperature=0.2: RAG cần factual, bám context, ít "sáng tạo".
TEMPERATURE = 0.2

MODEL = os.getenv("LLM_MODEL", "cc/claude-opus-4-8")

SYSTEM_PROMPT = """Bạn là trợ lý pháp luật, trả lời bằng tiếng Việt, CHỈ dựa trên ngữ cảnh được cung cấp.
Với mỗi khẳng định/sự thật, chèn ngay trích dẫn trong ngoặc vuông trỏ tới nguồn cụ thể
(ví dụ: [luat-phong-chong-ma-tuy-2021.md, Điều 23] hoặc [VnExpress, 2026]).

Nếu thông tin KHÔNG có trong ngữ cảnh, hãy trả lời 'Tôi không thể xác minh thông tin này
từ nguồn hiện có' thay vì suy đoán.

Quy tắc:
- Chỉ dùng thông tin trong ngữ cảnh.
- Mọi khẳng định PHẢI có citation.
- Nếu ngữ cảnh không đủ, nói rõ.
- Trình bày mạch lạc theo đoạn."""


# =============================================================================
# DOCUMENT REORDERING (chống lost in the middle)
# =============================================================================
def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks: quan trọng nhất ở ĐẦU và CUỐI, kém quan trọng ở GIỮA.

    Input (đã sort theo score giảm dần): [c0, c1, c2, c3, c4]
    Output: c0,c2,c4 (vị trí chẵn) + đảo ngược c3,c1 → [c0, c2, c4, c3, c1]
    => c0 (tốt nhất) ở đầu, c1 (tốt nhì) ở cuối, các chunk yếu hơn dồn vào giữa.
    """
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[0::2]          # 0, 2, 4, ...
    back = chunks[1::2][::-1]     # ..., 3, 1
    return front + back


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================
def format_context(chunks: list[dict]) -> str:
    """Format chunks thành context có nhãn nguồn để LLM cite."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        md = chunk.get("metadata", {}) or {}
        source = md.get("source", f"Source {i}")
        doc_type = md.get("type", "unknown")
        section = md.get("section", "")
        label = f"[Document {i} | Source: {source} | Type: {doc_type}"
        label += f" | {section}]" if section else "]"
        parts.append(f"{label}\n{chunk['content']}\n")
    return "\n---\n".join(parts)


# =============================================================================
# GENERATION
# =============================================================================
def _call_llm(system: str, user: str) -> str:
    """Gọi LLM qua router OpenAI-compatible. Tự lùi về temperature mặc định nếu bị từ chối."""
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=TEMPERATURE, top_p=TOP_P,
        )
    except Exception:
        # Một số model (Claude extended-thinking) chỉ chấp nhận temperature=1.
        resp = client.chat.completions.create(model=MODEL, messages=messages)
    return resp.choices[0].message.content


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Returns:
        {'answer': str, 'sources': list[dict], 'retrieval_source': str}
    """
    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Ngữ cảnh:\n{context}\n\n---\n\nCâu hỏi: {query}"

    answer = _call_llm(SYSTEM_PROMPT, user_message)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid"),
    }


if __name__ == "__main__":
    for q in [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
    ]:
        print(f"\n{'='*70}\nQ: {q}\n{'='*70}")
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
