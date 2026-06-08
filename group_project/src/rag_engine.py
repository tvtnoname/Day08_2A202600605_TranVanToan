import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Đảm bảo đã import config để load sys.path
from . import config

# Import các module từ personal_project thông qua personal_src
from personal_src.task5_semantic_search import semantic_search
from personal_src.task6_lexical_search import lexical_search
from personal_src.task7_reranking import rerank, rerank_rrf
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
        final_results = rerank(query, merged, top_k=top_k, method=rerank_method)
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
