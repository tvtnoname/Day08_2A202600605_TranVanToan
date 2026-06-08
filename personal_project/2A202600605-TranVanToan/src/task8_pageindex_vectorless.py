"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """
    Upload các tài liệu PDF pháp luật lên PageIndex.
    """
    from pageindex import PageIndexClient
    if not PAGEINDEX_API_KEY:
        print("⚠ PAGEINDEX_API_KEY is not set in .env")
        return
        
    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    
    # PageIndex chỉ hỗ trợ tệp PDF
    pdf_dir = STANDARDIZED_DIR.parent / "landing" / "legal"
    for pdf_file in pdf_dir.rglob("*.pdf"):
        if pdf_file.name.startswith("."):
            continue
        print(f"Uploading: {pdf_file.name}")
        try:
            res = client.submit_document(file_path=str(pdf_file))
            print(f"  ✓ Uploaded: {pdf_file.name}, doc_id: {res.get('doc_id')}")
        except Exception as e:
            print(f"  ✗ Failed to upload {pdf_file.name}: {e}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    from pageindex import PageIndexClient
    import time
    
    if not PAGEINDEX_API_KEY or PAGEINDEX_API_KEY.startswith("pi_xxx"):
        raise ValueError("PAGEINDEX_API_KEY is not set or invalid.")
        
    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    
    # 1. Lấy danh sách tài liệu đang có trên PageIndex
    docs_res = client.list_documents(limit=50)
    documents = docs_res.get("documents", [])
    if not documents:
        return []
        
    all_results = []
    # 2. Thực hiện truy vấn trên từng tài liệu và tổng hợp kết quả
    for doc in documents:
        doc_id = doc.get("id")
        if not doc_id:
            continue
        try:
            sub_res = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = sub_res.get("retrieval_id")
            if not retrieval_id:
                continue
                
            # Polling kết quả truy vấn
            for _ in range(10):  # Chờ tối đa 10s cho mỗi tài liệu
                ret_res = client.get_retrieval(retrieval_id)
                status = ret_res.get("status")
                if status == "completed":
                    for r in ret_res.get("results", []):
                        all_results.append({
                            "content": r.get("text") or r.get("content") or "",
                            "score": float(r.get("score") or 0.0),
                            "metadata": r.get("metadata") or {},
                            "source": "pageindex"
                        })
                    break
                elif status == "failed":
                    break
                time.sleep(1)
        except Exception as e:
            print(f"Error querying doc {doc_id} on PageIndex: {e}")
            
    # Sắp xếp kết quả theo điểm số tương đồng giảm dần
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
