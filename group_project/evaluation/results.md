# RAG Evaluation Results

- **Framework:** DeepEval (LLM-as-a-judge)
- **Judge model:** `cc/claude-haiku-4-5-20251001` (temperature=0)
- **Golden dataset:** 16 câu hỏi
- **Threshold pass:** 0.7
- **Metrics:** Faithfulness, Answer Relevancy, Contextual Recall, Contextual Precision

## So sánh A/B (điểm trung bình)

| Config | Faithfulness | Answer Relevancy | Contextual Recall | Contextual Precision | **Overall** |
|---|---|---|---|---|---|
| `hybrid_rerank` | 0.935 | 0.571 | 0.798 | 0.665 | **0.742** |
| `hybrid_no_rerank` | 0.910 | 0.626 | 0.767 | 0.511 | **0.704** |

**Cấu hình tốt nhất theo overall: `hybrid_rerank` (0.742).**

> `hybrid_rerank` vs `hybrid_no_rerank`: chênh lệch overall = +0.039. Reranking giúp cải thiện chất lượng.

## Worst performers (config `hybrid_rerank`)

### ❌ avg=0.25 — Các chất ma tuý tuyệt đối cấm sử dụng (nhóm I) gồm những chất tiêu biểu nào?
- **Answer:** Dựa trên các nguồn đã truy xuất, câu trả lời là: - Đơn vị được tiến hành các hoạt động hợp pháp liên quan đến ma túy vì mục đích quốc phòng, an ninh được quy định tại Điều 23 của Nghị định này có trác...
- **Metric yếu nhất:** answer_relevancy = 0.0
- **Lý do:** The score is 0.00 because the actual output completely fails to address the input question. The input asks for examples of absolutely prohibited narcotic substances in Group I, but the output instead discusses legal procedures, storage requ

### ❌ avg=0.44 — Tội mua bán trái phép chất ma tuý theo Điều 251 có khung hình phạt cao nhất là gì?
- **Answer:** Dựa trên các nguồn đã truy xuất, câu trả lời là: - Người nào tàng trữ trái phép chất ma túy mà không nhằm mục đích mua bán, vận chuyển, sản xuất trái phép chất ma túy thuộc một trong các trường hợp...
- **Metric yếu nhất:** contextual_recall = 0.0
- **Lý do:** The score is 0.00 because the retrieval context in node(s) in retrieval context does not contain information about the maximum penalties for illegal drug trafficking mentioned in the expected output. While the context references penalties o

### ❌ avg=0.58 — Thủ đô của nước Pháp tên là gì?
- **Answer:** Dựa trên các nguồn đã truy xuất, câu trả lời là: - ang các nước tham gia Điều 12 Công ước năm 1988 của Liên hợp quốc về chống buôn bán bất hợp pháp các chất ma túy và các chất hướng thần), bao gồm cả ...
- **Metric yếu nhất:** answer_relevancy = 0.0
- **Lý do:** The score is 0.00 because the actual output is entirely irrelevant to the input question, which asks for the capital of France in Vietnamese. Instead of providing the straightforward answer 'Paris' (or 'Paris' in French), the output discuss

## Phân tích & Đề xuất cải tiến

- **Contextual Recall thấp** → retriever bỏ sót evidence: tăng `top_k`, cải thiện chunking, hoặc bổ sung query expansion/HyDE.
- **Contextual Precision thấp** → context lẫn nhiễu: siết rerank, giảm `top_k`, hoặc pre-filter theo metadata.
- **Faithfulness thấp** → câu trả lời vượt chứng cứ: siết prompt grounding / abstention.
- **Answer Relevancy thấp** → câu trả lời lan man: rút gọn, bám sát câu hỏi (giảm số câu trích).

_(Báo cáo tự sinh bởi `eval_pipeline.py`. Thành viên 4 bổ sung phân tích sâu + sơ đồ kiến trúc.)_