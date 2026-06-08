# ⚖️ Trợ Lý Pháp Luật Phòng Chống Ma Tuý & Tin Tức Nghệ Sĩ

Chào mừng bạn đến với **DrugLaw RAG Chatbot**! Đây là ứng dụng tìm kiếm và giải đáp câu hỏi thông minh, kết hợp dữ liệu pháp luật chính thống và tin tức báo chí Việt Nam về các chất cấm.

---

### 🚀 Tính Năng Nổi Bật

1.  **Truy xuất Hybrid (Dense + Sparse)**: Kết hợp tìm kiếm ngữ nghĩa (Semantic) qua *ChromaDB* và tìm kiếm từ khóa (Lexical) qua *BM25* giúp tìm đúng tài liệu tham khảo.
2.  **Reranking Nâng Cao**: Sử dụng mô hình *Cross-Encoder* để sắp xếp lại tài liệu có độ liên quan cao nhất lên đầu.
3.  **Hội thoại liên tục (Memory)**: Trích xuất ngữ cảnh từ lịch sử trò chuyện để viết lại câu hỏi tiếp nối tự động.
4.  **Trích dẫn minh bạch (Citations)**: Câu trả lời luôn đi kèm các thẻ nguồn dẫn tới điều khoản luật hoặc bài báo tương ứng.
5.  **Cơ chế dự phòng (PageIndex Fallback)**: Tự động chuyển hướng sang tìm kiếm phi vector trên *PageIndex Cloud* khi điểm số hybrid dưới ngưỡng an toàn.

---

### 💡 Gợi ý câu hỏi để thử nghiệm

*   *“Hình phạt tù cho tội tàng trữ trái phép chất ma tuý là bao nhiêu năm?”*
*   *“Nghị định 105/2021/NĐ-CP hướng dẫn chi tiết những nội dung gì?”*
*   *“Nghệ sĩ nào đã bị truy tố liên quan tới ma tuý gần đây?”*
*   *“Quy trình cai nghiện bắt buộc gồm những giai đoạn nào?”*

---

> [!TIP]
> Bạn có thể tùy chỉnh các thông số như số lượng tài liệu `top_k`, ngưỡng điểm `score_threshold` và thuật toán xếp hạng tại bảng **Chat Settings** (Cài đặt) ở thanh điều khiển góc dưới bên trái!
