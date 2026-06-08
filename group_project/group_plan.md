# Kế hoạch Phân công Công việc Nhóm — Search Engine / RAG Chatbot

Kế hoạch này phân chia công việc đều cho nhóm 4 thành viên nhằm tối ưu hóa hiệu năng phát triển và **hoàn toàn tránh được xung đột mã nguồn (Git Merge Conflict)** bằng cách phân chia tệp tin/thư mục làm việc độc lập.

---

## 📂 Đề xuất cấu trúc thư mục nhóm (Phân tách rõ ràng)

```text
group_project/
│
├── src/                          <-- Làm việc bởi Thành viên 1
│   ├── __init__.py
│   ├── config.py                 <-- File cấu hình chung (LLM, threshold, paths)
│   └── rag_engine.py             <-- Core RAG Engine (tích hợp Rerank, PageIndex...)
│
├── evaluation/                   <-- Làm việc bởi Thành viên 3 & 4
│   ├── golden_dataset.json       <-- Bộ Dataset 15+ câu hỏi (Thành viên 4)
│   ├── eval_pipeline.py          <-- Script chạy đánh giá (Thành viên 3)
│   └── results.md                <-- Báo cáo kết quả và so sánh A/B (Thành viên 4)
│
├── app.py                        <-- Giao diện UI Chatbot Streamlit (Thành viên 2)
│
├── requirements.txt              <-- Khai báo các thư viện cài thêm (Cả nhóm)
├── group_plan.md                 <-- Bản kế hoạch phân công công việc này (Cả nhóm)
└── README.md                     <-- Hướng dẫn cài đặt và Diagram kiến trúc (Thành viên 4)
```

---

## 👥 Bảng phân công công việc chi tiết

| Thành viên | MSSV | Vai trò chính | Nhiệm vụ cụ thể | Tệp tin thao tác chính |
| :--- | :--- | :--- | :--- | :--- |
| **Thành viên 1** | *[Điền MSSV]* | **RAG Core Engineer**<br>(Nhóm trưởng) | - Tích hợp, chọn lọc codebase cá nhân.<br>- Viết core engine tại `src/rag_engine.py` xử lý Hybrid search (Dense + Sparse), Reranking (CrossEncoder/MMR) và PageIndex fallback.<br>- Định nghĩa template prompt sinh câu trả lời có Citation. | `src/rag_engine.py`<br>`src/config.py` |
| **Thành viên 2** | *[Điền MSSV]* | **UI/UX Chat Developer** | - Xây dựng giao diện Web Chatbot tương tác bằng **Streamlit** hoặc Chainlit tại `app.py`.<br>- Tích hợp cơ chế bộ nhớ hội thoại (`st.session_state` / conversation memory) cho các câu hỏi follow-up.<br>- Thiết kế phần hiển thị tài liệu nguồn đã sử dụng ở thanh sidebar hoặc accordion. | `app.py` |
| **Thành viên 3** | *[Điền MSSV]* | **Evaluation Engineer** | - Thiết lập và cấu hình framework đánh giá (DeepEval / Ragas).<br>- Viết script `evaluation/eval_pipeline.py` để tự động hóa việc đọc dataset, gọi pipeline RAG và tính toán 4 metrics cốt lõi.<br>- Thực thi đánh giá so sánh A/B Testing trên ít nhất 2 cấu hình (ví dụ: có Rerank vs không Rerank). | `evaluation/eval_pipeline.py` |
| **Thành viên 4** | *[Điền MSSV]* | **Data Analyst & Writer** | - Biên soạn 15+ cặp câu hỏi Q&A chất lượng lưu vào `evaluation/golden_dataset.json`.<br>- Thực hiện chạy thử nghiệm đánh giá cùng Thành viên 3.<br>- Phân tích các trường hợp tệ nhất (worst performers) và đề xuất cải tiến.<br>- Vẽ sơ đồ kiến trúc hệ thống và cập nhật báo cáo `evaluation/results.md` & `README.md`. | `evaluation/golden_dataset.json`<br>`evaluation/results.md`<br>`README.md` |

---

## 🛡️ Quy trình phối hợp và Kiểm soát xung đột Git

Để đảm bảo dự án hoạt động trơn tru mà không xảy ra xung đột code khi push:

1. **Khởi tạo và Thống nhất Interface (Ngày đầu tiên):**
   - Thành viên 1 thiết lập khung thư mục, khai báo hàm rỗng của `rag_engine.py` (ví dụ: `def generate_with_citation(query: str) -> dict: pass`).
   - Push khung dự án này lên nhánh `main`.

2. **Làm việc trên các Nhánh tính năng (Feature Branches):**
   Từng thành viên checkout sang nhánh riêng dựa trên vai trò:
   - Thành viên 1: `git checkout -b feature/rag-core`
   - Thành viên 2: `git checkout -b feature/chat-ui`
   - Thành viên 3: `git checkout -b feature/eval-script`
   - Thành viên 4: `git checkout -b feature/golden-data`

3. **Tích hợp độc lập:**
   - Thành viên 2 viết UI gọi hàm từ `src.rag_engine.generate_with_citation` (gọi hàm mẫu đã định nghĩa trước).
   - Thành viên 3 viết script kiểm thử cũng gọi `src.rag_engine.generate_with_citation`.
   - Vì mỗi thành viên sửa đổi trên các file hoàn toàn khác nhau (`app.py`, `eval_pipeline.py`, `rag_engine.py`, `golden_dataset.json`), các thao tác Git Merge/Push sẽ tự động được giải quyết (auto-merged) một cách hoàn hảo khi tạo **Pull Request (PR)**.
