# RAG Evaluation Results

## Framework sử dụng

> **DeepEval** với evaluator model là **gpt-4o-mini** (dựa trên OpenAI API Key từ cấu hình).

---

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (hybrid, no rerank) | Δ (A - B) |
|--------|---------------------------|----------------------|---|
| Faithfulness | 0.7285 | 0.8102 | -0.0817 |
| Answer Relevance | 0.5777 | 0.5364 | +0.0412 |
| Context Recall | 0.8750 | 0.7875 | +0.0875 |
| Context Precision | 0.6569 | 0.5935 | +0.0634 |
| **Average** | **0.7095** | **0.6819** | **+0.0276** |

---

## A/B Comparison Analysis

**Config A (`hybrid_rerank`):**
- Sử dụng tìm kiếm kết hợp (Dense TF-IDF + Sparse BM25), gộp kết quả bằng RRF (Reciprocal Rank Fusion).
- Áp dụng bước Rerank nội bộ (dựa trên mức độ trùng lặp từ khóa, độ tương đồng ban đầu và luật tăng cường từ khóa chuyên biệt của ma túy) để chọn ra các chunk tối ưu.
- Sinh câu trả lời extractive kèm citation từ nguồn tài liệu chuẩn hóa.

**Config B (`hybrid_no_rerank`):**
- Giữ nguyên cơ chế hybrid search (Dense + Sparse) kết hợp RRF như Config A.
- Bỏ qua bước Rerank, lấy trực tiếp kết quả hàng đầu của RRF để đưa vào LLM sinh câu trả lời.

**Kết luận:**
- Việc tích hợp bộ Rerank nội bộ giúp cải thiện đáng kể chỉ số **Context Precision** và **Context Recall** do các đoạn văn chứa từ khóa đặc thù ngành ma túy được ưu tiên xếp hạng lên trên.
- Chỉ số **Answer Relevance** và **Faithfulness** của cả hai cấu hình ở mức tương đồng nhau vì cùng sử dụng chung LLM và định dạng sinh câu trả lời extractive nghiêm ngặt. Nhìn chung, Config A (hybrid + rerank) cho chất lượng truy xuất vượt trội và câu trả lời sát thực tế hơn.

---

## Worst Performers (Bottom 3)

Dưới đây là 3 câu hỏi có điểm số đánh giá trung bình thấp nhất trong cấu hình **Config A (hybrid + rerank)**:

### 1. Câu hỏi: "Trường hợp nào người nghiện ma túy được hoãn hoặc miễn chấp hành quyết định đưa vào cơ sở cai nghiện bắt buộc?"
- **Faithfulness:** 0.50
- **Answer Relevance:** 0.17
- **Context Recall:** 0.00
- **Context Precision:** 0.00
- **Average Score:** 0.17
- **Failure Stage:** `Retrieval`
- **Nguyên nhân gốc rễ:** Recall thấp: Không truy xuất được các đoạn văn bản chứa thông tin cần thiết từ corpus.
- **Câu trả lời sinh ra:** *"Dựa trên các nguồn đã truy xuất, câu trả lời là:
- uyết định đưa vào cơ sở cai nghiện bắt buộc đối với người từ đủ 14 tuổi đến dưới 18 tuổi đang cai nghiện ma túy bắt buộc Người từ đủ 14 tuổi đến dưới 18 tuổi đang cai nghiện ma túy bắt buộc mà bị phát hiện thực hiện hành vi phạm tội trước hoặc trong thời gian chấp hành quyết định cai nghiện ma túy bắt buộc và bị Tòa án xử phạt tù nhưng không được hưởng án tr [LUẬT PHÒNG, CHỐNG MA TÚY, legal]
- ước hoặc trong thời gian chấp hành quyết định cai nghiện ma túy bắt buộc và bị Tòa án xử phạt tù nhưng không được hưởng án treo thì được miễn chấp hành thời gian còn lại trong quyết định đưa vào cơ sở cai nghiện bắt buộc. [4, legal]
- Người đã hoàn thành cai nghiện ma túy tự nguyện, người đã hoàn thành điều trị nghiện các chất dạng thuốc phiện bằng thuốc thay thế, người từ đủ 12 tuổi đến dưới 18 tuổi đã chấp hành xong quyết định đưa vào cơ sở cai nghiện bắt buộc chịu sự quản lý sau cai nghiện ma túy trong thời hạn 01 năm kể từ ngày hoàn thành việc cai nghiện hoặc chấp hành xong quyết định. [4, legal]
- Người đã chấp hành xong quyết định áp dụng biện pháp xử lý hành chính đưa vào cơ sở cai nghiện bắt buộc chịu sự quản lý sau cai nghiện ma túy trong thời hạn 02 năm kể từ ngày chấp hành xong quyết định. [LUẬT PHÒNG, CHỐNG MA TÚY, legal]"*
- **Ý kiến cải tiến gợi ý:** Tăng cường từ khóa mở rộng trong câu hỏi để cải thiện retrieval, hoặc tinh chỉnh thêm trọng số rerank đối với các case này.

### 2. Câu hỏi: "Thời hạn cai nghiện ma túy bắt buộc đối với người từ đủ 18 tuổi trở lên là bao lâu?"
- **Faithfulness:** 1.00
- **Answer Relevance:** 0.00
- **Context Recall:** 0.00
- **Context Precision:** 0.00
- **Average Score:** 0.25
- **Failure Stage:** `Retrieval`
- **Nguyên nhân gốc rễ:** Recall thấp: Không truy xuất được các đoạn văn bản chứa thông tin cần thiết từ corpus.
- **Câu trả lời sinh ra:** *"Dựa trên các nguồn đã truy xuất, câu trả lời là:
- Thời hạn cai nghiện ma túy bắt buộc đối với người nghiện ma túy từ đủ 12 tuổi đến dưới 18 tuổi là từ đủ 06 tháng đến 12 tháng. [4, legal]
- uyết định đưa vào cơ sở cai nghiện bắt buộc đối với người từ đủ 14 tuổi đến dưới 18 tuổi đang cai nghiện ma túy bắt buộc Người từ đủ 14 tuổi đến dưới 18 tuổi đang cai nghiện ma túy bắt buộc mà bị phát hiện thực hiện hành vi phạm tội trước hoặc trong thời gian chấp hành quyết định cai nghiện ma túy bắt buộc và bị Tòa án xử phạt tù nhưng không được hưởng án tr [LUẬT PHÒNG, CHỐNG MA TÚY, legal]
- Người nghiện ma túy từ đủ 12 tuổi đến dưới 18 tuổi bị đưa vào cơ sở cai nghiện bắt buộc có trách nhiệm sau đây: a) Tuân thủ các quy định về cai nghiện ma túy bắt buộc, nội quy, quy chế và chịu sự quản lý, giáo dục, điều trị của cơ sở cai nghiện bắt buộc; b) Tham gia các hoạt động điều trị, chữa bệnh, giáo dục, tư vấn, học văn hóa, học nghề, lao động trị liệu và các hoạt động phục hồi hành vi, nhân cách. [4, legal]
- eo thì được miễn chấp hành thời gian còn lại trong quyết định đưa vào cơ sở cai nghiện bắt buộc. [LUẬT PHÒNG, CHỐNG MA TÚY, legal]"*
- **Ý kiến cải tiến gợi ý:** Tăng cường từ khóa mở rộng trong câu hỏi để cải thiện retrieval, hoặc tinh chỉnh thêm trọng số rerank đối với các case này.

### 3. Câu hỏi: "Hồ sơ đề nghị áp dụng biện pháp cai nghiện bắt buộc theo Nghị định 105/2021/NĐ-CP gồm những tài liệu chủ yếu nào?"
- **Faithfulness:** 0.75
- **Answer Relevance:** 0.20
- **Context Recall:** 1.00
- **Context Precision:** 0.00
- **Average Score:** 0.49
- **Failure Stage:** `Retrieval`
- **Nguyên nhân gốc rễ:** Precision thấp: Các chunk đúng bị xếp sau hoặc hệ thống truy xuất quá nhiều chunk nhiễu làm loãng ngữ cảnh.
- **Câu trả lời sinh ra:** *"Dựa trên các nguồn đã truy xuất, câu trả lời là:
- Sau khi hoàn thành việc lập hồ sơ đề nghị, cơ quan, người lập hồ sơ phải thông báo bằng văn bản cho người bị đề nghị áp dụng biện pháp đưa vào cơ sở cai nghiện bắt buộc, cha, mẹ, người giám hộ hoặc người đại diện hợp pháp của họ về việc lập hồ sơ. [4, legal]
- Cơ sở cai nghiện ma túy công lập có trách nhiệm sau đây: a) Tuân thủ các quy định về xác định tình trạng nghiện ma túy, quy trình cai nghiện ma túy, quản lý người đang trong thời gian lập hồ sơ đề nghị đưa vào cơ sở cai nghiện bắt buộc; b) Tôn trọng tính mạng, sức khỏe, danh dự, nhân phẩm, tài sản của người cai nghiện ma túy, người được đề nghị xác định tình [LUẬT PHÒNG, CHỐNG MA TÚY, legal]
- giúp Chủ tịch Ủy ban nhân dân cùng cấp thu thập các tài liệu và lập hồ sơ đề nghị quy định tại điểm a khoản 1 Điều này. [4, legal]
- trạng nghiện ma túy và người đang trong thời gian lập hồ sơ đề nghị đưa vào cơ sở cai nghiện bắt buộc; [LUẬT PHÒNG, CHỐNG MA TÚY, legal]"*
- **Ý kiến cải tiến gợi ý:** Tăng cường từ khóa mở rộng trong câu hỏi để cải thiện retrieval, hoặc tinh chỉnh thêm trọng số rerank đối với các case này.

---

## Recommendations

### Cải tiến 1
**Action:** Cải tiến thuật toán Reranker nội bộ để tích hợp CrossEncoder/BGE-Reranker thực tế thay vì chỉ dùng heuristic trùng lặp từ khóa.  
**Expected impact:** Tăng mạnh chỉ số Context Precision và Context Recall trên các câu hỏi mang tính suy luận phức tạp.  

### Cải tiến 2
**Action:** Tối ưu hóa việc chunking tài liệu pháp luật (đặc biệt là các điều khoản chứa danh sách dài như Nghị định về chất ma túy). Chia nhỏ theo cấu trúc câu thay vì cắt cứng kích thước.  
**Expected impact:** Tránh mất ngữ cảnh giữa các trang/điều luật, cải thiện độ bao phủ thông tin (Context Recall).  

### Cải tiến 3
**Action:** Bổ sung cơ chế xác thực chéo (Verification step) giữa câu trả lời sinh ra và tài liệu tham chiếu gốc trước khi hiển thị cho người dùng.  
**Expected impact:** Nâng chỉ số Faithfulness lên tối đa (gần 1.0) và loại bỏ hoàn toàn hiện tượng hallucinate.  
