# Báo cáo cá nhân — Day 6

## Dữ liệu và pipeline
- Chủ đề, nguồn/quyền, số trang và quy tắc tính trang: Corpus gồm 7 chương về sức khỏe ung thư từ *Cancer Concepts: A Guidebook for the Non-Oncologist*, University of Massachusetts Chan Medical School eScholarship ([nguồn sách](https://publishing.escholarship.umassmed.edu/plugins/books/1/)). Các PDF ghi giấy phép CC BY-NC-SA 4.0; khi chia sẻ cần ghi công, giữ giấy phép tương thích và tuân thủ điều kiện phi thương mại. Có 55 trang vật lý; tính 38 trang nội dung chương có văn bản, tổng cộng 124.315 ký tự sau chuẩn hóa. Loại trang thought questions, glossary-only và references-only theo [manifest](../../data/manifest.csv). Corpus được kiểm tra bằng `python verify_corpus.py`.
- Chuẩn hóa, lỗi và giới hạn: Trích xuất PDF bằng `pypdf 6.21.4`, chuẩn hóa Unicode NFKC, line ending và whitespace; giữ số trang PDF vật lý. Không dùng OCR. Trong 38 trang được chọn, không ghi nhận trang trích xuất lỗi hoặc rỗng. Kết quả phụ thuộc vào chất lượng text layer của các PDF; kiểm tra corpus: 55 trang vật lý, 38 trang nội dung, 124.315 ký tự.
- Embedding model/provider/version/số chiều, vector store/version, metric, lệnh chạy và chi phí: Google Gemini `gemini-embedding-001`, SDK `google-genai 3.3.0`, vector 768 chiều. Vector store PostgreSQL 16.16 + pgvector 0.8.7, client `psycopg 3.3.8`; cosine distance (`<=>`), exact search. Chạy bằng Docker Compose trong `starter/`, cấu hình `DATABASE_URL` và `GEMINI_API_KEY` trong `starter/.env` (không commit file này); lệnh chính: `cd starter && python -m src.run --phase dev`, sau đó `python -m src.run --phase test --run-id 20261113T222402Z --selected-strategy A --selection-criterion "Highest dev Evidence Hit@3; break ties by Hit@1"`. 
- Metadata bắt buộc, trường lọc bổ sung, nguồn giá trị và lý do chọn: Mỗi chunk giữ `document_id`, `source` (đường dẫn PDF), `page` (trang PDF vật lý), `chapter`, cùng `source_url`, `license` và chỉ số chunk. Trường lọc bổ sung là `chapter`: giá trị lấy từ tên chương trong manifest, còn chapter filter của câu hỏi chỉ được suy ra từ nội dung câu hỏi. Chọn chapter để giới hạn retrieval vào chương được hỏi và để truy dấu chunk về tài liệu/trang. Không dùng gold source hoặc đáp án làm filter.

## Cấu hình và benchmark
- A / B / C (tham số cụ thể, khác biệt duy nhất B–C):
  - A: cửa sổ cố định 256 từ, overlap 50 từ; không áp dụng chapter filter.
  - B: chunk theo đoạn, tối đa 256 từ; không áp dụng chapter filter.
  - C: cùng chunk và vectors với B; chỉ khác ở việc áp dụng chapter filter nếu câu hỏi nêu rõ chương.
- ≥20 câu: phân chia ≥15 dev / ≥5 test, câu metadata filter và cách kiểm chứng evidence: Benchmark có 20 câu đã xác minh, 15 dev và 5 test; 9 câu có yêu cầu chapter filter (3 trong test). Gold evidence quote được kiểm tra tự động trên PDF và trang khai báo. Hit@1/Evidence Hit@3 tính là hit khi kết quả khớp tài liệu, trang gold và evidence quote trong chunk. Đây là kiểm tra exact quote, có thể bỏ sót evidence tương đương hoặc evidence bị tách qua chunk. Câu hỏi và split nằm trong [questions.jsonl](../../benchmark/questions.jsonl).
- Tiêu chí chọn chiến lược đã chốt trước khi xem test: Chọn Evidence Hit@3 dev cao nhất; nếu hòa thì chọn Hit@1 dev cao hơn. Dựa trên dev, chọn A trước khi chạy test của run `20261113T222402Z`.

## Kết quả
| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count | Thời gian | Chi phí |
|---|---|---:|---:|---:|---|---|
| A | dev | 8/15 | 14/15 | 110 | Chunk 0.005s; embed 4.813s; store 0.105s |  |
| B | dev | 4/15 | 7/15 | 167 | Chunk 0.005s; embed 6.381s; store 0.092s |  |
| C | dev | 4/15 | 7/15 | 167 | Chunk 0.005s; reuse B embeddings; store 0.094s |  |
| A | test | 3/5 | 5/5 | 110 | Dùng index từ dev |  |
| B | test | 1/5 | 2/5 | 167 | Dùng index từ dev |  |
| C | test | 1/5 | 2/5 | 167 | Dùng index từ dev |  |

| Strategy | Coherence trung bình /2 (n) | Metadata đủ và đúng (n/N) | Filter tuân thủ (n/N hoặc N/A) | Grounding 0/1/2 (số câu) và trung bình /2 |
|---|---:|---:|---|---|
| A | 1.0 (10) | 15/15 | N/A (không bật filter; 3/3 kết quả tình cờ cùng chapter) | 0: 0, 1: 0, 2: 5; trung bình 2.0 (n=5) |
| B | 1.1 (10) | 15/15 | N/A (không bật filter; 2/3 kết quả tình cờ cùng chapter) | 0: 2, 1: 2, 2: 1; trung bình 0.8 (n=5) |
| C | 1.1 (10) | 15/15 | 3/3 | 0: 2, 1: 2, 2: 1; trung bình 0.8 (n=5) |

Thời gian là số đo của một lần chạy local, không phải benchmark ổn định. Chunking 0.005s; embed/store là số theo strategy; query+retrieval là 6.405s cho toàn pha dev và 2.201s cho toàn pha test (không phải thời gian riêng từng strategy). API cost không có trong runner và chưa được xác minh từ billing. Grounding là đánh giá retrieval evidence, không phải độ chính xác câu trả lời do LLM sinh.

Đường dẫn trace, nhãn thủ công và cách chọn mẫu: Toàn bộ trace và cấu hình run nằm trong [results/data-ingestion-20261113T222402Z](../../results/data-ingestion-20261113T222402Z/), gồm `dev_traces.jsonl`, `test_traces.jsonl` và `run_metadata.json`. Bảng chi tiết nhãn ở [manual_audit.csv](../../results/data-ingestion-20261113T222402Z/manual_audit.csv); các nhãn được đề xuất với hỗ trợ AI và đã được sinh viên xem lại, chấp thuận. Metadata được kiểm tra trên 5 câu test × top-3 × 3 strategies. Coherence lấy 10 vị trí đầu tiên theo `(document_id, page)` trong các trang manifest; tại mỗi trang dùng anchor bảy từ đầu tiên bắt đầu từ word 40 trở đi, đánh giá chunk chứa anchor theo thang 0–2. Grounding đánh giá top-3 cho 5 câu test theo thang 0 = không có evidence, 1 = một phần/thiếu điều kiện, 2 = đủ evidence và truy được nguồn. Với A/B, filter compliance không áp dụng vì các strategy không bật filter; số trong ngoặc chỉ kết quả tình cờ khớp chapter. C bật filter và tuân thủ 3/3 câu test có filter.

## Phân tích và kết luận
- So sánh A–B và B–C; 2 ví dụ sai có ID câu và nguồn: A cao hơn B trên dev (Evidence Hit@3 14/15 so với 7/15) và test (5/5 so với 2/5). B tạo 167 chunks, median 84 từ; A tạo 110 chunks. B/C có 31 chunk chỉ 6 từ trở xuống, nhiều chunk chỉ chứa tiêu đề, làm retrieval dễ trả về kết quả không có evidence. q09 (Environmental and Infectious Causes of Malignancy, PDF p.1): B/C truy hồi chunk tiêu đề và bỏ lỡ evidence bắt đầu “Up to two-thirds...”. q20 (The Pathology of Cancer, PDF p.1): B/C cũng trả về chunk tiêu đề, bỏ lỡ evidence về precursor lesion. So sánh B–C cô lập tác động filter: cùng chunk/vectors, C tăng tuân thủ filter trong test từ 2/3 kết quả tình cờ ở B lên 3/3, nhưng Hit@1/Hit@3 không đổi (1/5, 2/5); filter không sửa vấn đề chunk tiêu đề.
- Chiến lược chọn, đánh đổi, điều kiện áp dụng: Chọn A bằng tiêu chí dev đã định trước; A đạt 5/5 Evidence Hit@3 trên test nhỏ này. A phù hợp hơn trong thí nghiệm hiện tại khi ưu tiên thu hồi evidence, dù cửa sổ cố định có thể cắt giữa đoạn/câu. B/C giữ ranh giới đoạn tốt hơn theo thiết kế nhưng các đoạn ngắn/tiêu đề gây retrieval kém. C hữu ích khi truy vấn chỉ rõ một chapter và cần ép kết quả trong phạm vi đó; không cải thiện chất lượng evidence nếu chunking chưa tốt. Không thể kết luận A tối ưu tổng quát từ 5 câu test.
- Hạn chế của dữ liệu, nhãn, test nhỏ, đo thời gian/chi phí: Test chỉ có 5 câu, cùng corpus/chủ đề; split theo câu hỏi chứ không theo tài liệu/trang, và có các chương/trang liên quan giữa dev/test nên có nguy cơ tương quan. Corpus chỉ gồm 7 chương và không đại diện mọi dạng tài liệu PDF. Nhãn coherence/grounding có hỗ trợ AI, dù đã được sinh viên xem lại và chấp thuận; vẫn có yếu tố chủ quan. Exact quote có thể bỏ sót evidence tương đương. Thời gian chỉ từ một lần chạy local, không tách thành benchmark phần cứng ổn định. Chi phí API chưa kiểm tra từ provider billing, không được suy ra từ thời gian chạy.
- Nếu làm lại hoặc mở rộng, cải tiến gì? Tăng benchmark và chia split theo chương/tài liệu để giảm tương quan; xây dựng thêm câu hỏi đa chương và kiểm tra rõ yêu cầu filter. Sửa paragraph chunking để bỏ/gộp title-only chunks và tránh đoạn cực ngắn, sau đó chạy lại dev/test mới với tiêu chí chọn được ghi trước. Chuẩn hóa hướng dẫn chấm và mời người chấm độc lập cho coherence/grounding; lưu chi phí thực tế từ trang billing. Thử hybrid retrieval hoặc reranking như thí nghiệm riêng, không thay đổi nhiều yếu tố cùng lúc.
