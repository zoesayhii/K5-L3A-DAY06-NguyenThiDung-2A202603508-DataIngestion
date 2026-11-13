import os
from pypdf import PdfReader

DATA_DIR = "data/"

# Định nghĩa 3 cấu hình chunking theo chuẩn bài lab
CHUNK_CONFIGS = {
    "config_1_small_fine": {"chunk_size": 256, "chunk_overlap": 50},
    "config_2_medium_standard": {"chunk_size": 512, "chunk_overlap": 100},
    "config_3_large_context": {"chunk_size": 1024, "chunk_overlap": 128},
}

def simple_word_splitter(text, chunk_size, chunk_overlap):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - chunk_overlap
    return chunks

def test_chunking_strategies():
    # Lấy thử file cancer-biology.pdf để test chunking
    sample_file = "cancer-biology.pdf"
    file_path = os.path.join(DATA_DIR, sample_file)
    
    if not os.path.exists(file_path):
        print(f"Không tìm thấy file mẫu {sample_file} trong thư mục {DATA_DIR}")
        return

    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            full_text += extracted + "\n"

    print(f"--- Thử nghiệm Chunking trên file: {sample_file} ---")
    for config_name, cfg in CHUNK_CONFIGS.items():
        chunks = simple_word_splitter(
            full_text, 
            chunk_size=cfg["chunk_size"], 
            chunk_overlap=cfg["chunk_overlap"]
        )
        print(f"[{config_name}] Chunk Size: {cfg['chunk_size']}, Overlap: {cfg['chunk_overlap']} => Tổng số tạo ra: {len(chunks)} chunks")

if __name__ == "__main__":
    test_chunking_strategies()