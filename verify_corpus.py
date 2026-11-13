import csv
from pathlib import Path

from pypdf import PdfReader
from starter.src.ingestion import normalize_text

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "data" / "manifest.csv"


def verify_pdf_corpus():
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Corpus manifest is missing: {MANIFEST_PATH}")

    with MANIFEST_PATH.open(encoding="utf-8", newline="") as file:
        manifest = list(csv.DictReader(file))
    if not manifest:
        raise ValueError(f"Corpus manifest contains no documents: {MANIFEST_PATH}")

    physical_pages = 0
    content_pages = 0
    content_characters = 0
    for row in manifest:
        source_id = row["source_id"]
        path = (ROOT / row["source_path"]).resolve()
        if ROOT / "data" not in path.parents or not path.is_file():
            raise FileNotFoundError(f"Invalid or missing PDF for {source_id}: {path}")

        reader = PdfReader(str(path))
        physical_pages += len(reader.pages)
        selected_pages = {
            int(page.strip()) for page in row["content_page_numbers"].split(",") if page.strip()
        }
        if not selected_pages or max(selected_pages) > len(reader.pages):
            raise ValueError(f"Invalid content page selection for {source_id}")

        characters = 0
        for page_number in selected_pages:
            text = normalize_text(reader.pages[page_number - 1].extract_text() or "")
            if not text:
                raise ValueError(f"Selected page is empty: {source_id} page {page_number}")
            characters += len(text)

        if len(selected_pages) != int(row["content_page_count"]):
            raise ValueError(f"Content page count mismatch for {source_id}")
        content_pages += len(selected_pages)
        content_characters += characters
        print(
            f"- {row['title']}: {len(reader.pages)} physical pages; "
            f"{len(selected_pages)} content pages; {characters} normalized characters "
            f"(manifest: {row['extracted_characters']})"
        )

    print(
        f"\nCorpus total: {physical_pages} physical pages; {content_pages} eligible content pages; "
        f"{content_characters} normalized characters."
    )
    if content_pages < 30:
        raise ValueError(f"Corpus has {content_pages} content pages; at least 30 are required")
    print("Corpus meets the 30-content-page requirement.")


if __name__ == "__main__":
    verify_pdf_corpus()
