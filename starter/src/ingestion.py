"""Read and normalize source data. TODO: implement functions."""
import unicodedata
import re
from pypdf import PdfReader
from pathlib import Path
from .models import DocumentPage


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """TODO: extract each physical page; keep its page number and source.

    Detect and record pages with no text. OCR, when needed, is a separate step.
    """
    # raise NotImplementedError("TODO: load_pdf")
    pages = []
    reader = PdfReader(str(path))
    total_pages = len(reader.pages)

    for idx, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        text = normalize_text(raw_text)
        is_empty = len(text.strip()) == 0

        pages.append(DocumentPage(
            document_id=document_id,
            source=source,
            page=idx + 1,
            text=text,
            metadata={
                "is_empty": is_empty,
                "total_pages": total_pages,
                "document_id": document_id,
            }
        ))
    return pages


def normalize_text(text: str) -> str:
    """TODO: normalize Unicode and whitespace without discarding useful structure."""
    # raise NotImplementedError("TODO: normalize_text")
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
