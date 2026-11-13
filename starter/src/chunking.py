"""Implement fixed-size and structure-aware chunking."""
import hashlib
from .models import Chunk, DocumentPage


def _chunk_id(page: DocumentPage, strategy: str, chunk_index: int, text: str) -> str:
    identity = f"{page.document_id}\0{page.page}\0{strategy}\0{chunk_index}\0{text}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split text into overlapping windows and retain page metadata."""
    if size <= 0:
        raise ValueError("size must be greater than 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be between 0 and size - 1")

    words = page.text.split()
    chunks = []
    i = 0
    chunk_idx = 0
    while i < len(words):
        chunk_words = words[i:i + size]
        chunk_text = " ".join(chunk_words)
        strategy = "fixed_size"
        chunk = Chunk(
            chunk_id=_chunk_id(page, strategy, chunk_idx, chunk_text),
            document_id=page.document_id,
            source=page.source,
            page=page.page,
            text=chunk_text,
            metadata={**page.metadata, "chunk_index": chunk_idx, "strategy": strategy}
        )
        chunks.append(chunk)
        i += size - overlap
        chunk_idx += 1
    return chunks


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Group paragraphs, split oversized paragraphs and retain metadata."""
    if max_size <= 0:
        raise ValueError("max_size must be greater than 0")

    paragraphs = page.text.split("\n\n")
    chunks = []
    chunk_idx = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        words = para.split()
        if len(words) <= max_size:
            strategy = "by_paragraph"
            chunk = Chunk(
                chunk_id=_chunk_id(page, strategy, chunk_idx, para),
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=para,
                metadata={**page.metadata, "chunk_index": chunk_idx, "strategy": strategy}
            )
            chunks.append(chunk)
            chunk_idx += 1
        else:
            i = 0
            while i < len(words):
                part_words = words[i:i + max_size]
                part_text = " ".join(part_words)
                strategy = "by_paragraph"
                chunk = Chunk(
                    chunk_id=_chunk_id(page, strategy, chunk_idx, part_text),
                    document_id=page.document_id,
                    source=page.source,
                    page=page.page,
                    text=part_text,
                    metadata={**page.metadata, "chunk_index": chunk_idx, "strategy": strategy}
                )
                chunks.append(chunk)
                i += max_size
                chunk_idx += 1
    return chunks