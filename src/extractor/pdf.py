"""Phase 1: extract a PDF into inspectable, page-level JSONL records."""

import json
from pathlib import Path

import pymupdf


def extract_pdf(pdf_path: Path, output_path: Path, low_text_threshold: int = 50) -> dict:
    """Write every page, including blank ones; return extraction statistics."""
    if low_text_threshold < 0:
        raise ValueError("The low-text threshold must be nonnegative.")
    if pdf_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must be different files.")

    total_characters = 0
    low_text_pages = []
    with pymupdf.open(pdf_path) as document:
        if not document.is_pdf:
            raise ValueError("The input must be a PDF.")
        if document.needs_pass:
            raise ValueError("This PDF requires a password; use an unlocked copy.")

        page_count = len(document)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents accidentally overwriting an earlier extraction.
        with output_path.open("x", encoding="utf-8") as output:
            for page_number, page in enumerate(document, start=1):
                # Position-based reading order is a useful starting point for a novel.
                text = page.get_text("text", sort=True)
                record = {"source": pdf_path.name, "page": page_number, "text": text}
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_characters += len(text)
                if len(text.strip()) < low_text_threshold or not text.strip():
                    low_text_pages.append(page_number)

    return {
        "pages": page_count,
        "characters": total_characters,
        "low_text_pages": low_text_pages,
    }
