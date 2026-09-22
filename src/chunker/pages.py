"""Phase 2 baseline: one nonempty PDF page per chunk."""

import json
from pathlib import Path


def chunk_pages(input_path: Path, output_path: Path) -> dict:
    """Validate page records and write chunks without changing their text."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must be different files.")

    chunks = []
    seen = set()
    page_count = 0
    skipped = 0
    # Validate before writing so malformed input does not leave partial output.
    with input_path.open(encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Line {line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number}: expected a page object.")
            source = record.get("source")
            page = record.get("page")
            text = record.get("text")
            if not isinstance(source, str) or not source.strip():
                raise ValueError(f"Line {line_number}: source must be a nonempty string.")
            if type(page) is not int or page < 1:
                raise ValueError(f"Line {line_number}: page must be a positive integer.")
            if not isinstance(text, str):
                raise ValueError(f"Line {line_number}: text must be a string.")
            identity = (source, page)
            if identity in seen:
                raise ValueError(f"Line {line_number}: duplicate source/page: {identity}.")
            seen.add(identity)
            page_count += 1
            if not text.strip():
                skipped += 1
                continue
            chunks.append({
                "chunk_id": f"{source}:page:{page}",
                "source": source,
                "page": page,
                "text": text,
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as output:
        for chunk in chunks:
            output.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    return {"pages": page_count, "chunks": len(chunks), "skipped_blank_pages": skipped}
