"""Minimal folder-backed document storage; only completed books are listed."""

import json
from pathlib import Path
import re
import shutil
from uuid import uuid4

from src.extractor.pdf import extract_pdf
from src.chunker.pages import chunk_pages
from src.embedder.pipeline import embed_chunks
from src.indexer.index import build_index


class DocumentStore:
    def __init__(self, config_path: Path):
        self.config_path = config_path.resolve()
        settings = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.root = self.resolve(settings.get("documents_dir", "data/documents"))
        self.existing_index = self.resolve(settings["rag_index_dir"]) if settings.get("rag_index_dir") else None

    def resolve(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.config_path.parent / path

    def list(self) -> list[dict]:
        documents = []
        # Expose the existing book without copying or regenerating its vectors.
        if self.existing_index and (self.existing_index / "manifest.json").is_file():
            manifest = json.loads((self.existing_index / "manifest.json").read_text(encoding="utf-8"))
            documents.append({"document_id": "existing", "name": self.existing_index.name,
                              "chunks": manifest["vectors"]})
        for metadata in sorted(self.root.glob("*/document.json")):
            if (metadata.parent / "index/manifest.json").is_file():
                documents.append(json.loads(metadata.read_text(encoding="utf-8")))
        return documents

    def index_for(self, document_id: str) -> Path:
        if document_id == "existing" and self.existing_index and (self.existing_index / "manifest.json").is_file():
            return self.existing_index
        if re.fullmatch(r"[0-9a-f]{32}", document_id):
            folder = self.root / document_id
            if (folder / "document.json").is_file() and (folder / "index/manifest.json").is_file():
                return folder / "index"
        raise LookupError("Document not found or not ready.")

    def prepare(self, stream, filename: str) -> dict:
        # Never interpret a client-supplied filename as a server filesystem path.
        name = filename.replace("\\", "/").rsplit("/", 1)[-1]
        if not name or Path(name).suffix.lower() != ".pdf" or any(c in name for c in '<>:"|?*\x00'):
            raise ValueError("Provide a valid PDF filename.")
        document_id = uuid4().hex
        folder = self.root / document_id
        raw = folder / "raw"
        raw.mkdir(parents=True)
        pdf = raw / name
        with pdf.open("xb") as output:
            shutil.copyfileobj(stream, output)
        with pdf.open("rb") as saved_pdf:
            if b"%PDF-" not in saved_pdf.read(1024):
                raise ValueError("The uploaded file has no PDF header.")
        extracted = folder / "pages.jsonl"
        chunks = folder / "chunks.jsonl"
        try:
            stats = extract_pdf(pdf, extracted)
        except RuntimeError as error:
            raise ValueError(f"Unable to extract PDF: {error}") from error
        chunk_pages(extracted, chunks)
        embed_chunks(chunks, folder / "embeddings", self.config_path)
        index = build_index(folder / "embeddings", folder / "index")
        result = {"document_id": document_id, "name": name, "pages": stats["pages"], "chunks": index["vectors"]}
        temporary = folder / "document.tmp"
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(folder / "document.json")
        return result
