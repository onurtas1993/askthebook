"""Phase 4: build, persist, and reload an exact inner-product FAISS index."""

import hashlib
import json
from pathlib import Path

import faiss
import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_embeddings(directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    dimensions, count = manifest.get("dimensions"), manifest.get("chunks")
    if type(dimensions) is not int or dimensions < 1 or type(count) is not int or count < 1:
        raise ValueError("Embedding manifest must specify positive dimensions and chunk count.")
    if manifest.get("normalized") is not True:
        raise ValueError("This index requires normalized embeddings for cosine similarity.")
    vectors, mapping, seen = [], [], set()
    with (directory / "embeddings.jsonl").open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Line {number}: expected an embedding object.")
            for key in ("chunk_id", "source", "text"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"Line {number}: missing or invalid {key}.")
            if type(row.get("page")) is not int or row["page"] < 1:
                raise ValueError(f"Line {number}: invalid page number.")
            if row["chunk_id"] in seen:
                raise ValueError(f"Duplicate chunk ID: {row['chunk_id']}")
            seen.add(row["chunk_id"])
            vector = row.get("embedding")
            if not isinstance(vector, list) or len(vector) != dimensions:
                raise ValueError(f"Line {number}: incorrect vector dimension.")
            if not all(type(value) in (int, float) for value in vector):
                raise ValueError(f"Line {number}: vector must contain numbers.")
            vectors.append(vector)
            # FAISS labels are insertion positions, not PDF page numbers.
            mapping.append({"vector_id": len(mapping), **{
                key: row[key] for key in ("chunk_id", "source", "page", "text")
            }})
    if len(vectors) != count:
        raise ValueError("Embedding row count does not match its completion manifest.")
    matrix = np.ascontiguousarray(vectors, dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings contain non-finite values after float32 conversion.")
    if not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5, rtol=0):
        raise ValueError("Embeddings must have unit length; regenerate invalid embeddings.")
    return matrix, mapping, manifest


def load_index(directory: Path):
    """Reload a completed local index and its matching ordered metadata."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for filename in ("index.faiss", "chunks.jsonl"):
        if sha256(directory / filename) != manifest["file_sha256"][filename]:
            raise ValueError(f"{filename} does not match the saved manifest.")
    # Python file I/O also supports Windows paths containing Unicode characters.
    index = faiss.deserialize_index(np.frombuffer((directory / "index.faiss").read_bytes(), dtype=np.uint8))
    mapping = [json.loads(line) for line in (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    if not isinstance(index, faiss.IndexFlatIP) or index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError("Expected an exact inner-product index.")
    if index.ntotal != manifest["vectors"] or index.d != manifest["dimensions"] or len(mapping) != index.ntotal:
        raise ValueError("Index dimensions or count do not match metadata.")
    if [row["vector_id"] for row in mapping] != list(range(index.ntotal)):
        raise ValueError("Chunk mapping is not in vector insertion order.")
    if len({row["chunk_id"] for row in mapping}) != len(mapping):
        raise ValueError("Duplicate chunk IDs in mapping.")
    return index, mapping, manifest


def build_index(input_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    matrix, mapping, embedding_manifest = load_embeddings(input_dir)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)  # Exact flat indexes require no training step.
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "index.faiss").write_bytes(faiss.serialize_index(index).tobytes())
    with (output_dir / "chunks.jsonl").open("x", encoding="utf-8") as output:
        for row in mapping:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Verify every vector and metadata row after disk serialization, before marking complete.
    restored = faiss.deserialize_index(np.frombuffer((output_dir / "index.faiss").read_bytes(), dtype=np.uint8))
    np.testing.assert_array_equal(restored.reconstruct_n(0, restored.ntotal), matrix)
    saved_mapping = [json.loads(line) for line in (output_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    if saved_mapping != mapping:
        raise ValueError("Saved chunk mapping differs from its input order.")
    manifest = {
        "index_type": "IndexFlatIP", "metric": "inner_product", "normalized": True,
        "similarity": "cosine", "dtype": "float32", "vectors": index.ntotal,
        "dimensions": index.d, "faiss_version": faiss.__version__,
        "numpy_version": np.__version__, "embedding_manifest": embedding_manifest,
        "embeddings_sha256": sha256(input_dir / "embeddings.jsonl"),
        "file_sha256": {name: sha256(output_dir / name) for name in ("index.faiss", "chunks.jsonl")},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
