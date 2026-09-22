"""Phase 3: embed page chunks locally, without indexing or answer generation."""

import hashlib
import json
import math
from pathlib import Path
from urllib.error import URLError

from .runtime import load_config, local_server


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_chunks(path: Path) -> list[dict]:
    chunks, seen = [], set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Line {number}: expected a chunk object.")
        for key in ("chunk_id", "source", "text"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Line {number}: {key} must be a nonempty string.")
        if type(row.get("page")) is not int or row["page"] < 1:
            raise ValueError(f"Line {number}: page must be a positive integer.")
        if row["chunk_id"] in seen:
            raise ValueError(f"Duplicate chunk ID: {row['chunk_id']}")
        seen.add(row["chunk_id"])
        chunks.append(row)
    if not chunks:
        raise ValueError("No chunks to embed.")
    return chunks


def normalize(vector: list) -> list[float]:
    if len(vector) != 1024 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in vector):
        raise ValueError("Expected 1024 finite embedding values from Qwen3-Embedding-0.6B.")
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        raise ValueError("The server returned a zero embedding.")
    return [x / norm for x in vector]


def embed_chunks(input_path: Path, output_dir: Path, config_path: Path,
                 context_size: int = 4096, limit: int | None = None) -> dict:
    if not 2 <= context_size <= 32768:
        raise ValueError("Context size must be between 2 and 32768 tokens.")
    if limit is not None and limit < 1:
        raise ValueError("Limit must be positive.")
    config = load_config(config_path)
    chunks = read_chunks(input_path)
    if limit is not None:
        chunks = chunks[:limit]
    # A fresh output directory protects previous runs and separates partial results.
    output_dir.mkdir(parents=True, exist_ok=False)
    print("Loading external llama.cpp model...", flush=True)
    with local_server(config, context_size, output_dir / "llama-server.log") as client:
        end_token = client.request("/tokenize", {
            "content": "<|endoftext|>", "add_special": False, "parse_special": True,
        })["tokens"]
        if end_token != [151643]:
            raise ValueError("Unexpected Qwen tokenizer: document-ending token does not match.")

        tokenized = []
        for chunk in chunks:
            # Tokenize plain document text without a chat template or query instruction.
            tokens = client.tokens(chunk["text"]) + end_token
            if len(tokens) > context_size:
                raise ValueError(
                    f"{chunk['chunk_id']} has {len(tokens)} tokens; configured limit is "
                    f"{context_size}. No truncation performed. Choose a larger --context-size."
                )
            tokenized.append(tokens)
        maximum = max(map(len, tokenized))
        print(f"Checked {len(chunks)} chunks; longest input: {maximum} tokens (including end token).", flush=True)

        with (output_dir / "embeddings.jsonl").open("x", encoding="utf-8") as output:
            for index, (chunk, tokens) in enumerate(zip(chunks, tokenized), 1):
                result = client.request("/v1/embeddings", {
                    "model": "book-embeddings", "input": tokens, "encoding_format": "float",
                })
                if result.get("usage", {}).get("prompt_tokens") != len(tokens):
                    raise ValueError("Server token count differs from the full prepared input.")
                rows = result.get("data", [])
                if len(rows) != 1:
                    raise ValueError("Expected one embedding per page.")
                row = {**chunk, "token_count": len(tokens), "embedding": normalize(rows[0]["embedding"])}
                output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                if index == 1 or index % 25 == 0 or index == len(chunks):
                    print(f"Embedded {index}/{len(chunks)} pages", flush=True)

    # Write manifest last: its presence means the run finished successfully.
    model_hash = file_sha256(Path(config["embedding_model_path"]))
    summary = {
        "model": "Qwen3-Embedding-0.6B", "model_file": Path(config["embedding_model_path"]).name,
        "model_sha256": model_hash,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "chunks": len(chunks), "dimensions": 1024, "normalized": True,
        "pooling": "last", "document_end_token": 151643,
        "document_prompt": "", "context_size": context_size, "max_tokens": maximum,
        "runtime": "external llama-server", "format": "jsonl", "limit": limit,
    }
    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
