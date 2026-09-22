"""Phases 5–6: local question embedding, Top-K retrieval, and chunk inspection."""

import json
from pathlib import Path
from urllib.error import URLError

import numpy as np

from src.embedder.pipeline import file_sha256, normalize
from src.embedder.runtime import load_config, local_server
from src.indexer.index import load_index


# The model publisher's default retrieval query prompt; documents use no prompt.
QUERY_PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"


def embed_question(client, question: str, context_size: int) -> tuple[list[float], int]:
    if not question.strip():
        raise ValueError("Question must not be empty.")
    end = client.request("/tokenize", {
        "content": "<|endoftext|>", "add_special": False, "parse_special": True,
    })["tokens"]
    if end != [151643]:
        raise ValueError("Unexpected Qwen document-ending token.")
    tokens = client.tokens(QUERY_PREFIX + question) + end
    if len(tokens) > context_size:
        raise ValueError(f"Question plus instruction needs {len(tokens)} tokens; limit is {context_size}. No truncation performed.")
    result = client.request("/v1/embeddings", {
        "model": "book-embeddings", "input": tokens, "encoding_format": "float",
    })
    if result.get("usage", {}).get("prompt_tokens") != len(tokens):
        raise ValueError("Server did not process the full question input.")
    if len(result.get("data", [])) != 1:
        raise ValueError("Expected exactly one question embedding.")
    return normalize(result["data"][0]["embedding"]), len(tokens)


def search_vector(index, mapping: list[dict], vector: list[float], top_k: int) -> list[dict]:
    if type(top_k) is not int or top_k < 1:
        raise ValueError("Top-K must be a positive integer.")
    query = np.ascontiguousarray([vector], dtype=np.float32)
    if query.shape != (1, index.d) or not np.isfinite(query).all():
        raise ValueError("Question vector has invalid dimensions or values.")
    if not np.isclose(np.linalg.norm(query), 1, atol=1e-5, rtol=0):
        raise ValueError("Question vector must be normalized.")
    # Avoid FAISS's -1 sentinel when K exceeds the number of indexed pages.
    scores, labels = index.search(query, min(top_k, index.ntotal))
    return [
        {**mapping[int(label)], "rank": rank, "score": float(score)}
        for rank, (label, score) in enumerate(zip(labels[0], scores[0]), 1)
    ]


def retrieve(question: str, index_dir: Path, config_path: Path, output_dir: Path,
             top_k: int = 5) -> dict:
    if not question.strip() or type(top_k) is not int or top_k < 1:
        raise ValueError("Provide a nonempty question and a positive Top-K.")
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    index, mapping, manifest = load_index(index_dir)
    settings = manifest["embedding_manifest"]
    if (settings.get("model") != "Qwen3-Embedding-0.6B"
            or settings.get("pooling") != "last"
            or settings.get("document_end_token") != 151643
            or settings.get("normalized") is not True
            or settings.get("document_prompt") != ""
            or index.d != 1024):
        raise ValueError("Index embedding settings are incompatible with this Qwen retriever.")
    context_size = settings["context_size"]
    if type(context_size) is not int or not 2 <= context_size <= 32768:
        raise ValueError("Invalid context size in embedding manifest.")
    config = load_config(config_path)
    if file_sha256(Path(config["embedding_model_path"])) != settings["model_sha256"]:
        raise ValueError("Configured GGUF differs from the model used to build this index.")
    output_dir.mkdir(parents=True, exist_ok=False)
    with local_server(config, context_size, output_dir / "llama-server.log") as client:
        vector, tokens = embed_question(client, question, context_size)
    results = search_vector(index, mapping, vector, top_k)
    report = {
        "question": question, "query_prefix": QUERY_PREFIX, "query_tokens": tokens,
        "top_k_requested": top_k, "top_k_returned": len(results),
        "model_sha256": settings["model_sha256"],
        "index_sha256": manifest["file_sha256"]["index.faiss"],
        "similarity": "cosine", "results": results,
    }
    (output_dir / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return report
