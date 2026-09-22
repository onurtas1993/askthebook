"""Phase 8: compose retrieval and generation without hiding their artifacts."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from uuid import uuid4

from src.answerer.generate import generate, load_settings
from src.embedder.runtime import load_config
from src.retriever.search import retrieve


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config.local.json"


def configured_path(settings: dict, key: str, config_path: Path) -> Path:
    value = settings.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Set {key} in the local configuration.")
    path = Path(value).expanduser()
    return path if path.is_absolute() else config_path.parent / path


def ask(question: str, *, config_path: Path = DEFAULT_CONFIG,
        top_k: int | None = None, output_dir: Path | None = None,
        index_dir: Path | None = None, model: str | None = None) -> dict:
    """Return an answer, evidence, and diagnostics for one independent question."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a nonempty string.")
    config_path = Path(config_path).resolve()
    settings = load_settings(config_path)
    load_config(config_path)  # Catch missing local executables/model before making a run.
    index_dir = Path(index_dir) if index_dir is not None else configured_path(settings, "rag_index_dir", config_path)
    k = settings.get("rag_top_k", 5) if top_k is None else top_k
    if type(k) is not int or k < 1:
        raise ValueError("Top-K must be a positive integer.")
    if output_dir is None:
        root = configured_path(settings, "rag_output_dir", config_path)
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        output_dir = root / name
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    status_path = output_dir / "run.json"
    status = {"question": question, "top_k": k, "stage": "retrieval", "status": "running"}

    def save_status():
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    save_status()
    started = time.monotonic()
    try:
        retrieval = retrieve(question, index_dir, config_path, output_dir / "retrieval", k)
        status["retrieval_seconds"] = round(time.monotonic() - started, 3)
        status["stage"] = "generation"
        save_status()
        # retrieve() has stopped its Qwen subprocess before Gemma is called.
        generation_started = time.monotonic()
        answer = generate(output_dir / "retrieval" / "results.json", config_path, output_dir / "generation", model=model)
        status["generation_seconds"] = round(time.monotonic() - generation_started, 3)
        result = {**answer, "sources": retrieval["results"], "output_dir": str(output_dir),
                  "retrieval_seconds": status["retrieval_seconds"],
                  "generation_seconds": status["generation_seconds"]}
        (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        status.update(status="complete", stage="complete")
        save_status()
        return result
    except (Exception, KeyboardInterrupt) as error:
        status.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}")
        save_status()
        raise
