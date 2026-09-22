"""Own a temporary, offline llama-server process; communicate over loopback HTTP."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


class Client:
    def __init__(self, port: int, key: str):
        self.url = f"http://127.0.0.1:{port}"
        self.key = key
        # Never route local book text through a configured system HTTP proxy.
        self.opener = build_opener(ProxyHandler({}))

    def request(self, route: str, body=None, timeout=120):
        request = Request(
            self.url + route,
            data=None if body is None else json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"},
        )
        with self.opener.open(request, timeout=timeout) as response:
            return json.load(response)

    def tokens(self, text: str) -> list[int]:
        return self.request("/tokenize", {
            "content": text, "add_special": False, "parse_special": False,
        })["tokens"]


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("llama_cpp_bin_dir", "embedding_model_path"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"Missing configuration setting: {key}")
        resolved = Path(config[key]).expanduser()
        if not resolved.is_absolute():
            resolved = path.resolve().parent / resolved
        config[key] = str(resolved)
    executable = Path(config["llama_cpp_bin_dir"]) / "llama-server.exe"
    if not executable.is_file():
        raise ValueError(f"Missing executable: {executable}")
    if not Path(config["embedding_model_path"]).is_file():
        raise ValueError("Configured embedding_model_path does not exist.")
    return config


@contextmanager
def local_server(config: dict, context_size: int, log_path: Path):
    # Ask Windows for an available loopback port, then pass it to our subprocess.
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    key = secrets.token_hex(24)
    client = Client(port, key)
    args = [
        str(Path(config["llama_cpp_bin_dir"]) / "llama-server.exe"),
        "--model", config["embedding_model_path"], "--offline",
        "--embedding", "--pooling", "last", "--alias", "book-embeddings",
        "--host", "127.0.0.1", "--port", str(port), "--api-key", key,
        "--ctx-size", str(context_size), "--batch-size", str(context_size),
        "--ubatch-size", str(context_size), "--parallel", "1",
        "--gpu-layers", "all", "--no-webui",
    ]
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            args, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            deadline = time.monotonic() + 120
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"llama-server exited. See {log_path}")
                try:
                    if client.request("/health", timeout=2).get("status") == "ok":
                        break
                except (HTTPError, URLError, TimeoutError):
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError(f"Model loading timed out. See {log_path}")
                time.sleep(0.25)
            yield client
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
