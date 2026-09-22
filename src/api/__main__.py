"""Launch the server; processing is accessible only through HTTP clients."""

import argparse
import json
from pathlib import Path

import uvicorn

from .app import create_app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "config.local.json")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    host = settings.get("api_host", "127.0.0.1")
    port = settings.get("api_port", 8000)
    if host not in ("127.0.0.1", "localhost", "::1"):
        parser.error("This unauthenticated first version supports loopback hosting only.")
    if type(port) is not int or not 1 <= port <= 65535:
        parser.error("api_port must be between 1 and 65535.")
    uvicorn.run(create_app(args.config), host=host, port=port, workers=1)


if __name__ == "__main__":
    main()
