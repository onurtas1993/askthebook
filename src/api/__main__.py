"""Launch the server; processing is accessible only through HTTP clients."""

import argparse
import json
import sys
from pathlib import Path

import uvicorn

from .app import create_app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[2]
    parser.add_argument("--config", type=Path, default=base / "config.json")
    args = parser.parse_args()
    try:
        settings = json.loads(args.config.read_text(encoding="utf-8"))
    except FileNotFoundError:
        parser.error(f"Configuration not found: {args.config}. Keep config.json beside the executable or use --config PATH.")
    host = settings.get("api_host", "127.0.0.1")
    port = settings.get("api_port", 8000)
    if host not in ("127.0.0.1", "localhost", "::1"):
        parser.error("This unauthenticated first version supports loopback hosting only.")
    if type(port) is not int or not 1 <= port <= 65535:
        parser.error("api_port must be between 1 and 65535.")
    uvicorn.run(create_app(args.config), host=host, port=port, workers=1)


if __name__ == "__main__":
    main()
