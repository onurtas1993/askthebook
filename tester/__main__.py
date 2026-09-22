import argparse
import json
import os
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4


def main():
    parser = argparse.ArgumentParser(description="Test the running AskTheBook HTTP server")
    parser.add_argument("--url", default=os.getenv("ASKTHEBOOK_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--timeout", type=float, default=3600, help="Seconds to wait for preparation or an answer")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("documents")
    upload = commands.add_parser("add")
    upload.add_argument("pdf", type=Path)
    question = commands.add_parser("ask")
    question.add_argument("question")
    question.add_argument("--document", required=True)
    question.add_argument("--model")
    question.add_argument("--top-k", type=int)
    args = parser.parse_args()
    body, headers, route = None, {}, "/documents"
    try:
        if args.command == "add":
            boundary = uuid4().hex
            filename = args.pdf.name
            if any(c in filename for c in '\r\n"'):
                raise ValueError("Invalid filename.")
            body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\nContent-Type: application/pdf\r\n\r\n'.encode()
                    + args.pdf.read_bytes() + f'\r\n--{boundary}--\r\n'.encode())
            headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        elif args.command == "ask":
            route = "/ask"
            payload = {"question": args.question, "document_id": args.document}
            if args.model is not None:
                payload["model"] = args.model
            if args.top_k is not None:
                payload["top_k"] = args.top_k
            body = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        request = Request(args.url.rstrip("/") + route, data=body, headers=headers)
        with build_opener(ProxyHandler({})).open(request, timeout=args.timeout) as response:
            result = json.load(response)
    except HTTPError as error:
        parser.exit(1, f"Server returned HTTP {error.code}: {error.read().decode('utf-8', errors='replace')}\n")
    except (OSError, URLError, ValueError) as error:
        parser.exit(1, f"Request failed: {error}\nCheck that the server is running and --url is correct.\n")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
