"""Send one chat request to a running vLLM server and check the reply.

Usage: python scripts/smoke_test.py [--url http://localhost:8000] [--api-key KEY]
Exits 0 if the server answers with non-empty text, 1 otherwise.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request


def call(url: str, api_key: str | None, body: dict | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()
    base = args.url.rstrip("/")

    try:
        # Ask the server which model it serves, so the script works for any variant.
        model = call(f"{base}/v1/models", args.api_key)["data"][0]["id"]
        reply = call(
            f"{base}/v1/chat/completions",
            args.api_key,
            {
                "model": model,
                "messages": [
                    {"role": "user", "content": "What is the capital of France?"}
                ],
                "max_tokens": 20,
                "temperature": 0,
            },
        )
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"FAIL could not reach {base}: {e}")
        return 1

    text = reply["choices"][0]["message"]["content"] or ""
    if not text.strip():
        print(f"FAIL {model} returned an empty response")
        return 1

    tokens = reply["usage"]["completion_tokens"]
    print(f"OK   {model} ({tokens} tokens): {text.strip()!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
