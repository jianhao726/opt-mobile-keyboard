"""Launch the HTML keyboard with `python web_app.py`."""

import argparse
import json
import logging
import os
import time
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parent
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
MAX_BODY_BYTES = 128 * 1024
MAX_TEXT_LENGTH = 2048
MAX_CACHE_SIZE = 1024
MODEL_LOCK = Lock()


class InvalidRequest(ValueError):
    pass


class ModelUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_model(variant):
    # The UI and its launch instructions work even before the ML stack is installed.
    try:
        from predictor import OPTKeyboardPredictor
    except ImportError as exc:
        raise ModelUnavailable("Install the packages in requirements.txt, then retry predictions.") from exc

    try:
        return OPTKeyboardPredictor.from_variant(variant=variant, device="cpu")
    except Exception as exc:
        logging.exception("Could not load the %s keyboard model", variant)
        raise ModelUnavailable(
            "Could not load OPT-125M. Check the model files and the first-run download, then retry."
        ) from exc


def is_valid_text(value):
    return isinstance(value, str) and len(value) <= MAX_TEXT_LENGTH


def clean_vocabulary(cache):
    if not isinstance(cache, dict) or len(cache) > MAX_CACHE_SIZE:
        raise InvalidRequest(f"Send up to {MAX_CACHE_SIZE} vocabulary entries.")
    return {
        word.lower(): min(count, 1_000_000)
        for word, count in cache.items()
        if 2 <= len(word) <= 64 and word.isalpha() and type(count) is int and count > 0
    }


def candidate(pair):
    word, _ = pair
    return {"word": word}


def predict(payload):
    if not isinstance(payload, dict):
        raise InvalidRequest("Send a JSON object.")
    started = time.perf_counter()
    draft = payload.get("draft", "")
    context = payload.get("context", [])
    variant = payload.get("variant", "lora_int8")
    weight = payload.get("cacheWeight", 0.35)
    spelling = payload.get("spelling", True)

    if not is_valid_text(draft):
        raise InvalidRequest(f"Drafts can contain up to {MAX_TEXT_LENGTH} characters.")
    if not isinstance(context, list) or len(context) > 2 or not all(map(is_valid_text, context)):
        raise InvalidRequest("Send the last two messages or fewer as context.")
    if variant not in ("lora_int8", "lora_fp32", "base"):
        raise InvalidRequest("Choose a supported model variant.")
    if type(weight) not in (int, float) or not 0 <= weight <= 1:
        raise InvalidRequest("Choose a personalization boost between 0 and 1.")
    if not isinstance(spelling, bool):
        raise InvalidRequest("Spelling must be true or false.")
    cache = clean_vocabulary(payload.get("cache", {}))
    trusted = payload.get("trustedWords", [])
    if not isinstance(trusted, list) or len(trusted) > MAX_CACHE_SIZE or not all(
        isinstance(w, str) and 2 <= len(w) <= 64 and w.isalpha() for w in trusted
    ):
        raise InvalidRequest("Send up to 1024 confirmed vocabulary words.")
    previous = " ".join(message.strip() for message in context if message.strip())
    prefix = previous + " " if previous else ""
    model_input = prefix + (draft if draft.strip() or not prefix else "")

    # Each inference uses exactly one profile. The predictor also mutates its logit cache.
    with MODEL_LOCK:
        model = load_model(variant)
        model.user_freq_cache = cache
        model.trusted_words = {word.lower() for word in trusted}
        try:
            suggestions, latency, mode = model.suggest(model_input, top_k=3, cache_weight=weight)
            punctuation = model.suggest_punctuation(model_input) if draft.strip() else None
            corrections = model.spelling_suggestions(draft, prefix) if spelling and draft else []
        finally:
            model.user_freq_cache = {}
            model.trusted_words = set()
    return {
        "suggestions": [candidate(pair) for pair in suggestions],
        "punctuation": candidate(punctuation) if punctuation else None,
        "corrections": corrections,
        "mode": mode,
        "latencyMs": round((time.perf_counter() - started) * 1000, 2),
        "modelLabel": model.variant_info["label"],
        "actualVariant": model.variant,
        "warning": getattr(model, "load_warning", ""),
        "footprint": model.variant_info["size_label"],
    }


class KeyboardHandler(BaseHTTPRequestHandler):
    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def reply(self, status, content_type, body):
        self.send_response(status)
        self.send_cors_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # The user may have typed again and cancelled this request.

    def reply_json(self, status, data):
        self.reply(status, "application/json; charset=utf-8", json.dumps(data).encode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        asset = STATIC_FILES.get(urlsplit(self.path).path)
        if asset is None:
            self.reply_json(HTTPStatus.NOT_FOUND, {"error": "Page not found."})
            return
        filename, mime_type = asset
        try:
            body = (ROOT / "web" / filename).read_bytes()
        except FileNotFoundError:
            self.reply_json(HTTPStatus.NOT_FOUND, {"error": "File not found."})
            return
        self.reply(HTTPStatus.OK, mime_type, body)

    def do_POST(self):
        if urlsplit(self.path).path != "/api/predict":
            self.reply_json(HTTPStatus.NOT_FOUND, {"error": "Endpoint not found."})
            return
        if self.headers.get_content_type() != "application/json":
            self.reply_json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "Use application/json."})
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            length = -1
        if not 0 < length <= MAX_BODY_BYTES:
            status = HTTPStatus.REQUEST_ENTITY_TOO_LARGE if length > MAX_BODY_BYTES else HTTPStatus.BAD_REQUEST
            self.reply_json(status, {"error": "Send a JSON body of up to 128 KiB."})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, RecursionError):
            self.reply_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON body."})
            return
        try:
            result = predict(payload)
        except InvalidRequest as exc:
            self.reply_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except ModelUnavailable as exc:
            self.reply_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc)})
        except Exception:
            logging.exception("Keyboard prediction failed")
            self.reply_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Prediction failed. Try again."})
        else:
            self.reply_json(HTTPStatus.OK, result)

    def do_HEAD(self):
        self.send_response(HTTPStatus.OK)
        self.send_cors_headers()
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()


def main():
    on_hf = "SPACE_ID" in os.environ
    on_cloud = (
        os.environ.get("RENDER", "").lower() == "true"
        or "PORT" in os.environ
        or on_hf
    )
    host = "0.0.0.0" if on_cloud else "127.0.0.1"
    if "PORT" in os.environ:
        default_port = int(os.environ["PORT"])
    elif on_hf:
        default_port = 7860
    elif on_cloud:
        default_port = 10000
    else:
        default_port = 8765

    parser = argparse.ArgumentParser(description="Run the HTML keyboard and prediction API.")
    parser.add_argument("--host", default=host, help="Override the listening host")
    parser.add_argument("--port", type=int, default=default_port, help="Override the listening port")
    args = parser.parse_args()

    os.chdir(ROOT)  # The original predictor resolves its LoRA adapter from the working directory.
    with ThreadingHTTPServer((args.host, args.port), KeyboardHandler) as server:
        print(f"Keyboard Lab is running at http://{args.host}:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
