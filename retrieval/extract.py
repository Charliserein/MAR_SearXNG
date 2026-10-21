"""Bounded image verification and caption extraction with durable HTTP bodies."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import math
import os
from pathlib import Path
import tempfile
import threading
import time
from urllib.parse import urldefrag, urljoin, urlparse
import warnings

from bs4 import BeautifulSoup
import httpx
import numpy as np
from PIL import Image

from reverse_image.utils import compare_images

MAX_CANDIDATES = 5
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_HTML_BYTES = 4 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
EXTRACTION_VERSION = 1
_URL_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


class ExtractionError(RuntimeError):
    def __init__(self, reason: str, **details):
        super().__init__(reason)
        self.reason = reason
        self.details = details


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ExtractionError("time_budget_exceeded")
    return remaining


def _url(value):
    if not isinstance(value, str):
        raise ExtractionError("invalid_url")
    parsed = urlparse(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.netloc
            or parsed.username or parsed.password):
        raise ExtractionError("invalid_url")
    return urldefrag(value.strip())[0]


def _blob_metadata(path, body, response):
    return {
        "status": "ok", "sha256": hashlib.sha256(body).hexdigest(),
        "path": str(path), "bytes": len(body),
        "content_type": response.headers.get("content-type", ""),
        "final_url": str(response.url),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }


def _read_cached(cache, key, blob_dir, limit):
    cached = cache.get("http_body", key)
    if not cached or cached.get("status") != "ok":
        return None
    digest = cached.get("sha256", "")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)):
        return None
    # Resolve only content-addressed files inside the caller's blob directory.
    path = blob_dir / digest
    try:
        if path.stat().st_size > limit:
            raise ExtractionError("body_too_large", limit_bytes=limit, cache="hit")
        body = path.read_bytes()
    except (FileNotFoundError, OSError):
        return None
    if hashlib.sha256(body).hexdigest() != digest:
        return None
    return body, {**cached, "cache": "hit"}


def _fetch(url, *, client, cache, blob_dir, limit, timeout, deadline):
    url = _url(url)
    key = {"version": 1, "url": url}
    with _LOCKS_GUARD:
        lock = _URL_LOCKS.setdefault(url, threading.Lock())
    if not lock.acquire(timeout=_remaining(deadline)):
        raise ExtractionError("time_budget_exceeded")
    try:
        _remaining(deadline)
        cached = _read_cached(cache, key, blob_dir, limit)
        if cached is not None:
            return cached
        chunks, size = [], 0
        request_timeout = min(timeout, _remaining(deadline))
        with client.stream("GET", url, timeout=request_timeout) as response:
            response.raise_for_status()
            declared = response.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > limit:
                raise ExtractionError("body_too_large", limit_bytes=limit)
            # Do not buffer to a fixed chunk size: a slow stream must yield each
            # received chunk so that the wall-clock budget can be checked.
            for chunk in response.iter_bytes():
                _remaining(deadline)
                size += len(chunk)
                if size > limit:
                    raise ExtractionError("body_too_large", limit_bytes=limit)
                chunks.append(chunk)
            _remaining(deadline)
            body = b"".join(chunks)
            digest = hashlib.sha256(body).hexdigest()
            path = blob_dir / digest
            blob_dir.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix=".body-", dir=blob_dir)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(body)
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            metadata = _blob_metadata(path, body, response)
            cache.put("http_body", key, metadata)
            return body, {**metadata, "cache": "miss"}
    finally:
        lock.release()


def _clean(value):
    return " ".join(value.split()) if isinstance(value, str) else ""


def _image_urls(tag, page_url):
    for attribute in ("src", "data-src", "data-original", "data-lazy-src",
                      "data-hi-res-src", "data-raw-src"):
        value = tag.get(attribute)
        if isinstance(value, str) and value.strip():
            yield urldefrag(urljoin(page_url, value.strip()))[0]
    for attribute in ("srcset", "data-srcset", "data-lazy-srcset"):
        value = tag.get(attribute)
        if isinstance(value, str):
            for part in value.split(","):
                pieces = part.strip().split()
                if pieces:
                    yield urldefrag(urljoin(page_url, pieces[0]))[0]


def _caption(html, *, page_url, image_urls):
    soup = BeautifulSoup(html, "html.parser")
    title = _clean(soup.title.get_text(" ", strip=True)) if soup.title else ""
    for tag in soup.find_all("img"):
        if not set(_image_urls(tag, page_url)).intersection(image_urls):
            continue
        figure = tag.find_parent("figure")
        if figure is not None:
            caption = figure.find("figcaption")
            text = _clean(caption.get_text(" ", strip=True)) if caption else ""
            if text:
                return text, title, "figcaption"
        for attribute in ("alt", "title"):
            text = _clean(tag.get(attribute))
            if text:
                return text, title, attribute
    return "", title, ""


def _format_captions(entries):
    return "\n\n".join(
        f"[Re{i}] Source: {entry['source']}\nTitle: {entry['title']}\n"
        f"URL: {entry['page_url']}\nContent: {entry['caption']}"
        for i, entry in enumerate(entries, 1)
    )


def extract_matches_cached(image_path, matches, *, cache, blob_dir,
                           max_seconds=45, timeout=8, similarity_cutoff=10):
    """Verify at most five candidates without scanning other page images.

    Successful response bodies remain reusable even if decoding/extraction later
    fails. An error with no extracted evidence returns ``error``, never an empty
    success. When some evidence succeeds, error events remain attached and
    ``partial`` is true.
    """
    if (not isinstance(max_seconds, (int, float)) or max_seconds <= 0
            or not math.isfinite(max_seconds)
            or not isinstance(timeout, (int, float)) or timeout <= 0
            or not math.isfinite(timeout)):
        raise ValueError("max_seconds and timeout must be positive finite values")
    if not isinstance(similarity_cutoff, (int, float)) or similarity_cutoff <= 0:
        raise ValueError("similarity_cutoff must be positive")
    config = {
        "version": EXTRACTION_VERSION, "max_candidates": MAX_CANDIDATES,
        "max_seconds": max_seconds, "request_timeout": timeout,
        "max_image_bytes": MAX_IMAGE_BYTES, "max_html_bytes": MAX_HTML_BYTES,
        "max_image_pixels": MAX_IMAGE_PIXELS, "similarity_cutoff": similarity_cutoff,
    }
    matches = list(matches)
    blob_dir = Path(blob_dir).expanduser().resolve()
    deadline = time.monotonic() + max_seconds
    entries, events, seen = [], [], set()
    if len(matches) > MAX_CANDIDATES:
        events.append({"status": "limited", "stage": "candidate_selection",
                       "candidate_count": len(matches), "selected_count": MAX_CANDIDATES})
    client_options = {
        "trust_env": False, "follow_redirects": True, "max_redirects": 5,
        "headers": {"User-Agent": "Mozilla/5.0 (compatible; MARResearch/1.0)"},
    }
    if os.getenv("MAR_HTTP_PROXY"):
        client_options["proxy"] = os.environ["MAR_HTTP_PROXY"]
    try:
        with Image.open(image_path) as opened:
            if opened.width * opened.height > MAX_IMAGE_PIXELS:
                raise ExtractionError("image_dimensions_too_large")
            original = opened.convert("RGB")
        with original, httpx.Client(**client_options) as client:
            original_array = np.asarray(original)
            for index, match in enumerate(matches[:MAX_CANDIDATES]):
                context = {"page_url": match.link, "image_url": match.image_url,
                           "candidate_index": index}
                stage = "image_download"
                try:
                    _remaining(deadline)
                    image_url, page_url = _url(match.image_url), _url(match.link)
                    image_body, image_meta = _fetch(
                        image_url, client=client, cache=cache, blob_dir=blob_dir,
                        limit=MAX_IMAGE_BYTES, timeout=timeout, deadline=deadline)
                    stage = "image_decode"
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", Image.DecompressionBombWarning)
                        with Image.open(io.BytesIO(image_body)) as opened_candidate:
                            if opened_candidate.width * opened_candidate.height > MAX_IMAGE_PIXELS:
                                raise ExtractionError("image_dimensions_too_large")
                            candidate = opened_candidate.convert("RGB")
                    stage = "image_similarity"
                    with candidate:
                        similar, difference = compare_images(
                            original_array, original, np.asarray(candidate),
                            candidate, similarity_cutoff)
                    if not math.isfinite(float(difference)):
                        raise ExtractionError("image_comparison_failed")
                    _remaining(deadline)
                    if not similar:
                        events.append({**context, "status": "filtered", "stage": stage,
                                       "image_difference": float(difference),
                                       "image_cache": image_meta["cache"]})
                        continue
                    stage = "page_download"
                    page_body, page_meta = _fetch(
                        page_url, client=client, cache=cache, blob_dir=blob_dir,
                        limit=MAX_HTML_BYTES, timeout=timeout, deadline=deadline)
                    content_type = page_meta["content_type"].split(";", 1)[0].strip().lower()
                    if content_type and content_type not in {"text/html", "application/xhtml+xml"}:
                        raise ExtractionError("unexpected_page_content_type")
                    stage = "caption"
                    _remaining(deadline)
                    caption, title, origin = _caption(
                        page_body, page_url=page_meta["final_url"],
                        image_urls={image_url, urldefrag(image_meta["final_url"])[0]})
                    _remaining(deadline)
                    source = match.source or urlparse(page_url).netloc
                    if caption and (source, caption) not in seen:
                        seen.add((source, caption))
                        entries.append({
                            "page_url": page_url, "image_url": image_url,
                            "source": source, "title": title or match.title,
                            "caption": caption, "caption_origin": origin,
                            "image_difference": float(difference),
                            "image_body_sha256": image_meta["sha256"],
                            "page_body_sha256": page_meta["sha256"],
                        })
                    events.append({**context, "status": "ok" if caption else "no_caption",
                                   "stage": stage, "image_cache": image_meta["cache"],
                                   "page_cache": page_meta["cache"]})
                except Exception as exc:
                    event = {**context, "status": "error", "stage": stage,
                             "error_type": type(exc).__name__}
                    if isinstance(exc, ExtractionError):
                        event.update(reason=exc.reason, **exc.details)
                    elif isinstance(exc, httpx.HTTPStatusError):
                        event.update(reason="http_error", http_status=exc.response.status_code)
                    elif isinstance(exc, httpx.TimeoutException):
                        event["reason"] = "request_timeout"
                    events.append(event)
                    if isinstance(exc, ExtractionError) and exc.reason == "time_budget_exceeded":
                        break
    except Exception as exc:
        event = {"status": "error", "stage": "initialization",
                 "error_type": type(exc).__name__}
        if isinstance(exc, ExtractionError):
            event.update(reason=exc.reason, **exc.details)
        events.append(event)
    errors = any(event["status"] == "error" for event in events)
    status = ("ok" if entries else "error" if errors
              else "no_caption" if matches else "no_results")
    return {"status": status, "captions": _format_captions(entries),
            "entries": entries, "events": events, "matches": len(matches),
            "partial": bool(entries and errors), "extraction_config": config}
