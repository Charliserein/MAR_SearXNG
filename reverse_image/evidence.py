"""Reverse image evidence with inspectable URLs, filtering and failure states."""
import asyncio
from pathlib import Path

from serper_lens import SerperLensClient


def caption_text(caption):
    if not isinstance(caption, dict):
        return ""
    for key in ("caption_node", "alt_node", *caption):
        value = caption.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def format_captions(entries):
    return "\n\n".join(
        f"[Re{i}] Source: {e['source']}\nTitle: {e['title']}\nURL: {e['page_url']}\nContent: {e['caption']}"
        for i, e in enumerate(entries, 1))


def extract_matches(image_path, matches, hashing_cutoff, similarity_cutoff):
    import numpy as np
    from PIL import Image
    from types import SimpleNamespace
    from utils import get_img_content_req, compare_images
    from download_inverse_annotations_dirs import get_captions_and_process

    entries, events, seen = [], [], set()
    with Image.open(image_path) as original:
        original.load()
        original_array = np.asarray(original)
        for match in matches:
            event = {"page_url": match.link, "image_url": match.image_url}
            try:
                candidate, candidate_pil = get_img_content_req(match.image_url)
                if isinstance(candidate, int):
                    events.append({**event, "status": "error", "stage": "image_download"})
                    continue
                try:
                    similar, diff = compare_images(original_array, original, candidate, candidate_pil, similarity_cutoff)
                finally:
                    candidate_pil.close()
                if not similar:
                    events.append({**event, "status": "filtered", "stage": "image_similarity"})
                    continue
                entry = get_captions_and_process(
                    match.image_url, match.link, SimpleNamespace(page_title=""), "", 0, hashing_cutoff)
                if entry.get("fetch_error"):
                    events.append({**event, "status": "error", "stage": "page_download"})
                    continue
                caption = caption_text(entry.get("caption", {}))
                if caption and (match.source, caption) not in seen:
                    seen.add((match.source, caption))
                    entries.append({**event, "source": match.source, "title": entry.get("title", match.title),
                                    "caption": caption, "image_difference": float(diff)})
                events.append({**event, "status": "ok" if caption else "no_caption", "stage": "caption"})
            except Exception as exc:
                events.append({**event, "status": "error", "stage": "extraction", "error_type": type(exc).__name__})
    return entries, events


async def collect_reverse_evidence(image_path, serper_key, imgbb_key, max_results=20,
                                   hashing_cutoff=5, similarity_cutoff=10):
    matches = await SerperLensClient(serper_key, max_results).search(Path(image_path), imgbb_key)
    if not matches:
        return {"status": "no_results", "captions": "", "matches": 0, "entries": [], "events": []}
    entries, events = await asyncio.to_thread(extract_matches, image_path, matches, hashing_cutoff, similarity_cutoff)
    status = "ok" if entries else "error" if any(e["status"] == "error" for e in events) else "no_caption"
    return {"status": status, "captions": format_captions(entries), "matches": len(matches),
            "entries": entries, "events": events}
