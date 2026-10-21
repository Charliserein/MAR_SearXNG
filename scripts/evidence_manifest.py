"""Validate frozen reverse-image evidence before it can enter inference."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from retrieval.provenance import evidence_kind

READY_STATUSES = {"author_annotation", "ok", "no_results", "no_caption"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def summarize(records):
    return {"count": len(records), "statuses": dict(Counter(r["status"] for r in records.values())),
            "with_captions": sum(bool(r.get("captions", "").strip()) for r in records.values()),
            "ready": sum(r["status"] in READY_STATUSES for r in records.values())}


def attach_evidence(samples, manifest, input_sha256, strict=False):
    if manifest.get("schema_version") != 1 or manifest.get("input_sha256") != input_sha256:
        raise ValueError("Reverse evidence belongs to a different sample file or schema")
    records = manifest.get("records", {})
    if set(records) != {s["sample_id"] for s in samples}:
        raise ValueError("Reverse evidence must contain exactly the selected sample IDs")
    merged = []
    for sample in samples:
        record = records[sample["sample_id"]]
        if record.get("image_sha256") != sample["image_sha256"]:
            raise ValueError(f"Reverse evidence image changed: {sample['sample_id']}")
        captions = record.get("captions", "")
        if not isinstance(captions, str) or record.get("captions_sha256") != text_digest(captions):
            raise ValueError(f"Reverse evidence text changed: {sample['sample_id']}")
        status = record.get("status", "missing")
        if status in {"ok", "author_annotation"} and not captions.strip():
            raise ValueError(f"Empty captions marked successful: {sample['sample_id']}")
        if status in {"no_results", "no_caption"} and captions.strip():
            raise ValueError(f"Unexpected captions for empty status: {sample['sample_id']}")
        if strict and status not in READY_STATUSES:
            raise ValueError(f"Reverse search not ready: {sample['sample_id']} ({status})")
        if status == "author_annotation" and captions != sample.get("captions", ""):
            raise ValueError(f"Author annotations were modified: {sample['sample_id']}")
        merged.append({**sample, "captions": captions, "reverse_image_status": status,
                       "reverse_image_source": record.get("source"),
                       "image_evidence_kind": evidence_kind(record),
                       "reverse_image_captions_sha256": record["captions_sha256"]})
    return merged
