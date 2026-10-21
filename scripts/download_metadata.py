#!/usr/bin/env python3
"""从固定作者版本恢复样本文本与作者图注；不下载图片、不覆盖现有输入。"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
AUTHOR_REVISION = "3bb8c69e437d1b4a5a31d412210f0de3b2003847"
AUTHOR_BASE = f"https://raw.githubusercontent.com/Averdgr/MAR_KDD/{AUTHOR_REVISION}/"
SOURCES = {
    "DGM4-author1000": ("data_processing/DGM4/test_final.json",
        "5644105fb9334bd072e4a7a7c5a8bd9cebff1c62c1865a64495b3f939b6a5b01"),
    "MMFakeBench-author1000": ("data_processing/MMFakeBench/test_set.json",
        "fefcea882fb1ddd3e97282880a191c83037fc32aa7475dfa9d92926ffd353595"),
}
MAPPING = {"orig": "original", "original": "original",
           "text_attribute": "textual_veracity_distortion",
           "textual_veracity_distortion": "textual_veracity_distortion",
           "face_swap": "visual_veracity_distortion", "face_attribute": "visual_veracity_distortion",
           "visual_veracity_distortion": "visual_veracity_distortion",
           "text_swap": "mismatch", "mismatch": "mismatch"}


def serialized(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def build_metadata(name, root, body):
    relative, expected = SOURCES[name]
    if hashlib.sha256(body).hexdigest() != expected:
        raise ValueError(f"作者文件哈希不匹配：{name}")
    root = Path(root)
    index = json.loads((root / "sample-index.json").read_text())
    authors = json.loads(body)
    if len(authors) != index["count"] or len(index["records"]) != index["count"]:
        raise ValueError("作者文件与固定索引数量不匹配")
    rows, records = [], {}
    for entry in index["records"]:
        row = {**authors[entry["author_index"]],
               **{k: v for k, v in entry.items() if k != "author_index"}}
        row["ground_truth"] = MAPPING[row["fake_cls"]]
        row["captions"] = row.get("captions") or ""
        if not isinstance(row["captions"], str):
            raise ValueError("作者图注应为字符串")
        row["image_path"] = str((root / "images" /
            (row["sample_id"] + Path(row["zip_member"]).suffix.lower())).resolve())
        status = "author_annotation" if row["captions"].strip() else "pending"
        row["reverse_image_status"] = status
        rows.append(row)
        records[row["sample_id"]] = {
            "sample_id": row["sample_id"], "image_sha256": row["image_sha256"],
            "captions": row["captions"],
            "captions_sha256": hashlib.sha256(row["captions"].encode()).hexdigest(),
            "status": status, "source": "author_annotation" if row["captions"].strip() else "none",
            "entries": [], "events": [],
        }
    samples_bytes = serialized(rows)
    evidence = {"schema_version": 1,
        "input_sha256": hashlib.sha256(samples_bytes).hexdigest(), "records": records,
        "summary": {"count": len(rows),
            "statuses": dict(Counter(x["status"] for x in records.values())),
            "with_captions": sum(bool(x["captions"].strip()) for x in records.values()),
            "ready": sum(x["status"] == "author_annotation" for x in records.values())},
        "preparation": {"source": AUTHOR_BASE + relative, "source_sha256": expected,
                        "scope": "author_annotations_only"}}
    return samples_bytes, evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=(*SOURCES, "all"), default="all")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    args = parser.parse_args(argv)
    selected = SOURCES if args.dataset == "all" else (args.dataset,)
    for name in selected:
        directory = args.data_root / name
        samples_path = directory / "samples.json"
        if samples_path.exists() or (directory / "reverse-evidence.json").exists():
            print(f"保留现有输入与证据，未覆盖：{directory}")
            continue
        body = urllib.request.urlopen(AUTHOR_BASE + SOURCES[name][0], timeout=60).read()
        samples_bytes, evidence = build_metadata(name, directory, body)
        # 使用排他创建，防止覆盖同时启动的任务准备的输入。
        with samples_path.open("xb") as stream:
            stream.write(samples_bytes)
        with (directory / "reverse-evidence.json").open("xb") as stream:
            stream.write(serialized(evidence))
        print(json.dumps({"dataset": name, "samples": str(samples_path),
                          "evidence": evidence["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
