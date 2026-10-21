#!/usr/bin/env python3
"""下载固定版本的官方 Qwen3.8 FP8；支持断点续传与逐文件校验。"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPO = "Qwen/Qwen3.8-27B-FP8"
REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
DEFAULT_OUTPUT = ROOT / "models" / "Qwen3.8-27B-FP8"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_hash(item: dict) -> str | None:
    return item.get("lfs", {}).get("sha256")


def verify_file(path: Path, item: dict) -> tuple[bool, str | None]:
    if not path.is_file() or path.stat().st_size != item["size"]:
        return False, None
    actual = sha256_file(path)
    expected = expected_hash(item)
    if expected:
        return actual == expected, actual
    # Hugging Face 元数据的小文件使用 Git blob SHA-1。
    digest = hashlib.sha1()
    digest.update(f"blob {item['size']}\0".encode())
    with path.open("rb") as stream:
        digest.update(stream.read())
    return digest.hexdigest() == item["blobId"], actual


def download_large_ranges(partial: Path, url: str, item: dict) -> None:
    """直接写入互不重叠的范围，避免合并临时块时复制大文件。"""
    state_path = partial.with_name(partial.name + ".ranges.json")
    size = item["size"]
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["sha256"] != expected_hash(item) or state["size"] != size:
            raise RuntimeError("分段状态与固定权重不一致")
    else:
        prefix = partial.stat().st_size if partial.exists() else 0
        if prefix > size:
            raise RuntimeError("已有断点大于目标权重")
        state = {"sha256": expected_hash(item), "size": size, "prefix": prefix, "done": []}
        state_path.write_text(json.dumps(state))
    prefix = state["prefix"]
    step = 64 * 1024**2
    ranges = [(start, min(start + step, size) - 1) for start in range(prefix, size, step)]
    done = set(state["done"])
    lock = threading.Lock()
    descriptor = os.open(partial, os.O_CREAT | os.O_RDWR, 0o644)
    os.ftruncate(descriptor, size)

    def fetch(bounds: tuple[int, int]) -> None:
        start, end = bounds
        if start in done:
            return
        for attempt in range(1, 7):
            try:
                request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(request, timeout=90) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise ValueError("分段范围不匹配")
                    position = start
                    while chunk := response.read(4 * 1024**2):
                        if position + len(chunk) > end + 1:
                            raise ValueError("分段响应过长")
                        view = memoryview(chunk)
                        while view:
                            written = os.pwrite(descriptor, view, position)
                            position += written
                            view = view[written:]
                    if position != end + 1:
                        raise ValueError("分段响应不完整")
                with lock:
                    done.add(start)
                    state["done"] = sorted(done)
                    scratch = state_path.with_suffix(".tmp")
                    scratch.write_text(json.dumps(state))
                    os.replace(scratch, state_path)
                    if len(done) % 8 == 0 or len(done) == len(ranges):
                        complete = prefix + sum(b - a + 1 for a, b in ranges if a in done)
                        print(f"分段进度 {item['rfilename']} {complete / 1024**3:.2f}/{size / 1024**3:.2f} GiB", flush=True)
                return
            except (OSError, urllib.error.URLError, ValueError):
                if attempt == 6:
                    raise RuntimeError(f"分段下载失败: {item['rfilename']} {start}-{end}") from None
                time.sleep(min(15, attempt * 2))

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(fetch, ranges))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def download_file(out: Path, item: dict) -> dict:
    relative = item["rfilename"]
    target = out / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    valid, actual = verify_file(target, item)
    if valid:
        print(f"已校验 {relative}", flush=True)
        return {"path": relative, "size": item["size"], "sha256": actual}
    partial = target.with_name(target.name + ".partial")
    if item["size"] == 0:
        partial.touch()
    if item.get("_mirror_revision"):
        url = f"https://modelscope.cn/api/v1/models/{REPO}/repo?" + urllib.parse.urlencode({
            "Revision": item["_mirror_revision"], "FilePath": relative,
        })
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    else:
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{urllib.parse.quote(relative)}"
        opener = urllib.request.build_opener()
    if item.get("_mirror_revision") and item["size"] > 1024**3:
        download_large_ranges(partial, url, item)
        valid, actual = verify_file(partial, item)
        if not valid:
            raise RuntimeError(f"分段权重最终 SHA-256 校验失败: {relative}")
        os.replace(partial, target)
        partial.with_name(partial.name + ".ranges.json").unlink(missing_ok=True)
        print(f"下载并校验 {relative}", flush=True)
        return {"path": relative, "size": item["size"], "sha256": actual}
    for attempt in range(1, 7):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > item["size"]:
            partial.unlink()
            offset = 0
        try:
            if offset < item["size"]:
                headers = {"User-Agent": "MAR-KDD-cached-retrieval/1.0"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                request = urllib.request.Request(url, headers=headers)
                with opener.open(request, timeout=90) as response:
                    if offset and response.status != 206:
                        offset = 0
                    elif offset and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("服务器返回的断点位置不匹配")
                    with partial.open("ab" if offset else "wb") as stream:
                        while chunk := response.read(4 * 1024 * 1024):
                            stream.write(chunk)
            valid, actual = verify_file(partial, item)
            if not valid:
                if partial.stat().st_size >= item["size"]:
                    partial.unlink()
                raise ValueError("文件大小或哈希不匹配")
            os.replace(partial, target)
            print(f"下载并校验 {relative}", flush=True)
            return {"path": relative, "size": item["size"], "sha256": actual}
        except (OSError, urllib.error.URLError, ValueError) as exc:
            # 不输出请求 URL、认证信息或服务端响应正文。
            print(f"重试 {relative}: {type(exc).__name__} ({attempt}/6)", flush=True)
            if attempt == 6:
                raise RuntimeError(f"下载失败: {relative}") from None
            time.sleep(min(30, attempt * 3))
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--source", choices=["huggingface", "modelscope"], default="modelscope")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output / "upstream-metadata.json"
    if args.verify_only:
        metadata = json.loads(metadata_path.read_text())
    else:
        url = f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true"
        with urllib.request.urlopen(url, timeout=60) as response:
            metadata = json.load(response)
        if metadata["sha"] != REVISION:
            raise RuntimeError("模型 revision 与固定版本不一致")
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    files = [
        item for item in metadata["siblings"]
        if item["rfilename"] not in {".gitattributes", "README.md"}
    ]
    for item in files:
        relative = Path(item["rfilename"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("模型清单包含非法路径")
    if not args.verify_only and args.source == "modelscope":
        url = f"https://modelscope.cn/api/v1/models/{REPO}/repo/files?Revision=master&Recursive=true"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=60) as response:
            mirror = json.load(response)
        mirror_files = {item["Path"]: item for item in mirror["Data"]["Files"]}
        for item in files:
            if expected_hash(item):
                candidate = mirror_files.get(item["rfilename"], {})
                if candidate.get("Sha256") != expected_hash(item) or candidate.get("Size") != item["size"]:
                    raise RuntimeError(f"官方 ModelScope 镜像与固定 HF revision 哈希不符: {item['rfilename']}")
                item["_mirror_revision"] = candidate["Revision"]
        (args.output / "modelscope-metadata.json").write_text(json.dumps(mirror, ensure_ascii=False, indent=2))
    if args.verify_only:
        verified = []
        for item in files:
            valid, digest = verify_file(args.output / item["rfilename"], item)
            if not valid:
                raise RuntimeError(f"校验失败: {item['rfilename']}")
            verified.append({"path": item["rfilename"], "size": item["size"], "sha256": digest})
    else:
        remaining = sum(
            max(0, item["size"] - (
                (args.output / item["rfilename"]).stat().st_size
                if (args.output / item["rfilename"]).exists() else 0
            ))
            for item in files
        )
        free = shutil.disk_usage(args.output).free
        if free < remaining + 10 * 1024**3:
            raise RuntimeError("可用磁盘不足：下载后需至少保留 10 GiB")
        print(f"官方模型 {REPO}@{REVISION}; 文件 {len(files)}; 总字节 {sum(f['size'] for f in files)}", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            verified = list(pool.map(lambda item: download_file(args.output, item), files))
    manifest = {
        "repo": REPO,
        "revision": REVISION,
        "precision": "official FP8",
        "verified_at_unix": time.time(),
        "source": f"https://huggingface.co/{REPO}/tree/{REVISION}",
        "download_source": args.source,
        "file_count": len(verified),
        "total_bytes": sum(item["size"] for item in verified),
        "files": sorted(verified, key=lambda item: item["path"]),
    }
    (args.output / "download-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"模型完整，逐文件校验通过：{len(verified)} 个文件", flush=True)


if __name__ == "__main__":
    main()
