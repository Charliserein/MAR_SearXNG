#!/usr/bin/env python3
"""从官方 ZIP 或现有图片恢复固定作者样本，不重新抽样或修改标签。"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import re
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("DGM4-author1000", "MMFakeBench-author1000")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".restore-", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def restore_dataset(source, target, archives_root, *, reuse_existing=False,
                    in_place=False):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target and not in_place:
        raise ValueError("覆盖原清单必须显式指定 --in-place；默认写入 data-local/")
    input_path = source / "samples.json"
    original_hash = sha256(input_path)
    samples = json.loads(input_path.read_text(encoding="utf-8"))
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    evidence_path = source / "reverse-evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.exists() else None
    if evidence is not None and evidence.get("input_sha256") != original_hash:
        raise ValueError("输入与图像证据哈希不一致，拒绝恢复")
    identifiers = [row["sample_id"] for row in samples]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("样本编号重复")
    images = target / "images"
    images.mkdir(parents=True, exist_ok=True)
    archives = {}
    with ExitStack() as stack:
        for sample in samples:
            sample_id = sample["sample_id"]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", sample_id):
                raise ValueError("样本编号包含不安全的路径字符")
            archive_name, member = sample["archive"], sample["zip_member"]
            extension = Path(member).suffix.lower()
            if extension not in {".jpg", ".jpeg", ".png", ".webp"}:
                raise ValueError(f"不支持的图像扩展名：{sample_id}")
            destination = images / (sample_id + extension)
            expected = sample["image_sha256"]
            if not destination.is_file() or sha256(destination) != expected:
                old_image = Path(sample["image_path"])
                if reuse_existing and old_image.is_file() and sha256(old_image) == expected:
                    body = old_image.read_bytes()
                else:
                    relative = Path(archive_name)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError("压缩包路径必须是下载目录内的相对路径")
                    archive = Path(archives_root).resolve() / relative
                    if archive_name not in archives:
                        if not archive.is_file():
                            raise FileNotFoundError(f"缺少官方压缩包：{archive}；下载说明见 README")
                        archives[archive_name] = stack.enter_context(zipfile.ZipFile(archive))
                    opened = archives[archive_name]
                    if opened.getinfo(member).file_size > 100 * 1024 * 1024:
                        raise ValueError(f"图片超过 100 MiB：{sample_id}")
                    body = opened.read(member)
                if hashlib.sha256(body).hexdigest() != expected:
                    raise ValueError(f"图片 SHA-256 不匹配：{sample_id}；清单不会更新")
                with tempfile.NamedTemporaryFile(dir=images, prefix=".image-", delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(body)
                try:
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)
            sample["image_path"] = str(destination)
    # 全部图片成功后再写样本文件，失败时保留原输入；已提取图片供下一次复用。
    output_path = target / "samples.json"
    write_json(output_path, samples)
    restored_hash = sha256(output_path)
    manifest.update(samples_sha256=restored_hash, recovery={
        "source_samples_sha256": original_hash,
        "method": "exact_zip_members_and_image_sha256",
        "labels_and_selection_unchanged": True,
    })
    write_json(target / "manifest.json", manifest)
    if evidence is not None:
        evidence["input_sha256"] = restored_hash
        write_json(target / "reverse-evidence.json", evidence)
    return {"dataset": source.name, "count": len(samples),
            "output": str(target), "samples_sha256": restored_hash}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--source-root", type=Path, default=ROOT / "data")
    parser.add_argument("--archives-root", type=Path, default=ROOT / "downloads")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data-local")
    parser.add_argument("--reuse-existing", action="store_true",
                        help="允许读取清单中仍存在且哈希一致的原图片，避免重新下载")
    parser.add_argument("--in-place", action="store_true",
                        help="显式将路径写回 source-root，只用于没有运行实验的新副本")
    args = parser.parse_args(argv)
    output = args.source_root if args.in_place else args.output_root
    selected = DATASETS if args.dataset == "all" else (args.dataset,)
    try:
        for name in selected:
            result = restore_dataset(args.source_root / name, output / name,
                                     args.archives_root, reuse_existing=args.reuse_existing,
                                     in_place=args.in_place)
            print(json.dumps(result, ensure_ascii=False))
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"恢复失败：{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
