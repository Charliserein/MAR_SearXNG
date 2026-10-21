"""Local entry point for one news item or a bounded JSON dataset batch."""
import argparse
import json
import logging
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retrieval.provenance import evidence_kind


def build_parser():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="MAR multimodal fake-news detection")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="News claim for a single-image analysis")
    source.add_argument("--input", type=Path, help="JSON array of dataset samples")
    parser.add_argument("--image", type=Path, help="Image paired with --text")
    parser.add_argument("--captions-file", type=Path, help="Optional cached reverse-image evidence")
    parser.add_argument("--evidence-kind", default="unspecified",
                        choices=("unspecified", "author_annotation", "native_reverse_image",
                                 "text_to_image_with_visual_verification"))
    parser.add_argument("--image-root", type=Path, default=ROOT / "dataset" / "MMFakeBench")
    parser.add_argument("--limit", type=int, default=1, help="Maximum dataset samples (default: 1)")
    parser.add_argument("--rounds", type=int, default=int(os.getenv("MAR_ROUNDS", "3")))
    parser.add_argument("--base-url", default=os.getenv("MAR_BASE_URL", "http://127.0.0.1:19888/v1"))
    parser.add_argument("--model", default=os.getenv("MAR_MODEL", "Qwen3.8-27B"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "inference.jsonl")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file")
    return parser


def load_samples(args):
    if args.limit < 1 or args.rounds < 1:
        raise ValueError("--limit and --rounds must be positive")
    if args.text is not None:
        if args.image is None:
            raise ValueError("--text requires --image")
        captions = args.captions_file.read_text(encoding="utf-8") if args.captions_file else ""
        rows = [{"text": args.text, "image_path": str(args.image.resolve()), "captions": captions,
                 "image_evidence_kind": args.evidence_kind}]
    else:
        if args.image or args.captions_file:
            raise ValueError("--image and --captions-file are only valid with --text")
        data = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("--input must contain a JSON array")
        rows = data[:args.limit]
    if not rows:
        raise ValueError("No samples selected")
    prepared = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            raise ValueError(f"Sample {index}: missing text")
        if not row.get("image_path"):
            raise ValueError(f"Sample {index}: missing image_path")
        image = Path(row["image_path"])
        if not image.is_absolute():
            image = args.image_root / image
        if not image.is_file():
            raise ValueError(f"Sample {index}: image does not exist: {image}")
        prepared.append((row, image.resolve()))
    return prepared


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        rows = load_samples(args)
        if args.output.exists() and not args.overwrite:
            raise ValueError(f"Output already exists: {args.output}; select another path or use --overwrite")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    from NewsAnalyzer import NewsAnalyzer

    analyzer = NewsAnalyzer(
        base_url=args.base_url,
        api_key=os.getenv("MAR_API_KEY", "sk-local"),
        model=args.model,
        temperature=args.temperature,
        n_rounds=args.rounds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    failed = 0
    with args.output.open("w" if args.overwrite else "x", encoding="utf-8") as output:
        for index, (sample, image) in enumerate(rows):
            record = dict(sample)
            try:
                result = analyzer.analyze(sample["text"], str(image), sample.get("captions", ""),
                                          evidence_kind=evidence_kind(sample))
                record.update({key: value for key, value in result.items() if key != "image"})
                record["status"] = "ok"
                print(f"[{index + 1}/{len(rows)}] {record['label']}")
            except Exception as exc:
                failed += 1
                # Preserve failures in the output instead of silently dropping them.
                record.update(status="error", error_type=type(exc).__name__)
                logging.error("Sample %s failed (%s); verify backend, model and search settings", index, type(exc).__name__)
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
    print(f"Output: {args.output}; completed: {len(rows) - failed}; failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
