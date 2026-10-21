"""固定分层子集、独立后台评估、断点保护和周期报告。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
from scripts.evidence_manifest import attach_evidence, summarize, text_digest

DATASETS = ("DGM4-author1000", "MMFakeBench-author1000")
DEFAULT_OUTPUT = ROOT / "outputs/author2000-qwen38"


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_json(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def stratified_sample(rows, size, seed, dataset):
    """最大余数分配类别名额，再用固定种子在类别内部抽样。"""
    if not 0 < size <= len(rows):
        raise ValueError("每套抽样数必须在 1 和原始样本数之间")
    groups = {}
    for row in rows:
        groups.setdefault(row["ground_truth"], []).append(row)
    counts = {label: len(group) for label, group in groups.items()}
    quotas = {label: size * count // len(rows) for label, count in counts.items()}
    remainder = size - sum(quotas.values())
    priority = sorted(counts, key=lambda label: (-(size * counts[label] % len(rows)), label))
    for label in priority[:remainder]:
        quotas[label] += 1
    rng = random.Random(int(hashlib.sha256(f"{seed}:{dataset}".encode()).hexdigest(), 16))
    selected_ids = set()
    for label in sorted(groups):
        ordered = sorted(groups[label], key=lambda row: row["sample_id"])
        selected_ids.update(row["sample_id"] for row in rng.sample(ordered, quotas[label]))
    selected = [{**row, "source_dataset": dataset} for row in rows
                if row["sample_id"] in selected_ids]
    return selected, quotas


def prepare_experiment(output, *, per_dataset=1000, seed=20261020):
    output = Path(output).resolve()
    plan_path = output / "experiment.json"
    if plan_path.exists():
        plan = read_json(plan_path)
        if plan["per_dataset"] != per_dataset or plan["seed"] != seed:
            raise ValueError("已有实验的抽样参数不同，请使用新目录")
        verify_inputs(output, plan)
        return plan
    selections, evidence, manifests, inputs = {}, {}, {}, {}
    for name in DATASETS:
        source = ROOT / "data" / name
        source_path = source / "samples.json"
        rows = read_json(source_path)
        if len({row["sample_id"] for row in rows}) != len(rows):
            raise ValueError(f"{name}: 样本 ID 重复")
        selected, quotas = stratified_sample(rows, per_dataset, seed, name)
        source_evidence = read_json(source / "reverse-evidence.json")
        if source_evidence is not None:
            attach_evidence(rows, source_evidence, digest(source_path), strict=False)
        records = {}
        for row in selected:
            if digest(row["image_path"]) != row["image_sha256"]:
                raise ValueError("图片内容改变：" + row["sample_id"])
            record = ((source_evidence or {}).get("records", {}).get(row["sample_id"]))
            if record is None:
                caption = row.get("captions", "")
                record = {"sample_id": row["sample_id"], "image_sha256": row["image_sha256"],
                          "captions": caption, "captions_sha256": text_digest(caption),
                          "status": "author_annotation" if caption.strip() else "pending",
                          "source": "author_annotation" if caption.strip() else "none",
                          "entries": [], "events": []}
            records[row["sample_id"]] = {**record, "source_dataset": name}
        selected_path = output / "inputs" / name / "samples.json"
        write_json(selected_path, selected)
        selected_evidence = {
            "schema_version": 1, "input_sha256": digest(selected_path),
            "records": records, "summary": summarize(records),
            "preparation": {"frozen_from": str(source / "reverse-evidence.json"),
                            "source_sha256": digest(source / "reverse-evidence.json") if source_evidence else None,
                            "missing_evidence_preserved": True},
        }
        manifest = {
            "dataset": name, "count": len(selected), "seed": seed,
            "parent_count": len(rows), "split": "stratified_author_test_subset",
            "selection_rule": "largest_remainder_by_ground_truth_then_seeded_within_class",
            "rng": "random.Random(int(sha256(f'{seed}:{dataset}'),16))",
            "class_counts": quotas,
            "subclass_counts": dict(Counter(row["fake_cls"] for row in selected)),
            "source_samples": [str(source_path)], "source_sha256": digest(source_path),
            "source_manifest": read_json(source / "manifest.json", {}),
            "samples_sha256": digest(selected_path),
            "purpose": "免费检索变体的固定子集实验；缺失图像证据不伪装为完成",
        }
        write_json(selected_path.with_name("reverse-evidence.json"), selected_evidence)
        write_json(selected_path.with_name("manifest.json"), manifest)
        selections[name], evidence[name], manifests[name] = selected, records, manifest
    combined = [row for pair in zip(*(selections[name] for name in DATASETS)) for row in pair]
    if len({row["sample_id"] for row in combined}) != len(combined):
        raise ValueError("两套数据存在重复样本 ID")
    combined_dir = output / "inputs/combined"
    write_json(combined_dir / "samples.json", combined)
    all_evidence = {sid: record for records in evidence.values() for sid, record in records.items()}
    write_json(combined_dir / "reverse-evidence.json", {
        "schema_version": 1, "input_sha256": digest(combined_dir / "samples.json"),
        "records": all_evidence, "summary": summarize(all_evidence),
        "preparation": {"frozen_selection": True, "searxng_images_is_native_reverse": False},
    })
    write_json(combined_dir / "manifest.json", {
        "dataset": "DGM4 + MMFakeBench 作者测试集分层子集",
        "count": len(combined), "seed": seed, "split": "stratified_author_test_subset",
        "purpose": "免费检索变体，三轮 MAR；标签只用于分层抽样与事后评分",
        "selection": "各数据集按原类别比例最大余数分配，固定随机抽样，按数据集交错排列",
        "source_samples": [str(ROOT / "data" / name / "samples.json") for name in DATASETS],
        "source_datasets": manifests,
        "class_counts": dict(Counter(row["ground_truth"] for row in combined)),
        "subclass_counts": dict(Counter(row["fake_cls"] for row in combined)),
        "dataset_counts": {name: len(rows) for name, rows in selections.items()},
        "samples_sha256": digest(combined_dir / "samples.json"),
    })
    for path in sorted((output / "inputs").rglob("*.json")):
        inputs[str(path.relative_to(output))] = digest(path)
    plan = {"schema_version": 1, "created_at": now(), "seed": seed,
            "per_dataset": per_dataset, "total": len(combined),
            "datasets": list(DATASETS), "input_files": inputs,
            "combined_input": str(combined_dir / "samples.json"),
            "evaluation_output": str(output / "evaluation"),
            "evidence_summary": summarize(all_evidence),
            "first4_sample_ids": [row["sample_id"] for row in combined[:4]],
            "first8_sample_ids": [row["sample_id"] for row in combined[:8]],
            "first20_sample_ids": [row["sample_id"] for row in combined[:20]]}
    write_json(plan_path, plan)
    return plan


def verify_inputs(output, plan):
    for relative, expected in plan["input_files"].items():
        path = output / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError("冻结输入已改变：" + relative)


def code_snapshot():
    paths = [*ROOT.joinpath("MAR_framework").rglob("*.py"),
             *ROOT.joinpath("retrieval").rglob("*.py"),
             ROOT / "scripts/evaluate_batch.py", ROOT / "scripts/evidence_manifest.py",
             Path(__file__).resolve(),
             ROOT / "configs/runtime.json", ROOT / "scripts/serve_qwen.sh"]
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(set(paths))}


def execution_snapshot(output, plan, workers):
    environment = {key: value for key, value in os.environ.items() if key.startswith("MAR_")}
    return {
        "schema_version": 1, "experiment_sha256": digest(output / "experiment.json"),
        "input_files": plan["input_files"], "code_sha256": code_snapshot(),
        "environment_sha256": json_hash(environment),
        "runtime": {"workers": workers, "rounds": 3, "attempts": 2,
                    "strict_evidence": False, "strict_web": True, "error_retry_rounds": 2,
                    "model": os.getenv("MAR_MODEL", "Qwen3.8-27B"),
                    "base_url": os.getenv("MAR_BASE_URL", "http://127.0.0.1:19888/v1"),
                    "search_provider": os.getenv("MAR_SEARCH_PROVIDER", "searxng"),
                    "cache_mode": os.getenv("MAR_CACHE_MODE", "online"),
                    "searxng_url": os.getenv("MAR_SEARXNG_URL", "http://127.0.0.1:18888")},
    }


def freeze_or_verify(output, plan, workers):
    verify_inputs(output, plan)
    current = execution_snapshot(output, plan, workers)
    snapshot_path = output / "run-snapshot.json"
    old = read_json(snapshot_path)
    if old is not None and old != current:
        raise ValueError("源码、输入、运行环境或配置已改变，禁止混合续跑；请使用新实验目录")
    if old is None:
        for relative in current["code_sha256"]:
            path = output / "frozen_source" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / relative).read_bytes())
        write_json(snapshot_path, current)
    return current


def process_token(pid):
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        rest = raw[raw.rfind(")") + 2:].split()
        if rest[0] == "Z":
            return None
        return rest[19]
    except (OSError, IndexError):
        return None


def running(output):
    state = read_json(output / "supervisor.json", {})
    pid = state.get("pid")
    return bool(pid and process_token(pid) == state.get("process_token") and process_token(pid))


def evaluation_command(plan, workers, retry_errors):
    command = [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "scripts/evaluate_batch.py"),
               "--input", plan["combined_input"], "--output", plan["evaluation_output"],
               "--rounds", "3", "--workers", str(workers), "--attempts", "2",
               "--no-strict-evidence", "--strict-web"]
    if retry_errors:
        command.append("--retry-errors")
    return command


def records_from(output):
    records = []
    for path in sorted((output / "evaluation/records").glob("*.json")):
        try:
            records.append(read_json(path))
        except (ValueError, OSError):
            continue
    return records


def benchmarks(plan, records):
    by_id = {r["sample_id"]: r for r in records}
    result = {}
    for count in (4, 8, 20):
        ids = plan.get(f"first{count}_sample_ids", [])
        selected = [by_id[sid] for sid in ids if sid in by_id]
        if len(selected) != len(ids) or not selected:
            result[f"first{count}"] = {"available": False, "records": len(selected), "target": len(ids)}
            continue
        started = min(datetime.fromisoformat(r["started_at"]) for r in selected)
        finished = max(datetime.fromisoformat(r["finished_at"]) for r in selected)
        wall = max(0, (finished - started).total_seconds())
        successes = sum(r["status"] == "ok" for r in selected)
        result[f"first{count}"] = {
            "available": True, "target": len(ids), "ok": successes,
            "errors": len(ids) - successes, "wall_seconds": wall,
            "wall_definition": "该组最早样本开始至最晚样本完成，包含组内并发和等待",
            "completed_per_hour": len(ids) * 3600 / wall if wall else None,
            "successful_per_hour": successes * 3600 / wall if wall else None,
            "samples": [{"sample_id": r["sample_id"], "source_dataset": r.get("source_dataset"),
                         "seconds": r["seconds"], "status": r["status"],
                         "attempt": r.get("attempt")} for r in selected],
        }
    return result


def update_progress(output, plan, state, child_pid=None, render=False):
    records = records_from(output)
    ok = sum(r["status"] == "ok" for r in records)
    errors = len(records) - ok
    pending = plan["total"] - len(records)
    retryable = errors if state["phase_index"] < 2 and state["status"] == "running" else 0
    benchmark = benchmarks(plan, records)
    value = {
        "schema_version": 1, "status": state["status"], "updated_at": now(),
        "started_at": state["started_at"], "elapsed_seconds": state["elapsed_seconds"],
        "active_session_started_at": state.get("active_session_started_at"),
        "session_count": state["session_count"], "workers": state["workers"], "rounds": 3,
        "total": plan["total"], "completed_records": len(records), "ok": ok,
        "errors": errors, "pending": pending, "retryable_errors": retryable,
        "pending_to_attempt": pending + retryable, "phase_index": state["phase_index"],
        "evaluator_pid": child_pid,
        "error_types": dict(Counter(r.get("error_type", "unknown") for r in records if r["status"] != "ok")),
        "benchmark": benchmark,
        "note": "错误不计为模型预测成功；初次完成后最多两轮仅重试错误",
    }
    if state.get("finished_at"):
        value["finished_at"] = state["finished_at"]
    write_json(output / "progress.json", value)
    write_json(output / "benchmark.json", benchmark)
    write_json(output / "evaluation/runtime-status.json", value)
    if render and all((output / "evaluation" / name).exists() for name in
                      ("samples.json", "config.json", "dataset-manifest.json")):
        try:
            with (output / "report.log").open("ab") as log:
                result = subprocess.run(
                    [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/report_evaluation.py"),
                     plan["evaluation_output"]], cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, timeout=120, check=False)
            value["report"] = {"updated_at": now(), "returncode": result.returncode,
                               "code_sha256": digest(ROOT / "scripts/report_evaluation.py"),
                               "docx": str(output / "report.docx")}
            document = output / "evaluation/report.docx"
            if result.returncode == 0 and document.exists():
                temporary = output / "report.docx.tmp"
                shutil.copyfile(document, temporary)
                temporary.replace(output / "report.docx")
        except Exception as exc:
            value["report"] = {"updated_at": now(), "error_type": type(exc).__name__}
        write_json(output / "progress.json", value)
    return value


def supervise(output, workers):
    plan = read_json(output / "experiment.json")
    with (output / ".supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze_or_verify(output, plan, workers)
        old = read_json(output / "supervisor-state.json", {})
        previous_seconds = old.get("elapsed_seconds", 0)
        session_start = time.monotonic()
        state = {**old, "status": "running", "workers": workers,
                 "started_at": old.get("started_at", now()), "active_session_started_at": now(),
                 "elapsed_seconds": previous_seconds, "phase_index": old.get("phase_index", 0),
                 "session_count": old.get("session_count", 0) + 1}
        state.pop("finished_at", None)
        write_json(output / "supervisor.json", {"pid": os.getpid(),
                   "process_token": process_token(os.getpid()), "started_at": now()})
        (output / "supervisor.pid").write_text(str(os.getpid()) + "\n")
        write_json(output / "supervisor-state.json", state)
        child = None
        try:
            while state["phase_index"] <= 2:
                with (output / "evaluation.log").open("ab") as log:
                    child = subprocess.Popen(
                        evaluation_command(plan, workers, state["phase_index"] > 0),
                        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True)
                write_json(output / "evaluator.json", {"pid": child.pid,
                           "process_token": process_token(child.pid), "started_at": now(),
                           "phase_index": state["phase_index"]})
                last_report = 0.0
                last_count = 0
                while child.poll() is None:
                    state["elapsed_seconds"] = previous_seconds + time.monotonic() - session_start
                    write_json(output / "supervisor-state.json", state)
                    count = len(list((output / "evaluation/records").glob("*.json")))
                    if time.monotonic() - last_report >= 60 or count - last_count >= 25 or (
                            count >= 4 and last_count < 4) or (
                            count >= 8 and last_count < 8) or (count >= 20 and last_count < 20):
                        update_progress(output, plan, state, child.pid, render=True)
                        last_report, last_count = time.monotonic(), count
                    if (output / "stop.request.json").exists():
                        state["status"] = "stopped_by_user"
                        try:
                            if child.poll() is None:
                                os.killpg(child.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        try:
                            child.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            try:
                                os.killpg(child.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            child.wait()
                        active = read_json(output / "evaluation/progress.json", {}).get("active", {})
                        write_json(output / "evaluation/status.json", {
                            "status": "stopped_by_user", "interrupted": list(active),
                            "finished_at": now(), "total": plan["total"]})
                        break
                    time.sleep(2)
                state["elapsed_seconds"] = previous_seconds + time.monotonic() - session_start
                if state["status"] == "stopped_by_user":
                    break
                records = records_from(output)
                errors = sum(r["status"] != "ok" for r in records)
                state["last_evaluator_returncode"] = child.returncode
                if child.returncode not in (0, 1) or len(records) != plan["total"]:
                    state["status"] = "failed"
                    state["failure_reason"] = "evaluator_exit_before_complete_phase"
                    break
                if not errors or state["phase_index"] == 2:
                    state["status"] = "completed" if not errors else "completed_with_errors"
                    break
                state["phase_index"] += 1
                write_json(output / "supervisor-state.json", state)
                update_progress(output, plan, state, render=True)
        except BaseException as exc:
            state["status"] = "failed"
            state["failure_reason"] = type(exc).__name__
            # A surviving evaluator remains discoverable and blocks a second
            # launch until it exits; we do not silently duplicate its workload.
            raise
        finally:
            state["elapsed_seconds"] = previous_seconds + time.monotonic() - session_start
            state["finished_at"] = now()
            write_json(output / "supervisor-state.json", state)
            update_progress(output, plan, state, render=True)


def start(output, workers, per_dataset=1000, seed=20261020):
    output = Path(output).resolve()
    if running(output):
        return {"status": "already_running", **read_json(output / "supervisor.json")}
    child = read_json(output / "evaluator.json", {})
    if (child.get("pid") and process_token(child["pid"])
            and process_token(child["pid"]) == child.get("process_token")):
        raise RuntimeError("上次 evaluator 仍在运行，请等待它退出，不能重复启动")
    plan = read_json(output / "experiment.json")
    if plan is None:
        plan = prepare_experiment(output, per_dataset=per_dataset, seed=seed)
    elif plan.get("per_dataset", per_dataset) != per_dataset or plan.get("seed", seed) != seed:
        raise ValueError("启动参数与已冻结的抽样参数不同，请指定原参数或使用新目录")
    snapshot = freeze_or_verify(output, plan, workers)
    state = read_json(output / "supervisor-state.json", {})
    if state.get("status") in {"completed", "completed_with_errors"}:
        return {"status": state["status"], "message": "计划的初次评估及错误重试已结束"}
    with (output / ".launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if running(output):
            return {"status": "already_running", **read_json(output / "supervisor.json")}
        (output / "stop.request.json").unlink(missing_ok=True)
        with (output / "supervisor.log").open("ab") as log:
            process = subprocess.Popen(
                ["nohup", "setsid", str(ROOT / ".venv/bin/python"), "-u",
                 str(Path(__file__).resolve()), "_supervise", "--output", str(output),
                 "--workers", str(workers)],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        # This is the PID of setsid's non-forking child; the supervisor rewrites
        # its own identity once its lock and snapshot checks have passed.
        write_json(output / "supervisor.json", {
            "pid": process.pid, "process_token": process_token(process.pid), "started_at": now()})
        (output / "supervisor.pid").write_text(str(process.pid) + "\n")
    return {"status": "started", "pid": process.pid,
            "snapshot_sha256": digest(output / "run-snapshot.json"),
            "snapshot_content_sha256": json_hash(snapshot),
            "output": str(output), "log": str(output / "supervisor.log")}


def live_evaluator(output):
    child = read_json(output / "evaluator.json", {})
    pid = child.get("pid")
    if not pid or not process_token(pid) or process_token(pid) != child.get("process_token"):
        return None
    try:
        if os.getpgid(pid) != pid:
            return None
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        expected = str(output / "evaluation").encode()
        if str(ROOT / "scripts/evaluate_batch.py").encode() not in command or expected not in command:
            return None
    except (ProcessLookupError, OSError):
        return None
    return child


def stop(output):
    if running(output):
        write_json(output / "stop.request.json", {"requested_at": now(), "reason": "user_stop"})
        return {"status": "stop_requested", "pid": read_json(output / "supervisor.json")["pid"]}
    child = live_evaluator(output)
    if child is None:
        return {"status": "not_running"}
    try:
        os.killpg(child["pid"], signal.SIGTERM)
    except ProcessLookupError:
        return {"status": "not_running"}
    deadline = time.monotonic() + 15
    while process_token(child["pid"]) == child["process_token"] and time.monotonic() < deadline:
        time.sleep(0.2)
    if live_evaluator(output) is not None:
        try:
            os.killpg(child["pid"], signal.SIGKILL)
        except ProcessLookupError:
            pass
    state = read_json(output / "supervisor-state.json", {})
    state.update(status="stopped_by_user", finished_at=now(), stop_reason="orphan_evaluator_stopped")
    write_json(output / "supervisor-state.json", state)
    return {"status": "stopped_orphan_evaluator", "pid": child["pid"]}


def main():
    load_dotenv(ROOT / ".env")
    bypass = {part.strip() for part in
              (os.getenv("NO_PROXY", "") + "," + os.getenv("no_proxy", "")).split(",") if part.strip()}
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = ",".join(sorted(bypass | {"127.0.0.1", "localhost"}))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "start", "status", "stop", "restart", "_supervise"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument("--per-dataset", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "prepare":
        result = prepare_experiment(output, per_dataset=args.per_dataset, seed=args.seed)
    elif args.command == "_supervise":
        supervise(output, args.workers)
        return
    elif args.command == "status":
        result = {"running": running(output), "output": str(output),
                  "progress": read_json(output / "progress.json", {}),
                  "state": read_json(output / "supervisor-state.json", {}),
                  "snapshot_sha256": digest(output / "run-snapshot.json")
                  if (output / "run-snapshot.json").exists() else None}
    elif args.command == "stop":
        result = stop(output)
    else:
        if args.command == "restart":
            stop(output)
            deadline = time.monotonic() + 180
            while running(output) and time.monotonic() < deadline:
                time.sleep(1)
            if running(output):
                raise RuntimeError("后台任务仍在停止中，请稍后重新执行 start")
        result = start(output, args.workers, args.per_dataset, args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
