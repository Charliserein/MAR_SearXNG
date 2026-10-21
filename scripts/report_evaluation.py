"""Generate a self-contained Chinese report from persisted predictions."""
import argparse
import base64
from collections import Counter
import contextlib
import csv
from datetime import datetime, timedelta, timezone
import html
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from urllib.parse import urlparse
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

LABELS = ["original", "textual_veracity_distortion", "visual_veracity_distortion", "mismatch"]
NAMES = {"original": "真实", "textual_veracity_distortion": "文字篡改",
         "visual_veracity_distortion": "图像篡改", "mismatch": "图文不符"}
PAPER = {
    "title": "MAR: Metacognitive Agentic Reasoning for Multimodal Fake News Detection",
    "doi": "https://doi.org/10.1145/3770855.3818025",
    "local_pdf": "/home/c/czh/MAR_KDD/18 - MAR- Metacognitive Agentic Reasoning for Multimodal Fake News Detection.pdf",
    "metric_source": "论文 §5.2、表 1（二分类，印刷页 467）、表 2（四分类，印刷页 468）",
    "averaging_source": "https://github.com/Averdgr/MAR_KDD/blob/3bb8c69e437d1b4a5a31d412210f0de3b2003847/MAR_framework/main_eval.py",
    "metrics": ["macro_f1", "accuracy", "macro_precision", "macro_recall"],
    "averaging": "F1、Precision、Recall 均按类别做宏平均；零分母置零。",
    "test_samples_per_dataset": 1000,
    "backbone": "Qwen3-VL-32B-Instruct-FP8",
    "rounds": 3,
    "temperature": 0,
    "reported_results": {
        "DGM4": {
            "binary": {"macro_f1": .7275, "accuracy": .7570, "macro_precision": .7200, "macro_recall": .7445},
            "four_class": {"macro_f1": .4774, "accuracy": .6080, "macro_precision": .4811, "macro_recall": .5067},
        },
        "MMFakeBench": {
            "binary": {"macro_f1": .7805, "accuracy": .8130, "macro_precision": .7769, "macro_recall": .7845},
            "four_class": {"macro_f1": .6802, "accuracy": .6820, "macro_precision": .6962, "macro_recall": .6933},
        },
    },
}
TERMINAL_STATUSES = {"completed", "completed_with_errors", "stopped_by_user", "failed"}


def atomic_bytes(path, value):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + "-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_text(path, value):
    atomic_bytes(path, value.encode("utf-8"))


def read_jsonl(path):
    """Ignore only an unfinished final append while an evaluation is running."""
    if not path.exists():
        return []
    data = path.read_bytes()
    rows, lines = [], data.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            if index == len(lines) - 1 and not line.endswith(b"\n"):
                break
            raise
    return rows


def parsed_time(value):
    if not isinstance(value, str):
        return None
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return result if result.tzinfo else result.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def sample_wall_seconds(record):
    start, finish = parsed_time(record.get("started_at")), parsed_time(record.get("finished_at"))
    if start and finish and finish >= start:
        return (finish - start).total_seconds()
    value = record.get("seconds")
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 else None


def latency_summary(values):
    return {
        "count": len(values), "mean": float(np.mean(values)) if values else None,
        "median": float(np.median(values)) if values else None,
        "p95": float(np.percentile(values, 95)) if values else None,
        "min": min(values) if values else None, "max": max(values) if values else None,
    }


def sample_dataset(sample, manifest):
    for field in ("dataset", "source_dataset", "dataset_name"):
        value = sample.get(field)
        if isinstance(value, str) and value.strip():
            for known in ("MMFakeBench", "DGM4"):
                if known.lower() in value.lower():
                    return known
            return value.strip()
    sid = str(sample.get("sample_id", "")).lower()
    for prefix, name in (("author-dgm4-", "DGM4"), ("author-mmfakebench-", "MMFakeBench")):
        if sid.startswith(prefix):
            return name
    description = str(manifest.get("dataset", ""))
    known = [name for name in ("DGM4", "MMFakeBench") if name.lower() in description.lower()]
    return known[0] if len(known) == 1 else "未标明数据集"


def dataset_metrics(samples, records, manifest):
    groups = {}
    for sample in samples:
        groups.setdefault(sample_dataset(sample, manifest), []).append(sample)
    result = {}
    for name, group in sorted(groups.items()):
        identities = {sample["sample_id"] for sample in group}
        selected_records = [record for record in records if record["sample_id"] in identities]
        metrics = compute_metrics(group, selected_records)
        metrics["evidence"] = evidence_metrics(selected_records)
        durations = [sample_wall_seconds(record) for record in selected_records
                     if record["status"] == "ok" and record.get("result", {}).get("current_round") == 3
                     and len(record.get("result", {}).get("short_term_memories", [])) == 3]
        metrics["full_three_round_sample_wall_seconds"] = latency_summary(
            [value for value in durations if value is not None])
        result[name] = metrics
    return result


def run_timing(samples, records, config, sessions, status, runtime_status, progress):
    """Use concurrent session wall time; never sum per-sample durations as elapsed."""
    declared = runtime_status.get("elapsed_seconds")
    if isinstance(declared, (float, int)) and math.isfinite(declared) and declared >= 0:
        elapsed = float(declared)
        source = "runtime-status.elapsed_seconds"
        current_status = runtime_status.get("status", "running")
        final = current_status in TERMINAL_STATUSES
    else:
        intervals = []
        for session in sessions:
            start, finish = parsed_time(session.get("started_at")), parsed_time(session.get("finished_at"))
            if start and finish and finish >= start:
                intervals.append((start, finish))
        terminal = bool(status.get("finished_at")) or status.get("status") in TERMINAL_STATUSES
        if not intervals:
            starts = [parsed_time(record.get("started_at")) for record in records]
            finishes = [parsed_time(record.get("finished_at")) for record in records]
            observed = parsed_time(progress.get("timestamp"))
            if observed:
                finishes.append(observed)
                starts.extend(observed - timedelta(seconds=float(active["seconds"]))
                              for active in progress.get("active", {}).values()
                              if isinstance(active.get("seconds"), (int, float)))
            starts, finishes = [value for value in starts if value], [value for value in finishes if value]
            if starts and finishes and max(finishes) >= min(starts):
                intervals.append((min(starts), max(finishes)))
            source = "observed_record_and_heartbeat_span"
        else:
            source = "sessions_interval_union"
            observed = parsed_time(progress.get("timestamp"))
            if not terminal and observed and observed > max(end for _, end in intervals):
                active_start = parsed_time(runtime_status.get("active_session_started_at"))
                starts = [parsed_time(record.get("started_at")) for record in records]
                last_end = max(end for _, end in intervals)
                starts = [value for value in starts if value and value >= last_end]
                if active_start or starts:
                    intervals.append((active_start or min(starts), observed))
        elapsed, merged = 0.0, []
        for start, finish in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(finish, merged[-1][1]))
            else:
                merged.append((start, finish))
        elapsed = sum((finish - start).total_seconds() for start, finish in merged)
        current_status = status.get("status", "completed" if terminal else "running")
        final = terminal
    valid = [record for record in records if record["status"] == "ok"]
    full_round_records = [record for record in valid
                          if record.get("result", {}).get("current_round") == 3
                          and len(record.get("result", {}).get("short_term_memories", [])) == 3]
    full_round_seconds = [sample_wall_seconds(record) for record in full_round_records]
    pending = max(0, len(samples) - len(records))
    remaining = runtime_status.get("pending_to_attempt", pending)
    remaining = max(0, remaining) if isinstance(remaining, int) else pending
    effective = elapsed / len(records) if records and elapsed > 0 else None
    eta = (remaining * effective if effective is not None and not final else
           0.0 if remaining == 0 else None)
    return {
        "session_seconds": elapsed, "session_seconds_source": source,
        "session_seconds_final": final, "run_status": current_status,
        "successful_samples_per_hour": len(valid) * 3600 / elapsed if elapsed > 0 else None,
        "finalized_records_per_hour": len(records) * 3600 / elapsed if elapsed > 0 else None,
        "effective_seconds_per_successful_sample": elapsed / len(valid) if valid and elapsed > 0 else None,
        "effective_seconds_per_finalized_record": effective,
        "eta_remaining_samples": remaining, "eta_remaining_seconds": eta,
        "eta_basis": "待处理样本数 × 累计并发墙钟 / 已结束记录数；运行错误另列，不视为成功。",
        "full_three_round_sample_wall_seconds": latency_summary(
            [value for value in full_round_seconds if value is not None]),
        "single_sample_timing_definition": "每条完整三轮样本从开始到最终结束的墙钟时间，包含模型排队、检索和推理。",
        "throughput_timing_definition": "累计评估墙钟 / 完成数量，体现样本并发收益，不能作为单条延迟。",
    }


def duration(value):
    if value is None:
        return "待估算"
    if value >= 3600:
        return f"{value / 3600:.2f} 小时"
    if value >= 60:
        return f"{value / 60:.1f} 分钟"
    return f"{value:.1f} 秒"


def wilson(correct, count):
    if not count:
        return [None, None]
    z, p = 1.959963984540054, correct / count
    center = (p + z * z / (2 * count)) / (1 + z * z / count)
    margin = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / (1 + z * z / count)
    return [center - margin, center + margin]


def classification_metrics(truth, predictions, labels, total):
    if not truth:
        return {"accuracy": None, "correct": 0, "correct_over_all_selected": 0,
                "macro_precision": None, "macro_recall": None,
                "macro_f1": None, "confusion_matrix": [[0] * len(labels) for _ in labels], "per_class": {}}
    precision, recall, f1, support = precision_recall_fscore_support(truth, predictions, labels=labels, zero_division=0)
    correct = sum(a == b for a, b in zip(truth, predictions))
    return {"accuracy": float(accuracy_score(truth, predictions)), "correct": correct,
            "correct_over_all_selected": correct / total if total else None,
            "macro_precision": float(np.mean(precision)), "macro_recall": float(np.mean(recall)),
            "macro_f1": float(np.mean(f1)), "accuracy_wilson_95": wilson(correct, len(truth)),
            "confusion_matrix": confusion_matrix(truth, predictions, labels=labels).tolist(),
            "per_class": {str(label): {"precision": float(precision[i]), "recall": float(recall[i]),
                                       "f1": float(f1[i]), "support": int(support[i])} for i, label in enumerate(labels)}}


def compute_metrics(samples, records):
    source = {row['sample_id']: row for row in samples}
    if len(source) != len(samples) or len({r['sample_id'] for r in records}) != len(records):
        raise ValueError("Duplicate sample or prediction ID")
    if any(r['sample_id'] not in source for r in records):
        raise ValueError("Prediction is not in the frozen selection")
    if any(r.get('ground_truth', source[r['sample_id']]['ground_truth']) != source[r['sample_id']]['ground_truth'] for r in records):
        raise ValueError("Ground truth differs from frozen selection")
    valid = [r for r in records if r['status'] == 'ok']
    if any(r.get('prediction') not in LABELS for r in valid):
        raise ValueError("Invalid prediction marked successful")
    truth = [source[r['sample_id']]['ground_truth'] for r in valid]
    predictions = [r['prediction'] for r in valid]
    return {"selected": len(samples), "attempted": len(records), "valid": len(valid),
            "errors": sum(r['status'] != 'ok' for r in records), "pending": len(samples) - len(records),
            "coverage": len(valid) / len(samples) if samples else 0,
            "class_counts": dict(Counter(row['ground_truth'] for row in samples)),
            "prediction_counts": dict(Counter(predictions)),
            "four_class": classification_metrics(truth, predictions, LABELS, len(samples)),
            "binary": classification_metrics([int(x != 'original') for x in truth],
                                             [int(x != 'original') for x in predictions], [0, 1], len(samples))}


def percent(value):
    return "—" if value is None else f"{value * 100:.1f}%"


def escape(value):
    return html.escape(str(value))


def is_diagnostic_error(event):
    return event.get('status') == 'error' or 'error_type' in event


def evidence_metrics(records):
    valid = [r for r in records if r['status'] == 'ok']
    events = [e for r in records for e in r.get('retrieval_diagnostics', [])]
    return {
        'reverse_statuses': dict(Counter(r.get('reverse_image_status', 'unrecorded') for r in records)),
        'samples_with_reverse_evidence': sum(r.get('reverse_image_used', False) for r in valid),
        'samples_with_web_search': sum(any(e.get('stage') == 'search' for e in r.get('retrieval_diagnostics', [])) for r in valid),
        'samples_with_retained_web_evidence': sum(any('URL:' in item for item in r['result'].get('retrieved_knowledge_only', [])) for r in valid),
        'search_event_statuses': dict(Counter(e.get('status', 'error' if is_diagnostic_error(e) else 'unrecorded')
                                             for e in events if e.get('stage') == 'search')),
        'routing_decisions': dict(Counter(e.get('decision') for e in events if e.get('stage') == 'routing')),
    }


def table(headers, rows):
    return "<div class='scroll'><table><thead><tr>" + "".join(f"<th>{escape(x)}</th>" for x in headers) + \
        "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{escape(x)}</td>" for x in row) + "</tr>" for row in rows) + "</tbody></table></div>"


def source_class(sample):
    value = sample.get('source_class', sample.get('fake_cls', sample.get('ground_truth', '未记录')))
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def dataset_profile(manifest, samples):
    """Describe only the current frozen selection and its recorded provenance."""
    description = str(manifest.get('dataset') or '清单所列数据集')
    purpose = str(manifest.get('purpose', ''))
    diagnostic = bool(manifest.get('diagnostic_subset') or 'smoke' in purpose.lower())
    if diagnostic:
        description += '（流程试跑子集）'
    source_values = [manifest.get(key, '') for key in
                     ('dataset', 'source', 'source_url', 'upstream_repository')]
    source_values.extend(manifest.get('source_samples', []))
    source_text = ' '.join(str(value) for value in source_values).lower()
    known_datasets = [name for name in ('DGM4', 'MMFakeBench') if name.lower() in source_text]
    metadata = manifest.get('metadata_sha256', manifest.get('source_metadata_sha256'))
    return {
        'description': description,
        'diagnostic_only': diagnostic,
        'datasets': known_datasets,
        'subclass_counts': dict(Counter(source_class(sample) for sample in samples)),
        'metadata_sha256': metadata,
        'revision': manifest.get('revision'),
        'seed': manifest.get('seed'),
        'source': manifest.get('source'),
        'source_url': manifest.get('source_url'),
        'source_samples': manifest.get('source_samples', []),
        'official_split': manifest.get('official_split'),
    }


def dataset_sources_html(profile):
    links = [("MAR 官方代码", "https://github.com/Averdgr/MAR_KDD")]
    if 'DGM4' in profile['datasets']:
        links.append(("DGM4 官方数据", "https://huggingface.co/datasets/rshaojimmy/DGM4"))
    if 'MMFakeBench' in profile['datasets']:
        links.append(("MMFakeBench 官方数据", "https://huggingface.co/datasets/liuxuannan/MMFakeBench"))
    source_url = profile.get('source_url')
    if isinstance(source_url, str) and urlparse(source_url).scheme in ('http', 'https'):
        links.append(("清单记录的原始来源", source_url))
    return "<p>来源：" + "；".join(
        f"<a href='{escape(url)}'>{escape(label)}</a>" for label, url in links) + "。</p>"


def paper_metric_rows(metrics, reference=False):
    rows = []
    for dataset, group in metrics['by_dataset'].items():
        values = PAPER['reported_results'].get(dataset, {}) if reference else group
        for task, label in (('binary', '真假二分类'), ('four_class', '四分类')):
            if task not in values:
                continue
            rows.append([
                dataset, label, "论文 1000 条" if reference else f"{group['valid']} / {group['selected']}",
                *[percent(values[task][name]) for name in PAPER['metrics']],
            ])
    return rows


def timing_rows(metrics, config):
    runtime = metrics['runtime']
    full = runtime['full_three_round_sample_wall_seconds']
    return [
        ['累计并发墙钟', duration(runtime['session_seconds'])],
        ['墙钟数据来源', runtime['session_seconds_source']],
        ['配置并发 / 辩论轮数', f"{config['workers']} / {config['rounds']}"],
        ['完整三轮成功样本数', full['count']],
        ['单条完整三轮墙钟：平均 / 中位 / P95',
         " / ".join(duration(full[key]) for key in ('mean', 'median', 'p95'))],
        ['单条完整三轮墙钟：最短 / 最长',
         " / ".join(duration(full[key]) for key in ('min', 'max'))],
        ['成功吞吐率', f"{runtime['successful_samples_per_hour']:.2f} 条/小时"
         if runtime['successful_samples_per_hour'] is not None else '待首批完成后估算'],
        ['已结束吞吐率（含错误）', f"{runtime['finalized_records_per_hour']:.2f} 条/小时"
         if runtime['finalized_records_per_hour'] is not None else '待首批完成后估算'],
        ['并发后的有效每成功条秒数', duration(runtime['effective_seconds_per_successful_sample'])],
        ['并发后的有效每结束条秒数', duration(runtime['effective_seconds_per_finalized_record'])],
        ['预计剩余时间', duration(runtime['eta_remaining_seconds'])],
        ['ETA 对应待处理数量', runtime['eta_remaining_samples']],
    ]


def write_docx(output, metrics, samples, records, config, manifest):
    """Write a portable Chinese Word report using standard OOXML only."""
    word_ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    ET.register_namespace("w", word_ns)
    w = lambda name: "{" + word_ns + "}" + name
    document = ET.Element(w("document"))
    body = ET.SubElement(document, w("body"))

    def text(value):
        return "".join(character for character in str(value)
                       if character in "\t\n\r" or ord(character) >= 32)

    def paragraph(value, *, style=None, parent=body):
        node = ET.SubElement(parent, w("p"))
        if style:
            properties = ET.SubElement(node, w("pPr"))
            ET.SubElement(properties, w("pStyle"), {w("val"): style})
        run = ET.SubElement(node, w("r"))
        ET.SubElement(run, w("t"), {"{http://www.w3.org/XML/1998/namespace}space": "preserve"}).text = text(value)
        return node

    def word_table(headers, rows):
        node = ET.SubElement(body, w("tbl"))
        properties = ET.SubElement(node, w("tblPr"))
        ET.SubElement(properties, w("tblW"), {w("w"): "5000", w("type"): "pct"})
        borders = ET.SubElement(properties, w("tblBorders"))
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            ET.SubElement(borders, w(edge), {w("val"): "single", w("sz"): "4", w("color"): "D1DCE8"})
        grid = ET.SubElement(node, w("tblGrid"))
        column_width = str(9638 // len(headers))
        for _ in headers:
            ET.SubElement(grid, w("gridCol"), {w("w"): column_width})
        for index, row in enumerate([headers, *rows]):
            tr = ET.SubElement(node, w("tr"))
            if index == 0:
                ET.SubElement(ET.SubElement(tr, w("trPr")), w("tblHeader"))
            for value in row:
                cell = ET.SubElement(tr, w("tc"))
                cell_properties = ET.SubElement(cell, w("tcPr"))
                ET.SubElement(cell_properties, w("tcW"), {w("w"): column_width, w("type"): "dxa"})
                if index == 0:
                    ET.SubElement(cell_properties, w("shd"), {w("fill"): "EAF0F9"})
                paragraph(value, style="TableText", parent=cell)
        paragraph("")

    paragraph("MAR 图文检测运行报告", style="Title")
    paragraph("自动保存时间：" + metrics['generated_at'])
    paragraph(metrics['dataset']['description'])
    paragraph(metrics['comparison_scope'])
    if metrics['dataset']['diagnostic_only']:
        paragraph("本次仅作集成流程试跑，不用于估计模型准确率。")
    paragraph("运行进度", style="Heading1")
    word_table(["范围", "选中", "已结束", "成功", "错误", "待处理"], [
        ["全部", metrics['selected'], metrics['finalized'], metrics['valid'], metrics['errors'], metrics['pending']],
        *[[name, group['selected'], group['attempted'], group['valid'], group['errors'], group['pending']]
          for name, group in metrics['by_dataset'].items()],
    ])
    paragraph("成功指获得有效最终分类；运行错误单独统计，不充当真实或伪造类别。"
              "准确率及宏平均指标只以成功预测为分母；尚无成功预测时显示“—”。")
    paragraph("按论文口径的分数据集指标", style="Heading1")
    paragraph(PAPER['metric_source'] + "。" + PAPER['averaging'])
    paragraph("四分类固定真实、文字失真、视觉失真、图文不符四类；二分类把后三类合为假。"
              "阶段报告缺少的类别仍保留在宏平均中，不能把早期少量样本指标当最终结果。")
    headers = ["数据集", "任务", "成功/选中", "Macro-F1", "Accuracy", "Macro-Precision", "Macro-Recall"]
    word_table(headers, paper_metric_rows(metrics))
    if paper_metric_rows(metrics, reference=True):
        paragraph("论文 MAR 原配置参考值", style="Heading2")
        paragraph("以下是论文完整测试集参考值。骨干模型、检索和样本范围可能不同，不据此作公平优劣或显著性结论。")
        word_table(headers, paper_metric_rows(metrics, reference=True))
    paragraph("合并指标（补充）", style="Heading2")
    word_table(["任务", "Macro-F1", "Accuracy", "Macro-Precision", "Macro-Recall"], [
        [label, *[percent(metrics[task][key]) for key in PAPER['metrics']]]
        for task, label in (("binary", "真假二分类"), ("four_class", "四分类"))
    ])
    for name, group in metrics['by_dataset'].items():
        paragraph(name + " 四分类混淆矩阵", style="Heading2")
        paragraph("行是真实标签，列是预测标签；仅包含成功预测。")
        word_table(["真实→预测", *[NAMES[label] for label in LABELS]], [
            [NAMES[label], *group['four_class']['confusion_matrix'][index]]
            for index, label in enumerate(LABELS)])
    paragraph("完整流程耗时、吞吐与预计剩余时间", style="Heading1")
    word_table(["指标", "当前实测"], timing_rows(metrics, config))
    paragraph(metrics['runtime']['single_sample_timing_definition'])
    paragraph(metrics['runtime']['throughput_timing_definition'])
    paragraph(metrics['runtime']['eta_basis'] + " 估计会随检索、排队和重试变化；停止后不承诺剩余完成时间。"
              "单样本耗时之和仅用于累计工作量和模型等待占比，不作为并行总墙钟。")
    paragraph("检索与运行配置", style="Heading1")
    word_table(["项目", "配置或实测"], [
        ["推理模型", config['model']], ["推理服务", config['base_url']],
        ["温度", config['temperature']], ["搜索提供者", config.get('search_provider', '未记录')],
        ["缓存模式", config.get('cache_mode', '未记录')],
        ["使用反向图像说明的成功样本", metrics['evidence']['samples_with_reverse_evidence']],
        ["保留网页证据的成功样本", metrics['evidence']['samples_with_retained_web_evidence']],
        ["模型调用 / 错误", f"{metrics['runtime']['llm_calls']} / {metrics['runtime']['llm_errors']}"],
        ["输入 / 输出 tokens", f"{metrics['runtime']['input_tokens']} / {metrics['runtime']['output_tokens']}"],
        ["样本清单 SHA-256", config['input_sha256']],
        ["证据清单 SHA-256", config.get('reverse_evidence_sha256', '未记录')],
    ])
    paragraph("逐条完整流程状态与墙钟", style="Heading1")
    paragraph("错误记录的耗时是失败前已发生的耗时；只有状态成功且实际完成三轮才属于完整三轮延迟统计。")
    by_id = {record['sample_id']: record for record in records}
    detail_rows = []
    for sample in samples:
        record = by_id.get(sample['sample_id'])
        if record is None:
            detail_rows.append([sample_dataset(sample, manifest), sample['sample_id'], '待处理', '—', '—'])
        else:
            rounds = record.get('result', {}).get('current_round', '—')
            detail_rows.append([sample_dataset(sample, manifest), sample['sample_id'],
                                '成功' if record['status'] == 'ok' else '错误',
                                rounds, duration(sample_wall_seconds(record))])
    word_table(["数据集", "样本编号", "状态", "实际轮数", "单条墙钟"], detail_rows)
    paragraph("来源与审计文件", style="Heading1")
    paragraph("论文：" + PAPER['doi'])
    paragraph("宏平均实现：" + PAPER['averaging_source'])
    paragraph("本地论文：" + PAPER['local_pdf'])
    paragraph("完整指标：metrics.json；逐条结果：details.csv；网页报告：report.html；"
              "模型及检索配置：config.json；样本来源：dataset-manifest.json；原始推理：records/ 与 calls/。")
    section = ET.SubElement(body, w("sectPr"))
    ET.SubElement(section, w("pgSz"), {w("w"): "11906", w("h"): "16838"})
    ET.SubElement(section, w("pgMar"), {w("top"): "1134", w("bottom"): "1134",
                                     w("left"): "1134", w("right"): "1134"})
    xml = ET.tostring(document, encoding="utf-8", xml_declaration=True)
    styles = f'''<?xml version="1.0" encoding="UTF-8"?>
<w:styles xmlns:w="{word_ns}">
<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:eastAsia="Microsoft YaHei"/><w:sz w:val="21"/></w:rPr></w:rPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:pPr><w:spacing w:after="100" w:line="300" w:lineRule="auto"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="36"/><w:color w:val="153B69"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="240" w:after="120"/></w:pPr><w:rPr><w:b/><w:sz w:val="28"/><w:color w:val="153B69"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/></w:pPr><w:rPr><w:b/><w:sz w:val="24"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="TableText"><w:name w:val="Table text"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:after="50"/></w:pPr><w:rPr><w:sz w:val="18"/></w:rPr></w:style>
</w:styles>'''
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", '''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>''')
        archive.writestr("_rels/.rels", '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>''')
        archive.writestr("word/document.xml", xml)
        archive.writestr("word/styles.xml", styles)
        archive.writestr("word/_rels/document.xml.rels", '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>''')
    atomic_bytes(output / "report.docx", buffer.getvalue())


def generate_report(output):
    output = Path(output)
    samples = json.loads((output / "samples.json").read_text())
    config = json.loads((output / "config.json").read_text())
    manifest = json.loads((output / "dataset-manifest.json").read_text())
    reference_path = output / "reference-captions.json"
    references = json.loads(reference_path.read_text()) if reference_path.exists() else {}
    records = [json.loads(p.read_text()) for p in sorted((output / "records").glob("*.json"))]
    attempts = [json.loads(p.read_text()) for p in sorted((output / "attempts").glob("*.json"))]
    metrics = compute_metrics(samples, records)
    metrics['evidence'] = evidence_metrics(records)
    metrics['by_dataset'] = dataset_metrics(samples, records, manifest)
    metrics['paper_protocol'] = PAPER
    full_author_tests = (
        set(metrics['by_dataset']) == {"DGM4", "MMFakeBench"}
        and all(group['selected'] == PAPER['test_samples_per_dataset']
                for group in metrics['by_dataset'].values()))
    metrics['selection_scope'] = "full_author_test_lists" if full_author_tests else "selected_subset"
    metrics['comparison_scope'] = (
        f"本次使用 {config.get('model', '未记录模型')}、{config.get('search_provider', '未记录检索')} "
        f"与持久缓存，选中 {len(samples)} 条；论文使用每数据集 1,000 条及 "
        "Qwen3-VL-32B-Instruct-FP8。此处按论文指标口径报告"
        + ("作者公开测试清单的全量变体" if full_author_tests else "所选子集的变体")
        + "，不能视作论文原模型和检索配置的严格复现。")
    metrics['generated_at'] = datetime.now(timezone.utc).isoformat()
    profile = dataset_profile(manifest, samples)
    metrics['dataset'] = profile
    dataset_description = escape(profile['description'])
    seed_description = f"固定种子 {escape(profile['seed'])}" if profile['seed'] is not None else "清单未记录随机种子"
    metrics['selection'] = {key: manifest[key] for key in
                            ('split', 'source_split', 'official_split', 'parent_count', 'eligible_count',
                             'selection_rule', 'selection_bias', 'seed', 'selection', 'purpose', 'source_samples')
                            if key in manifest}
    status_path = output / 'status.json'
    execution_status = json.loads(status_path.read_text()) if status_path.exists() else {}
    runtime_path = output / 'runtime-status.json'
    runtime_status = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    progress_path = output / 'progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    stopped = (execution_status.get('status') == 'stopped_by_user'
               or runtime_status.get('status') == 'stopped_by_user')
    metrics['execution'] = execution_status
    metrics['finalized'] = len(records)
    if stopped:
        metrics['attempted'] = len(records) + len(execution_status.get('interrupted', []))
    word_counts = [len(row['text'].split()) for row in samples]
    metrics['data_profile'] = {'words_min': min(word_counts), 'words_median': float(np.median(word_counts)),
                               'words_max': max(word_counts), 'under_10_words': sum(n < 10 for n in word_counts),
                               'image_width_median': float(np.median([row['image_size'][0] for row in samples])),
                               'image_height_median': float(np.median([row['image_size'][1] for row in samples]))}
    valid = [r for r in records if r['status'] == 'ok']
    seconds = [value for r in valid if (value := sample_wall_seconds(r)) is not None]
    all_diagnostics = [d for r in attempts for d in r.get('retrieval_diagnostics', [])]
    sessions = read_jsonl(output / "sessions.jsonl")
    timing = run_timing(samples, records, config, sessions, execution_status, runtime_status, progress)
    elapsed, elapsed_final = timing['session_seconds'], timing['session_seconds_final']
    calls = [row for p in sorted((output / "calls").glob("*.jsonl")) for row in read_jsonl(p)]
    metrics['runtime'] = {
        "session_seconds": elapsed, "attempts": len(attempts),
        "session_seconds_final": elapsed_final,
        "initial_failures": sum(r['attempt'] == 1 and r['status'] != 'ok' for r in attempts),
        "retry_attempts": sum(r['attempt'] > 1 for r in attempts),
        "sample_seconds_mean": float(np.mean(seconds)) if seconds else None,
        "sample_seconds_p50": float(np.percentile(seconds, 50)) if seconds else None,
        "sample_seconds_p95": float(np.percentile(seconds, 95)) if seconds else None,
        "sample_seconds_max": max(seconds) if seconds else None,
        "llm_calls": len(calls), "llm_errors": sum(c['status'] != 'ok' for c in calls),
        "input_tokens": sum(c.get('usage', {}).get('input_tokens', 0) for c in calls),
        "output_tokens": sum(c.get('usage', {}).get('output_tokens', 0) for c in calls),
        "truncated_responses": sum(c.get('response_metadata', {}).get('finish_reason') == 'length' for c in calls),
        "samples_with_retrieval_diagnostics": sum(any(is_diagnostic_error(e) for e in r.get('retrieval_diagnostics', [])) for r in valid),
        "retrieval_diagnostic_counts": dict(Counter(d['stage'] for d in all_diagnostics)),
        "search_error_types": dict(Counter(d.get('error_type', 'unknown') for d in all_diagnostics if d['stage'] == 'search' and is_diagnostic_error(d))),
        "search_error_messages": dict(Counter(
            'no_results' if 'No results found' in d.get('error', '') else
            'timeout' if 'timeout' in d.get('error', '').lower() else
            'transport' if 'RequestError' in d.get('error', '') or 'error sending request' in d.get('error', '') else 'other'
            for d in all_diagnostics if d['stage'] == 'search' and is_diagnostic_error(d))),
        "samples_with_retained_web_evidence": sum(any('URL:' in item for item in r['result'].get('retrieved_knowledge_only', [])) for r in valid),
        "samples_with_summary_fallback": sum(bool(r['result'].get('runtime_warnings')) for r in valid),
        "completed_configured_rounds": sum(r['result'].get('current_round') == config['rounds'] and len(r['result'].get('short_term_memories', [])) == config['rounds'] for r in valid),
        "completed_three_rounds": timing['full_three_round_sample_wall_seconds']['count'],
        **timing,
    }
    metrics['subclasses'] = {}
    sample_classes = {sample['sample_id']: source_class(sample) for sample in samples}
    for subclass in sorted(profile['subclass_counts']):
        group = [r for r in valid if sample_classes[r['sample_id']] == subclass]
        metrics['subclasses'][subclass] = {
            'selected': profile['subclass_counts'][subclass], 'valid': len(group),
            'four_class_correct': sum(r['prediction'] == r['ground_truth'] for r in group),
            'binary_correct': sum((r['prediction'] != 'original') == (r['ground_truth'] != 'original') for r in group),
        }
    metrics['judgment_distributions'] = {
        label: {dimension: dict(Counter(r['result'].get('judgments', {}).get(dimension, {}).get('conclusion', 'missing')
                                       for r in valid if r['ground_truth'] == label))
                for dimension in ('text', 'image', 'consistency')}
        for label in LABELS
    }
    completed_calls = []
    for record in valid:
        path = output / 'calls' / f"{record['sample_id']}-{record['attempt']}.jsonl"
        if path.exists():
            for call in read_jsonl(path):
                if (datetime.fromisoformat(record['started_at']) <= datetime.fromisoformat(call['timestamp'])
                        <= datetime.fromisoformat(record['finished_at'])):
                    completed_calls.append(call)
    sample_seconds = sum(seconds)
    metrics['runtime'].update(
        completed_sample_llm_seconds=sum(c['seconds'] for c in completed_calls),
        completed_sample_seconds=sample_seconds,
        llm_wait_fraction=sum(c['seconds'] for c in completed_calls) / sample_seconds if sample_seconds else None,
        calls_per_completed_sample=len(completed_calls) / len(valid) if valid else None,
        output_tokens_per_completed_sample=sum(c.get('usage', {}).get('output_tokens', 0) for c in completed_calls) / len(valid) if valid else None,
    )
    metrics['baselines'] = {
        'always_fake_selected_accuracy': sum(s['ground_truth'] != 'original' for s in samples) / len(samples),
        'always_fake_valid_accuracy': sum(r['ground_truth'] != 'original' for r in valid) / len(valid) if valid else None,
        'majority_four_class_valid_accuracy': max(Counter(r['ground_truth'] for r in valid).values()) / len(valid) if valid else None,
    }
    # Fixed selection: first two errors per ground-truth class in sample-ID order.
    cases = []
    for label in LABELS:
        cases.extend([r for r in valid if r['ground_truth'] == label and r['prediction'] != label][:2])
    metrics['case_sample_ids'] = [r['sample_id'] for r in cases]
    metrics['original_caption_references_available'] = len(references)
    atomic_text(output / "metrics.json", json.dumps(metrics, ensure_ascii=False, indent=2))
    with io.StringIO(newline="") as stream:
        fields = ["sample_id", "news_id", "ground_truth", "prediction", "correct", "status", "attempt", "seconds",
                  "llm_calls", "input_tokens", "output_tokens", "reverse_image_status", "reverse_image_source",
                  "reverse_image_used", "reverse_image_captions_sha256", "retained_web_urls",
                  "text", "original_text", "text_reason", "image_reason", "consistency_reason", "error"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            result = record.get('result', {})
            row = {key: record.get(key, "") for key in fields}
            row.update(correct=record['prediction'] == record['ground_truth'] if record['status'] == 'ok' else "",
                       retained_web_urls=sum(item.count('URL:') for item in result.get('retrieved_knowledge_only', [])),
                       original_text=references.get(record['sample_id'], {}).get('original_text', ''),
                       **{key: result.get(key, "") for key in ('text_reason', 'image_reason', 'consistency_reason')})
            writer.writerow(row)
        atomic_text(output / "details.csv", "\ufeff" + stream.getvalue())
    four, binary, runtime = metrics['four_class'], metrics['binary'], metrics['runtime']
    complete = metrics['valid'] == len(samples)
    parts = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
             "<title>MAR 图文测试分析报告</title><style>body{font:16px/1.7 system-ui,'Noto Sans CJK SC',sans-serif;color:#172334;background:#f2f5fa;margin:0}main{max-width:1140px;margin:auto;padding:36px 24px}h1{font-size:32px;line-height:1.3}h2{margin-top:38px;color:#153b69}h3{margin-bottom:8px}.card,details{background:white;border:1px solid #dbe3ee;border-radius:10px;padding:20px;margin:16px 0}.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}.kpis .card{margin:0}.value{font-size:30px;font-weight:700;color:#155c9b}table{border-collapse:collapse;width:100%;background:white}th,td{border:1px solid #dbe3ee;padding:9px 12px;text-align:left;vertical-align:top}th{background:#eaf0f9}.scroll{overflow:auto}.muted{color:#586779}code{overflow-wrap:anywhere;background:#e9eef5;padding:2px 4px}a{color:#145da3}img{max-width:360px;max-height:250px;object-fit:contain}.bad{color:#a92c29}.good{color:#16724b}summary{cursor:pointer;font-weight:600}blockquote{margin:12px 0;padding:10px 18px;border-left:3px solid #92b4d8;background:#f7f9fc}input,select{font:inherit;padding:8px;margin:5px;max-width:90%}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}@media print{body{background:white}main{max-width:none}.controls{display:none}details{break-inside:avoid}h2{break-after:avoid}}</style><main>",
             f"<p class='muted'>MAR 实测 · {dataset_description} · {seed_description}</p>",
             f"<h1>MAR 图文测试分析报告：{metrics['valid']} 条完整结果</h1>",
             "<p class='bad'><strong>流程验证范围：</strong>本次仅作集成流程试跑，不用于估计模型准确率。下列分类指标只描述这次选中的样本，不能视作两个完整测试集的评测结果。</p>" if profile['diagnostic_only'] else "",
             f"<p><strong>实验范围：</strong>{escape(metrics['comparison_scope'])}</p>",
             f"<p class='bad'><strong>抽样范围：</strong>{escape(manifest['selection_bias'])}</p>" if manifest.get('selection_bias') else "",
             f"<p>{'测试已按用户要求停止，以下为本次收尾报告' if stopped else '测试已完成' if complete else '阶段报告：测试尚未全部完成'}。原计划 {len(samples)} 条，成功 {metrics['valid']} 条，运行失败 {metrics['errors']} 条，未获得最终预测 {metrics['pending']} 条。标签不进入推理输入。</p>",
             "<div class='kpis'>" + ''.join(f"<div class='card'>{title}<div class='value'>{value}</div></div>" for title, value in [
                 ('成功覆盖率', percent(metrics['coverage'])), ('四分类准确率', percent(four['accuracy'])),
                 ('四分类 Macro-F1', percent(four['macro_f1'])), ('真假二分类准确率', percent(binary['accuracy']))]) + "</div>",
             "<h2>如何理解结果</h2>",
             f"<p>准确率仅以成功返回的 {metrics['valid']} 条为分母，四分类正确 {four['correct']} 条。未完成样本不算预测错误，也不算预测正确。二分类将三类伪造合为“假”，真实为“真”。运行错误不充当任何类别预测。</p>",
             f"<p>本次使用 {dataset_description}，选中 {len(samples)} 条，类别和完成数量以冻结的样本清单为准。划分记录：{escape(manifest.get('split', manifest.get('source_split', '未记录')))}；官方原始划分：{escape(manifest.get('official_split', '未记录'))}；路径恢复方法：{escape(manifest.get('path_recovery', '未记录'))}。</p>",
             f"<p>成功样本中，反向搜图说明进入推理 {metrics['evidence']['samples_with_reverse_evidence']} 条，保留网页证据 {runtime['samples_with_retained_web_evidence']} 条。实时网络检索随时间变化。论文原文的模型、协议和结果表仍须逐项核对，此报告不自动代表论文指标复现。</p>"]
    if metrics['valid']:
        ci = four['accuracy_wilson_95']
        parts.append(f"<p>四分类准确率的 Wilson 95% 参考区间为 {percent(ci[0])}～{percent(ci[1])}。该区间仅作小样本误差参考，未校正分层抽样设计，不能直接外推整个数据集。</p>")
    if stopped:
        parts.append(f"<p>停止时有 {len(execution_status.get('interrupted', []))} 条正在运行、{execution_status.get('not_started', 0)} 条尚未开始。中途停止是用户选择，不计作模型运行失败；其已发生的调用开销仍保留在成本统计中。早停和执行顺序可能影响已完成子集的类别分布，不代表原定 {len(samples)} 条结果。</p>")
    if (complete or stopped) and metrics['valid']:
        real, visual = four['per_class']['original'], four['per_class']['visual_veracity_distortion']
        fake_baseline = metrics['baselines']['always_fake_valid_accuracy']
        gain = (binary['accuracy'] - fake_baseline) * 100
        parts.append("<div class='card'><h3>本轮测试的主要结论</h3>" +
                     f"<p>现有 {metrics['valid']} 条完整结果中，真假判断正确 {binary['correct']} 条；相对于同一子集始终判假的 {percent(fake_baseline)} 基线，准确率差值为 {gain:+.1f} 个百分点。这是描述性比较，不能据此声称统计显著改进。</p>" +
                     f"<p>真实新闻召回率为 {percent(real['recall'])}，图像篡改召回率为 {percent(visual['recall'])}。模型为 {runtime['samples_with_retained_web_evidence']} 条保留过网页证据，{metrics['evidence']['samples_with_reverse_evidence']} 条使用了反向搜图说明。需要同时关注误报、漏检和证据质量。</p></div>")
    parts.extend([
        "<h2>按论文口径的分数据集结果</h2>",
        f"<p>{escape(PAPER['metric_source'])}。{escape(PAPER['averaging'])}"
        "本次固定二类和四类标签集合，阶段报告尚未出现的类别仍参与宏平均；无成功预测时指标显示“—”。</p>",
        table(["数据集", "选中", "已结束", "成功", "错误", "待处理"], [
            [name, group['selected'], group['attempted'], group['valid'], group['errors'], group['pending']]
            for name, group in metrics['by_dataset'].items()]),
        table(["数据集", "任务", "成功 / 选中", "Macro-F1", "Accuracy", "Macro-Precision", "Macro-Recall"],
              paper_metric_rows(metrics)),
    ])
    if paper_metric_rows(metrics, reference=True):
        parts.extend([
            "<h3>论文 MAR 原配置参考值</h3><p>论文完整测试集参考值如下。模型、检索及样本范围可能不同，此表不构成公平优劣或显著性比较。</p>",
            table(["数据集", "任务", "范围", "Macro-F1", "Accuracy", "Macro-Precision", "Macro-Recall"],
                  paper_metric_rows(metrics, reference=True)),
            f"<p>依据：<a href='{PAPER['doi']}'>MAR 论文</a>；"
            f"<a href='{PAPER['averaging_source']}'>作者指标计算代码</a>。</p>",
        ])
    parts.extend(["<h2>分类表现（合并，补充）</h2>", table(['类别', '选中数量', '成功数量', 'Precision', 'Recall', 'F1'],
        [[NAMES[label], metrics['class_counts'].get(label, 0), four['per_class'].get(label, {}).get('support', 0),
          *[percent(four['per_class'].get(label, {}).get(key)) for key in ('precision', 'recall', 'f1')]] for label in LABELS]),
        "<h3>四分类混淆矩阵</h3><p class='muted'>行是真实标签，列是预测标签。对角线为正确预测。</p>",
        table(['四分类汇总指标', '值'], [[name, percent(four[key])] for name, key in
              [('Accuracy', 'accuracy'), ('Macro-Precision', 'macro_precision'), ('Macro-Recall', 'macro_recall'), ('Macro-F1', 'macro_f1')]]),
        table(['真实标签 → 预测标签', *[NAMES[x] for x in LABELS]], [[NAMES[label], *four['confusion_matrix'][i]] for i, label in enumerate(LABELS)]),
        "<h3>真假二分类</h3>", table(['真实标签 → 预测标签', '真', '假'], [['真', *binary['confusion_matrix'][0]], ['假', *binary['confusion_matrix'][1]]]),
        table(['指标', '值'], [['Macro-F1', percent(binary['macro_f1'])], ['Macro-Precision', percent(binary['macro_precision'])],
                              ['Macro-Recall', percent(binary['macro_recall'])], ['假新闻 Precision', percent(binary['per_class'].get('1', {}).get('precision'))],
                              ['假新闻 Recall', percent(binary['per_class'].get('1', {}).get('recall'))],
                              ['始终判假基线（全部选中样本）', percent(metrics['baselines']['always_fake_selected_accuracy'])],
                              ['始终判假基线（当前成功样本）', percent(metrics['baselines']['always_fake_valid_accuracy'])]]),
        "<h3>原始篡改细类</h3>", table(['原始类别', '成功 / 选中', '四分类正确', '真假二分类正确'],
            [[name, f"{group['valid']} / {group['selected']}", group['four_class_correct'], group['binary_correct']]
             for name, group in metrics['subclasses'].items()]),
        "<h2>完整三轮耗时与并发吞吐</h2>", table(['指标', '当前实测'], timing_rows(metrics, config)),
        f"<p>{escape(runtime['single_sample_timing_definition'])} "
        f"{escape(runtime['throughput_timing_definition'])} {escape(runtime['eta_basis'])}"
        "预计剩余时间会随排队、检索和重试变化。单样本耗时之和不作为并行总墙钟。</p>",
        "<h2>运行成本与检索</h2>", table(['项目', '实测'], [
            ['运行会话累计墙钟时间' if elapsed_final else '累计运行墙钟（尚未结束）', f"{elapsed / 60:.1f} 分钟（并发 {config['workers']}）"],
            ['成功样本平均 / 中位 / P95 耗时', ' / '.join(f"{runtime[k]:.1f} 秒" if runtime[k] is not None else '—' for k in ('sample_seconds_mean', 'sample_seconds_p50', 'sample_seconds_p95'))],
            ['完成配置轮数且具有对应短期记忆', f"{runtime['completed_configured_rounds']} / {metrics['valid']}"],
            ['初次运行失败 / 额外尝试', f"{runtime['initial_failures']} / {runtime['retry_attempts']}"],
            ['模型调用 / 错误 / 长度截断', f"{runtime['llm_calls']} / {runtime['llm_errors']} / {runtime['truncated_responses']}"],
            ['输入 / 输出 token', f"{runtime['input_tokens']:,} / {runtime['output_tokens']:,}"],
            ['成功样本平均调用次数 / 输出 token', f"{runtime['calls_per_completed_sample']:.1f} / {runtime['output_tokens_per_completed_sample']:,.0f}" if valid else '—'],
            ['成功样本内模型调用等待占比', percent(runtime['llm_wait_fraction'])],
            ['成功样本中至少保留一条网页证据', f"{runtime['samples_with_retained_web_evidence']} / {metrics['valid']}"],
            ['成功样本中反向搜图说明进入推理', f"{metrics['evidence']['samples_with_reverse_evidence']} / {metrics['valid']}"],
            ['反向搜图状态（已结束样本）', json.dumps(metrics['evidence']['reverse_statuses'], ensure_ascii=False)],
            ['搜索请求状态（已结束样本）', json.dumps(metrics['evidence']['search_event_statuses'], ensure_ascii=False)],
            ['检索路由选择', json.dumps(metrics['evidence']['routing_decisions'], ensure_ascii=False)],
            ['成功样本中记录过检索异常或解析降级', f"{runtime['samples_with_retrieval_diagnostics']} / {metrics['valid']}"],
            ['已结束尝试的检索诊断事件', json.dumps(runtime['retrieval_diagnostic_counts'], ensure_ascii=False)],
            ['搜索错误类型', json.dumps(runtime['search_error_types'], ensure_ascii=False)],
            ['搜索错误消息归类', json.dumps(runtime['search_error_messages'], ensure_ascii=False)],
            ['摘要段落回退样本', runtime['samples_with_summary_fallback']]]),
        f"<p>模型调用等待占比由成功样本各次调用的客户端耗时之和除以样本耗时之和得到，包含服务排队和通信，并非纯 GPU 计算时间。配置的样本并发数为 {config['workers']}；输入 token 按每次调用累计，重复上下文也计入。阶段报告的总调用数可能包含尚未结束样本已经完成的调用。</p>",
        "<p>No results found 既可能来自没有匹配网页，也可能来自搜索引擎响应限制，不能仅凭这一消息确定原因。检索诊断与最终运行失败分别统计；有证据返回也不等于证据能支持正确判断。保存的是搜索摘要，不是网页全文。</p>",
        "<h2>误判样例</h2><p>以下按固定规则选择：每类按样本编号取前两条误判，不根据解释是否好看挑选。模型理由为原始输出，不代表已经得到人工事实核验。</p>"])
    for record in cases:
        result = record['result']
        data = base64.b64encode(Path(record['image_path']).read_bytes()).decode()
        original = references.get(record['sample_id'], {}).get('original_text')
        reference_html = (f"<p class='muted'>官方同ID原始图注（仅用于事后分析）：{escape(original)}</p>"
                          if original and original != record['text'] else "")
        parts.append(f"<div class='card'><h3>{escape(record['sample_id'])}：{NAMES[record['ground_truth']]} → {NAMES[record['prediction']]}</h3><img src='data:image/jpeg;base64,{data}' alt='该样本新闻配图'><blockquote>{escape(record['text'])}</blockquote>" +
                     reference_html +
                     table(['判断维度', '模型结论', '模型原始理由'], [[name, result.get('judgments', {}).get(key, {}).get('conclusion', ''), result.get(reason, '')]
                     for name, key, reason in [('文本', 'text', 'text_reason'), ('图像', 'image', 'image_reason'), ('一致性', 'consistency', 'consistency_reason')]]) + "</div>")
    analysis_path = output / "analysis-notes.html"
    if analysis_path.exists():
        parts.append(analysis_path.read_text())
    for fragment in ('deployment-and-commands.html', 'research-plan.html'):
        fragment_path = output / fragment
        if fragment_path.exists():
            parts.append(fragment_path.read_text())
    parts.extend([f"<h2>项目实际执行流程</h2><p>同一个 {escape(config['model'])} 模型通过不同角色提示词承担多项任务：领域分析 → 检索路由 → 网页搜索、嵌入预筛和相关性评分 → 支持者／质疑者各自草拟、自我批评和修订 → 短期摘要和长期记忆更新，按配置循环 {config['rounds']} 轮 → 汇总 → 文本、图像、一致性三个代理裁决。长期记忆仅在当前样本的轮次之间更新，每条新闻重新初始化，没有跨测试样本学习。各条实际完成轮数和检索结果以保存的记录为准。</p>",
        "<p>保留上游提示词和“文本不真实优先，其次图像不真实，再判断图文一致性”的决策顺序。模型误判不等同于程序 bug；本次没有按测试标签调整提示词。</p>",
        "<h2>复现配置与数据来源</h2>", table(['项目', '配置'], [
            ['模型', config['model']], ['推理温度 / 辩论轮数', f"{config['temperature']} / {config['rounds']}"],
            ['推理服务', config['base_url']],
            ['嵌入模型', f"{config.get('embedding_model', 'BGE-base-en-v1.5')} / {config['embedding_device']}"],
            ['搜索提供者', config.get('search_provider', '未记录')],
            ['DDGS 后端', config.get('search_backends', '未记录') if config.get('search_provider', 'ddgs') == 'ddgs' else '不适用'],
            ['检索缓存模式', config.get('cache_mode', '未记录')],
            ['反向搜图', f"选中样本中 {config.get('reverse_image_caption_count', 0)} 条有说明；严格模式 {config.get('strict_evidence', False)}"],
            ['数据 revision', profile['revision'] or '未记录'],
            ['来源 metadata SHA-256', json.dumps(profile['metadata_sha256'], ensure_ascii=False) if profile['metadata_sha256'] is not None else '未记录'],
            ['来源清单', profile['source'] or json.dumps(profile['source_samples'], ensure_ascii=False) if profile['source'] or profile['source_samples'] else '未记录'],
            ['样本清单 SHA-256', config['input_sha256']], ['证据清单 SHA-256', config.get('reverse_evidence_sha256')],
            ['细类分布（当前选中样本）', json.dumps(profile['subclass_counts'], ensure_ascii=False)]]),
        dataset_sources_html(profile),
        f"<p>原始类别及映射以冻结的数据清单为准；选中 {len(samples)} 条，来自 {manifest.get('unique_news_ids', '未记录')} 个不同新闻ID。选择在推理前冻结，分类失败或误判不会触发样本替换。作者的历史反向搜图说明与本次新增检索分别记录，真实无匹配不保证能够形成有效证据。</p>",
        f"<p>本次保存 {len(references)} 条事后分析参考于 <code>reference-captions.json</code>，原始文本同时附在 CSV 中；参考来源以该文件记录为准。参考文本和真实标签均未进入 MAR 推理。</p>" if references else "<p>本次没有保存单独的事后分析参考图注文件。</p>",
        f"<p>选中的 {len(samples)} 条样本文本词数范围 {min(word_counts)}～{max(word_counts)}，中位数 {np.median(word_counts):.0f} 词；其中 {sum(n < 10 for n in word_counts)} 条不足10词。它们不包含完整报道上下文或统一发布日期。图片宽、高的分别中位数为 {metrics['data_profile']['image_width_median']:.0f}、{metrics['data_profile']['image_height_median']:.0f} 像素。这些数据特征限制了实体消歧、历史事实核查和细微篡改识别。</p>",
        "<p>文件：<a href='report.docx'>中文 Word 报告</a> · <a href='metrics.json'>完整指标 JSON</a> · <a href='details.csv'>逐条结果 CSV</a> · <a href='predictions.jsonl'>预测及推理记录 JSONL</a> · <a href='config.json'>代码摘要与运行配置</a> · <a href='dataset-manifest.json'>数据来源清单</a></p>",
        "<h2>全部样本</h2><div class='controls'><input id='query' placeholder='搜索编号或新闻文本'><select id='filter'><option value='all'>全部</option><option value='wrong'>仅误判</option><option value='correct'>仅正确</option><option value='error'>仅运行失败</option></select></div>"])
    for record in records:
        kind = 'error' if record['status'] != 'ok' else 'correct' if record['prediction'] == record['ground_truth'] else 'wrong'
        label = NAMES.get(record['prediction'], '运行失败')
        result = record.get('result', {})
        parts.append(f"<details class='sample' data-kind='{kind}'><summary>{escape(record['sample_id'])} · {NAMES[record['ground_truth']]} → {label} · {record['seconds']:.0f}秒</summary><p>{escape(record['text'])}</p>" +
                     f"<p>反向搜图状态：{escape(record.get('reverse_image_status', '未记录'))}；说明进入推理：{escape(record.get('reverse_image_used', False))}；保留网页 URL 数：{sum(item.count('URL:') for item in result.get('retrieved_knowledge_only', []))}。</p>" +
                     table(['维度', '理由'], [[name, result.get(key, '')] for name, key in [('文本', 'text_reason'), ('图像', 'image_reason'), ('一致性', 'consistency_reason')]]) +
                     f"<p><a href='records/{escape(record['sample_id'])}.json'>完整推理记录</a></p>{'<pre>' + escape(record.get('error', '')) + '</pre>' if kind == 'error' else ''}</details>")
    parts.append("<script>function filter(){const q=document.getElementById('query').value.toLowerCase(),f=document.getElementById('filter').value;document.querySelectorAll('.sample').forEach(x=>x.hidden=!(x.textContent.toLowerCase().includes(q)&&(f==='all'||x.dataset.kind===f)))}document.getElementById('query').addEventListener('input',filter);document.getElementById('filter').addEventListener('change',filter);</script></main></html>")
    atomic_text(output / "report.html", "\n".join(parts))
    write_docx(output, metrics, samples, records, config, manifest)
    print(json.dumps({"report": str(output / 'report.html'), "docx": str(output / 'report.docx'),
                      "valid": metrics['valid'], "four_class_accuracy": four['accuracy'],
                      "binary_accuracy": binary['accuracy']}, ensure_ascii=False))
    return metrics


def generate_initial_report(experiment_root):
    """Produce a zero-result document without creating evaluation/config.json."""
    experiment_root = Path(experiment_root).resolve()
    plan = json.loads((experiment_root / "experiment.json").read_text())
    source = Path(plan['combined_input'])
    snapshot_path = experiment_root / "run-snapshot.json"
    snapshot = json.loads(snapshot_path.read_text()).get("runtime", {}) if snapshot_path.exists() else {}
    defaults_path = Path(__file__).resolve().parents[1] / "configs/runtime.json"
    defaults = json.loads(defaults_path.read_text()) if defaults_path.exists() else {}
    evaluation = Path(plan.get("evaluation_output", experiment_root / "evaluation"))
    if (evaluation / "config.json").exists():
        metrics = generate_report(evaluation)
        atomic_bytes(experiment_root / "report.docx", (evaluation / "report.docx").read_bytes())
        return metrics
    config = {
        "model": snapshot.get("model", defaults.get("model_name", "Qwen3.8-27B")),
        "base_url": snapshot.get("base_url", defaults.get("model_base_url", "http://127.0.0.1:19888/v1")),
        "rounds": snapshot.get("rounds", 3), "workers": snapshot.get("workers", 4),
        "temperature": 0, "embedding_device": "待实际运行配置确认",
        "search_provider": snapshot.get("search_provider", "searxng"),
        "cache_mode": snapshot.get("cache_mode", "online"),
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "report_only_configuration": True,
    }
    evidence_path = source.with_name("reverse-evidence.json")
    if evidence_path.exists():
        config["reverse_evidence_sha256"] = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="mar-initial-report-") as temporary:
        target = Path(temporary)
        atomic_bytes(target / "samples.json", source.read_bytes())
        atomic_bytes(target / "dataset-manifest.json", source.with_name("manifest.json").read_bytes())
        atomic_text(target / "config.json", json.dumps(config, ensure_ascii=False))
        atomic_text(target / "runtime-status.json", json.dumps({"status": "prepared", "elapsed_seconds": 0}))
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = generate_report(target)
        atomic_bytes(experiment_root / "report.docx", (target / "report.docx").read_bytes())
        atomic_bytes(experiment_root / "initial-report.html", (target / "report.html").read_bytes())
        atomic_bytes(experiment_root / "initial-metrics.json", (target / "metrics.json").read_bytes())
    print(json.dumps({"docx": str(experiment_root / "report.docx"), "valid": metrics['valid'],
                      "pending": metrics['pending'], "phase": "prepared"}, ensure_ascii=False))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--initial", action="store_true", help="从实验计划生成初始零完成文档，不创建评估配置")
    arguments = parser.parse_args()
    (generate_initial_report if arguments.initial else generate_report)(arguments.output)
