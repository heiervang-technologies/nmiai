#!/usr/bin/env python3
"""Analyze accounting submission logs and infer likely scorer-critical misses."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = ROOT / "tasks" / "accounting"
SERVER_DIR = TASK_DIR / "server"
PLAYBOOK_DIR = SERVER_DIR / "playbooks"
DEFAULT_LOG_DIR = Path("/tmp/accounting-logs")
DEFAULT_OUT_DIR = TASK_DIR / "analysis"
RESULTS_TSV = TASK_DIR / "autoresearch_results.tsv"
STATE_PATH = DEFAULT_OUT_DIR / "analyzer_state.json"

OMISSION_PATTERNS = [
    (re.compile(r"\ba[uú]n no\b", re.IGNORECASE), "explicit_missing_followup"),
    (re.compile(r"\bnot added\b", re.IGNORECASE), "explicit_missing_followup"),
    (re.compile(r"\bopen\b|\babierta\b|\bopened\b", re.IGNORECASE), "left_open"),
    (re.compile(r"ready to deliver|ready for delivery|listo para entregar", re.IGNORECASE), "not_delivered"),
]

FAMILY_HINTS = {
    "travel_expense": {
        "expected_prompt_fields": [
            (re.compile(r"per diem|indemnizaci[oó]n diaria|diett", re.IGNORECASE), "per_diem"),
            (re.compile(r"\b\d+\s*(day|days|d[ií]as|dager)\b", re.IGNORECASE), "duration"),
            (re.compile(r"berg(en)?|oslo|[åa]lesund|departure|salida", re.IGNORECASE), "departure_or_destination"),
        ],
        "message_rules": [
            (re.compile(r"ratetype|satskategori|sats eller satskategori", re.IGNORECASE), "rate_type"),
            (re.compile(r"uten kostnader|without costs", re.IGNORECASE), "cost_lines_or_allowances"),
            (re.compile(r"kun reiseregning", re.IGNORECASE), "travel_expense_type_or_travel_details"),
            (re.compile(r"levere|deliver", re.IGNORECASE), "delivered_state"),
            (re.compile(r"per diem|a[uú]n no aparece a[nñ]adida", re.IGNORECASE), "per_diem_completion"),
        ],
        "hypotheses": [
            "Scorer likely checks that the travel expense is not only created, but typed correctly as travel, populated with per diem or cost details, and delivered or otherwise finalized.",
            "Repeated validation errors around `rateType.id` suggest the per diem object is structurally incomplete even when the travel expense itself is created.",
        ],
    },
    "timesheet": {
        "expected_prompt_fields": [
            (re.compile(r"timesheet|time sheet|log hours|registrer timer|heures|horas", re.IGNORECASE), "hours"),
            (re.compile(r"project|prosjekt", re.IGNORECASE), "project"),
            (re.compile(r"activity|aktivitet", re.IGNORECASE), "activity"),
        ],
        "message_rules": [
            (re.compile(r"activity", re.IGNORECASE), "activity"),
            (re.compile(r"project", re.IGNORECASE), "project"),
            (re.compile(r"employee", re.IGNORECASE), "employee"),
            (re.compile(r"hours", re.IGNORECASE), "hours"),
        ],
        "hypotheses": [
            "Scorer likely checks the resolved employee, project, activity, date, and exact hours, not just whether `/timesheet/entry` was posted.",
            "Combined prompts may also require downstream invoice linkage after the hours are logged.",
        ],
    },
}

GENERIC_FIELD_RULES = [
    (re.compile(r"vattype", re.IGNORECASE), "vat_type"),
    (re.compile(r"paymenttype", re.IGNORECASE), "payment_type"),
    (re.compile(r"costcategory", re.IGNORECASE), "cost_category"),
    (re.compile(r"customer", re.IGNORECASE), "customer"),
    (re.compile(r"department", re.IGNORECASE), "department"),
    (re.compile(r"project", re.IGNORECASE), "project"),
    (re.compile(r"activity", re.IGNORECASE), "activity"),
    (re.compile(r"employee", re.IGNORECASE), "employee"),
    (re.compile(r"ratetype", re.IGNORECASE), "rate_type"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--results-tsv", type=Path, default=RESULTS_TSV)
    parser.add_argument("--state-path", type=Path, default=STATE_PATH)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=20)
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def load_playbook_families() -> list[str]:
    families = []
    for path in sorted(PLAYBOOK_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            families.append(payload.get("family") or path.stem)
        except Exception:
            families.append(path.stem)
    return families


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"seen_families": [], "logged_batches": [], "last_fingerprint": None}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_embedded_json(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def successful_writes(calls: list[dict[str, Any]]) -> int:
    return sum(1 for call in calls if call.get("method") in {"POST", "PUT", "DELETE"} and 200 <= int(call.get("status", 0)) < 300)


def collect_omissions(final_message: str) -> list[str]:
    hits = []
    for pattern, label in OMISSION_PATTERNS:
        if final_message and pattern.search(final_message):
            hits.append(label)
    return hits


def extract_validation_entries(call: dict[str, Any]) -> list[dict[str, str]]:
    raw_error = call.get("error")
    parsed = parse_embedded_json(raw_error)
    entries = []
    if isinstance(parsed, dict) and parsed.get("validationMessages"):
        for item in parsed["validationMessages"]:
            entries.append(
                {
                    "path": call.get("path", ""),
                    "field": str(item.get("field") or ""),
                    "message": str(item.get("message") or ""),
                }
            )
    elif raw_error:
        entries.append({"path": call.get("path", ""), "field": "", "message": str(raw_error)})
    elif int(call.get("status", 0)) >= 400:
        entries.append(
            {
                "path": call.get("path", ""),
                "field": "",
                "message": f"HTTP {call.get('status')} with no captured body",
            }
        )
    return entries


def infer_missing_fields(
    family: str,
    prompt: str,
    final_message: str,
    validation_entries: list[dict[str, str]],
) -> tuple[Counter, Counter]:
    direct = Counter()
    prompt_required = Counter()
    family_hints = FAMILY_HINTS.get(family, {})

    for pattern, field_name in family_hints.get("expected_prompt_fields", []):
        if pattern.search(prompt or ""):
            prompt_required[field_name] += 1

    combined_texts = [final_message or ""]
    for entry in validation_entries:
        combined_texts.append(" ".join([entry.get("path", ""), entry.get("field", ""), entry.get("message", "")]))

    for text in combined_texts:
        for pattern, field_name in family_hints.get("message_rules", []):
            if pattern.search(text):
                direct[field_name] += 1
        for pattern, field_name in GENERIC_FIELD_RULES:
            if pattern.search(text):
                direct[field_name] += 1

    return direct, prompt_required


def normalize_pattern(entry: dict[str, str]) -> str:
    field = entry.get("field", "").strip()
    message = re.sub(r"\s+", " ", entry.get("message", "").strip())
    path = entry.get("path", "").strip()
    if field and message:
        return f"{path}: {field} -> {message}"
    if message:
        return f"{path}: {message}" if path else message
    return path or "unknown_error"


def load_logs(log_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(log_dir.glob("*.json")):
        if path.name == "summary.jsonl":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_path"] = str(path)
        records.append(payload)
    return records


def ensure_results_header(path: Path) -> None:
    if path.exists():
        return
    path.write_text(
        "timestamp\tbatch_id\tfamily\tmodel\tproxy_clean_rate\tapi_errors\tsuccessful_writes\tdashboard_score\tsubmissions_used\tattachment_present\tstatus\tdescription\n",
        encoding="utf-8",
    )


def append_results_rows(path: Path, batch_id: str, family_rows: list[dict[str, Any]], state: dict[str, Any]) -> None:
    logged = set(state.get("logged_batches", []))
    if batch_id in logged:
        return
    ensure_results_header(path)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        for row in family_rows:
            writer.writerow(
                [
                    datetime.now(timezone.utc).isoformat(),
                    batch_id,
                    row["family"],
                    "openai/gpt-5.4",
                    f"{row['proxy_clean_rate']:.4f}",
                    row["api_errors"],
                    row["successful_writes"],
                    "",
                    0,
                    str(row["attachment_present"]).lower(),
                    row["status"],
                    row["description"],
                ]
            )
    logged.add(batch_id)
    state["logged_batches"] = sorted(logged)


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Accounting Log Analysis",
        "",
        f"Analyzed `{summary['total_runs']}` runs from `{summary['log_dir']}`.",
        "",
        "## Global",
        "",
        f"- Seen families: {', '.join(summary['seen_families']) if summary['seen_families'] else 'none'}",
        f"- New families since last run: {', '.join(summary['new_families_since_last_run']) if summary['new_families_since_last_run'] else 'none'}",
        f"- Unseen playbook families: {', '.join(summary['unseen_families']) if summary['unseen_families'] else 'none'}",
        f"- Empty attachment runs: {summary['empty_attachment_runs']}",
        "",
        "## Alerts",
        "",
    ]
    for alert in summary["alerts"]:
        lines.append(f"- {alert}")
    if not summary["alerts"]:
        lines.append("- none")
    lines.extend(["", "## Families", ""])
    for family in summary["families"]:
        lines.append(f"### {family['family']}")
        lines.append("")
        lines.append(f"- Runs: {family['runs']}")
        lines.append(f"- Proxy clean rate: {family['proxy_clean_rate']:.1%}")
        lines.append(f"- Likely full runs: {family['likely_full_runs']}")
        lines.append(f"- Likely partial runs: {family['likely_partial_runs']}")
        lines.append(f"- Mean API errors: {family['mean_api_errors']:.2f}")
        lines.append(
            f"- Prompt-required fields: {', '.join(item['field'] for item in family['prompt_required_fields'][:6]) or 'none'}"
        )
        lines.append(
            f"- Missing-field hypotheses: {', '.join(item['field'] for item in family['likely_missing_fields'][:6]) or 'none'}"
        )
        if family["top_error_patterns"]:
            lines.append(f"- Top error: {family['top_error_patterns'][0]['pattern']}")
        for hypothesis in family.get("scorer_hypotheses", []):
            lines.append(f"- Hypothesis: {hypothesis}")
        lines.append("")
    return "\n".join(lines) + "\n"


def build_summary(
    records: list[dict[str, Any]],
    playbook_families: list[str],
    state: dict[str, Any],
    log_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], str, str]:
    families: dict[str, dict[str, Any]] = {}
    seen_families = set()
    empty_attachment_runs = 0
    fingerprint = "|".join(record.get("timestamp", "") for record in records)
    batch_id = records[-1]["timestamp"] if records else datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    for record in records:
        family = ((record.get("plan") or {}).get("family") or "unknown").strip()
        seen_families.add(family)
        calls = ((record.get("api_stats") or {}).get("calls") or [])
        writes = successful_writes(calls)
        api_errors = int((record.get("api_stats") or {}).get("errors_4xx", 0))
        final_message = ((record.get("result") or {}).get("final_message") or "")
        omissions = collect_omissions(final_message)
        validation_entries = []
        for call in calls:
            validation_entries.extend(extract_validation_entries(call))
        missing_fields, prompt_required = infer_missing_fields(family, record.get("prompt", ""), final_message, validation_entries)

        files = record.get("files") or []
        attachment_present = bool(files)
        empty_attachment = any((not f.get("content_len")) and not f.get("content_base64") for f in files)
        if empty_attachment:
            empty_attachment_runs += 1

        bucket = families.setdefault(
            family,
            {
                "family": family,
                "runs": 0,
                "clean_runs": 0,
                "likely_full_runs": 0,
                "likely_partial_runs": 0,
                "api_errors_total": 0,
                "api_calls_total": 0,
                "successful_writes_total": 0,
                "attachment_runs": 0,
                "empty_attachment_runs": 0,
                "error_patterns": Counter(),
                "missing_fields": Counter(),
                "prompt_required": Counter(),
                "prompt_examples": [],
            },
        )
        bucket["runs"] += 1
        bucket["api_errors_total"] += api_errors
        bucket["api_calls_total"] += int((record.get("api_stats") or {}).get("total_calls", 0))
        bucket["successful_writes_total"] += writes
        bucket["attachment_runs"] += int(attachment_present)
        bucket["empty_attachment_runs"] += int(empty_attachment)
        if api_errors <= 2 and writes >= 1:
            bucket["clean_runs"] += 1
        if api_errors == 0 and writes >= 1 and not omissions:
            bucket["likely_full_runs"] += 1
        if writes >= 1 and (api_errors > 0 or omissions):
            bucket["likely_partial_runs"] += 1
        for entry in validation_entries:
            bucket["error_patterns"].update([normalize_pattern(entry)])
        bucket["missing_fields"].update(missing_fields)
        bucket["prompt_required"].update(prompt_required)
        bucket["prompt_examples"].append(
            {
                "timestamp": record.get("timestamp"),
                "prompt": record.get("prompt", "")[:220],
                "omissions": omissions,
            }
        )

    family_rows = []
    rendered_families = []
    alerts = []
    for family_name in sorted(families):
        bucket = families[family_name]
        runs = bucket["runs"] or 1
        proxy_clean_rate = bucket["clean_runs"] / runs
        likely_missing_fields = [
            {"field": field, "count": count} for field, count in bucket["missing_fields"].most_common(8)
        ]
        prompt_required_fields = [
            {"field": field, "count": count} for field, count in bucket["prompt_required"].most_common(8)
        ]
        top_error_patterns = [
            {"pattern": pattern, "count": count} for pattern, count in bucket["error_patterns"].most_common(8)
        ]
        status = "keep" if proxy_clean_rate >= 0.7 else "watch"
        if proxy_clean_rate < 0.4:
            status = "manual_review"
        description = (
            f"{family_name}: clean_rate={proxy_clean_rate:.1%}, "
            f"likely_missing={','.join(item['field'] for item in likely_missing_fields[:4]) or 'none'}"
        )
        family_rows.append(
            {
                "family": family_name,
                "proxy_clean_rate": proxy_clean_rate,
                "api_errors": bucket["api_errors_total"],
                "successful_writes": bucket["successful_writes_total"],
                "attachment_present": bucket["attachment_runs"] > 0,
                "status": status,
                "description": description,
            }
        )
        rendered_families.append(
            {
                "family": family_name,
                "runs": bucket["runs"],
                "proxy_clean_rate": proxy_clean_rate,
                "likely_full_runs": bucket["likely_full_runs"],
                "likely_partial_runs": bucket["likely_partial_runs"],
                "mean_api_errors": bucket["api_errors_total"] / runs,
                "mean_api_calls": bucket["api_calls_total"] / runs,
                "attachment_runs": bucket["attachment_runs"],
                "empty_attachment_runs": bucket["empty_attachment_runs"],
                "top_error_patterns": top_error_patterns,
                "likely_missing_fields": likely_missing_fields,
                "prompt_required_fields": prompt_required_fields,
                "prompt_examples": bucket["prompt_examples"][-3:],
                "scorer_hypotheses": FAMILY_HINTS.get(family_name, {}).get("hypotheses", []),
            }
        )
        if proxy_clean_rate < 0.7:
            alerts.append(
                f"{family_name} clean rate is {proxy_clean_rate:.1%}; likely missing fields: {', '.join(item['field'] for item in likely_missing_fields[:4]) or 'none'}"
            )
        if family_name == "travel_expense" and likely_missing_fields:
            alerts.append(
                "travel_expense remains partial: focus on delivered_state, rate_type, travel_expense typing, and per-diem completion before broad retries"
            )

    previous_seen = set(state.get("seen_families", []))
    new_families = sorted(seen_families - previous_seen)
    if new_families:
        alerts.append(f"New families seen: {', '.join(new_families)}")
    unseen_families = sorted(set(playbook_families) - seen_families)
    if unseen_families:
        alerts.append(f"Still unseen families: {', '.join(unseen_families)}")
    if empty_attachment_runs:
        alerts.append(f"{empty_attachment_runs} runs had empty attachments; treat file-based tasks as manual review")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "log_dir": str(log_dir),
        "total_runs": len(records),
        "seen_families": sorted(seen_families),
        "new_families_since_last_run": new_families,
        "unseen_families": unseen_families,
        "unknown_families": sorted(seen_families - set(playbook_families)),
        "empty_attachment_runs": empty_attachment_runs,
        "alerts": alerts,
        "families": rendered_families,
    }
    return summary, family_rows, fingerprint, batch_id


def write_outputs(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "latest_log_analysis.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "latest_log_analysis.md").write_text(render_markdown(summary), encoding="utf-8")
    (output_dir / "current_alerts.json").write_text(json.dumps({"alerts": summary["alerts"]}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_once(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    state = load_state(args.state_path)
    playbook_families = load_playbook_families()
    records = load_logs(args.log_dir)
    summary, family_rows, fingerprint, batch_id = build_summary(records, playbook_families, state, args.log_dir)
    changed = fingerprint != state.get("last_fingerprint")
    append_results_rows(args.results_tsv, batch_id, family_rows, state)
    state["seen_families"] = sorted(set(state.get("seen_families", [])) | set(summary["seen_families"]))
    state["last_fingerprint"] = fingerprint
    save_state(args.state_path, state)
    write_outputs(summary, args.output_dir)
    return summary, changed


def main() -> None:
    args = parse_args()
    if args.watch:
        while True:
            summary, changed = run_once(args)
            if changed:
                print(
                    f"[{summary['generated_at']}] analyzed {summary['total_runs']} runs; "
                    f"families={','.join(summary['seen_families']) or 'none'}"
                )
            time.sleep(max(args.interval, 1))
    else:
        summary, _ = run_once(args)
        if args.print_json:
            print(json.dumps(summary, indent=2, ensure_ascii=False))
        else:
            print(f"Wrote {args.output_dir / 'latest_log_analysis.json'}")
            print(f"Wrote {args.output_dir / 'latest_log_analysis.md'}")
            for family in summary["families"]:
                missing = ",".join(item["field"] for item in family["likely_missing_fields"][:4]) or "none"
                print(
                    f"{family['family']}: clean_rate={family['proxy_clean_rate']:.1%} "
                    f"runs={family['runs']} likely_missing={missing}"
                )


if __name__ == "__main__":
    main()
