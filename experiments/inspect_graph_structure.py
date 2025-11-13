#!/usr/bin/env python3
"""Utility to inspect the session graph export structure."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Stream JSON objects from a file that may contain JSONL content."""
    with path.open("r", encoding="utf-8") as handle:
        text = handle.read().strip()
        if not text:
            return
        if text[0] == "{" and text[-1] == "}":
            # Heuristic: treat as JSONL by splitting on newlines.
            for line in text.splitlines():
                line = line.strip()
                if line:
                    yield json.loads(line)
        else:
            raise ValueError("Unsupported file format; expected JSON objects per line.")


def summarize_graph(records: Iterable[Dict[str, Any]], limit: int = 3) -> Dict[str, Any]:
    """Collect structural statistics for the session graph export."""
    top_level_keys = Counter()
    timeline_payload_keys = Counter()
    summary_entry_keys = Counter()
    summary_context_keys = Counter()
    retrieved_entry_keys = Counter()
    retrieved_context_keys = Counter()

    total_records = 0
    sample_records: List[Dict[str, Any]] = []

    for record in records:
        total_records += 1
        top_level_keys.update(record.keys())
        if len(sample_records) < limit:
            sample_records.append(record)

        timeline_relation = record.get("timeline_relation") or {}
        if isinstance(timeline_relation, dict):
            payload = timeline_relation.get("timeline_payload") or {}
            if isinstance(payload, dict):
                timeline_payload_keys.update(payload.keys())
                for entry in payload.get("summary_entries", []) or []:
                    if isinstance(entry, dict):
                        summary_entry_keys.update(entry.keys())
                        for ctx in entry.get("context", []) or []:
                            if isinstance(ctx, dict):
                                summary_context_keys.update(ctx.keys())

            for entry in timeline_relation.get("retrieved_entries", []) or []:
                if isinstance(entry, dict):
                    retrieved_entry_keys.update(entry.keys())
                    for ctx in entry.get("context", []) or []:
                        if isinstance(ctx, dict):
                            retrieved_context_keys.update(ctx.keys())

    return {
        "total_records": total_records,
        "top_level_keys": dict(top_level_keys),
        "timeline_payload_keys": dict(timeline_payload_keys),
        "summary_entry_keys": dict(summary_entry_keys),
        "summary_context_keys": dict(summary_context_keys),
        "retrieved_entry_keys": dict(retrieved_entry_keys),
        "retrieved_context_keys": dict(retrieved_context_keys),
        "sample_records": sample_records,
    }


def truncate_vector(record: Dict[str, Any]) -> None:
    """Replace large vector payloads with their length for readability."""
    vector = record.get("vector")
    if isinstance(vector, list):
        record["vector"] = f"<vector length={len(vector)}>"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect session graph export structure")
    parser.add_argument("graph_path", type=Path, help="Path to graph JSONL export")
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of sample records to print in full",
    )
    args = parser.parse_args()

    graph_path: Path = args.graph_path
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

    summary = summarize_graph(read_jsonl(graph_path), limit=args.limit)

    print(f"Total records: {summary['total_records']}")
    print("Top-level key counts:")
    for key, count in summary["top_level_keys"].items():
        print(f"  - {key}: {count}")

    print("\nTimeline payload key counts:")
    for key, count in summary["timeline_payload_keys"].items():
        print(f"  - {key}: {count}")

    print("\nSummary entry key counts:")
    for key, count in summary["summary_entry_keys"].items():
        print(f"  - {key}: {count}")

    print("\nSummary context key counts:")
    for key, count in summary["summary_context_keys"].items():
        print(f"  - {key}: {count}")

    print("\nRetrieved entry key counts:")
    for key, count in summary["retrieved_entry_keys"].items():
        print(f"  - {key}: {count}")

    print("\nRetrieved context key counts:")
    for key, count in summary["retrieved_context_keys"].items():
        print(f"  - {key}: {count}")

    for idx, record in enumerate(summary["sample_records"][: args.limit], start=1):
        cleaned = dict(record)
        truncate_vector(cleaned)
        print(f"\nSample record {idx}:")
        print(json.dumps(cleaned, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
