import json
from pathlib import Path
from typing import Dict, List, Any


def build_report(
    anchors_total: int,
    filled_text: int,
    filled_checkboxes: int,
    filled_tables: int,
    unmatched_anchors: List[Dict[str, Any]],
    unused_json_keys: List[str],
    mapping_summary: List[Dict[str, Any]],
    actions_counts: Dict[str, int],
) -> Dict[str, object]:
    return {
        "anchors_total": anchors_total,
        "filled_text": filled_text,
        "filled_checkboxes": filled_checkboxes,
        "filled_tables": filled_tables,
        "unmatched_anchors": unmatched_anchors,
        "unused_json_keys": unused_json_keys,
        "mapping_summary": mapping_summary,
        "actions": actions_counts,
    }


def write_report(path: Path, report: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
