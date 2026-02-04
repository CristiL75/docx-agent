from typing import Dict, List, Any


def merge_mappings(
    anchors: List[Dict[str, object]],
    heuristic: Dict[str, Dict[str, object]],
    llm: Dict[str, object],
    heuristic_threshold: float = 90.0,
    llm_threshold: float = 0.4,
) -> Dict[str, str]:
    final: Dict[str, str] = {}

    llm_items = llm.get("items") if isinstance(llm, dict) else None
    llm_by_anchor = {}
    if isinstance(llm_items, list):
        for item in llm_items:
            anchor_id = str(item.get("anchor_id") or "")
            if anchor_id:
                llm_by_anchor[anchor_id] = item

    for anchor in anchors:
        anchor_id = str(anchor.get("anchor_id") or "")
        if not anchor_id:
            continue

        heuristic_item = heuristic.get(anchor_id)
        if heuristic_item:
            score = float(heuristic_item.get("score") or 0.0)
            if score >= heuristic_threshold:
                final[anchor_id] = str(heuristic_item.get("json_key"))
                continue

        llm_item = llm_by_anchor.get(anchor_id)
        if llm_item:
            confidence = float(llm_item.get("confidence") or 0.0)
            json_key = llm_item.get("json_key")
            if json_key is not None and confidence >= llm_threshold:
                final[anchor_id] = str(json_key)

    return final


def build_mapping_report(labels: List[str], data_keys: List[str], mapping: Dict[str, str]) -> Dict[str, object]:
    missing_labels = [label for label in labels if label not in mapping or not mapping.get(label)]
    used_keys = {v for v in mapping.values() if v}
    unused_keys = [key for key in data_keys if key not in used_keys]
    return {
        "total_labels": len(labels),
        "mapped_labels": len(mapping),
        "missing_labels": missing_labels,
        "unused_keys": unused_keys,
    }
