import json
import re
import string
from typing import Dict, List, Any

from rapidfuzz import process, fuzz

from src.llm.hf_model import HFModel


def _normalize_text(value: str) -> str:
    text = value.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _heuristic_mapping(
    anchors: List[Dict[str, object]],
    data_keys: List[str],
    threshold: int = 78,
) -> Dict[str, Dict[str, object]]:
    norm_keys = [_normalize_text(k) for k in data_keys]
    mapping: Dict[str, Dict[str, object]] = {}

    for anchor in anchors:
        label = str(anchor.get("label_text") or "")
        anchor_id = str(anchor.get("anchor_id") or "")
        if not anchor_id or not label:
            continue
        norm_label = _normalize_text(label)
        if not norm_label:
            continue
        matches = process.extract(norm_label, norm_keys, scorer=fuzz.WRatio, limit=2)
        if not matches:
            continue
        best_match, best_score, best_idx = matches[0]
        best_key = data_keys[best_idx]
        ambiguous = best_score < threshold
        if len(matches) > 1 and matches[1][1] == best_score:
            ambiguous = True

        mapping[anchor_id] = {
            "label_text": label,
            "json_key": best_key,
            "score": float(best_score),
            "ambiguous": ambiguous,
        }

    return mapping


def _extract_json(text: str) -> Dict[str, str]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _parse_llm_items(response: str) -> List[Dict[str, Any]]:
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        return []
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []
    return items


def heuristic_map(
    anchors: List[Dict[str, object]],
    data_keys: List[str],
    threshold: int = 78,
) -> Dict[str, Dict[str, object]]:
    return _heuristic_mapping(anchors, data_keys, threshold=threshold)


def llm_map_ambiguous(
    heuristic_mapping: Dict[str, Dict[str, object]],
    data_keys: List[str],
    model: HFModel,
    batch_size: int = 8,
) -> Dict[str, List[Dict[str, Any]]]:
    norm_keys = [_normalize_text(k) for k in data_keys]
    ambiguous = []
    for anchor_id, meta in heuristic_mapping.items():
        if meta.get("ambiguous") is not True:
            continue
        label_text = str(meta.get("label_text") or "")
        norm_label = _normalize_text(label_text)
        candidates: List[str] = []
        if norm_label:
            matches = process.extract(norm_label, norm_keys, scorer=fuzz.WRatio, limit=8)
            candidates = [data_keys[idx] for _, _, idx in matches]
        ambiguous.append(
            {
                "anchor_id": anchor_id,
                "label_text": label_text,
                "candidates": candidates,
            }
        )

    items: List[Dict[str, Any]] = []
    if not ambiguous:
        return {"items": items}

    if model and model.available():
        for i in range(0, len(ambiguous), batch_size):
            batch = ambiguous[i : i + batch_size]
            prompt = (
                "Return ONLY strict JSON with schema: "
                '{"items":[{"anchor_id":"...","json_key":"...|null","confidence":0.0}]}.\n'
                "Rules:\n"
                "- Choose the single best JSON key from each anchor's candidates list.\n"
                "- Prefer exact semantic match to the anchor label (Romanian).\n"
                "- If the label is a section title or not a field, use null.\n"
                "- Confidence between 0.0 and 1.0.\n\n"
                "Examples:\n"
                "Label: 'Către' -> json_key: 'Catre - denumirea autoritatii contractante si adresa completa'\n"
                "Label: 'reprezentanţi ai ofertantului' -> json_key: 'Denumirea / numele ofertantului'\n"
                "Label: 'Data' -> json_key: 'Data'\n\n"
                f"Anchors: {batch}\n"
            )
            response = model.generate(prompt)
            parsed_items = _parse_llm_items(response)
            batch_ids = {b["anchor_id"] for b in batch}
            for item in parsed_items:
                anchor_id = str(item.get("anchor_id") or "")
                json_key = item.get("json_key")
                confidence = item.get("confidence")
                if not anchor_id or anchor_id not in batch_ids:
                    continue
                if json_key is not None and json_key not in data_keys:
                    continue
                if not isinstance(confidence, (int, float)):
                    confidence = 0.0
                items.append(
                    {
                        "anchor_id": anchor_id,
                        "json_key": json_key,
                        "confidence": max(0.0, min(1.0, float(confidence))),
                    }
                )

    return {"items": items}
