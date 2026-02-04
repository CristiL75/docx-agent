import json
import re
import string
import unicodedata
from typing import Dict, List, Any

from rapidfuzz import process, fuzz

from src.llm.hf_model import HFModel


def _infer_tags_from_text(text: str) -> List[str]:
    t = _normalize_text(text)
    tags: List[str] = []
    if not t:
        return tags
    if any(x in t for x in ("suma", "lei", "valoare", "tva", "taxa")):
        tags.append("money")
    if any(x in t for x in ("%", "procent")):
        tags.append("percent")
    if any(x in t for x in ("data", "ziua", "luna", "anul", "an")):
        tags.append("date")
    if any(x in t for x in ("durata", "zile", "luni")):
        tags.append("duration")
    if any(x in t for x in ("servici", "furniz")):
        tags.append("service")
    if any(x in t for x in ("adresa", "sediu", "domiciliu")):
        tags.append("address")
    if any(x in t for x in ("denumirea", "numele", "operator", "ofertant", "achizitor")):
        tags.append("entity")
    if any(x in t for x in ("procedura", "contract", "achizitie")):
        tags.append("process")
    return tags


def _normalize_text(value: str) -> str:
    text = value.lower()
    text = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _heuristic_mapping(
    anchors: List[Dict[str, object]],
    data_keys: List[str],
    threshold: int = 78,
) -> Dict[str, Dict[str, object]]:
    norm_keys = [_normalize_text(k) for k in data_keys]
    mapping: Dict[str, Dict[str, object]] = {}

    key_tags = {k: _infer_tags_from_text(k) for k in data_keys}

    for anchor in anchors:
        label = str(anchor.get("label_text") or "")
        nearby = str(anchor.get("nearby_text") or "")
        anchor_id = str(anchor.get("anchor_id") or "")
        if not anchor_id or not label:
            continue
        norm_label = _normalize_text(label)
        norm_nearby = _normalize_text(nearby)
        if not norm_label and not norm_nearby:
            continue
        label_tags = set(_infer_tags_from_text(label + " " + nearby))
        filtered_keys = [k for k in data_keys if (not label_tags) or (label_tags & set(key_tags.get(k, [])))]
        filtered_norm_keys = [_normalize_text(k) for k in filtered_keys]
        candidates = [norm_label]
        if norm_nearby:
            candidates.append(norm_nearby)
        matches = process.extract(
            " ".join(candidates),
            filtered_norm_keys or norm_keys,
            scorer=fuzz.token_set_ratio,
            limit=3,
        )
        if not matches:
            continue
        best_match, best_score, best_idx = matches[0]
        best_key = (filtered_keys or data_keys)[best_idx]
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
    key_tags = {k: _infer_tags_from_text(k) for k in data_keys}
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
        label_tags = _infer_tags_from_text(label_text)
        if label_tags:
            candidates = [k for k in candidates if set(label_tags) & set(key_tags.get(k, []))] or candidates
        ambiguous.append(
            {
                "anchor_id": anchor_id,
                "label_text": label_text,
                "expected_tags": label_tags,
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
                "- Respect expected_tags if present.\n"
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


def llm_map_all(
    anchors: List[Dict[str, object]],
    data_keys: List[str],
    model: HFModel,
    batch_size: int = 8,
) -> Dict[str, List[Dict[str, Any]]]:
    norm_keys = [_normalize_text(k) for k in data_keys]
    key_tags = {k: _infer_tags_from_text(k) for k in data_keys}
    items: List[Dict[str, Any]] = []

    candidates_payload = []
    for anchor in anchors:
        anchor_id = str(anchor.get("anchor_id") or "")
        label_text = str(anchor.get("label_text") or "")
        nearby_text = str(anchor.get("nearby_text") or "")
        if not anchor_id or not label_text:
            continue
        norm_label = _normalize_text(label_text)
        norm_nearby = _normalize_text(nearby_text)
        if not norm_label and not norm_nearby:
            continue
        query = " ".join([t for t in (norm_label, norm_nearby) if t])
        matches = process.extract(query, norm_keys, scorer=fuzz.token_set_ratio, limit=8)
        candidates = [data_keys[idx] for _, _, idx in matches]
        label_tags = _infer_tags_from_text(label_text + " " + nearby_text)
        if label_tags:
            candidates = [k for k in candidates if set(label_tags) & set(key_tags.get(k, []))] or candidates
        candidates_payload.append(
            {
                "anchor_id": anchor_id,
                "label_text": label_text,
                "expected_tags": label_tags,
                "candidates": candidates,
            }
        )

    if not candidates_payload:
        return {"items": items}

    if model and model.available():
        for i in range(0, len(candidates_payload), batch_size):
            batch = candidates_payload[i : i + batch_size]
            prompt = (
                "Return ONLY strict JSON with schema: "
                '{"items":[{"anchor_id":"...","json_key":"...|null","confidence":0.0}]}.\n'
                "Rules:\n"
                "- Choose the single best JSON key from each anchor's candidates list.\n"
                "- Prefer exact semantic match to the anchor label (Romanian).\n"
                "- Respect expected_tags if present.\n"
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
