import json
import random
import re
import unicodedata
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import process, fuzz

from src.llm.hf_model import HFModel
from src.validate import (
    is_date,
    is_money,
    is_percent,
    is_numericish,
    is_orgish,
    is_personish,
    is_role_title,
    is_addressish,
)


def _normalize_text(value: str) -> str:
    text = value.lower()
    text = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))
    return " ".join(text.split())


def _expected_types_for_paragraph(text: str, spans: List[Dict[str, Any]]) -> Dict[str, str]:
    norm = _normalize_text(text)
    expected: Dict[str, str] = {}

    if "durata de" in norm and "zile" in norm and "pana la data" in norm:
        spans_sorted = sorted(spans, key=lambda s: s.get("start_char", 0))
        if len(spans_sorted) >= 2:
            expected[spans_sorted[0]["span_id"]] = "DURATION_DAYS"
            expected[spans_sorted[1]["span_id"]] = "DATE_UNTIL"

    for span in spans:
        sid = span.get("span_id")
        if not sid:
            continue
        if sid in expected:
            continue
        if "subsemnatul" in norm:
            expected[sid] = "PERSON_NAME"
        elif "catre" in norm:
            expected[sid] = "ADDRESSEE"
        elif "ofertantul" in norm:
            expected[sid] = "ORG_NAME"
        elif "tva" in norm or "taxa pe valoarea adaugata" in norm:
            expected[sid] = "MONEY"
        elif "in calitate de" in norm:
            expected[sid] = "ROLE_TITLE"
        elif "oferta pentru si in numele" in norm:
            expected[sid] = "ORG_NAME"

    return expected


def _type_check(expected_type: Optional[str], value: Any) -> bool:
    if expected_type is None:
        return True
    if expected_type == "DURATION_DAYS":
        return is_numericish(value) and not is_date(value)
    if expected_type == "DATE_UNTIL":
        return is_date(value)
    if expected_type == "PERSON_NAME":
        return is_personish(value)
    if expected_type == "ORG_NAME":
        return is_orgish(value)
    if expected_type == "MONEY":
        return is_money(value)
    if expected_type == "ROLE_TITLE":
        return is_role_title(value)
    if expected_type == "ADDRESSEE":
        return (is_orgish(value) or is_addressish(value)) and not is_date(value)
    return True


def _parse_date(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
    return None


def _llm_candidates(model: HFModel, span: Dict[str, Any], data_keys: List[str]) -> List[Dict[str, Any]]:
    if not model or not model.available():
        return []
    prompt = (
        "Return ONLY strict JSON with schema: "
        '{"candidates":[{"key":"...","confidence":0.0}]}\n'
        "Pick candidates for the blank based on context.\n"
        f"Context: {span.get('left_context','')} [BLANK] {span.get('right_context','')}\n"
        f"Keys: {', '.join(data_keys[:50])}"
    )
    response = model.generate(prompt, max_new_tokens=200, temperature=0.0)
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        return []
    items = payload.get("candidates") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []
    cleaned = []
    for item in items:
        key = item.get("key") if isinstance(item, dict) else None
        conf = item.get("confidence") if isinstance(item, dict) else 0.0
        if isinstance(key, str) and key in data_keys:
            cleaned.append({"key": key, "confidence": float(conf)})
    return cleaned


def map_field_spans(
    spans: List[Dict[str, Any]],
    data_norm: Dict[str, Any],
    model: Optional[HFModel] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    random.seed(seed)
    data_keys = list(data_norm.keys())
    key_norms = {k: _normalize_text(k) for k in data_keys}

    spans_by_para: Dict[Tuple, List[Dict[str, Any]]] = {}
    for span in spans:
        loc = span.get("location") or {}
        key = (loc.get("type"), loc.get("section_idx"), loc.get("header_footer"), loc.get("table_idx"), span.get("paragraph_idx"))
        spans_by_para.setdefault(key, []).append(span)

    expected_types: Dict[str, str] = {}
    for _, group in spans_by_para.items():
        paragraph_text = group[0].get("paragraph_text", "") if group else ""
        expected_types.update(_expected_types_for_paragraph(paragraph_text, group))

    mapping: Dict[str, Optional[str]] = {}
    computed_values: Dict[str, Any] = {}
    type_mismatch_prevented: List[Dict[str, Any]] = []

    # First pass: map all non-duration spans
    for span in spans:
        span_id = span.get("span_id")
        if not span_id:
            continue
        if span.get("blank_kind") == "checkbox":
            continue
        expected_type = expected_types.get(span_id)
        if expected_type == "DURATION_DAYS":
            continue

        context = f"{span.get('left_context','')} {span.get('right_context','')}"
        query = _normalize_text(context)
        matches = process.extract(query, list(key_norms.values()), scorer=fuzz.WRatio, limit=10)
        candidates = [data_keys[idx] for _, _, idx in matches]

        if expected_type == "ADDRESSEE":
            candidates = [
                k
                for k in candidates
                if any(tok in key_norms[k] for tok in ("catre", "autoritate", "denumire", "adresa"))
            ] or candidates

        chosen = None
        for key in candidates:
            value = data_norm.get(key)
            if isinstance(value, (list, dict)):
                continue
            if not _type_check(expected_type, value):
                type_mismatch_prevented.append(
                    {
                        "span_id": span_id,
                        "key": key,
                        "expected_type": expected_type,
                        "value": value,
                    }
                )
                continue
            chosen = key
            break

        if chosen is None and model is not None:
            llm_cands = _llm_candidates(model, span, data_keys)
            for cand in llm_cands:
                key = cand.get("key")
                value = data_norm.get(key)
                if isinstance(value, (list, dict)):
                    continue
                if not _type_check(expected_type, value):
                    type_mismatch_prevented.append(
                        {
                            "span_id": span_id,
                            "key": key,
                            "expected_type": expected_type,
                            "value": value,
                        }
                    )
                    continue
                chosen = key
                break

        mapping[span_id] = chosen

    # Second pass: duration days computed if missing
    for _, group in spans_by_para.items():
        group_sorted = sorted(group, key=lambda s: s.get("start_char", 0))
        if not group_sorted:
            continue
        if "durata de" not in _normalize_text(group_sorted[0].get("paragraph_text", "")):
            continue
        if len(group_sorted) < 2:
            continue
        span_duration = group_sorted[0]
        span_date = group_sorted[1]
        sid_duration = span_duration.get("span_id")
        sid_date = span_date.get("span_id")
        if expected_types.get(sid_duration) != "DURATION_DAYS":
            continue
        if mapping.get(sid_duration):
            continue

        date_key = mapping.get(sid_date)
        date_value = data_norm.get(date_key) if date_key else None
        date_until = _parse_date(date_value)
        data_completarii = None
        for key in data_keys:
            if "data completarii" in key_norms[key] or "data completare" in key_norms[key]:
                data_completarii = _parse_date(data_norm.get(key))
                if data_completarii:
                    break

        if date_until and data_completarii:
            days = (date_until - data_completarii).days
            if days >= 0:
                computed_values[sid_duration] = str(days)
                mapping[sid_duration] = "__computed__"

    unmatched = [s for s in spans if mapping.get(s.get("span_id")) in (None, "") and s.get("blank_kind") != "checkbox"]

    return {
        "mapping": mapping,
        "computed_values": computed_values,
        "expected_types": expected_types,
        "unmatched": unmatched,
        "type_mismatch_prevented": type_mismatch_prevented,
    }
