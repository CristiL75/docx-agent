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
    infer_slot_type,
    value_matches_type,
)


def _normalize_text(value: str) -> str:
    text = value.lower()
    text = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))
    return " ".join(text.split())


def _tokens(value: str) -> List[str]:
    return [t for t in re.split(r"\W+", _normalize_text(value)) if t]


def _jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb))


def _candidate_scores(
    slot: Dict[str, Any],
    data_keys: List[str],
    data_norm: Dict[str, Any],
) -> List[Dict[str, Any]]:
    context = f"{slot.get('left_context','')} {slot.get('right_context','')}"
    norm_context = _normalize_text(context)
    context_tokens = _tokens(context)

    boosts = []
    if "catre" in norm_context:
        boosts.append(("catre", 0.2))
        boosts.append(("autoritate", 0.2))
    if "cif" in norm_context:
        boosts.append(("cif", 0.3))
    if "cod cpv" in norm_context or "cpv" in norm_context:
        boosts.append(("cpv", 0.3))
    if re.search(r"\bnr\b", norm_context):
        boosts.append(("nr", 0.15))
        boosts.append(("numar", 0.15))
    if "valabila" in norm_context and "pana la data" in norm_context:
        boosts.append(("data expir", 0.2))
        boosts.append(("valabil", 0.2))
        boosts.append(("durata", 0.15))

    results: List[Dict[str, Any]] = []
    for key in data_keys:
        value = data_norm.get(key)
        if isinstance(value, (list, dict)):
            continue
        if not value_matches_type(value, infer_slot_type(slot)):
            continue
        norm_key = _normalize_text(key)
        key_tokens = _tokens(key)
        score_j = _jaccard(context_tokens, key_tokens)
        score_f = fuzz.token_set_ratio(norm_context, norm_key) / 100.0
        score = 0.55 * score_f + 0.45 * score_j
        for token, boost in boosts:
            if token in norm_key:
                score += boost
        results.append({"key": key, "score": float(score)})

    results.sort(key=lambda r: (-r["score"], r["key"]))
    return results[:5]


def _expected_types_for_paragraph(text: str, spans: List[Dict[str, Any]]) -> Dict[str, str]:
    norm = _normalize_text(text)
    expected: Dict[str, str] = {}

    if "durata de" in norm and "zile" in norm and "pana la data" in norm:
        spans_sorted = sorted(spans, key=lambda s: s.get("start_char", 0))
        if len(spans_sorted) >= 2:
            expected[spans_sorted[0]["span_id"]] = "NUMBER"
            expected[spans_sorted[1]["span_id"]] = "DATE"

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


def _type_check(expected_type: Optional[str], value: Any, slot: Dict[str, Any]) -> bool:
    if expected_type is None:
        return value_matches_type(value, infer_slot_type(slot))
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
    return value_matches_type(value, expected_type)


def _parse_date(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
    return None


def _llm_choose_key(
    model: HFModel,
    span: Dict[str, Any],
    slot_type: str,
    candidate_keys: List[str],
) -> Optional[Dict[str, Any]]:
    if not model or not model.available():
        return None
    bad_examples = [
        "Avoid DATE for MONEY/NUMBER slots",
        "Avoid MONEY for PERSON/ROLE slots",
        "Avoid ADDRESS for MONEY/NUMBER slots",
    ]
    prompt = (
        "Return ONLY strict JSON with schema: "
        '{"best_key":"...|null","confidence":0.0,"reason_short":"..."}\n'
        "Pick the best key for the blank, or null if none fit.\n"
        f"Slot type: {slot_type}\n"
        f"Context left: {span.get('left_context','')}\n"
        f"Context right: {span.get('right_context','')}\n"
        f"Keys: {', '.join(candidate_keys[:30])}\n"
        f"Bad mappings to avoid: {', '.join(bad_examples)}"
    )
    response = model.generate(prompt, max_new_tokens=200)
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    best_key = payload.get("best_key")
    if not isinstance(best_key, str) or best_key not in candidate_keys:
        return None
    conf = payload.get("confidence")
    try:
        conf_val = float(conf)
    except (TypeError, ValueError):
        conf_val = 0.0
    return {"key": best_key, "confidence": conf_val, "reason_short": payload.get("reason_short")}


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
    candidates_by_span: Dict[str, List[Dict[str, Any]]] = {}
    computed_values: Dict[str, Any] = {}
    type_mismatch_prevented: List[Dict[str, Any]] = []

    repeatable_keys = {
        k
        for k in data_keys
        if "data completarii" in key_norms[k] or "data completare" in key_norms[k]
    }
    used_keys: Dict[str, List[str]] = {}
    span_context_tokens: Dict[str, List[str]] = {}
    # First pass: collect candidates
    sorted_spans = sorted(spans, key=lambda s: str(s.get("span_id") or ""))

    for span in sorted_spans:
        span_id = span.get("span_id")
        if not span_id:
            continue
        if span.get("blank_kind") == "checkbox":
            continue
        context = f"{span.get('left_context','')} {span.get('right_context','')}"
        span_context_tokens[span_id] = _tokens(context)

        heuristic_candidates = _candidate_scores(span, data_keys, data_norm)
        if not heuristic_candidates:
            query = _normalize_text(context)
            matches = process.extract(query, list(key_norms.values()), scorer=fuzz.WRatio, limit=10)
            heuristic_candidates = [
                {"key": data_keys[idx], "score": float(score) / 100.0}
                for _, score, idx in matches
            ]
        candidates_by_span[span_id] = heuristic_candidates

    # Global assignment (greedy with reuse penalties)
    span_order: List[Tuple[str, float]] = []
    for span in sorted_spans:
        span_id = span.get("span_id")
        if not span_id or span.get("blank_kind") == "checkbox":
            continue
        cand_list = candidates_by_span.get(span_id, [])
        top1 = cand_list[0]["score"] if len(cand_list) > 0 else 0.0
        top2 = cand_list[1]["score"] if len(cand_list) > 1 else 0.0
        span_order.append((span_id, top1 - top2))
    span_order.sort(key=lambda x: (-x[1], x[0]))

    span_map = {s.get("span_id"): s for s in sorted_spans}

    for span_id, _gap in span_order:
        span = span_map.get(span_id)
        if not span:
            continue
        expected_type = expected_types.get(span_id)
        cand_list = candidates_by_span.get(span_id, [])
        if not cand_list:
            mapping[span_id] = None
            continue

        best_key = None
        best_score = -1.0
        for cand in cand_list:
            key = cand.get("key")
            if not key:
                continue
            value = data_norm.get(key)
            if isinstance(value, (list, dict)):
                continue
            if not _type_check(expected_type, value, span):
                type_mismatch_prevented.append(
                    {
                        "span_id": span_id,
                        "key": key,
                        "expected_type": expected_type,
                        "value": value,
                    }
                )
                continue
            score = float(cand.get("score", 0.0))
            if key in used_keys and key not in repeatable_keys:
                score -= 0.25
                existing_ctx = used_keys.get(key, [])
                if existing_ctx:
                    sim = _jaccard(span_context_tokens.get(span_id, []), existing_ctx)
                    if sim < 0.3:
                        score -= 0.25
            if score > best_score:
                best_score = score
                best_key = key

        if best_key is None:
            mapping[span_id] = None
            continue

        top1 = cand_list[0]["score"] if len(cand_list) > 0 else 0.0
        top2 = cand_list[1]["score"] if len(cand_list) > 1 else 0.0
        ambiguous = (top1 - top2) < 0.08
        if ambiguous and model is not None:
            slot_type = infer_slot_type(span).value
            cand_keys = [c["key"] for c in cand_list]
            llm_pick = _llm_choose_key(model, span, slot_type, cand_keys)
            if llm_pick:
                key = llm_pick.get("key")
                value = data_norm.get(key)
                if not isinstance(value, (list, dict)) and _type_check(expected_type, value, span):
                    best_key = key

        mapping[span_id] = best_key
        if best_key and best_key not in repeatable_keys:
            used_keys[best_key] = span_context_tokens.get(span_id, [])

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
        if expected_types.get(sid_duration) not in {"DURATION_DAYS", "NUMBER"}:
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
        "candidates": candidates_by_span,
        "computed_values": computed_values,
        "expected_types": expected_types,
        "unmatched": unmatched,
        "type_mismatch_prevented": type_mismatch_prevented,
    }
