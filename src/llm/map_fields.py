import json
import re
import string
import unicodedata
from typing import Dict, List, Any, Tuple, Optional

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
    if any(x in t for x in ("valabil", "valabilitate")):
        tags.append("validity")
    if any(x in t for x in ("servici", "furniz")):
        tags.append("service")
    if any(x in t for x in ("plata", "termen", "conditii")):
        tags.append("payment")
    if any(x in t for x in ("penalit" , "doband")):
        tags.append("penalty")
    if any(x in t for x in ("adresa", "sediu", "domiciliu")):
        tags.append("address")
    if any(x in t for x in ("banca", "asigur", "parafata")):
        tags.append("bank")
    if any(x in t for x in ("denumirea", "numele", "operator", "ofertant", "achizitor")):
        tags.append("entity")
    if any(x in t for x in ("autoritate", "catre")):
        tags.append("authority")
    if "ofert" in t:
        tags.append("offer")
    if "subcontract" in t:
        tags.append("subcontract")
    if any(
        x in t
        for x in (
            "reprezentant",
            "imputernicit",
            "semnatar",
            "persoana",
            "nume",
            "functie",
            "director",
            "calitate",
            "dl",
            "dna",
        )
    ):
        tags.append("person")
        tags.append("role")
    if any(x in t for x in ("cif", "cnp", "registrul", "comert", "serie", "numar", "bi", "ci")):
        tags.append("id")
    if any(x in t for x in ("procedura", "contract", "achizitie")):
        tags.append("process")
    return tags


FIELD_TYPES = {
    "DATE",
    "MONEY",
    "PERCENT",
    "NUMBER",
    "ORG",
    "PERSON",
    "ADDRESS",
    "CHECKBOX_GROUP",
    "TABLE",
    "TEXT",
}


def _infer_field_type(anchor: Dict[str, object]) -> str:
    label = str(anchor.get("label_text") or "")
    nearby = str(anchor.get("nearby_text") or "")
    kind = str(anchor.get("kind") or "")
    placeholder_span = anchor.get("placeholder_span") or {}
    placeholder = str(placeholder_span.get("text", ""))
    text = _normalize_text(" ".join([label, nearby]))

    if kind == "checkbox" or "|_|" in label or "☐" in label or "[ ]" in label:
        return "CHECKBOX_GROUP"
    if kind == "table":
        return "TABLE"
    if "/" in placeholder or "data" in text:
        return "DATE"
    if "%" in label or "procent" in text:
        return "PERCENT"
    if any(x in text for x in ("suma", "lei", "valoare", "tva", "taxa")):
        return "MONEY"
    if any(x in text for x in ("numar", "nr", "cif", "cnp", "cod", "serie", "bi", "ci")):
        return "NUMBER"
    if any(x in text for x in ("adresa", "sediu", "domiciliu")):
        return "ADDRESS"
    if any(x in text for x in ("subsemnat", "reprezentant", "imputernicit", "semnatar", "dl", "dna", "functie")):
        return "PERSON"
    if any(x in text for x in ("operator", "ofertant", "autoritate", "societate", "s.c", "achizitor")):
        return "ORG"
    return "TEXT"


def _infer_key_type(key: str) -> str:
    t = _normalize_text(key)
    if "data" in t:
        return "DATE"
    if any(x in t for x in ("tva", "suma", "valoare", "lei")):
        return "MONEY"
    if "%" in key or "procent" in t:
        return "PERCENT"
    if any(x in t for x in ("numar", "nr", "cif", "cnp", "cod", "serie", "bi", "ci")):
        return "NUMBER"
    if any(x in t for x in ("adresa", "sediu", "domiciliu")):
        return "ADDRESS"
    if any(x in t for x in ("reprezentant", "imputernicit", "semnatar", "nume", "functie", "director")):
        return "PERSON"
    if any(x in t for x in ("operator", "ofertant", "autoritate", "societate", "achizitor", "s.c")):
        return "ORG"
    if any(x in t for x in ("tabel", "lista")):
        return "TABLE"
    return "TEXT"


def _value_matches_type(value: object, field_type: str) -> bool:
    if value is None:
        return True
    text = str(value)
    if field_type == "DATE":
        return bool(re.search(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{4}\b", text))
    if field_type == "MONEY":
        return bool(re.search(r"\b\d+[\d\s\.]*\b", text))
    if field_type == "PERCENT":
        return "%" in text or bool(re.search(r"\b\d+(?:\.\d+)?\b", text))
    if field_type == "NUMBER":
        return bool(re.search(r"\b\d+\b", text))
    return True


def _type_compatible(field_type: str, key_type: str) -> bool:
    if field_type == "TABLE":
        return key_type == "TABLE"
    if field_type == "CHECKBOX_GROUP":
        return True
    if field_type == "DATE":
        return key_type == "DATE"
    if field_type == "MONEY":
        return key_type in {"MONEY", "NUMBER"}
    if field_type == "PERCENT":
        return key_type in {"PERCENT", "NUMBER"}
    if field_type == "NUMBER":
        return key_type in {"NUMBER", "MONEY", "PERCENT"}
    if field_type == "PERSON":
        return key_type == "PERSON"
    if field_type == "ORG":
        return key_type == "ORG"
    if field_type == "ADDRESS":
        return key_type == "ADDRESS"
    return True


def _context_score(field_type: str, key_type: str) -> float:
    if field_type == "TABLE":
        return 1.0 if key_type == "TABLE" else 0.0
    if key_type == "TABLE":
        return 0.0
    return 1.0


def _score_candidate(
    text_similarity: float,
    field_type: str,
    key_type: str,
) -> float:
    type_score = 1.0 if _type_compatible(field_type, key_type) else 0.0
    context = _context_score(field_type, key_type)
    return 0.7 * text_similarity + 0.2 * type_score + 0.1 * context


def _parse_llm_candidates(response: str) -> List[Dict[str, Any]]:
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        return []
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []
    return items


def _normalize_text(value: str) -> str:
    text = value.lower()
    text = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _heuristic_mapping(
    anchors: List[Dict[str, object]],
    data_norm: Dict[str, Any],
    data_keys: List[str],
    llm_candidates: Optional[Dict[str, List[str]]] = None,
    threshold: float = 0.6,
) -> Dict[str, Dict[str, object]]:
    norm_keys = [_normalize_text(k) for k in data_keys]
    mapping: Dict[str, Dict[str, object]] = {}

    key_tags = {k: _infer_tags_from_text(k) for k in data_keys}
    key_types = {k: _infer_key_type(k) for k in data_keys}
    llm_candidates = llm_candidates or {}

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
        field_type = _infer_field_type(anchor)
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
            limit=10,
        )
        if not matches:
            continue
        base_candidates: List[Tuple[str, float]] = []
        for _, score, idx in matches:
            base_candidates.append(((filtered_keys or data_keys)[idx], float(score) / 100.0))

        extra = llm_candidates.get(anchor_id, [])
        all_candidates: List[Tuple[str, float]] = base_candidates + [(k, 0.5) for k in extra if k in data_keys]
        seen = set()
        deduped: List[Tuple[str, float]] = []
        for k, s in all_candidates:
            if k in seen:
                continue
            seen.add(k)
            deduped.append((k, s))

        best_key = None
        best_score = 0.0
        for key, sim in deduped:
            key_type = key_types.get(key, "TEXT")
            if field_type == "TABLE":
                value = data_norm.get(key)
                if not isinstance(value, list) or not any(isinstance(v, dict) for v in value):
                    continue
            if field_type == "DATE" and key_type != "DATE":
                continue
            if field_type in {"MONEY", "PERCENT", "NUMBER"}:
                if not _value_matches_type(data_norm.get(key), field_type):
                    continue
            if field_type == "PERSON" and key_type != "PERSON":
                continue
            if field_type == "ORG" and key_type == "PERSON":
                continue

            score = _score_candidate(sim, field_type, key_type)
            if score > best_score:
                best_score = score
                best_key = key

        ambiguous = best_score < threshold or best_key is None

        mapping[anchor_id] = {
            "label_text": label,
            "json_key": best_key,
            "score": float(best_score * 100.0),
            "ambiguous": ambiguous,
            "field_type": field_type,
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
    data_norm: Dict[str, Any],
    data_keys: List[str],
    llm_candidates: Optional[Dict[str, List[str]]] = None,
    threshold: float = 0.6,
) -> Dict[str, Dict[str, object]]:
    return _heuristic_mapping(anchors, data_norm, data_keys, llm_candidates=llm_candidates, threshold=threshold)


def llm_suggest_candidates(
    anchors: List[Dict[str, object]],
    data_keys: List[str],
    heuristic_mapping: Dict[str, Dict[str, object]],
    model: HFModel,
    batch_size: int = 8,
    top_k: int = 5,
) -> Dict[str, List[str]]:
    ambiguous = [
        a
        for a in anchors
        if heuristic_mapping.get(str(a.get("anchor_id") or ""), {}).get("ambiguous")
    ]
    if not ambiguous or not model or not model.available():
        return {}

    results: Dict[str, List[str]] = {}
    for i in range(0, len(ambiguous), batch_size):
        batch = ambiguous[i : i + batch_size]
        batch_payload = [
            {
                "anchor_id": a.get("anchor_id"),
                "label_text": a.get("label_text"),
                "nearby_text": a.get("nearby_text"),
            }
            for a in batch
        ]
        prompt = (
            "Return ONLY strict JSON with schema: "
            '{"items":[{"anchor_id":"...","candidates":["key1","key2"]}]}.\n'
            "Rules:\n"
            f"- For each anchor, return up to {top_k} keys from the provided list.\n"
            "- Use semantic similarity to the Romanian label.\n"
            "- If unsure, return an empty list.\n\n"
            f"Keys: {data_keys}\n\n"
            f"Anchors: {batch_payload}\n"
        )
        response = model.generate(prompt)
        parsed = _parse_llm_candidates(response)
        for item in parsed:
            anchor_id = str(item.get("anchor_id") or "")
            cand = item.get("candidates")
            if not anchor_id or not isinstance(cand, list):
                continue
            cleaned = [c for c in cand if isinstance(c, str) and c in data_keys][:top_k]
            results[anchor_id] = cleaned

    return results


def composite_map(
    anchors: List[Dict[str, object]],
    data_norm: Dict[str, Any],
    data_keys: List[str],
    model: HFModel,
    fuzzy_threshold: float = 0.6,
) -> Dict[str, object]:
    base = _heuristic_mapping(anchors, data_norm, data_keys, llm_candidates=None, threshold=fuzzy_threshold)
    llm_candidates = llm_suggest_candidates(anchors, data_keys, base, model)
    final_map = _heuristic_mapping(anchors, data_norm, data_keys, llm_candidates=llm_candidates, threshold=fuzzy_threshold)
    mapping_final = {aid: meta.get("json_key") for aid, meta in final_map.items() if meta.get("json_key")}
    return {
        "mapping_final": mapping_final,
        "mapping_heuristic": final_map,
        "mapping_llm": {"candidates": llm_candidates},
    }


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
