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
    "DATE_PARTS",
    "MONEY",
    "PERCENT",
    "NUMBER",
    "ORG_NAME",
    "ORG_ADDRESS",
    "PERSON_NAME",
    "PERSON_ADDRESS",
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
    norm_label = _normalize_text(label)
    norm_nearby = _normalize_text(nearby)
    text = norm_label if len(norm_label) >= 5 else _normalize_text(" ".join([label, nearby]))

    if kind in {"checkbox", "checkbox_group"} or "|_|" in label or "☐" in label or "[ ]" in label:
        return "CHECKBOX_GROUP"
    if "tabel" in text or "lista" in text:
        return "TABLE"
    if any(x in text for x in ("ziua", "luna", "anul")):
        return "DATE_PARTS"
    if "/" in placeholder or "data" in text:
        return "DATE"
    if "%" in label or "procent" in text:
        return "PERCENT"
    if any(x in text for x in ("suma", "lei", "valoare", "tva", "taxa")):
        return "MONEY"
    if any(x in text for x in ("adresa", "sediu", "domiciliu")):
        return "ORG_ADDRESS"
    if any(x in text for x in ("domiciliat", "cnp", "bi", "ci")):
        return "PERSON_ADDRESS"
    if any(x in text for x in ("subsemnat", "reprezentant", "imputernicit", "semnatar", "dl", "dna", "functie")):
        return "PERSON_NAME"
    if any(x in text for x in ("operator economic", "ofertant", "s.c", "societate")):
        return "ORG_NAME"
    if any(x in text for x in ("banca", "autoritate", "achizitor", "contractant", "beneficiar")):
        return "ORG_NAME"
    if "cpv" in text:
        return "NUMBER"
    if any(x in text for x in ("numar", "nr", "cif", "cnp", "cod", "serie", "bi", "ci")):
        return "NUMBER"
    return "TEXT"


def _infer_key_type(key: str) -> str:
    t = _normalize_text(key)
    if "tabel" in t or "lista" in t:
        return "TABLE"
    if any(x in t for x in ("zi", "luna", "an")) and "data" in t:
        return "DATE_PARTS"
    if "data" in t:
        return "DATE"
    if any(x in t for x in ("tva", "suma", "valoare", "lei")):
        return "MONEY"
    if "%" in key or "procent" in t:
        return "PERCENT"
    if "cpv" in t:
        return "NUMBER"
    if any(x in t for x in ("numar", "nr", "cif", "cnp", "cod", "serie", "bi", "ci")):
        return "NUMBER"
    if any(x in t for x in ("adresa", "sediu", "domiciliu")):
        return "ORG_ADDRESS"
    if any(x in t for x in ("domiciliat", "cnp", "bi", "ci")):
        return "PERSON_ADDRESS"
    if any(x in t for x in ("operator", "ofertant", "autoritate", "societate", "achizitor", "s.c", "contractant", "beneficiar", "banca", "asigur")):
        if "denumire" in t or "nume" in t:
            return "ORG_NAME"
    if any(x in t for x in ("reprezentant", "imputernicit", "semnatar", "nume", "functie", "director")):
        return "PERSON_NAME"
    if any(x in t for x in ("operator", "ofertant", "autoritate", "societate", "achizitor", "s.c")):
        return "ORG_NAME"
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
    if field_type == "DATE_PARTS":
        return key_type in {"DATE", "DATE_PARTS"}
    if field_type == "MONEY":
        return key_type in {"MONEY", "NUMBER"}
    if field_type == "PERCENT":
        return key_type in {"PERCENT", "NUMBER"}
    if field_type == "NUMBER":
        return key_type in {"NUMBER", "MONEY", "PERCENT"}
    if field_type == "PERSON_NAME":
        return key_type == "PERSON_NAME"
    if field_type == "PERSON_ADDRESS":
        return key_type == "PERSON_ADDRESS"
    if field_type == "ORG_NAME":
        return key_type == "ORG_NAME"
    if field_type == "ORG_ADDRESS":
        return key_type == "ORG_ADDRESS"
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
    label: str,
    nearby: str,
    key: str,
) -> float:
    type_score = 1.0 if _type_compatible(field_type, key_type) else 0.0
    context = _context_score(label, nearby, key)
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
        if norm_label == "data":
            exact_key = next((k for k in data_keys if _normalize_text(k) == "data"), None)
            if exact_key:
                mapping[anchor_id] = {
                    "label_text": label,
                    "json_key": exact_key,
                    "score": 100.0,
                    "ambiguous": False,
                    "field_type": field_type,
                }
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
            value = data_norm.get(key)
            if field_type != "TABLE" and isinstance(value, (list, dict)):
                continue
            if field_type == "DATE" and key_type != "DATE":
                continue
            if field_type in {"MONEY", "PERCENT", "NUMBER"}:
                if not _value_matches_type(data_norm.get(key), field_type):
                    continue
            if field_type == "PERSON_NAME" and key_type != "PERSON_NAME":
                continue
            if field_type == "PERSON_ADDRESS" and key_type != "PERSON_ADDRESS":
                continue
            if field_type == "ORG_NAME" and key_type != "ORG_NAME":
                continue
            if field_type == "ORG_ADDRESS" and key_type != "ORG_ADDRESS":
                continue
            if field_type not in {"TEXT", "CHECKBOX_GROUP"} and not _type_compatible(field_type, key_type):
                continue

            score = _score_candidate(sim, field_type, key_type, label, nearby, key)
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
) -> Dict[str, List[Dict[str, Any]]]:
    ambiguous = [
        a
        for a in anchors
        if heuristic_mapping.get(str(a.get("anchor_id") or ""), {}).get("ambiguous")
    ]
    if not ambiguous or not model or not model.available():
        return {}

    results: Dict[str, List[Dict[str, Any]]] = {}
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
            '{"items":[{"anchor_id":"...","candidates":[{"key":"...","confidence":0.0}]}]}.\n'
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
            cleaned: List[Dict[str, Any]] = []
            for entry in cand:
                if not isinstance(entry, dict):
                    continue
                key = entry.get("key")
                conf = entry.get("confidence")
                if not isinstance(key, str) or key not in data_keys:
                    continue
                if not isinstance(conf, (int, float)):
                    conf = 0.0
                cleaned.append({"key": key, "confidence": float(conf)})
            results[anchor_id] = cleaned[:top_k]

    return results


def _context_score(label: str, nearby: str, key: str) -> float:
    label_tokens = set(_normalize_text(label).split())
    nearby_tokens = set(_normalize_text(nearby).split())
    key_tokens = set(_normalize_text(key).split())
    tokens = label_tokens | nearby_tokens
    if not tokens or not key_tokens:
        return 0.0
    overlap = tokens & key_tokens
    return min(1.0, len(overlap) / max(1, len(key_tokens)))


def _hard_gate(
    field_type: str,
    label: str,
    key: str,
    value: Any,
) -> bool:
    if isinstance(value, (list, dict)):
        return field_type == "TABLE"

    norm_key = _normalize_text(key)
    norm_label = _normalize_text(label)

    if field_type in {"DATE", "DATE_PARTS"}:
        if norm_key.startswith("data"):
            return True
        if isinstance(value, str) and re.search(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{4}\b", value):
            return True
        return False

    if field_type == "PERCENT":
        if isinstance(value, str) and "%" in value:
            return True
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, str) and value.strip().replace(".", "").isdigit() and len(value.strip()) <= 6:
            return True
        return False

    if field_type == "NUMBER":
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, str) and value.strip().replace("/", "").replace("-", "").isdigit() and len(value.strip()) <= 12:
            return True
        return False

    if field_type == "ORG_NAME":
        if any(x in norm_key for x in ("imputernicit", "subsemnat", "reprezentant")):
            return False
        if any(x in norm_key for x in ("ofertant", "operator economic", "contractant")):
            return True
    if field_type == "PERSON_NAME":
        if any(x in norm_key for x in ("imputernicit", "subsemnat", "reprezentant", "semnatar")):
            return True
        if any(x in norm_key for x in ("ofertant", "operator economic", "contractant")):
            return False

    return True


def _build_candidates(
    anchors: List[Dict[str, object]],
    data_norm: Dict[str, Any],
    data_keys: List[str],
    llm_candidates: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    key_types = {k: _infer_key_type(k) for k in data_keys}
    candidates: Dict[str, List[Dict[str, Any]]] = {}

    for anchor in anchors:
        anchor_id = str(anchor.get("anchor_id") or "")
        label = str(anchor.get("label_text") or "")
        nearby = str(anchor.get("nearby_text") or "")
        if not anchor_id or not label:
            continue
        field_type = _infer_field_type(anchor)
        query = " ".join([_normalize_text(label), _normalize_text(nearby)]).strip()
        matches = process.extract(query, [_normalize_text(k) for k in data_keys], scorer=fuzz.token_set_ratio, limit=10)
        base: List[Tuple[str, float]] = [(data_keys[idx], float(score) / 100.0) for _, score, idx in matches]

        llm_list = llm_candidates.get(anchor_id, [])
        llm_map = {c.get("key"): float(c.get("confidence") or 0.0) for c in llm_list if isinstance(c, dict)}

        merged: Dict[str, Dict[str, Any]] = {}
        for key, score_text in base:
            merged.setdefault(key, {"key": key, "score_text": score_text, "score_llm": 0.0})
        for key, score_llm in llm_map.items():
            if key not in merged:
                merged[key] = {"key": key, "score_text": 0.0, "score_llm": score_llm}
            else:
                merged[key]["score_llm"] = max(merged[key]["score_llm"], score_llm)

        cand_list: List[Dict[str, Any]] = []
        for key, meta in merged.items():
            value = data_norm.get(key)
            if not _hard_gate(field_type, label, key, value):
                continue
            key_type = key_types.get(key, "TEXT")
            score_type = 1.0 if _type_compatible(field_type, key_type) else 0.0
            score_context = _context_score(label, nearby, key)
            total = (
                0.45 * meta.get("score_text", 0.0)
                + 0.25 * meta.get("score_llm", 0.0)
                + 0.20 * score_type
                + 0.10 * score_context
            )
            cand_list.append(
                {
                    "key": key,
                    "score_text": meta.get("score_text", 0.0),
                    "score_llm": meta.get("score_llm", 0.0),
                    "score_type": score_type,
                    "score_context": score_context,
                    "total": total,
                    "field_type": field_type,
                }
            )

        cand_list.sort(key=lambda c: (-c["total"], c["key"]))
        candidates[anchor_id] = cand_list

    return candidates


def solve_global_mapping(
    anchors: List[Dict[str, object]],
    candidates: Dict[str, List[Dict[str, Any]]],
    threshold_global: float = 0.55,
) -> Tuple[Dict[str, str], Dict[str, int]]:
    def _section_id(a: Dict[str, object]) -> Tuple:
        loc = a.get("location") or {}
        return (
            loc.get("type"),
            loc.get("section_idx"),
            loc.get("header_footer"),
            loc.get("table_idx"),
        )

    anchor_map = {str(a.get("anchor_id")): a for a in anchors if a.get("anchor_id")}

    stability: List[Tuple[str, float]] = []
    for anchor_id, cand_list in candidates.items():
        best = cand_list[0]["total"] if cand_list else 0.0
        second = cand_list[1]["total"] if len(cand_list) > 1 else 0.0
        stability.append((anchor_id, best - second))
    stability.sort(key=lambda x: (-x[1], x[0]))

    mapping_final: Dict[str, str] = {}
    chosen_scores: Dict[str, float] = {}
    for anchor_id, _ in stability:
        cand_list = candidates.get(anchor_id, [])
        if not cand_list:
            continue
        best = cand_list[0]
        if best["total"] >= threshold_global:
            mapping_final[anchor_id] = best["key"]
            chosen_scores[anchor_id] = best["total"]

    def _is_incompatible(t1: str, t2: str) -> bool:
        person_types = {"PERSON_NAME", "PERSON_ADDRESS"}
        org_types = {"ORG_NAME", "ORG_ADDRESS"}
        if (t1 in person_types and t2 in org_types) or (t2 in person_types and t1 in org_types):
            return True
        return False

    def _anchor_keywords(a: Dict[str, object], tokens: Tuple[str, ...]) -> bool:
        text = _normalize_text(str(a.get("label_text") or "") + " " + str(a.get("nearby_text") or ""))
        return any(tok in text for tok in tokens)

    conflicts_found = 0
    repairs_made = 0

    for _ in range(3):
        conflict_anchors: List[str] = []

        # Type collisions per section
        section_to_keys: Dict[Tuple, Dict[str, List[str]]] = {}
        for aid, key in mapping_final.items():
            a = anchor_map.get(aid)
            if not a:
                continue
            sec = _section_id(a)
            section_to_keys.setdefault(sec, {}).setdefault(key, []).append(aid)

        for sec, key_map in section_to_keys.items():
            for key, aids in key_map.items():
                if len(aids) < 2:
                    continue
                types = [candidates[aid][0]["field_type"] for aid in aids if candidates.get(aid)]
                for i in range(len(types)):
                    for j in range(i + 1, len(types)):
                        if _is_incompatible(types[i], types[j]):
                            conflict_anchors.extend([aids[i], aids[j]])

        # Section requirements
        for sec in { _section_id(a) for a in anchors }:
            sec_anchors = [a for a in anchors if _section_id(a) == sec]
            if any(_anchor_keywords(a, ("operator economic", "denumirea numele")) for a in sec_anchors):
                has_org = any(
                    candidates.get(aid, [{}])[0].get("field_type") == "ORG_NAME"
                    and aid in mapping_final
                    and any(tok in _normalize_text(mapping_final[aid]) for tok in ("ofertant", "operator economic", "contractant"))
                    for aid in mapping_final
                )
                if not has_org:
                    conflict_anchors.extend([a.get("anchor_id") for a in sec_anchors if a.get("anchor_id")])

            if any(_anchor_keywords(a, ("subsemnat" ,)) for a in sec_anchors):
                has_person = any(
                    candidates.get(aid, [{}])[0].get("field_type") == "PERSON_NAME"
                    and aid in mapping_final
                    and any(tok in _normalize_text(mapping_final[aid]) for tok in ("imputernicit", "subsemnat", "reprezentant"))
                    for aid in mapping_final
                )
                if not has_person:
                    conflict_anchors.extend([a.get("anchor_id") for a in sec_anchors if a.get("anchor_id")])

        if not conflict_anchors:
            break

        conflicts_found += 1
        # repair weakest anchors first
        unique_conflicts = sorted(set(a for a in conflict_anchors if a))
        unique_conflicts.sort(key=lambda aid: (chosen_scores.get(aid, 0.0), aid))
        repaired_any = False
        for aid in unique_conflicts:
            cand_list = candidates.get(aid, [])
            if not cand_list:
                continue
            current_key = mapping_final.get(aid)
            for cand in cand_list[1:]:
                mapping_final[aid] = cand["key"]
                chosen_scores[aid] = cand["total"]
                repaired_any = True
                repairs_made += 1
                break
            if not repaired_any and current_key:
                mapping_final.pop(aid, None)
        if not repaired_any:
            break

    # Role-pattern cluster enforcement
    role_clusters_detected = 0
    role_repairs_made = 0
    def _cluster_sort_key(a: Dict[str, object]) -> Tuple[int, str]:
        loc = a.get("location") or {}
        return (loc.get("paragraph_idx") or -1, str(a.get("anchor_id") or ""))

    def _pick_best_of_type(aid: str, allowed: Tuple[str, ...]) -> Optional[str]:
        cand_list = candidates.get(aid, [])
        for cand in cand_list:
            if cand.get("field_type") in allowed:
                return cand.get("key")
        return None

    clusters: Dict[int, List[Dict[str, object]]] = {}
    for a in anchors:
        cid = a.get("cluster_id")
        if cid is None:
            continue
        clusters.setdefault(int(cid), []).append(a)

    for _, items in clusters.items():
        if not items:
            continue
        role_pattern = items[0].get("role_pattern")
        if not role_pattern:
            continue
        role_clusters_detected += 1
        items.sort(key=_cluster_sort_key)

        if role_pattern == "PERSON_THEN_ORG":
            for pos, a in enumerate(items[:2]):
                aid = str(a.get("anchor_id") or "")
                if not aid:
                    continue
                required = ("PERSON_NAME",) if pos == 0 else ("ORG_NAME", "ORG_ADDRESS")
                current = mapping_final.get(aid)
                current_type = None
                if current:
                    for cand in candidates.get(aid, []):
                        if cand.get("key") == current:
                            current_type = cand.get("field_type")
                            break
                if current_type in required:
                    continue
                conflicts_found += 1
                next_key = _pick_best_of_type(aid, required)
                if next_key:
                    mapping_final[aid] = next_key
                    repairs_made += 1
                    role_repairs_made += 1
                else:
                    if aid in mapping_final:
                        mapping_final.pop(aid, None)
                        role_repairs_made += 1

        if role_pattern == "ORG_THEN_DATE":
            for pos, a in enumerate(items):
                aid = str(a.get("anchor_id") or "")
                if not aid:
                    continue
                required = ("ORG_NAME",) if pos == 0 else ("DATE_PARTS",)
                current = mapping_final.get(aid)
                current_type = None
                if current:
                    for cand in candidates.get(aid, []):
                        if cand.get("key") == current:
                            current_type = cand.get("field_type")
                            break
                if current_type in required:
                    continue
                conflicts_found += 1
                next_key = _pick_best_of_type(aid, required)
                if next_key:
                    mapping_final[aid] = next_key
                    repairs_made += 1
                    role_repairs_made += 1
                else:
                    if aid in mapping_final:
                        mapping_final.pop(aid, None)
                        role_repairs_made += 1

        if role_pattern == "ORG_ONLY":
            for a in items:
                aid = str(a.get("anchor_id") or "")
                if not aid:
                    continue
                required = ("ORG_NAME",)
                current = mapping_final.get(aid)
                current_type = None
                if current:
                    for cand in candidates.get(aid, []):
                        if cand.get("key") == current:
                            current_type = cand.get("field_type")
                            break
                if current_type in required:
                    continue
                conflicts_found += 1
                next_key = _pick_best_of_type(aid, required)
                if next_key:
                    mapping_final[aid] = next_key
                    repairs_made += 1
                    role_repairs_made += 1
                else:
                    if aid in mapping_final:
                        mapping_final.pop(aid, None)
                        role_repairs_made += 1

        if role_pattern == "MONEY_THEN_PERCENT":
            for pos, a in enumerate(items[:2]):
                aid = str(a.get("anchor_id") or "")
                if not aid:
                    continue
                required = ("MONEY",) if pos == 0 else ("PERCENT",)
                current = mapping_final.get(aid)
                current_type = None
                if current:
                    for cand in candidates.get(aid, []):
                        if cand.get("key") == current:
                            current_type = cand.get("field_type")
                            break
                if current_type in required:
                    continue
                conflicts_found += 1
                next_key = _pick_best_of_type(aid, required)
                if next_key:
                    mapping_final[aid] = next_key
                    repairs_made += 1
                    role_repairs_made += 1
                else:
                    if aid in mapping_final:
                        mapping_final.pop(aid, None)
                        role_repairs_made += 1

    stats = {
        "conflicts_found": conflicts_found,
        "repairs_made": repairs_made,
        "role_clusters_detected": role_clusters_detected,
        "role_repairs_made": role_repairs_made,
    }
    return mapping_final, stats


def composite_map(
    anchors: List[Dict[str, object]],
    data_norm: Dict[str, Any],
    data_keys: List[str],
    model: HFModel,
    fuzzy_threshold: float = 0.6,
) -> Dict[str, object]:
    base = _heuristic_mapping(anchors, data_norm, data_keys, llm_candidates=None, threshold=fuzzy_threshold)
    llm_candidates = llm_suggest_candidates(anchors, data_keys, base, model)
    candidates = _build_candidates(anchors, data_norm, data_keys, llm_candidates)
    mapping_final, stats = solve_global_mapping(anchors, candidates)
    anchor_map = {str(a.get("anchor_id")): a for a in anchors if a.get("anchor_id")}
    final_map: Dict[str, Dict[str, Any]] = {}
    for aid, cand_list in candidates.items():
        best = cand_list[0] if cand_list else {}
        final_map[aid] = {
            "label_text": (anchor_map.get(aid, {}) or {}).get("label_text"),
            "json_key": mapping_final.get(aid),
            "score": float(best.get("total", 0.0)) * 100.0,
            "ambiguous": aid not in mapping_final,
            "field_type": best.get("field_type", "TEXT"),
        }
    return {
        "mapping_final": mapping_final,
        "mapping_heuristic": final_map,
        "mapping_llm": {"candidates": llm_candidates},
        "mapping_stats": stats,
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
