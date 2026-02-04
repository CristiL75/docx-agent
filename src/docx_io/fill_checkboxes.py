import re
from typing import Dict, Any, List, Tuple

from docx.text.paragraph import Paragraph
from rapidfuzz import process, fuzz

from src.docx_io.fill_text import replace_span_across_runs
from src.docx_io.traverse import TextContainer
from src.data.normalize import normalize_key

CHECKBOX_PATTERN = "|_|"
CHECKBOX_MARKED = "|x|"


def _find_checkbox_occurrences(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start = 0
    while True:
        idx = text.find(CHECKBOX_PATTERN, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(CHECKBOX_PATTERN)))
        start = idx + len(CHECKBOX_PATTERN)
    return spans


def _line_spans(text: str) -> List[Tuple[int, int, str]]:
    spans = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        start = cursor
        end = cursor + len(line)
        spans.append((start, end, line.rstrip("\n")))
        cursor = end
    if not spans:
        spans.append((0, len(text), text))
    return spans


def _get_mapping_value(text: str, data: Dict[str, Any], mapping: Dict[str, str]) -> Any:
    full_label = text.strip(" :\t")
    label = text.splitlines()[0].strip(" :\t") if text.splitlines() else full_label
    key = mapping.get(full_label) or mapping.get(label, label)
    value = data.get(key)
    if value is None:
        value = data.get(normalize_key(key))
    return value


def fill_checkboxes_in_container(container: TextContainer, data: Dict[str, Any], mapping: Dict[str, str]) -> int:
    paragraph = container.obj
    if not isinstance(paragraph, Paragraph):
        return 0

    text = paragraph.text
    if CHECKBOX_PATTERN not in text:
        return 0

    value = _get_mapping_value(text, data, mapping)
    if not isinstance(value, str):
        return 0
    normalized = value.strip().lower()

    line_spans = _line_spans(text)
    checkbox_spans = _find_checkbox_occurrences(text)
    if not checkbox_spans:
        return 0

    groups: List[List[Tuple[str, Tuple[int, int]]]] = []
    current_group: List[Tuple[str, Tuple[int, int]]] = []

    for start, end, line in line_spans:
        if CHECKBOX_PATTERN in line:
            option_text = line.replace(CHECKBOX_PATTERN, "").replace(CHECKBOX_MARKED, "").strip()
            for s, e in checkbox_spans:
                if s >= start and e <= end:
                    current_group.append((option_text, (s, e)))
        else:
            if current_group:
                groups.append(current_group)
                current_group = []

    if current_group:
        groups.append(current_group)

    if not groups:
        return 0

    best_global_score = 0
    best_span: Tuple[int, int] = (-1, -1)
    for group in groups:
        option_texts = [o[0] for o in group]
        best = process.extractOne(normalized, option_texts, scorer=fuzz.token_set_ratio)
        if not best:
            continue
        _, best_score, best_idx = best
        if best_score > best_global_score:
            best_global_score = best_score
            best_span = group[best_idx][1]

    if best_global_score < 70 or best_span == (-1, -1):
        return 0

    span_start, span_end = best_span
    if replace_span_across_runs(paragraph, span_start, span_end, CHECKBOX_MARKED):
        return 1
    return 0


def fill_checkbox_groups(anchors: List[Dict[str, Any]], data: Dict[str, Any], mapping: Dict[str, str]) -> int:
    filled = 0
    for anchor in anchors:
        if anchor.get("kind") != "checkbox_group":
            continue
        label = str(anchor.get("label_text") or "")
        key = mapping.get(label)
        if not key:
            continue
        value = data.get(key) or data.get(normalize_key(key))
        if not isinstance(value, str):
            continue
        normalized = value.strip().lower()
        options = anchor.get("options") or []
        option_texts = [str(o.get("text") or "").strip() for o in options]
        if not option_texts:
            continue
        best = process.extractOne(normalized, option_texts, scorer=fuzz.token_set_ratio)
        if not best:
            continue
        _, best_score, best_idx = best
        if best_score < 70:
            continue
        opt = options[best_idx]
        span = opt.get("span") or {}
        container = opt.get("container")
        if not container:
            continue
        paragraph = container.obj
        if not isinstance(paragraph, Paragraph):
            continue
        if replace_span_across_runs(paragraph, span.get("start", 0), span.get("end", 0), CHECKBOX_MARKED):
            filled += 1
    return filled
