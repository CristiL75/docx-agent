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

    options: List[Tuple[str, Tuple[int, int]]] = []
    for start, end, line in line_spans:
        if CHECKBOX_PATTERN not in line:
            continue
        option_text = line.replace(CHECKBOX_PATTERN, "").replace(CHECKBOX_MARKED, "").strip()
        for s, e in checkbox_spans:
            if s >= start and e <= end:
                options.append((option_text, (s, e)))

    if not options:
        return 0

    option_texts = [o[0] for o in options]
    best = process.extractOne(normalized, option_texts, scorer=fuzz.token_set_ratio)
    if not best:
        return 0
    best_text, best_score, best_idx = best
    if best_score < 70:
        return 0

    span_start, span_end = options[best_idx][1]
    if replace_span_across_runs(paragraph, span_start, span_end, CHECKBOX_MARKED):
        return 1
    return 0
