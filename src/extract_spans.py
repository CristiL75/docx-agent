import re
from typing import Dict, List, Any, Tuple, Optional

from docx import Document

from src.docx_io.traverse import TextContainer, Location


BLANK_UNDERSCORES = re.compile(r"_{3,}")
BLANK_DOTS = re.compile(r"\.{3,}")
CHECKBOX_RE = re.compile(r"(\|_\||\[\s\]|\[x\]|\[X\]|☐|☑|□)")


def _location_to_dict(location: Location) -> Dict[str, object]:
    return {
        "type": location.type,
        "section_idx": location.section_idx,
        "header_footer": location.header_footer,
        "table_idx": location.table_idx,
        "row": location.row,
        "col": location.col,
        "paragraph_idx": location.paragraph_idx,
    }


def _concat_runs(container: TextContainer) -> str:
    paragraph = container.obj
    runs = getattr(paragraph, "runs", [])
    text = ""
    for run in runs:
        text += run.text
    return text


def _extract_spans_from_text(text: str) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    for m in BLANK_UNDERSCORES.finditer(text):
        spans.append((m.start(), m.end(), "underscore"))
    for m in BLANK_DOTS.finditer(text):
        spans.append((m.start(), m.end(), "dots"))
    for m in CHECKBOX_RE.finditer(text):
        spans.append((m.start(), m.end(), "checkbox"))
    spans.sort(key=lambda s: (s[0], s[1]))
    return spans


def extract_field_spans(containers: List[TextContainer], doc: Document) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []
    for container in containers:
        paragraph_text = _concat_runs(container)
        if not paragraph_text:
            continue
        raw_spans = _extract_spans_from_text(paragraph_text)
        if not raw_spans:
            continue
        for idx, (start, end, kind) in enumerate(raw_spans):
            left_context = paragraph_text[max(0, start - 80) : start]
            right_context = paragraph_text[end : min(len(paragraph_text), end + 80)]
            span = {
                "span_id": f"P{container.location.paragraph_idx}_S{idx}",
                "paragraph_idx": container.location.paragraph_idx,
                "span_index": idx,
                "start_char": start,
                "end_char": end,
                "blank_kind": kind,
                "left_context": left_context,
                "right_context": right_context,
                "paragraph_text": paragraph_text,
                "location": _location_to_dict(container.location),
            }
            spans.append(span)
    return spans
