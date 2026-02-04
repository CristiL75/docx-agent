import hashlib
import re
from typing import Dict, List, Any, Tuple, Optional

from docx import Document

from src.docx_io.traverse import TextContainer, Location


BLANK_UNDERSCORES = re.compile(r"_{2,}")
BLANK_DOTS = re.compile(r"\.{4,}")
BLANK_ELLIPSIS = re.compile(r"…{2,}")
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


def _run_spans(container: TextContainer) -> List[Tuple[int, int, int]]:
    paragraph = container.obj
    runs = getattr(paragraph, "runs", [])
    spans: List[Tuple[int, int, int]] = []
    cursor = 0
    for idx, run in enumerate(runs):
        start = cursor
        cursor += len(run.text)
        end = cursor
        spans.append((idx, start, end))
    return spans


def _extract_spans_from_text(text: str) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    for m in BLANK_UNDERSCORES.finditer(text):
        spans.append((m.start(), m.end(), "underscore"))
    for m in BLANK_DOTS.finditer(text):
        spans.append((m.start(), m.end(), "dots"))
    for m in BLANK_ELLIPSIS.finditer(text):
        spans.append((m.start(), m.end(), "ellipsis"))
    for m in CHECKBOX_RE.finditer(text):
        spans.append((m.start(), m.end(), "checkbox"))
    spans.sort(key=lambda s: (s[0], s[1]))
    return spans


def _make_slot_id(location: Dict[str, object], left: str, right: str, kind: str, raw: str) -> str:
    raw_key = f"{location}|{left}|{right}|{kind}|{raw}"
    return hashlib.sha1(raw_key.encode("utf-8")).hexdigest()[:16]


def extract_field_spans(containers: List[TextContainer], doc: Document) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []
    for container in containers:
        paragraph_text = _concat_runs(container)
        if not paragraph_text:
            if container.location.table_idx is not None:
                location = _location_to_dict(container.location)
                slot_id = _make_slot_id(location, "", "", "empty_cell", "")
                spans.append(
                    {
                        "span_id": slot_id,
                        "paragraph_idx": container.location.paragraph_idx,
                        "span_index": 0,
                        "start_char": 0,
                        "end_char": 0,
                        "blank_kind": "empty_cell",
                        "raw_placeholder_text": "",
                        "left_context": "",
                        "right_context": "",
                        "paragraph_text": "",
                        "location": location,
                        "run_idx_start": None,
                        "run_idx_end": None,
                    }
                )
            continue
        raw_spans = _extract_spans_from_text(paragraph_text)
        if not raw_spans:
            continue
        run_spans = _run_spans(container)
        for idx, (start, end, kind) in enumerate(raw_spans):
            left_context = paragraph_text[max(0, start - 80) : start]
            right_context = paragraph_text[end : min(len(paragraph_text), end + 80)]
            raw_placeholder_text = paragraph_text[start:end]
            run_start = None
            run_end = None
            for run_idx, r_start, r_end in run_spans:
                if run_start is None and start >= r_start and start <= r_end:
                    run_start = run_idx
                if end >= r_start and end <= r_end:
                    run_end = run_idx
            location = _location_to_dict(container.location)
            slot_id = _make_slot_id(location, left_context, right_context, kind, raw_placeholder_text)
            span = {
                "span_id": slot_id,
                "paragraph_idx": container.location.paragraph_idx,
                "span_index": idx,
                "start_char": start,
                "end_char": end,
                "blank_kind": kind,
                "raw_placeholder_text": raw_placeholder_text,
                "left_context": left_context,
                "right_context": right_context,
                "paragraph_text": paragraph_text,
                "location": location,
                "run_idx_start": run_start,
                "run_idx_end": run_end,
            }
            spans.append(span)
    return spans


def extract_slots(docx_path: str) -> List[Dict[str, Any]]:
    doc = Document(docx_path)
    from src.docx_io.traverse import iter_text_containers

    containers = list(iter_text_containers(doc))
    return extract_field_spans(containers, doc)
