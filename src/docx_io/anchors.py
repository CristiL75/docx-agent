import re
from typing import Dict, List, Optional, Tuple

from docx import Document

from src.docx_io.traverse import TextContainer, Location

PLACEHOLDER_UNDERSCORES = re.compile(r"_{4,}")
PLACEHOLDER_DOTS = re.compile(r"\.{4,}")
PLACEHOLDER_DATE = re.compile(r"_{4,}/_{4,}/_{4,}")
CHECKBOX_RE = re.compile(r"(\|_\||\[\s\]|\[x\]|\[X\]|☐|☑)")


def _line_spans(text: str) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        start = cursor
        end = cursor + len(line)
        spans.append((start, end, line.rstrip("\n")))
        cursor = end
    if not spans:
        spans.append((0, len(text), text))
    return spans


def _find_checkbox_spans(line: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    idx = 0
    while True:
        pos = line.find("|_|", idx)
        if pos == -1:
            break
        spans.append((pos, pos + 3))
        idx = pos + 3
    for match in re.finditer(r"\[\s\]|\[x\]|\[X\]|☐|☑", line):
        spans.append((match.start(), match.end()))
    return spans


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


def _concat_runs(container: TextContainer) -> Tuple[str, List[Tuple[int, int]]]:
    paragraph = container.obj
    runs = getattr(paragraph, "runs", [])
    spans: List[Tuple[int, int]] = []
    text = ""
    for run in runs:
        start = len(text)
        text += run.text
        end = len(text)
        spans.append((start, end))
    return text, spans


def _find_placeholder_spans(text: str) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    occupied = [False] * (len(text) + 1)

    def _reserve(start: int, end: int) -> None:
        for i in range(start, end):
            occupied[i] = True

    for match in PLACEHOLDER_DATE.finditer(text):
        spans.append((match.start(), match.end(), match.group(0)))
        _reserve(match.start(), match.end())

    for regex in (PLACEHOLDER_UNDERSCORES, PLACEHOLDER_DOTS):
        for match in regex.finditer(text):
            if any(occupied[match.start() : match.end()]):
                continue
            spans.append((match.start(), match.end(), match.group(0)))
            _reserve(match.start(), match.end())

    spans.sort(key=lambda s: s[0])
    return spans


def _extract_label_from_text(text: str, start: int) -> str:
    before = text[:start]
    lines = before.splitlines()
    if not lines:
        return ""
    return lines[-1].strip(" :\t")


def _get_table_cell_text(doc: Document, location: Location, col_offset: int) -> Optional[str]:
    if location.table_idx is None or location.row is None or location.col is None:
        return None
    col_idx = location.col + col_offset
    if col_idx < 0:
        return None
    if location.type == "header":
        section = doc.sections[location.section_idx]
        table = section.header.tables[location.table_idx]
    elif location.type == "footer":
        section = doc.sections[location.section_idx]
        table = section.footer.tables[location.table_idx]
    else:
        table = doc.tables[location.table_idx]
    if location.row >= len(table.rows) or col_idx >= len(table.rows[location.row].cells):
        return None
    return table.rows[location.row].cells[col_idx].text.strip()


def extract_anchors(containers: List[TextContainer], doc: Document) -> List[Dict[str, object]]:
    anchors: List[Dict[str, object]] = []
    anchor_id = 1
    prev_text: Optional[str] = None

    for container in containers:
        paragraph_text, _ = _concat_runs(container)
        text = paragraph_text
        if not text:
            prev_text = None
            continue

        placeholder_spans = _find_placeholder_spans(text)
        for start, end, placeholder in placeholder_spans:
            label = _extract_label_from_text(text, start)

            if container.location.table_idx is not None and not label:
                left_label = _get_table_cell_text(doc, container.location, -1)
                if left_label:
                    label = left_label

            if not label and prev_text:
                if "(" in prev_text and ")" in prev_text:
                    label = prev_text.strip()

            if not label:
                continue

            nearby = text[max(0, start - 100) : min(len(text), end + 100)]
            kind = "table" if container.location.table_idx is not None else "text"

            anchors.append(
                {
                    "anchor_id": f"A{anchor_id}",
                    "label_text": label,
                    "nearby_text": nearby[:200],
                    "placeholder_span": {"start": start, "end": end, "text": placeholder},
                    "location": _location_to_dict(container.location),
                    "kind": kind,
                    "container": container,
                }
            )
            anchor_id += 1

        if CHECKBOX_RE.search(text):
            options: List[Dict[str, object]] = []
            for start, end, line in _line_spans(text):
                if not CHECKBOX_RE.search(line):
                    continue
                spans = _find_checkbox_spans(line)
                option_text = (
                    line.replace("|_|", "")
                    .replace("[ ]", "")
                    .replace("[x]", "")
                    .replace("[X]", "")
                    .replace("☐", "")
                    .replace("☑", "")
                    .strip()
                )
                for s, e in spans:
                    options.append(
                        {
                            "text": option_text,
                            "span": {"start": start + s, "end": start + e},
                        }
                    )

            anchors.append(
                {
                    "anchor_id": f"A{anchor_id}",
                    "label_text": text.strip(),
                    "nearby_text": text[:200],
                    "placeholder_span": None,
                    "location": _location_to_dict(container.location),
                    "kind": "checkbox_group",
                    "options": options,
                    "container": container,
                }
            )
            anchor_id += 1

        prev_text = text

    return anchors
