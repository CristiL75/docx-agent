from typing import Dict, Any, List

from docx import Document

from src.docx_io.traverse import iter_text_containers
from src.docx_io.fill_text import replace_span_across_runs
from src.validate import is_date, is_money


def fill_spans_in_docx(
    doc: Document,
    spans: List[Dict[str, Any]],
    mapping: Dict[str, str],
    data_norm: Dict[str, Any],
    computed_values: Dict[str, Any],
    expected_types: Dict[str, str],
) -> int:
    def _loc_key(location: Dict[str, Any]) -> tuple:
        return (
            location.get("type"),
            location.get("section_idx"),
            location.get("header_footer"),
            location.get("table_idx"),
            location.get("row"),
            location.get("col"),
            location.get("paragraph_idx"),
        )

    location_map = {}
    for container in iter_text_containers(doc):
        location_map[_loc_key(container.location.__dict__)] = container

    spans_by_container: Dict[int, List[Dict[str, Any]]] = {}
    for span in spans:
        if span.get("blank_kind") == "checkbox":
            continue
        location = span.get("location")
        if not isinstance(location, dict):
            continue
        container = location_map.get(_loc_key(location))
        if not container:
            continue
        spans_by_container.setdefault(id(container), []).append({**span, "_container": container})

    filled = 0
    for container_spans in spans_by_container.values():
        container_spans.sort(key=lambda s: s.get("start_char", 0), reverse=True)
        for span in container_spans:
            span_id = span.get("span_id")
            if not span_id:
                continue
            key = mapping.get(span_id)
            if not key:
                continue
            if key == "__computed__":
                value = computed_values.get(span_id)
            else:
                value = data_norm.get(key)
            if value is None:
                continue

            expected = expected_types.get(span_id)
            if expected == "DURATION_DAYS" and is_date(value):
                raise SystemExit("DATE value in DURATION_DAYS span")
            if expected == "PERSON_NAME" and is_money(value):
                raise SystemExit("MONEY value in Subsemnatul span")

            if replace_span_across_runs(
                span["_container"].obj,
                int(span.get("start_char", 0)),
                int(span.get("end_char", 0)),
                str(value),
            ):
                filled += 1

    return filled
