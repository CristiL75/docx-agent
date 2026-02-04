from typing import Dict, Any, List

from docx import Document
from docx.oxml import OxmlElement
from docx.text.paragraph import Paragraph

from src.docx_io.traverse import iter_text_containers
from src.docx_io.fill_text import replace_span_across_runs
from src.validate import is_date, is_money, infer_slot_type, value_matches_type


def fill_spans_in_docx(
    doc: Document,
    spans: List[Dict[str, Any]],
    mapping: Dict[str, str],
    data_norm: Dict[str, Any],
    computed_values: Dict[str, Any],
    expected_types: Dict[str, str],
) -> tuple[int, List[Dict[str, Any]]]:
    def _placeholder_only(paragraph: Paragraph, raw: str) -> bool:
        if raw is None:
            return False
        return paragraph.text.strip() == str(raw).strip()

    def _insert_paragraph_after(paragraph: Paragraph, text: str) -> Paragraph:
        new_p = OxmlElement("w:p")
        paragraph._p.addnext(new_p)
        new_para = Paragraph(new_p, paragraph._parent)
        try:
            new_para.style = paragraph.style
        except Exception:
            pass
        if text:
            new_para.add_run(text)
        return new_para

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
    suspicious: List[Dict[str, Any]] = []
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

            slot_type = infer_slot_type(span).value
            if not value_matches_type(value, slot_type):
                suspicious.append(
                    {
                        "span_id": span_id,
                        "slot_type": slot_type,
                        "value": value,
                        "reason": "value does not match slot_type",
                    }
                )

            paragraph = span["_container"].obj
            raw_placeholder = span.get("raw_placeholder_text", "")
            value_text = str(value)

            if "\n" in value_text and _placeholder_only(paragraph, raw_placeholder):
                lines = value_text.split("\n")
                if replace_span_across_runs(
                    paragraph,
                    int(span.get("start_char", 0)),
                    int(span.get("end_char", 0)),
                    lines[0] if lines else "",
                ):
                    for extra in lines[1:]:
                        _insert_paragraph_after(paragraph, extra)
                    if raw_placeholder and raw_placeholder in paragraph.text:
                        suspicious.append(
                            {
                                "span_id": span_id,
                                "slot_type": slot_type,
                                "value": value,
                                "reason": "placeholder remains after fill",
                            }
                        )
                    filled += 1
                continue

            if replace_span_across_runs(
                paragraph,
                int(span.get("start_char", 0)),
                int(span.get("end_char", 0)),
                value_text,
            ):
                if raw_placeholder and raw_placeholder in paragraph.text:
                    suspicious.append(
                        {
                            "span_id": span_id,
                            "slot_type": slot_type,
                            "value": value,
                            "reason": "placeholder remains after fill",
                        }
                    )
                filled += 1

    return filled, suspicious


def fill_docx(
    template_path: str,
    slots: List[Dict[str, Any]],
    mapping: Dict[str, str],
    json_data: Dict[str, Any],
    output_path: str,
    computed_values: Dict[str, Any] | None = None,
    expected_types: Dict[str, str] | None = None,
) -> str:
    doc = Document(template_path)
    fill_spans_in_docx(
        doc,
        slots,
        mapping,
        json_data,
        computed_values or {},
        expected_types or {},
    )
    doc.save(output_path)
    return output_path
