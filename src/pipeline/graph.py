from pathlib import Path
from typing import Dict, Any, TypedDict, List, Optional

from docx import Document
from langgraph.graph import StateGraph, END

from src.docx_io.traverse import iter_text_containers
from src.docx_io.anchors import extract_anchors
from src.docx_io.fill_text import fill_text_container
from src.docx_io.fill_checkboxes import fill_checkboxes_in_container
from src.docx_io.fill_tables import fill_tables
from src.data.normalize import normalize_key, load_json, normalize_data
from src.llm.hf_model import HFModel
from src.llm.map_fields import heuristic_map, llm_map_ambiguous
from src.validate.mapping_rules import merge_mappings
from src.report.make_report import write_report, write_text_report, build_report


class State(TypedDict):
    template_path: Path
    data_path: Path
    anchors: List[Dict[str, Any]]
    data_norm: Dict[str, Any]
    mapping_heuristic: Dict[str, Dict[str, Any]]
    mapping_llm: Dict[str, Any]
    mapping_final: Dict[str, str]
    issues: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    out_path: Path
    report: Dict[str, Any]


class _InternalState(TypedDict, total=False):
    data_raw: Dict[str, Any]
    mapping_heuristic: Dict[str, Dict[str, Any]]
    mapping_llm: Dict[str, Any]


def _normalize_data_node(state: State, artifacts_dir: Path) -> State:
    raw = load_json(state["data_path"])
    data_norm, data_raw = normalize_data(raw)
    state["data_norm"] = data_norm
    state["issues"] = state.get("issues", [])
    internal: _InternalState = {"data_raw": data_raw}
    state.update(internal)
    write_report(artifacts_dir / "data_normalized.json", {"normalized": data_norm, "raw": data_raw})
    return state


def _extract_anchors_node(state: State, artifacts_dir: Path) -> State:
    doc = Document(str(state["template_path"]))
    containers = list(iter_text_containers(doc))
    anchors = extract_anchors(containers, doc)
    state["anchors"] = anchors
    anchors_report = [{k: v for k, v in a.items() if k != "container"} for a in anchors]
    write_report(artifacts_dir / "anchors.json", anchors_report)
    return state


def _heuristic_map_node(state: State, artifacts_dir: Path, threshold: int) -> State:
    data_keys = list(state["data_norm"].keys())
    heuristic = heuristic_map(state["anchors"], data_keys, threshold=threshold)
    state.update({"mapping_heuristic": heuristic})
    write_report(artifacts_dir / "mapping_heuristic.json", heuristic)
    return state


def _llm_map_ambiguous_node(state: State, artifacts_dir: Path, model_name: str) -> State:
    data_keys = list(state["data_norm"].keys())
    model = HFModel(model_name=model_name)
    llm_mapping = llm_map_ambiguous(state.get("mapping_heuristic", {}), data_keys, model)
    state.update({"mapping_llm": llm_mapping})
    write_report(artifacts_dir / "mapping_llm.json", llm_mapping)
    return state


def _validate_merge_node(
    state: State,
    artifacts_dir: Path,
    heuristic_threshold: float,
    llm_threshold: float,
) -> State:
    mapping_final = merge_mappings(
        state["anchors"],
        state.get("mapping_heuristic", {}),
        state.get("mapping_llm", {}),
        heuristic_threshold=heuristic_threshold,
        llm_threshold=llm_threshold,
    )
    state["mapping_final"] = mapping_final
    write_report(artifacts_dir / "mapping_final.json", mapping_final)

    unmatched = [
        {"label": a.get("label_text"), "location": a.get("location")}
        for a in state["anchors"]
        if a.get("anchor_id") not in mapping_final
    ]
    state["issues"] = unmatched
    return state


def _fill_docx_node(state: State, artifacts_dir: Path, dry_run: bool) -> State:
    if dry_run:
        return state

    doc = Document(str(state["template_path"]))
    normalized_data = {normalize_key(k): v for k, v in state["data_norm"].items()}
    mapping = state["mapping_final"]

    actions = state.get("actions", [])

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

    anchors_by_container: Dict[int, List[Dict[str, Any]]] = {}
    for anchor in state["anchors"]:
        if not anchor.get("placeholder_span"):
            continue
        location = anchor.get("location")
        if not isinstance(location, dict):
            continue
        container = location_map.get(_loc_key(location))
        if not container:
            continue
        anchors_by_container.setdefault(id(container), []).append({**anchor, "_container": container})

    filled_text = 0
    for container_anchors in anchors_by_container.values():
        container_anchors.sort(key=lambda a: a["placeholder_span"]["start"], reverse=True)
        for anchor in container_anchors:
            label = anchor.get("label_text")
            placeholder_span = anchor.get("placeholder_span")
            if not label or not placeholder_span:
                continue
            key = mapping.get(anchor.get("anchor_id"))
            if not key:
                continue
            value = normalized_data.get(normalize_key(key))
            if value is None:
                continue
            if fill_text_container(
                anchor["_container"],
                placeholder_span["start"],
                placeholder_span["end"],
                str(value),
            ):
                filled_text += 1
                actions.append({"type": "text", "label": label, "key": key, "value": value})

    label_mapping = {
        a["label_text"]: mapping.get(a["anchor_id"])
        for a in state["anchors"]
        if a.get("label_text") and a.get("anchor_id")
    }

    checkbox_filled = 0
    for container in iter_text_containers(doc):
        filled_here = fill_checkboxes_in_container(container, state["data_norm"], label_mapping)
        checkbox_filled += filled_here
        if filled_here:
            actions.append({"type": "checkbox", "label": container.text})

    table_filled = fill_tables(doc, state["data_norm"])

    doc.save(str(state["out_path"]))
    actions.append(
        {
            "type": "summary",
            "filled_text": filled_text,
            "filled_checkboxes": checkbox_filled,
            "filled_tables": table_filled,
        }
    )
    state["actions"] = actions
    return state


def _report_node(state: State, artifacts_dir: Path) -> State:
    mapping = state["mapping_final"]
    data_keys = list(state["data_norm"].keys())
    used_keys = {v for v in mapping.values() if v}
    unused_keys = [k for k in data_keys if k not in used_keys]

    llm_items = state.get("mapping_llm", {}).get("items", [])
    llm_by_anchor = {item.get("anchor_id"): item for item in llm_items if item.get("anchor_id")}

    mapping_summary = []
    for a in state["anchors"]:
        anchor_id = a.get("anchor_id")
        if not anchor_id:
            continue
        heuristic_item = state.get("mapping_heuristic", {}).get(anchor_id)
        llm_item = llm_by_anchor.get(anchor_id)
        mapping_summary.append(
            {
                "anchor_id": anchor_id,
                "label_text": a.get("label_text"),
                "heuristic": heuristic_item,
                "llm": llm_item,
                "final_key": mapping.get(anchor_id),
            }
        )

    summary = next((a for a in reversed(state.get("actions", [])) if a.get("type") == "summary"), {})
    action_counts = {
        "text": sum(1 for a in state.get("actions", []) if a.get("type") == "text"),
        "checkbox": sum(1 for a in state.get("actions", []) if a.get("type") == "checkbox"),
        "table": int(summary.get("filled_tables", 0)),
    }

    report = build_report(
        anchors_total=len(state["anchors"]),
        filled_text=int(summary.get("filled_text", action_counts["text"])),
        filled_checkboxes=int(summary.get("filled_checkboxes", action_counts["checkbox"])),
        filled_tables=int(summary.get("filled_tables", action_counts["table"])),
        unmatched_anchors=state.get("issues", []),
        unused_json_keys=unused_keys,
        mapping_summary=mapping_summary,
        actions_counts=action_counts,
    )

    state["report"] = report
    write_report(artifacts_dir / "report.json", report)
    write_text_report(artifacts_dir / "report.txt", report)
    write_report(artifacts_dir / "actions.json", state.get("actions", []))
    return state


def run_pipeline(
    input_docx: Path,
    input_json: Path,
    output_docx: Path,
    model_name: str,
    artifacts_dir: Path,
    dry_run: bool = False,
    strict: bool = False,
    heuristic_threshold: int = 90,
    llm_threshold: float = 0.4,
    repair_rounds: int = 1,
    repair_heuristic_threshold: int = 80,
    repair_llm_threshold: float = 0.25,
) -> Dict[str, Any]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    state: State = {
        "template_path": input_docx,
        "data_path": input_json,
        "anchors": [],
        "data_norm": {},
        "mapping_heuristic": {},
        "mapping_llm": {},
        "mapping_final": {},
        "issues": [],
        "actions": [],
        "out_path": output_docx,
        "report": {},
    }

    graph = StateGraph(State)
    graph.add_node("normalize_data_node", lambda s: _normalize_data_node(s, artifacts_dir))
    graph.add_node("extract_anchors_node", lambda s: _extract_anchors_node(s, artifacts_dir))
    graph.add_node("heuristic_map_node", lambda s: _heuristic_map_node(s, artifacts_dir, heuristic_threshold))
    graph.add_node("llm_map_ambiguous_node", lambda s: _llm_map_ambiguous_node(s, artifacts_dir, model_name))
    graph.add_node(
        "validate_merge_node",
        lambda s: _validate_merge_node(s, artifacts_dir, heuristic_threshold, llm_threshold),
    )
    graph.add_node("fill_docx_node", lambda s: _fill_docx_node(s, artifacts_dir, dry_run))
    graph.add_node("report_node", lambda s: _report_node(s, artifacts_dir))

    graph.set_entry_point("normalize_data_node")
    graph.add_edge("normalize_data_node", "extract_anchors_node")
    graph.add_edge("extract_anchors_node", "heuristic_map_node")
    graph.add_edge("heuristic_map_node", "llm_map_ambiguous_node")
    graph.add_edge("llm_map_ambiguous_node", "validate_merge_node")
    graph.add_edge("validate_merge_node", "fill_docx_node")
    graph.add_edge("fill_docx_node", "report_node")
    graph.add_edge("report_node", END)

    compiled = graph.compile()
    result = compiled.invoke(state)

    for _ in range(max(0, int(repair_rounds))):
        if not result.get("issues"):
            break
        _heuristic_map_node(result, artifacts_dir, threshold=repair_heuristic_threshold)
        _llm_map_ambiguous_node(result, artifacts_dir, model_name)
        _validate_merge_node(
            result,
            artifacts_dir,
            heuristic_threshold=repair_heuristic_threshold,
            llm_threshold=repair_llm_threshold,
        )
        _fill_docx_node(result, artifacts_dir, dry_run)
        _report_node(result, artifacts_dir)

    if strict and result.get("issues"):
        raise SystemExit(2)

    return result
