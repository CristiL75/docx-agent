import json
import unicodedata
from pathlib import Path
from typing import Any, Dict, Tuple


def normalize_key(value: str) -> str:
    text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return " ".join(text.lower().split())


def load_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _try_parse_json(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        trimmed = value.strip()
        return unicodedata.normalize("NFC", trimmed)
    if isinstance(value, list):
        return [_normalize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize_value(v) for k, v in value.items()}
    return value


def normalize_data(raw: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    normalized: Dict[str, Any] = {}
    for key, value in raw.items():
        parsed = _try_parse_json(value)
        normalized[key] = _normalize_value(parsed)
    return normalized, raw
