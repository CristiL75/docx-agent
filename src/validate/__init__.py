import re
import unicodedata
from datetime import datetime
from typing import Any


def _normalize_text(value: str) -> str:
	text = value.lower()
	text = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))
	return " ".join(text.split())


def is_date(value: Any) -> bool:
	if not isinstance(value, str):
		return False
	text = value.strip()
	for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
		try:
			datetime.strptime(text, fmt)
			return True
		except ValueError:
			continue
	return False


def is_money(value: Any) -> bool:
	if isinstance(value, (int, float)):
		return True
	if not isinstance(value, str):
		return False
	v = value.lower()
	if re.search(r"\b\d+[\d\s\.,]*\b", v) and ("lei" in v or "ron" in v or v.strip().replace(",", "").replace(".", "").isdigit()):
		return True
	return False


def is_percent(value: Any) -> bool:
	if not isinstance(value, str):
		return False
	return "%" in value


def is_numericish(value: Any, max_len: int = 10) -> bool:
	if isinstance(value, (int, float)):
		return True
	if not isinstance(value, str):
		return False
	text = value.strip()
	if len(text) > max_len:
		return False
	if len(text.split()) > 2:
		return False
	cleaned = text.replace("%", "").replace(".", "").replace(",", "")
	return cleaned.isdigit()


def is_orgish(value: Any) -> bool:
	if not isinstance(value, str):
		return False
	v = _normalize_text(value)
	return any(tok in v for tok in ("srl", "sa", "sc", "compania", "institutia", "societatea"))


def is_personish(value: Any) -> bool:
	if not isinstance(value, str):
		return False
	v = value.strip()
	if any(ch.isdigit() for ch in v):
		return False
	parts = [p for p in re.split(r"\s+", v) if p]
	return 1 <= len(parts) <= 4


def is_role_title(value: Any) -> bool:
	if not isinstance(value, str):
		return False
	v = value.strip()
	if not v:
		return False
	if any(ch.isdigit() for ch in v):
		return False
	if is_date(v):
		return False
	return len(v) <= 40


def is_addressish(value: Any) -> bool:
	if not isinstance(value, str):
		return False
	v = value.strip()
	if len(v) < 12:
		return False
	if "," in v:
		return True
	return any(ch.isdigit() for ch in v)
