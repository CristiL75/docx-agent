import re
import unicodedata
from datetime import datetime
from enum import Enum
from typing import Any, Dict


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


class SlotType(str, Enum):
    DATE = "DATE"
    DATE_PARTS = "DATE_PARTS"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    NUMBER = "NUMBER"
    ORG_NAME = "ORG_NAME"
    ORG_ADDRESS = "ORG_ADDRESS"
    PERSON_NAME = "PERSON_NAME"
    PERSON_ROLE = "PERSON_ROLE"
    CPV_CODE = "CPV_CODE"
    CHECKBOX_GROUP = "CHECKBOX_GROUP"
    UNKNOWN = "UNKNOWN"


def infer_slot_type(slot: Dict[str, Any]) -> SlotType:
    text = f"{slot.get('left_context','')} {slot.get('right_context','')} {slot.get('paragraph_text','')}"
    norm = _normalize_text(text)
    blank_kind = str(slot.get("blank_kind") or "")

    if blank_kind == "checkbox" or "|_|" in text or "☐" in text or "□" in text:
        return SlotType.CHECKBOX_GROUP
    if "cpv" in norm:
        return SlotType.CPV_CODE
    if any(tok in norm for tok in ("ziua", "luna", "anul")):
        return SlotType.DATE_PARTS
    if "data" in norm:
        return SlotType.DATE
    if "%" in text or "procent" in norm:
        return SlotType.PERCENT
    if any(tok in norm for tok in ("suma", "lei", "valoare", "tva", "taxa")):
        return SlotType.MONEY
    if any(tok in norm for tok in ("adresa", "sediu", "domiciliu", "strada", "judet")):
        return SlotType.ORG_ADDRESS
    if any(tok in norm for tok in ("in calitate de", "functie", "reprezentant", "imputernicit")):
        return SlotType.PERSON_ROLE
    if any(tok in norm for tok in ("subsemnatul", "dl", "dna", "nume")):
        return SlotType.PERSON_NAME
    if any(tok in norm for tok in ("ofertant", "operator economic", "contractant", "societate", "srl", "sa", "s.c")):
        return SlotType.ORG_NAME
    if any(tok in norm for tok in ("nr", "numar", "cod", "serie", "cif", "cui")):
        return SlotType.NUMBER
    return SlotType.UNKNOWN


def value_matches_type(value: Any, slot_type: SlotType | str) -> bool:
    if isinstance(slot_type, str):
        try:
            slot_type = SlotType(slot_type)
        except ValueError:
            slot_type = SlotType.UNKNOWN

    if slot_type == SlotType.UNKNOWN:
        return True
    if slot_type == SlotType.DATE:
        return is_date(value)
    if slot_type == SlotType.DATE_PARTS:
        return is_numericish(value, max_len=4) and not is_date(value)
    if slot_type == SlotType.MONEY:
        return is_money(value)
    if slot_type == SlotType.PERCENT:
        return is_percent(value) and is_numericish(str(value).replace("%", ""), max_len=6)
    if slot_type == SlotType.NUMBER:
        return is_numericish(value, max_len=12)
    if slot_type == SlotType.ORG_NAME:
        return is_orgish(value)
    if slot_type == SlotType.ORG_ADDRESS:
        return is_addressish(value)
    if slot_type == SlotType.PERSON_NAME:
        return is_personish(value)
    if slot_type == SlotType.PERSON_ROLE:
        return is_role_title(value)
    if slot_type == SlotType.CPV_CODE:
        return isinstance(value, str) and re.search(r"\b\d{8}-\d\b", value)
    if slot_type == SlotType.CHECKBOX_GROUP:
        return isinstance(value, str)
    return True
