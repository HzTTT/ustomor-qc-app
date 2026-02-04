from __future__ import annotations

import re


_ALIAS_SPLIT_RE = re.compile(r"[,\n;/|，；、]+")


def normalize_text(value: str) -> str:
    """Normalize category/tag text for stable matching."""
    if not value:
        return ""
    return re.sub(r"\s+", "", value.strip().lower())


def normalize_key(category: str, tag_name: str) -> str:
    cat = normalize_text(category)
    name = normalize_text(tag_name)
    if not cat and not name:
        return ""
    return f"{cat}::{name}"


def parse_aliases(raw: str) -> list[str]:
    if not raw:
        return []
    parts = _ALIAS_SPLIT_RE.split(raw)
    return [p.strip() for p in parts if p.strip()]
