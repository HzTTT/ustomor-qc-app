from __future__ import annotations

from datetime import datetime

from sqlalchemy import func
from sqlmodel import Session, select

try:
    # Runtime path when running from `app/` (e.g. `uvicorn main:app`)
    from models import TagCategory, TagDefinition
except ImportError:  # pragma: no cover
    # Compat: allow importing via `app.*` in some environments/tests.
    from app.models import TagCategory, TagDefinition


PRODUCT_QUALITY_CATEGORY_NAME = "产品质量投诉"
PRODUCT_QUALITY_SCOPE_PREFIX = "命中判定范围：用户收到货后反馈（含试穿/使用/洗后）；售前/未收货仅咨询不命中。命中标准："


def _normalize_product_quality_standard(standard: str) -> str:
    s = (standard or "").strip()
    if not s:
        return PRODUCT_QUALITY_SCOPE_PREFIX
    if s.startswith("命中判定范围：用户收到货后反馈"):
        return s
    if s.startswith("命中判定范围：") and "命中标准：" in s:
        # Replace legacy scope header, keep the actual standard part.
        parts = s.split("命中标准：", 1)
        rest = (parts[1] or "").strip()
        return PRODUCT_QUALITY_SCOPE_PREFIX + rest
    return PRODUCT_QUALITY_SCOPE_PREFIX + s


def normalize_standard_for_category(session: Session, category_id: int, standard: str) -> str:
    """Normalize tag standard based on category rules (idempotent)."""
    c = session.get(TagCategory, int(category_id))
    if not c:
        return standard
    if (c.name or "").strip() != PRODUCT_QUALITY_CATEGORY_NAME:
        return standard
    return _normalize_product_quality_standard(standard)


def _next_tag_sort_order(session: Session, category_id: int) -> int:
    max_sort = session.exec(
        select(func.max(TagDefinition.sort_order)).where(TagDefinition.category_id == int(category_id))
    ).one()
    if max_sort is None:
        return 0
    return int(max_sort) + 1


def update_tag_definition(
    session: Session,
    *,
    tag_id: int,
    name: str,
    standard: str,
    description: str,
    is_active: bool,
    category_id: int | None = None,
) -> TagDefinition:
    t = session.get(TagDefinition, int(tag_id))
    if not t:
        raise ValueError("标签不存在")

    new_name = (name or "").strip() or t.name
    new_category_id = int(category_id) if category_id is not None else int(t.category_id)

    # Validate category when moving
    if new_category_id != int(t.category_id):
        cat = session.get(TagCategory, new_category_id)
        if not cat:
            raise ValueError("所属分类不存在")

    # Duplicate name check (within target category)
    dup = session.exec(
        select(TagDefinition).where(
            TagDefinition.category_id == int(new_category_id),
            TagDefinition.name == new_name,
            TagDefinition.id != int(t.id),
        )
    ).first()
    if dup:
        if new_category_id != int(t.category_id):
            raise ValueError("目标分类已存在同名标签，请先重命名或使用“合并”功能")
        raise ValueError("当前分类已存在同名标签，请先重命名或使用“合并”功能")

    # Apply updates
    t.name = new_name
    t.standard = normalize_standard_for_category(session, int(new_category_id), standard)
    t.description = description
    t.is_active = bool(is_active)

    if new_category_id != int(t.category_id):
        t.category_id = int(new_category_id)
        t.sort_order = _next_tag_sort_order(session, int(new_category_id))

    t.updated_at = datetime.utcnow()
    session.add(t)
    return t
