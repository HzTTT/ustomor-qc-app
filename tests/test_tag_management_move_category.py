from sqlmodel import Session

import pytest

from models import TagCategory, TagDefinition
from tools.tag_management import update_tag_definition


def test_update_tag_can_move_category_and_reorder(session: Session):
    cat1 = TagCategory(name="分类1")
    cat2 = TagCategory(name="分类2")
    session.add(cat1)
    session.add(cat2)
    session.commit()
    session.refresh(cat1)
    session.refresh(cat2)

    t1 = TagDefinition(category_id=cat1.id, name="二级标签A", standard="s1", sort_order=5, is_active=True)
    t2 = TagDefinition(category_id=cat2.id, name="二级标签B", standard="s2", sort_order=7, is_active=True)
    session.add(t1)
    session.add(t2)
    session.commit()
    session.refresh(t1)
    session.refresh(t2)

    update_tag_definition(
        session,
        tag_id=int(t1.id),
        category_id=int(cat2.id),
        name="二级标签A",
        standard="s1x",
        description="d1",
        is_active=True,
    )
    session.commit()
    session.refresh(t1)

    assert int(t1.category_id) == int(cat2.id)
    assert t1.standard == "s1x"
    assert t1.description == "d1"
    assert int(t1.sort_order) == 8


def test_update_tag_move_category_duplicate_name_blocked(session: Session):
    cat1 = TagCategory(name="分类1")
    cat2 = TagCategory(name="分类2")
    session.add(cat1)
    session.add(cat2)
    session.commit()
    session.refresh(cat1)
    session.refresh(cat2)

    t1 = TagDefinition(category_id=cat1.id, name="重复名", standard="", is_active=True)
    t2 = TagDefinition(category_id=cat2.id, name="重复名", standard="", is_active=True)
    session.add(t1)
    session.add(t2)
    session.commit()
    session.refresh(t1)

    with pytest.raises(ValueError) as e:
        update_tag_definition(
            session,
            tag_id=int(t1.id),
            category_id=int(cat2.id),
            name="重复名",
            standard="",
            description="",
            is_active=True,
        )
    assert "同名标签" in str(e.value)


def test_update_tag_move_category_invalid_category(session: Session):
    cat1 = TagCategory(name="分类1")
    session.add(cat1)
    session.commit()
    session.refresh(cat1)

    t1 = TagDefinition(category_id=cat1.id, name="标签", standard="", is_active=True)
    session.add(t1)
    session.commit()
    session.refresh(t1)

    with pytest.raises(ValueError) as e:
        update_tag_definition(
            session,
            tag_id=int(t1.id),
            category_id=999999,
            name="标签",
            standard="",
            description="",
            is_active=True,
        )
    assert str(e.value) == "所属分类不存在"
