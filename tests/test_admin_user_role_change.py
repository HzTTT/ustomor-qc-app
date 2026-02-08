from sqlmodel import Session

from models import Role, User
from tools.user_admin import validate_role_change


def test_validate_role_change_blocks_demote_last_admin(session: Session):
    acting = User(username="a", email="a@example.com", name="A", role=Role.admin, password_hash="x")
    session.add(acting)
    session.commit()
    session.refresh(acting)

    err = validate_role_change(session=session, target_user=acting, new_role=Role.agent, acting_user=acting)
    assert err == "至少需要保留一个管理员账号"


def test_validate_role_change_allows_demote_when_other_admin_exists(session: Session):
    admin1 = User(username="a1", email="a1@example.com", name="A1", role=Role.admin, password_hash="x")
    admin2 = User(username="a2", email="a2@example.com", name="A2", role=Role.admin, password_hash="x")
    session.add(admin1)
    session.add(admin2)
    session.commit()
    session.refresh(admin1)
    session.refresh(admin2)

    err = validate_role_change(session=session, target_user=admin1, new_role=Role.supervisor, acting_user=admin2)
    assert err is None


def test_validate_role_change_blocks_inactive_user(session: Session):
    acting = User(username="a", email="a@example.com", name="A", role=Role.admin, password_hash="x")
    target = User(
        username="u",
        email="u@example.com",
        name="U",
        role=Role.agent,
        password_hash="x",
        is_active=False,
    )
    session.add(acting)
    session.add(target)
    session.commit()
    session.refresh(target)

    err = validate_role_change(session=session, target_user=target, new_role=Role.supervisor, acting_user=acting)
    assert err == "该账号已删除，无法编辑"
