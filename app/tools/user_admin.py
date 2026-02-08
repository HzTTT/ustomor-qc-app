from __future__ import annotations

from sqlmodel import Session, select

try:
    # Runtime path when running from `app/` (e.g. `uvicorn main:app`)
    from models import Role, User
except ImportError:  # pragma: no cover
    # Compat: allow importing via `app.*` in some environments/tests.
    from app.models import Role, User


def validate_role_change(
    session: Session,
    target_user: User,
    new_role: Role,
    acting_user: User,
) -> str | None:
    """Validate whether an admin is allowed to change target user's role.

    Returns an error message string when not allowed; otherwise None.
    """
    if getattr(target_user, "is_active", True) is False:
        return "该账号已删除，无法编辑"

    # Safety: always keep at least one active admin.
    target_role_val = getattr(target_user.role, "value", None) or str(target_user.role)
    new_role_val = getattr(new_role, "value", None) or str(new_role)

    if target_role_val == Role.admin.value and new_role_val != Role.admin.value:
        other_admin = session.exec(
            select(User).where(
                User.is_active == True,  # noqa: E712
                User.role == Role.admin,
                User.id != target_user.id,
            )
        ).first()
        if not other_admin:
            return "至少需要保留一个管理员账号"

    # UX safety: if demoting self, it will take effect immediately (DB-based auth).
    # Allow it, but the caller should avoid redirecting to admin-only pages.
    return None
