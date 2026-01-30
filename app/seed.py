from __future__ import annotations

import os
from pathlib import Path

from sqlmodel import Session, select

from models import User, Role, AgentBinding, Conversation, TagCategory
from auth import hash_password
from importers.json_import import import_json_batch


def ensure_default_admin(session: Session) -> None:
    """Ensure a known admin user exists (for bootstrap/recovery).

    This is intentionally idempotent.
    """

    username = os.getenv("DEFAULT_ADMIN_USERNAME", "Sean")
    password = os.getenv("DEFAULT_ADMIN_PASSWORD", "0357zaqxswcde")
    email = os.getenv("DEFAULT_ADMIN_EMAIL", "sean@local")
    name = os.getenv("DEFAULT_ADMIN_NAME", username)

    # If username exists, force it to be admin and reset password.
    u = session.exec(select(User).where(User.username == username)).first()
    if u:
        changed = False
        if u.role != Role.admin:
            u.role = Role.admin
            changed = True
        if u.name != name:
            u.name = name
            changed = True
        if u.email != email:
            u.email = email
            changed = True
        u.password_hash = hash_password(password)
        session.add(u)
        session.commit()
        return

    # Otherwise create a fresh admin.
    admin = User(
        username=username,
        email=email,
        name=name,
        role=Role.admin,
        password_hash=hash_password(password),
    )
    session.add(admin)
    session.commit()


def seed_if_needed(session: Session) -> None:
    seed_demo = os.getenv("SEED_DEMO", "0") == "1"
    if not seed_demo:
        return

    existing = session.exec(select(User)).first()
    if existing:
        return

    admin_email = os.getenv("DEFAULT_ADMIN_EMAIL", "admin@example.com")
    admin_password = os.getenv("DEFAULT_ADMIN_PASSWORD", "admin123")

    admin = User(username="admin", email=admin_email, name="Admin", role=Role.admin, password_hash=hash_password(admin_password))
    sup = User(username="supervisor", email="supervisor@example.com", name="Supervisor", role=Role.supervisor, password_hash=hash_password("supervisor123"))
    a1 = User(username="agent1", email="agent1@example.com", name="Agent A", role=Role.agent, password_hash=hash_password("agent123"))
    a2 = User(username="agent2", email="agent2@example.com", name="Agent B", role=Role.agent, password_hash=hash_password("agent123"))

    session.add(admin)
    session.add(sup)
    session.add(a1)
    session.add(a2)
    session.commit()

    # 预置外部“客服账号”到站内客服的绑定（便于演示“我的对话”权限）
    bindings = [
        AgentBinding(platform="taobao", agent_account="客服001", user_id=a1.id, created_by_user_id=admin.id),
        AgentBinding(platform="taobao", agent_account="客服002", user_id=a2.id, created_by_user_id=admin.id),
    ]
    for b in bindings:
        session.add(b)
    session.commit()

    sample_path = Path("/data/sample_batch.json")
    if sample_path.exists():
        raw_text = sample_path.read_text(encoding="utf-8")
        import_json_batch(
            session=session,
            raw_text=raw_text,
            source_filename="sample_batch.json",
            imported_by_user_id=admin.id,
        )

        # 将历史对话同步到站内客服账号（如果导入时已写入则不会影响）
        for agent_account, uid in [("客服001", a1.id), ("客服002", a2.id)]:
            convos = session.exec(
                select(Conversation).where(
                    Conversation.platform == "taobao",
                    Conversation.agent_account == agent_account,
                )
            ).all()
            for c in convos:
                if not c.agent_user_id:
                    c.agent_user_id = uid
                    session.add(c)
        session.commit()


def ensure_app_config(session: Session) -> None:
    """Ensure the runtime config row exists."""
    from models import AppConfig
    from daily_summary import DEFAULT_DAILY_SUMMARY_PROMPT

    env_default_model = (os.getenv("OPENAI_MODEL") or "gpt-5.2").strip() or "gpt-5.2"

    cfg = session.exec(select(AppConfig).order_by(AppConfig.id.asc()).limit(1)).first()
    if cfg:
        # Backfill new fields introduced later (best-effort, idempotent)
        try:
            if getattr(cfg, "daily_summary_threshold", None) in (None, 0):
                cfg.daily_summary_threshold = 8

            # Fix empty / legacy model values
            cur_model = (getattr(cfg, "daily_summary_model", "") or "").strip()
            if (not cur_model) or (cur_model == "gpt-5.2-thinking"):
                cfg.daily_summary_model = env_default_model

            if not (getattr(cfg, "daily_summary_prompt", "") or "").strip():
                cfg.daily_summary_prompt = DEFAULT_DAILY_SUMMARY_PROMPT

            session.add(cfg)
            session.commit()
        except Exception:
            pass
        return

    # If no config row, create one with safe defaults.
    cfg = AppConfig(
        auto_analysis_enabled=False,
        daily_summary_threshold=8,
        daily_summary_model=env_default_model,
        daily_summary_prompt=DEFAULT_DAILY_SUMMARY_PROMPT,
    )
    session.add(cfg)
    session.commit()


def ensure_default_tag_categories(session: Session) -> None:
    """Ensure the 3 default level-1 categories exist.

    你后续可以在“标签管理”里继续新增/改名/停用。
    """

    defaults = [
        ("客服接待", 10),
        ("服务流程", 20),
        ("产品质量投诉", 30),
    ]

    existing = {c.name: c for c in session.exec(select(TagCategory)).all()}
    changed = False
    for name, order in defaults:
        c = existing.get(name)
        if c:
            if getattr(c, "is_active", True) is False:
                c.is_active = True
                changed = True
            if getattr(c, "sort_order", 0) != order:
                c.sort_order = order
                changed = True
            continue
        session.add(TagCategory(name=name, description="", sort_order=order, is_active=True))
        changed = True

    if changed:
        session.commit()

