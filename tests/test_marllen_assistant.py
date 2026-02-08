import asyncio
from datetime import datetime, timedelta

from sqlmodel import Session

from models import (
    User,
    Role,
    AssistantMessage,
    AssistantJob,
    AssistantThread,
    Conversation,
    Message,
)
import marllen_assistant as ma


def _mk_user(session: Session) -> User:
    u = User(
        username="admin1",
        email="admin1@example.com",
        name="Admin",
        role=Role.admin,
        password_hash="x",
        is_active=True,
    )
    session.add(u)
    session.commit()
    session.refresh(u)
    return u


def test_fmt_dt_iso_returns_utc_timezone():
    # `main._fmt_dt_iso` must include timezone info; otherwise browsers parse it as local time,
    # causing elapsed time to be off by the user's UTC offset (e.g. +480 minutes in Shanghai).
    from datetime import timezone
    import main

    naive_utc = datetime(2026, 2, 7, 0, 0, 0)
    s1 = main._fmt_dt_iso(naive_utc)
    assert s1 is not None
    assert s1.endswith("Z") or s1.endswith("+00:00")

    aware_utc = datetime(2026, 2, 7, 0, 0, 0, tzinfo=timezone.utc)
    s2 = main._fmt_dt_iso(aware_utc)
    assert s2 is not None
    assert s2.endswith("Z") or s2.endswith("+00:00")


def test_assistant_thread_create_and_welcome(session: Session):
    u = _mk_user(session)
    t = ma.get_or_create_active_thread(session, owner_user_id=int(u.id))
    assert t.id is not None

    msgs = ma.list_thread_messages(session, thread_id=int(t.id), limit=10)
    assert msgs, "should seed a welcome message"
    assert msgs[0].role in ("assistant", "system")


def test_assistant_job_claim_and_mark_done(session: Session):
    u = _mk_user(session)
    t = ma.get_or_create_active_thread(session, owner_user_id=int(u.id))

    user_msg = AssistantMessage(thread_id=int(t.id), role="user", content="hi")
    session.add(user_msg)
    session.commit()
    session.refresh(user_msg)

    job = AssistantJob(
        thread_id=int(t.id),
        created_by_user_id=int(u.id),
        status="pending",  # type: ignore
        user_message_id=int(user_msg.id),
    )
    session.add(job)
    session.commit()
    session.refresh(job)

    claimed = ma.claim_one_assistant_job(session)
    assert claimed is not None
    st = getattr(claimed.status, "value", None) or str(claimed.status)
    assert str(st) == "running"

    assistant_msg = AssistantMessage(thread_id=int(t.id), role="assistant", content="ok")
    session.add(assistant_msg)
    session.commit()
    session.refresh(assistant_msg)

    ma.mark_assistant_job_done(session, claimed, assistant_message_id=int(assistant_msg.id))
    job2 = session.get(AssistantJob, int(claimed.id))
    assert job2 is not None
    st2 = getattr(job2.status, "value", None) or str(job2.status)
    assert str(st2) == "done"
    assert int(job2.assistant_message_id or 0) == int(assistant_msg.id)


def test_generate_assistant_reply_creates_message(session: Session, monkeypatch):
    u = _mk_user(session)
    t = ma.get_or_create_active_thread(session, owner_user_id=int(u.id))

    user_msg = AssistantMessage(thread_id=int(t.id), role="user", content="系统怎么导入数据？")
    session.add(user_msg)
    session.commit()
    session.refresh(user_msg)

    job = AssistantJob(
        thread_id=int(t.id),
        created_by_user_id=int(u.id),
        status="running",  # type: ignore
        user_message_id=int(user_msg.id),
    )
    session.add(job)
    session.commit()
    session.refresh(job)

    class _FakeAIError(RuntimeError):
        pass

    async def _fake_chat_completion(*args, **kwargs):
        return "OK"

    monkeypatch.setattr(ma, "_get_ai_client", lambda: (_FakeAIError, _fake_chat_completion))

    mid = asyncio.run(ma.generate_assistant_reply(session, job=job))
    assert mid
    m = session.get(AssistantMessage, int(mid))
    assert m is not None
    assert m.role == "assistant"
    assert (m.content or "").strip() == "OK"


def test_generate_assistant_reply_calls_ai_for_best_worst_question(session: Session, monkeypatch):
    u = _mk_user(session)
    t = ma.get_or_create_active_thread(session, owner_user_id=int(u.id))
    user_msg = AssistantMessage(
        thread_id=int(t.id),
        role="user",
        content="给我一份数据快照（概览）顺便告诉我最好的客服是谁？最差的客服是谁？为什么？",
    )
    session.add(user_msg)
    session.commit()
    session.refresh(user_msg)

    job = AssistantJob(
        thread_id=int(t.id),
        created_by_user_id=int(u.id),
        status="running",  # type: ignore
        user_message_id=int(user_msg.id),
    )
    session.add(job)
    session.commit()
    session.refresh(job)

    called = {"n": 0}

    class _FakeAIError(RuntimeError):
        pass

    async def _fake_chat_completion(*args, **kwargs):
        called["n"] += 1
        return "AI_OK"

    monkeypatch.setattr(ma, "_get_ai_client", lambda: (_FakeAIError, _fake_chat_completion))

    mid = asyncio.run(ma.generate_assistant_reply(session, job=job))
    assert mid
    m = session.get(AssistantMessage, int(mid))
    assert m is not None
    assert m.role == "assistant"
    assert (m.content or "").strip() == "AI_OK"
    assert int(called["n"] or 0) == 1


def test_generate_assistant_reply_does_not_attach_charts_without_chart_block(session: Session, monkeypatch):
    u = _mk_user(session)
    t = ma.get_or_create_active_thread(session, owner_user_id=int(u.id))
    user_msg = AssistantMessage(
        thread_id=int(t.id),
        role="user",
        content="给我一份数据快照（概览），顺便看一下最近7天环比。",
    )
    session.add(user_msg)
    session.commit()
    session.refresh(user_msg)

    job = AssistantJob(
        thread_id=int(t.id),
        created_by_user_id=int(u.id),
        status="running",  # type: ignore
        user_message_id=int(user_msg.id),
    )
    session.add(job)
    session.commit()
    session.refresh(job)

    class _FakeAIError(RuntimeError):
        pass

    async def _fake_chat_completion(*args, **kwargs):
        return "OK（不带 chart block）"

    monkeypatch.setattr(ma, "_get_ai_client", lambda: (_FakeAIError, _fake_chat_completion))

    mid = asyncio.run(ma.generate_assistant_reply(session, job=job))
    m = session.get(AssistantMessage, int(mid))
    assert m is not None
    charts = ((m.meta or {}).get("charts") or [])
    assert isinstance(charts, list)
    assert len(charts) == 0


def test_generate_assistant_reply_extracts_charts_from_chart_block(session: Session, monkeypatch):
    u = _mk_user(session)
    t = ma.get_or_create_active_thread(session, owner_user_id=int(u.id))
    user_msg = AssistantMessage(
        thread_id=int(t.id),
        role="user",
        content="给我一份数据快照（概览），顺便看一下最近7天环比。",
    )
    session.add(user_msg)
    session.commit()
    session.refresh(user_msg)

    job = AssistantJob(
        thread_id=int(t.id),
        created_by_user_id=int(u.id),
        status="running",  # type: ignore
        user_message_id=int(user_msg.id),
    )
    session.add(job)
    session.commit()
    session.refresh(job)

    class _FakeAIError(RuntimeError):
        pass

    async def _fake_chat_completion(*args, **kwargs):
        return (
            "OK（带 chart block）\n"
            "```chart\n"
            "[{\"type\":\"bar\",\"title\":\"示例\",\"labels\":[\"A\",\"B\"],\"values\":[1,2],\"unit\":\"次\"}]\n"
            "```"
        )

    monkeypatch.setattr(ma, "_get_ai_client", lambda: (_FakeAIError, _fake_chat_completion))

    mid = asyncio.run(ma.generate_assistant_reply(session, job=job))
    m = session.get(AssistantMessage, int(mid))
    assert m is not None
    assert "```chart" not in (m.content or "")

    charts = ((m.meta or {}).get("charts") or [])
    assert isinstance(charts, list)
    assert len(charts) == 1
    c0 = charts[0]
    assert isinstance(c0, dict)
    assert c0.get("type") == "bar"
    assert c0.get("labels") == ["A", "B"]
    assert c0.get("values") == [1, 2]

def test_generate_assistant_reply_calls_llm_for_platform_earliest_question(session: Session, monkeypatch):
    u = _mk_user(session)
    t = ma.get_or_create_active_thread(session, owner_user_id=int(u.id))

    user_msg = AssistantMessage(
        thread_id=int(t.id),
        role="user",
        content="抖音最早的聊天对话是什么时候开始的？淘宝的呢？你帮我直接搜索数据库",
    )
    session.add(user_msg)
    session.commit()
    session.refresh(user_msg)

    job = AssistantJob(
        thread_id=int(t.id),
        created_by_user_id=int(u.id),
        status="running",  # type: ignore
        user_message_id=int(user_msg.id),
    )
    session.add(job)
    session.commit()
    session.refresh(job)

    called = {"n": 0}

    class _FakeAIError(RuntimeError):
        pass

    async def _fake_chat_completion(*args, **kwargs):
        called["n"] += 1
        return "AI_OK"

    monkeypatch.setattr(ma, "_get_ai_client", lambda: (_FakeAIError, _fake_chat_completion))

    mid = asyncio.run(ma.generate_assistant_reply(session, job=job))
    assert mid
    m = session.get(AssistantMessage, int(mid))
    assert m is not None
    assert m.role == "assistant"
    assert (m.content or "").strip() == "AI_OK"
    assert int(called["n"] or 0) == 1


def test_generate_assistant_reply_routes_to_codex_when_configured(session: Session, monkeypatch):
    u = _mk_user(session)
    t = ma.get_or_create_active_thread(session, owner_user_id=int(u.id))

    user_msg = AssistantMessage(thread_id=int(t.id), role="user", content="帮我看看这个系统里 Marllen小助手 的实现在哪个文件？")
    session.add(user_msg)
    session.commit()
    session.refresh(user_msg)

    job = AssistantJob(
        thread_id=int(t.id),
        created_by_user_id=int(u.id),
        status="running",  # type: ignore
        user_message_id=int(user_msg.id),
    )
    session.add(job)
    session.commit()
    session.refresh(job)

    # Force codex upstream
    monkeypatch.setattr(ma, "ASSISTANT_UPSTREAM", "codex")

    called = {"ai": 0, "codex": 0}

    class _FakeAIError(RuntimeError):
        pass

    async def _fake_chat_completion(*args, **kwargs):
        called["ai"] += 1
        return "AI_SHOULD_NOT_BE_CALLED"

    async def _fake_codex(*, session, job, messages):
        called["codex"] += 1
        return ("CODEX_OK", {"source": "codex", "format": "md", "codex": {"task_id": "t1"}})

    monkeypatch.setattr(ma, "_get_ai_client", lambda: (_FakeAIError, _fake_chat_completion))
    monkeypatch.setattr(ma, "_assistant_generate_reply_via_codex", _fake_codex)

    mid = asyncio.run(ma.generate_assistant_reply(session, job=job))
    m = session.get(AssistantMessage, int(mid))
    assert m is not None
    assert (m.content or "").strip() == "CODEX_OK"
    assert (m.meta or {}).get("source") == "codex"
    assert int(called["codex"] or 0) == 1
    assert int(called["ai"] or 0) == 0


def test_build_codex_system_prompt_minimal_exact(session: Session):
    u = User(
        username="sean",
        email="sean@example.com",
        name="Sean",
        role=Role.admin,
        password_hash="x",
        is_active=True,
    )
    session.add(u)
    session.commit()
    session.refresh(u)

    prompt = ma.build_codex_system_prompt(user=u)
    assert prompt == (
        "你是 Marllen小助手（客服质检项目工作区内的 AI 助手）。\n"
        "- 不要修改代码/配置/数据库，不要执行任何 destructive 操作。\n"
        "-不要在回答里泄露敏感信息（密码、密钥、cookie、环境变量原文等）\n"
        "- 有数据的话请调用图表生成skills融入答案里\n"
        "【系统提示】\n"
        "当前用户：id=1，姓名=Sean，角色=admin"
    )

    # For human inspection (run with `pytest -s`)
    print("\n=== CODEX_SYSTEM_PROMPT ===\n" + prompt + "\n=== /CODEX_SYSTEM_PROMPT ===\n")


def test_assistant_build_codex_prompt_system_plus_last_user_question():
    system = (
        "你是 Marllen小助手（客服质检项目工作区内的 AI 助手）。\n"
        "- 不要修改代码/配置/数据库，不要执行任何 destructive 操作。\n"
        "-不要在回答里泄露敏感信息（密码、密钥、cookie、环境变量原文等）\n"
        "- 有数据的话请调用图表生成skills融入答案里\n"
        "【系统提示】\n"
        "当前用户：id=1，姓名=Sean，角色=admin"
    )
    msgs = [
        {"role": "system", "content": system},
        {"role": "assistant", "content": "你好，我是 Marllen小助手。"},
        {"role": "user", "content": "第一句问题"},
        {"role": "assistant", "content": "一些回答"},
        {"role": "user", "content": "最近客服表现如何？"},
    ]
    prompt = ma._assistant_build_codex_prompt(msgs)
    assert prompt == (
        system
        + "\n\n【对话上下文】\n"
        + "ASSISTANT: 你好，我是 Marllen小助手。\n\n"
        + "USER: 第一句问题\n\n"
        + "ASSISTANT: 一些回答\n\n"
        + "USER: 最近客服表现如何？"
    )
    # For human inspection (run with `pytest -s`)
    print("\n=== CODEX_PROMPT ===\n" + prompt + "\n=== /CODEX_PROMPT ===\n")


def test_system_prompt_does_not_include_snapshot_wording(session: Session):
    u = _mk_user(session)
    prompt = ma.build_system_prompt(session, user=u, question_hint="hi")
    assert "数据快照" not in (prompt or "")


def test_compact_thread_title(session: Session):
    assert ma.compact_thread_title("  第一行   \n第二行  ") == "第一行"
    out = ma.compact_thread_title("A" * 40, max_len=10)
    assert out.startswith("A" * 10)
    assert out.endswith("…")


def test_maybe_auto_title_thread_sets_title_and_meta(session: Session):
    u = _mk_user(session)
    t = AssistantThread(owner_user_id=int(u.id), title="Marllen小助手", is_archived=False, meta={})
    session.add(t)
    session.commit()
    session.refresh(t)

    ok = ma.maybe_auto_title_thread(t, "如何导入数据？（带 CID=123）")
    assert ok is True
    assert "如何导入数据" in (t.title or "")
    assert (t.meta or {}).get("auto_title") == 1

    # custom title should not be overridden
    t2 = AssistantThread(owner_user_id=int(u.id), title="自定义标题", is_archived=False, meta={})
    session.add(t2)
    session.commit()
    session.refresh(t2)
    ok2 = ma.maybe_auto_title_thread(t2, "新的问题")
    assert ok2 is False
    assert t2.title == "自定义标题"


def test_activate_thread_archives_other_active(session: Session):
    u = _mk_user(session)
    t1 = AssistantThread(owner_user_id=int(u.id), title="t1", is_archived=False, meta={})
    t2 = AssistantThread(owner_user_id=int(u.id), title="t2", is_archived=False, meta={})
    session.add(t1)
    session.add(t2)
    session.commit()
    session.refresh(t1)
    session.refresh(t2)

    ma.activate_thread(session, owner_user_id=int(u.id), thread=t1)
    session.commit()

    t1r = session.get(AssistantThread, int(t1.id))
    t2r = session.get(AssistantThread, int(t2.id))
    assert t1r is not None and t2r is not None
    assert bool(t1r.is_archived) is False
    assert bool(t2r.is_archived) is True


def test_list_threads_orders_active_first_then_recent(session: Session):
    u = _mk_user(session)
    now = datetime.utcnow()
    t_old = AssistantThread(
        owner_user_id=int(u.id),
        title="old",
        is_archived=True,
        updated_at=now - timedelta(days=3),
        meta={},
    )
    t_new = AssistantThread(
        owner_user_id=int(u.id),
        title="new",
        is_archived=True,
        updated_at=now - timedelta(days=1),
        meta={},
    )
    t_active = AssistantThread(
        owner_user_id=int(u.id),
        title="active",
        is_archived=False,
        updated_at=now - timedelta(days=10),
        meta={},
    )
    session.add(t_old)
    session.add(t_new)
    session.add(t_active)
    session.commit()

    rows = ma.list_threads(session, owner_user_id=int(u.id), limit=10)
    assert [r.title for r in rows[:3]] == ["active", "new", "old"]
