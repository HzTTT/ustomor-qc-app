from __future__ import annotations

import asyncio
import os
import json
import re
import time
import shlex
from datetime import datetime, timedelta
from typing import Any

from sqlmodel import Session, select

try:  # pragma: no cover
    from models import (  # type: ignore
        AIAnalysisJob,
        AssistantJob,
        AssistantMessage,
        AssistantThread,
        AgentBinding,
        DailyAISummaryJob,
        Conversation,
        ConversationAnalysis,
        ConversationTagHit,
        ImportRun,
        Message,
        Role,
        TagCategory,
        TagDefinition,
        TrainingTask,
        User,
    )
except Exception:  # pragma: no cover
    from app.models import (  # type: ignore
        AIAnalysisJob,
        AssistantJob,
        AssistantMessage,
        AssistantThread,
        AgentBinding,
        DailyAISummaryJob,
        Conversation,
        ConversationAnalysis,
        ConversationTagHit,
        ImportRun,
        Message,
        Role,
        TagCategory,
        TagDefinition,
        TrainingTask,
        User,
    )


ASSISTANT_NAME = "Marllen小助手"
ASSISTANT_MODEL = (
    (os.getenv("MARLLEN_ASSISTANT_MODEL") or "").strip()
    or (os.getenv("OPENAI_MODEL") or "").strip()
    or "gpt-5.2"
)
ASSISTANT_REASONING_EFFORT = (os.getenv("MARLLEN_ASSISTANT_REASONING_EFFORT", "low") or "low").strip().lower()
ASSISTANT_SAVE_TRACE = (os.getenv("MARLLEN_ASSISTANT_SAVE_TRACE", "0") or "0").strip().lower() in ("1", "true", "yes", "y", "on")
ASSISTANT_UPSTREAM = (os.getenv("MARLLEN_ASSISTANT_UPSTREAM", "openai") or "openai").strip().lower()
ASSISTANT_CODEX_CWD = (os.getenv("MARLLEN_ASSISTANT_CODEX_CWD", "/Users/marllenos/customor-qc-app") or "").strip()
ASSISTANT_CODEX_MODEL = (os.getenv("MARLLEN_ASSISTANT_CODEX_MODEL", "") or "").strip()
ASSISTANT_CODEX_SANDBOX = (os.getenv("MARLLEN_ASSISTANT_CODEX_SANDBOX", "workspace-write") or "workspace-write").strip().lower()
ASSISTANT_CODEX_APPROVAL = (os.getenv("MARLLEN_ASSISTANT_CODEX_APPROVAL", "never") or "never").strip().lower()
ASSISTANT_CODEX_OSS = (os.getenv("MARLLEN_ASSISTANT_CODEX_OSS", "0") or "0").strip().lower() in ("1", "true", "yes", "y", "on")
ASSISTANT_CODEX_LOCAL_PROVIDER = (os.getenv("MARLLEN_ASSISTANT_CODEX_LOCAL_PROVIDER", "") or "").strip().lower()

_ASSISTANT_TIMEOUT_S = 180
try:
    _ASSISTANT_TIMEOUT_S = int(os.getenv("MARLLEN_ASSISTANT_TIMEOUT", "180") or "180")
except Exception:
    _ASSISTANT_TIMEOUT_S = 180
ASSISTANT_TIMEOUT_S = max(30, min(900, _ASSISTANT_TIMEOUT_S))

_assistant_stale_default = max(180, ASSISTANT_TIMEOUT_S + 90)
try:
    ASSISTANT_JOB_STALE_SECONDS = int(
        (os.getenv("ASSISTANT_JOB_STALE_SECONDS") or os.getenv("MARLLEN_ASSISTANT_STALE_SECONDS") or str(_assistant_stale_default)).strip()
    )
except Exception:
    ASSISTANT_JOB_STALE_SECONDS = _assistant_stale_default
ASSISTANT_JOB_STALE_SECONDS = max(60, min(7200, int(ASSISTANT_JOB_STALE_SECONDS)))


CHART_BLOCK_RE = re.compile(r"```chart\s*(.*?)```", re.DOTALL | re.IGNORECASE)

def _role_str(role: Any) -> str:
    try:
        v = getattr(role, "value", None)
        if v is not None:
            return str(v).strip()
    except Exception:
        pass
    return str(role or "").strip()


def _user_role(user: User) -> str:
    return _role_str(getattr(user, "role", None))


def _user_is_agent(user: User) -> bool:
    return _user_role(user) == "agent"


def _user_can_manage(user: User) -> bool:
    return _user_role(user) in ("admin", "supervisor")


def _job_status_str(job: AssistantJob | None) -> str:
    if not job:
        return ""
    return str(getattr(getattr(job, "status", None), "value", None) or getattr(job, "status", None) or "").strip().lower()


def _job_is_canceled(job: AssistantJob | None) -> bool:
    if not job:
        return False
    try:
        extra = job.extra or {}
        if isinstance(extra, dict) and extra.get("canceled") in (1, "1", True, "true", "TRUE"):
            return True
    except Exception:
        pass

    err = (getattr(job, "last_error", None) or "").strip().lower()
    if err.startswith("canceled"):
        return True
    if "canceled by user" in err:
        return True
    return False


def _assistant_messages_to_instructions_and_input(messages: list[dict[str, str]]) -> tuple[str, str]:
    instructions_parts: list[str] = []
    transcript_parts: list[str] = []

    for m in (messages or []):
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "").strip().lower() or "user"
        content = m.get("content")
        if not isinstance(content, str):
            content = "" if content is None else str(content)

        if role == "system":
            if content.strip():
                instructions_parts.append(content.strip())
            continue

        label = "USER"
        if role == "assistant":
            label = "ASSISTANT"
        elif role == "developer":
            label = "DEVELOPER"
        elif role == "tool":
            label = "TOOL"
        transcript_parts.append(f"{label}: {content}".rstrip())

    return ("\n\n".join(instructions_parts).strip(), "\n\n".join(transcript_parts).strip())


def _assistant_build_codex_prompt(messages: list[dict[str, str]]) -> str:
    """Turn assistant messages into a single prompt for `codex exec`.

    IMPORTANT: Keep it minimal on purpose.
    We pass:
    1) the system instructions (1 block)
    2) the conversation transcript (last N messages)
    No internal context modules.
    """
    msgs = list(messages or [])

    def _coerce_content(v: Any) -> str:
        if isinstance(v, str):
            return v
        if v is None:
            return ""
        return str(v)

    system_prompt = ""
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = (_coerce_content(m.get("role")) or "").strip().lower()
        if role != "system":
            continue
        content = _coerce_content(m.get("content")).strip()
        if content:
            system_prompt = content
            break

    transcript_parts: list[str] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = (_coerce_content(m.get("role")) or "").strip().lower() or "user"
        if role == "system":
            continue
        content = _coerce_content(m.get("content")).strip()
        if not content:
            continue

        label = "USER"
        if role == "assistant":
            label = "ASSISTANT"
        elif role == "developer":
            label = "DEVELOPER"
        elif role == "tool":
            label = "TOOL"
        transcript_parts.append(f"{label}: {content}".rstrip())

    # Keep only the last N messages to control token usage.
    max_transcript = 18
    if len(transcript_parts) > max_transcript:
        transcript_parts = transcript_parts[-max_transcript:]

    out = (system_prompt or "").strip()
    if transcript_parts:
        if out:
            out += "\n\n"
        out += "【对话上下文】\n" + ("\n\n".join(transcript_parts).strip())
    return out.strip()


async def _assistant_generate_reply_via_codex(
    *,
    session: Session,
    job: AssistantJob,
    messages: list[dict[str, str]],
) -> tuple[str, dict[str, Any]]:
    """Generate assistant reply by calling local `codex exec` via clawd-relay."""
    from clawd_relay_client import exec_via_clawd_relay, ClawdRelayError  # type: ignore

    job_id = int(getattr(job, "id", 0) or 0)
    attempt = int(getattr(job, "attempts", 0) or 0)

    prompt = _assistant_build_codex_prompt(messages)
    codex_cwd = ASSISTANT_CODEX_CWD or "/Users/marllenos/customor-qc-app"
    sandbox = ASSISTANT_CODEX_SANDBOX if ASSISTANT_CODEX_SANDBOX in ("read-only", "workspace-write", "danger-full-access") else "workspace-write"
    approval = ASSISTANT_CODEX_APPROVAL if ASSISTANT_CODEX_APPROVAL in ("untrusted", "on-failure", "on-request", "never") else "never"
    out_path = f"/tmp/marllen_assistant_codex_job_{job_id}_attempt_{attempt}.md"
    log_path = f"/tmp/marllen_assistant_codex_job_{job_id}_attempt_{attempt}.log"

    argv: list[str] = [
        "codex",
        "-a",
        approval,
        "exec",
        "--skip-git-repo-check",
        "--color",
        "never",
    ]
    if ASSISTANT_CODEX_OSS:
        argv.append("--oss")
        if ASSISTANT_CODEX_LOCAL_PROVIDER in ("ollama", "lmstudio"):
            argv += ["--local-provider", ASSISTANT_CODEX_LOCAL_PROVIDER]
    argv += [
        "-s",
        sandbox,
        "-C",
        codex_cwd,
        "--output-last-message",
        out_path,
    ]
    if ASSISTANT_CODEX_MODEL:
        argv += ["-m", ASSISTANT_CODEX_MODEL]
    argv.append(prompt)

    cmd = " ".join(shlex.quote(a) for a in argv)
    # Keep stdout clean: write Codex logs to log_path, only cat the last message.
    bash_cmd = (
        f"{cmd} > /dev/null 2> {shlex.quote(log_path)}; "
        "rc=$?; "
        f"if [ $rc -ne 0 ]; then echo \"CODEX_EXIT_CODE=$rc\"; cat {shlex.quote(log_path)}; exit $rc; fi; "
        f"cat {shlex.quote(out_path)}; "
        f"rm -f {shlex.quote(out_path)} {shlex.quote(log_path)}"
    )

    if ASSISTANT_SAVE_TRACE:
        try:
            trace = {
                "ts": datetime.utcnow().isoformat(),
                "runner": "clawd-relay",
                "codex_cwd": codex_cwd,
                "sandbox": sandbox,
                "model": ASSISTANT_CODEX_MODEL or None,
                "oss": bool(ASSISTANT_CODEX_OSS),
                "local_provider": ASSISTANT_CODEX_LOCAL_PROVIDER or None,
                "prompt_chars": len(prompt),
                "prompt_preview": (prompt[:20000] + "…") if len(prompt) > 20000 else prompt,
                "argv": argv[:-1] + ["<PROMPT>"],
                "bash_cmd_preview": (bash_cmd[:2000] + "…") if len(bash_cmd) > 2000 else bash_cmd,
            }
            job.extra = {**(job.extra or {}), "trace_codex": trace}
            session.add(job)
            session.commit()
        except Exception:
            try:
                session.rollback()
            except Exception:
                pass

    t0 = time.time()
    try:
        res = await exec_via_clawd_relay(
            command=f"bash -lc {shlex.quote(bash_cmd)}",
            cwd=codex_cwd,
            timeout_s=max(30.0, float(ASSISTANT_TIMEOUT_S) + 120.0),
            timeout_ms=int(max(30, ASSISTANT_TIMEOUT_S + 60) * 1000),
            include_output=True,
            max_output_bytes=512 * 1024,
        )
    except ClawdRelayError:
        raise
    except Exception as e:
        raise ClawdRelayError(f"codex via clawd-relay 调用失败：{type(e).__name__}: {str(e)[:200]}") from e
    t1 = time.time()

    stdout = str(res.get("stdout") or "")
    text = stdout.strip()
    meta = {
        "source": "codex",
        "format": "md",
        "codex": {
            "cwd": codex_cwd,
            "sandbox": sandbox,
            "model": ASSISTANT_CODEX_MODEL or None,
            "duration_ms": res.get("durationMs"),
            "task_id": res.get("taskId") or res.get("requestId"),
            "elapsed_s": round(float(t1 - t0), 3),
        },
    }
    return text, meta


def compact_thread_title(text: str, *, max_len: int = 28) -> str:
    s = (text or "").strip()
    if not s:
        return ASSISTANT_NAME
    # Prefer the first line; keep it compact for list UI.
    s = (s.splitlines()[0] or "").strip()
    s = re.sub(r"\s+", " ", s)
    if len(s) > max_len:
        s = s[:max_len].rstrip() + "…"
    return s or ASSISTANT_NAME


def maybe_auto_title_thread(thread: AssistantThread, user_text: str) -> bool:
    cur = (getattr(thread, "title", None) or "").strip()
    if cur and cur not in (ASSISTANT_NAME, "Marllen小助手"):
        return False
    new_title = compact_thread_title(user_text)
    if not new_title or new_title in (ASSISTANT_NAME,):
        return False
    thread.title = new_title
    try:
        thread.meta = thread.meta or {}
        thread.meta["auto_title"] = 1
    except Exception:
        pass
    return True


def get_or_create_active_thread(session: Session, *, owner_user_id: int) -> AssistantThread:
    t = session.exec(
        select(AssistantThread)
        .where(AssistantThread.owner_user_id == owner_user_id, AssistantThread.is_archived == False)  # noqa: E712
        .order_by(AssistantThread.updated_at.desc(), AssistantThread.id.desc())
        .limit(1)
    ).first()
    if t:
        return t

    t = AssistantThread(
        owner_user_id=owner_user_id,
        title=ASSISTANT_NAME,
        is_archived=False,
        updated_at=datetime.utcnow(),
        meta={},
    )
    session.add(t)
    session.commit()
    session.refresh(t)

    # Seed a welcome message for better UX (no AI call needed).
    welcome = AssistantMessage(
        thread_id=int(t.id or 0),
        role="assistant",
        content=(
            f"你好，我是 {ASSISTANT_NAME}。\n"
            "\n"
            "你可以直接问我：\n"
            "\n"
            "- 最近一个月洗后缩水的客户投诉情况如何?\n"
            "- 最近每周各个平台的客户接待量是怎样的?\n"
            "- 这个系统有哪些功能?\n"
        ),
        meta={"seed": "welcome"},
    )
    session.add(welcome)
    session.commit()
    return t


def list_threads(session: Session, *, owner_user_id: int, limit: int = 50) -> list[AssistantThread]:
    try:
        limit = int(limit)
    except Exception:
        limit = 50
    limit = max(1, min(200, limit))

    rows = session.exec(
        select(AssistantThread)
        .where(AssistantThread.owner_user_id == owner_user_id)
        .order_by(AssistantThread.is_archived.asc(), AssistantThread.updated_at.desc(), AssistantThread.id.desc())
        .limit(limit)
    ).all()
    return list(rows)


def activate_thread(session: Session, *, owner_user_id: int, thread: AssistantThread) -> None:
    """Make the given thread the only active thread for this user (best-effort)."""
    now = datetime.utcnow()
    thread_id = int(getattr(thread, "id", 0) or 0)
    other_actives = session.exec(
        select(AssistantThread).where(
            AssistantThread.owner_user_id == owner_user_id,
            AssistantThread.is_archived == False,  # noqa: E712
            AssistantThread.id != thread_id,
        )
    ).all()
    for r in other_actives:
        r.is_archived = True
        r.updated_at = now
        session.add(r)

    thread.is_archived = False
    thread.updated_at = now
    session.add(thread)


def archive_and_create_new_thread(session: Session, *, owner_user_id: int) -> AssistantThread:
    # Archive all active threads for this user (keep history in DB).
    rows = session.exec(
        select(AssistantThread).where(
            AssistantThread.owner_user_id == owner_user_id,
            AssistantThread.is_archived == False,  # noqa: E712
        )
    ).all()
    now = datetime.utcnow()
    for r in rows:
        r.is_archived = True
        r.updated_at = now
        session.add(r)
    if rows:
        session.commit()

    return get_or_create_active_thread(session, owner_user_id=owner_user_id)


def list_thread_messages(session: Session, *, thread_id: int, limit: int = 80) -> list[AssistantMessage]:
    try:
        limit = int(limit)
    except Exception:
        limit = 80
    limit = max(10, min(300, limit))

    rows = session.exec(
        select(AssistantMessage)
        .where(AssistantMessage.thread_id == thread_id)
        .order_by(AssistantMessage.id.desc())
        .limit(limit)
    ).all()
    return list(reversed(rows))


def get_pending_job(session: Session, *, thread_id: int) -> AssistantJob | None:
    return session.exec(
        select(AssistantJob)
        .where(
            AssistantJob.thread_id == thread_id,
            AssistantJob.status.in_(["pending", "running"]),
        )
        .order_by(AssistantJob.created_at.desc(), AssistantJob.id.desc())
        .limit(1)
    ).first()


def _extract_charts(text: str) -> tuple[str, list[dict[str, Any]]]:
    raw = (text or "").strip()
    if not raw:
        return raw, []

    charts: list[dict[str, Any]] = []
    blocks = CHART_BLOCK_RE.findall(raw) or []
    for b in blocks[-3:]:
        s = (b or "").strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and isinstance(obj.get("charts"), list):
                for it in obj.get("charts")[:3]:
                    if isinstance(it, dict):
                        charts.append(it)
            elif isinstance(obj, list):
                for it in obj[:3]:
                    if isinstance(it, dict):
                        charts.append(it)
        except Exception:
            continue

    cleaned = CHART_BLOCK_RE.sub("", raw).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, charts[:3]


def build_system_prompt(session: Session, *, user: User, question_hint: str = "") -> str:
    role_val = _user_role(user)
    uid = int(user.id or 0) if getattr(user, "id", None) is not None else 0
    uname = (getattr(user, "name", None) or "").strip() or (getattr(user, "email", None) or "")

    can_manage = _user_can_manage(user)
    perm_hint = (
        "你拥有管理员/主管权限：可以给出改动/配置/操作建议，并可直接指导在本系统里如何完成；但仍然不要在回答里泄露敏感信息（密码、密钥、cookie、环境变量原文等），也不要编造数据。"
        if can_manage
        else "你是坐席账号：处于只读模式——不允许做任何改动（包括建议具体修改系统数据/配置/代码的操作步骤）；只做查询、分析、解释与排查建议。涉及改动时，明确说明需要管理员/主管执行。"
    )

    # Keep prompt compact. Prefer natural conversation; only add essential guardrails.
    return (
        "你在一个“客服质检 + 培训闭环 + 报表 + 多角色管理”的系统里充当内置助理（Marllen小助手）。用户常见诉求：分析近况 、解释报表口径、指导在页面里去哪点/怎么操作、询问系统功能。\n"
        "\n"
        f"当前用户：id={uid}，姓名={uname}，角色={role_val}\n"
        f"权限规则：{perm_hint}\n"
        "\n"
        "回答要求：\n"
        "- 默认简体中文；用 Markdown；结论先行。\n"
        "- 如果有适合用图标增强表达的地方，优先用 skill 生成图标；不要走工程化实现（例如引入图标库、手写复杂 SVG）。\n"
        "- 除了 ` ```chart ` 代码块，不要输出任何 JSON/dict。\n"
        "- 只要回答里引用了系统数据（统计/对比/趋势/分布/Top），就在末尾输出一个 ` ```chart ` 代码块：JSON 数组（最多 2 张图），字段：type(bar/line/pie)、title、labels、values、unit(可选)、note(可选)。不要写任何多余文字。\n"
    ).strip()


def build_codex_system_prompt(*, user: User) -> str:
    role_val = _user_role(user)
    uid = int(user.id or 0) if getattr(user, "id", None) is not None else 0
    uname = (getattr(user, "name", None) or "").strip() or (getattr(user, "email", None) or "")
    return (
        "你是 Marllen小助手（客服质检项目工作区内的 AI 助手）。\n"
        "- 不要修改代码/配置/数据库，不要执行任何 destructive 操作。\n"
        "-不要在回答里泄露敏感信息（密码、密钥、cookie、环境变量原文等）\n"
        "- 有数据的话请调用图表生成skills融入答案里\n"
        "【系统提示】\n"
        f"当前用户：id={uid}，姓名={uname}，角色={role_val}"
    ).strip()


def claim_one_assistant_job(session: Session) -> AssistantJob | None:
    """Best-effort claim: pick one pending job and mark running."""
    # Recovery: mark stale running jobs as error.
    try:
        stale_seconds = int(ASSISTANT_JOB_STALE_SECONDS)
        cutoff = datetime.utcnow() - timedelta(seconds=max(60, stale_seconds))
        stuck = session.exec(
            select(AssistantJob)
            .where(AssistantJob.status == "running")
            .where(AssistantJob.started_at.is_not(None))
            .where(AssistantJob.started_at < cutoff)
        ).all()
        # Some historical rows may incorrectly be "running" but missing started_at/attempts,
        # which would bypass the recovery above and cause the UI to look permanently stuck.
        stuck_no_started = session.exec(
            select(AssistantJob)
            .where(AssistantJob.status == "running")
            .where(AssistantJob.started_at.is_(None))
            .where(AssistantJob.created_at < cutoff)
        ).all()
        if stuck_no_started:
            stuck = list(stuck or []) + list(stuck_no_started or [])
        if stuck:
            for j in stuck:
                j.status = "error"  # type: ignore
                j.finished_at = datetime.utcnow()
                if getattr(j, "started_at", None) is None:
                    j.last_error = "任务异常恢复：运行中但缺少 started_at（可能 worker 重启或旧版本遗留）。已自动终止，请重试。"
                else:
                    mins = max(1, int(round(float(stale_seconds) / 60.0)))
                    j.last_error = f"任务超时自动恢复：已运行超过 {mins} 分钟仍未返回（可能 worker 重启或上游 AI 卡住）。已自动终止，请重试。"
                session.add(j)
            session.commit()
    except Exception:
        pass

    base_stmt = (
        select(AssistantJob)
        .where(AssistantJob.status == "pending")
        .order_by(AssistantJob.created_at.asc(), AssistantJob.id.asc())
        .limit(1)
    )
    try:
        job = session.exec(base_stmt.with_for_update(skip_locked=True)).first()
    except Exception:
        job = session.exec(base_stmt).first()
    if not job:
        return None

    job.status = "running"  # type: ignore
    job.started_at = datetime.utcnow()
    job.attempts = (job.attempts or 0) + 1
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


def mark_assistant_job_done(session: Session, job: AssistantJob, *, assistant_message_id: int) -> None:
    # Do not override an already-finished/canceled job (race: user cancels while worker is running).
    j = None
    try:
        if getattr(job, "id", None) is not None:
            j = session.get(AssistantJob, int(job.id))
    except Exception:
        j = None
    if j is None:
        j = job

    st = _job_status_str(j)
    if st in ("done", "error") or _job_is_canceled(j):
        return

    j.status = "done"  # type: ignore
    j.finished_at = datetime.utcnow()
    j.last_error = ""
    j.assistant_message_id = int(assistant_message_id)
    session.add(j)
    session.commit()


def mark_assistant_job_error(session: Session, job: AssistantJob, err: str) -> None:
    try:
        session.rollback()
    except Exception:
        pass

    j = None
    try:
        if getattr(job, "id", None) is not None:
            j = session.get(AssistantJob, int(job.id))
    except Exception:
        j = None
    if j is None:
        j = job

    # If user already canceled (or job already finished), do not override its final state/message.
    st = _job_status_str(j)
    if st in ("done", "error") and (_job_is_canceled(j) or getattr(j, "finished_at", None) is not None):
        return

    j.status = "error"  # type: ignore
    j.finished_at = datetime.utcnow()
    j.last_error = (err or "")[:2000]
    session.add(j)
    session.commit()


async def generate_assistant_reply(session: Session, *, job: AssistantJob) -> int:
    """Generate assistant reply and persist as AssistantMessage. Returns message id."""
    thread = session.get(AssistantThread, int(job.thread_id))
    if not thread or thread.is_archived:
        raise RuntimeError("thread not found or archived")

    user = session.get(User, int(job.created_by_user_id))
    if not user:
        raise RuntimeError("user not found")

    # If user canceled while job was in flight, stop early.
    fresh = None
    try:
        if getattr(job, "id", None) is not None:
            fresh = session.get(AssistantJob, int(job.id))
    except Exception:
        fresh = None
    if fresh and (_job_is_canceled(fresh) or _job_status_str(fresh) in ("done", "error")):
        raise RuntimeError("job canceled")

    # Build conversation context (last N messages)
    history = session.exec(
        select(AssistantMessage)
        .where(AssistantMessage.thread_id == thread.id)
        .order_by(AssistantMessage.id.asc())
    ).all()
    history = list(history or [])

    # Trim to control token usage
    max_ctx = 24
    if len(history) > max_ctx:
        history = history[-max_ctx:]

    # Use the triggering message to extract CID/task hints for extra context.
    trigger_msg = session.get(AssistantMessage, int(job.user_message_id)) if getattr(job, "user_message_id", None) else None
    question_hint = (trigger_msg.content if trigger_msg else "") or ""

    def _build_messages(system_prompt: str) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for mm in history:
            role = (mm.role or "user").strip().lower()
            if role not in ("user", "assistant", "system"):
                role = "user"
            content = (mm.content or "").strip()
            if not content:
                continue
            msgs.append({"role": role, "content": content})
        return msgs

    # Always generate via local Codex (via clawd-relay). No upstream LLM fallback.
    codex_system_prompt = build_codex_system_prompt(user=user)
    codex_messages = _build_messages(codex_system_prompt)
    text, codex_meta = await _assistant_generate_reply_via_codex(session=session, job=job, messages=codex_messages)
    if not (text or "").strip():
        raise RuntimeError("codex 返回为空")

    # Check cancellation again right before persisting (avoid late answers after user cancels).
    fresh2 = None
    try:
        if getattr(job, "id", None) is not None:
            fresh2 = session.get(AssistantJob, int(job.id))
    except Exception:
        fresh2 = None
    if fresh2 and (_job_is_canceled(fresh2) or _job_status_str(fresh2) in ("done", "error")):
        raise RuntimeError("job canceled")

    # Best-effort performance trace
    try:
        job.extra = {**(job.extra or {}), "codex": (codex_meta or {}).get("codex")}
        session.add(job)
        session.commit()
    except Exception:
        try:
            session.rollback()
        except Exception:
            pass

    cleaned, charts = _extract_charts(text)
    meta: dict[str, Any] = {**(codex_meta or {}), "reasoning_effort": ASSISTANT_REASONING_EFFORT}
    if charts:
        meta["charts"] = charts[:3]

    msg = AssistantMessage(
        thread_id=int(thread.id or 0),
        role="assistant",
        content=cleaned,
        meta=meta,
    )
    session.add(msg)
    thread.updated_at = datetime.utcnow()
    session.add(thread)
    session.commit()
    session.refresh(msg)
    return int(msg.id or 0)
