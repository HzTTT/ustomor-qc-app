from __future__ import annotations

import asyncio
import os
import json
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlmodel import Session, select

try:  # pragma: no cover
    from db import engine  # type: ignore
except Exception:  # pragma: no cover
    from app.db import engine  # type: ignore

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

_ASSISTANT_TIMEOUT_S = 1800
try:
    _ASSISTANT_TIMEOUT_S = int(os.getenv("MARLLEN_ASSISTANT_TIMEOUT", "1800") or "1800")
except Exception:
    _ASSISTANT_TIMEOUT_S = 1800
# Product expectation: allow long-running answers, up to 30 minutes.
ASSISTANT_TIMEOUT_S = max(30, min(1800, _ASSISTANT_TIMEOUT_S))

_assistant_stale_default = max(180, ASSISTANT_TIMEOUT_S + 90)
try:
    ASSISTANT_JOB_STALE_SECONDS = int(
        (os.getenv("ASSISTANT_JOB_STALE_SECONDS") or os.getenv("MARLLEN_ASSISTANT_STALE_SECONDS") or str(_assistant_stale_default)).strip()
    )
except Exception:
    ASSISTANT_JOB_STALE_SECONDS = _assistant_stale_default
ASSISTANT_JOB_STALE_SECONDS = max(60, min(7200, int(ASSISTANT_JOB_STALE_SECONDS)))


CHART_BLOCK_RE = re.compile(r"```chart\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def assistant_codex_paths(*, job_id: int, attempt: int) -> tuple[str, str]:
    """Shared file paths for Codex job artifacts (app+worker containers).

    We intentionally write logs into a shared mount (e.g. /workspace) so the web
    app can show real-time progress while the worker is running.
    """
    base = (os.getenv("MARLLEN_ASSISTANT_ARTIFACT_DIR") or "").strip()
    if not base:
        base = "/workspace/tmp/marllen_assistant" if os.path.isdir("/workspace") else "/tmp/marllen_assistant"
    base = base.rstrip("/")
    out_path = f"{base}/codex_job_{int(job_id)}_attempt_{int(attempt)}.md"
    log_path = f"{base}/codex_job_{int(job_id)}_attempt_{int(attempt)}.log"
    return out_path, log_path


def _job_stop_requested(job_id: int) -> bool:
    """Check if the job should stop (user canceled / recovered / externally finished)."""
    if not job_id:
        return False
    try:
        with Session(engine) as s:
            j = s.get(AssistantJob, int(job_id))
            if not j:
                return True
            st = _job_status_str(j)
            if st in ("done", "error"):
                return True
            if _job_is_canceled(j):
                return True
    except Exception:
        # Best-effort: if DB is temporarily unavailable, don't kill the process.
        return False
    return False


def _get_ai_client():
    """Late import to keep marllen_assistant import-light in tests/worker."""
    try:  # pragma: no cover
        from ai_client import AIError, chat_completion  # type: ignore
    except Exception:  # pragma: no cover
        from app.ai_client import AIError, chat_completion  # type: ignore
    return AIError, chat_completion


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


def _utc_iso_z(dt: datetime | None = None) -> str:
    """Serialize as ISO8601 with a trailing Z for UTC.

    We store DB timestamps as naive UTC (datetime.utcnow()) in this project.
    This helper is for JSON meta fields (e.g. heartbeats), not for DB columns.
    """
    d = dt or datetime.utcnow()
    try:
        # Keep it simple: naive UTC + 'Z'
        return d.replace(microsecond=0).isoformat() + "Z"
    except Exception:
        return str(d)


def _parse_utc_iso_maybe(s: Any) -> datetime | None:
    """Parse a best-effort UTC datetime from ISO strings like '...Z'."""
    if not s:
        return None
    try:
        ss = str(s).strip()
        if not ss:
            return None
        if ss.endswith("Z"):
            ss = ss[:-1] + "+00:00"
        d = datetime.fromisoformat(ss)
        # Normalize to naive UTC to match our DB timestamps.
        if getattr(d, "tzinfo", None) is not None:
            d = d.astimezone(timezone.utc).replace(tzinfo=None)
        return d
    except Exception:
        return None


def _job_heartbeat_dt(job: AssistantJob | None) -> datetime | None:
    if not job:
        return None
    try:
        extra = job.extra or {}
        if isinstance(extra, dict):
            return _parse_utc_iso_maybe(extra.get("heartbeat_at"))
    except Exception:
        return None
    return None


def _ensure_assistant_job_error_message(session: Session, job: AssistantJob, err: str) -> None:
    """Persist a visible assistant message for a failed job.

    Why: if the user refreshes or closes the assistant panel, the frontend may no longer
    be polling `/jobs/{id}`. Without a message, it looks like "nothing happened".
    """
    try:
        if getattr(job, "assistant_message_id", None):
            return
    except Exception:
        pass

    tid = int(getattr(job, "thread_id", 0) or 0)
    if not tid:
        return

    short = (err or "").strip()
    if len(short) > 240:
        short = short[:240].rstrip() + "…"
    if not short:
        short = "未知错误"

    # Keep the tone calm and practical; avoid overly technical wording.
    content = (
        "刚才这条没生成出来。\n"
        f"原因：{short}\n"
        "\n"
        "你可以直接再问一次；如果经常出现，通常是网络波动或上游 AI 卡住了。"
    )

    try:
        msg = AssistantMessage(
            thread_id=tid,
            role="assistant",
            content=content,
            meta={"type": "job_error", "job_id": int(getattr(job, "id", 0) or 0)},
        )
        session.add(msg)
        # Best-effort bump thread updated_at so it floats to the top in thread list.
        try:
            t = session.get(AssistantThread, tid)
            if t:
                t.updated_at = datetime.utcnow()
                session.add(t)
        except Exception:
            pass
        session.commit()
        session.refresh(msg)

        # Link back to job to make job API able to surface the message too.
        try:
            job.assistant_message_id = int(msg.id or 0)  # type: ignore
            session.add(job)
            session.commit()
        except Exception:
            try:
                session.rollback()
            except Exception:
                pass
    except Exception:
        try:
            session.rollback()
        except Exception:
            pass


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
    """Generate assistant reply by calling local `codex exec` directly (no clawd-relay)."""

    job_id = int(getattr(job, "id", 0) or 0)
    attempt = int(getattr(job, "attempts", 0) or 0)

    prompt = _assistant_build_codex_prompt(messages)
    # Prefer configured cwd; otherwise auto-pick a sensible workspace path.
    codex_cwd = (ASSISTANT_CODEX_CWD or "").strip()
    if codex_cwd and not os.path.isdir(codex_cwd):
        codex_cwd = ""
    if not codex_cwd:
        for p in ("/workspace", "/app", "/Users/marllenos/customor-qc-app"):
            if os.path.isdir(p):
                codex_cwd = p
                break
    if not codex_cwd:
        codex_cwd = os.getcwd()
    sandbox = ASSISTANT_CODEX_SANDBOX if ASSISTANT_CODEX_SANDBOX in ("read-only", "workspace-write", "danger-full-access") else "workspace-write"
    approval = ASSISTANT_CODEX_APPROVAL if ASSISTANT_CODEX_APPROVAL in ("untrusted", "on-failure", "on-request", "never") else "never"
    out_path, log_path = assistant_codex_paths(job_id=job_id, attempt=attempt)

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

    if ASSISTANT_SAVE_TRACE:
        try:
            trace = {
                "ts": datetime.utcnow().isoformat(),
                "runner": "local",
                "codex_cwd": codex_cwd,
                "sandbox": sandbox,
                "model": ASSISTANT_CODEX_MODEL or None,
                "oss": bool(ASSISTANT_CODEX_OSS),
                "local_provider": ASSISTANT_CODEX_LOCAL_PROVIDER or None,
                "prompt_chars": len(prompt),
                "prompt_preview": (prompt[:20000] + "…") if len(prompt) > 20000 else prompt,
                "argv": argv[:-1] + ["<PROMPT>"],
                "output_last_message": out_path,
                "stderr_log": log_path,
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
    # Run codex and keep stdout clean; capture stderr to log_path for debugging.
    ok = False
    try:
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass
        try:
            if os.path.exists(log_path):
                os.remove(log_path)
        except Exception:
            pass

        # Ensure artifact dir exists
        try:
            os.makedirs(os.path.dirname(log_path) or "/tmp", exist_ok=True)
        except Exception:
            pass

        stderr_f = open(log_path, "wb")
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=codex_cwd,
                stdout=stderr_f,
                stderr=stderr_f,
            )
            try:
                deadline = time.time() + float(ASSISTANT_TIMEOUT_S)
                tick_s = 0.8
                while True:
                    # Allow user to cancel while Codex is running.
                    if _job_stop_requested(job_id):
                        try:
                            proc.kill()
                        except Exception:
                            pass
                        raise RuntimeError("canceled by user")

                    now = time.time()
                    left = deadline - now
                    if left <= 0:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                        raise RuntimeError(f"codex 调用超时（{int(ASSISTANT_TIMEOUT_S)}s）")

                    try:
                        await asyncio.wait_for(proc.wait(), timeout=min(tick_s, max(0.2, left)))
                        break
                    except asyncio.TimeoutError:
                        continue
            except Exception:
                raise

            # Race safety: if user cancels right before Codex exits, do not proceed.
            if _job_stop_requested(job_id):
                raise RuntimeError("canceled by user")

            rc = int(getattr(proc, "returncode", 0) or 0)
        finally:
            try:
                stderr_f.close()
            except Exception:
                pass

        if rc != 0:
            err_tail = ""
            try:
                with open(log_path, "rb") as f:
                    b = f.read()[-16 * 1024 :]
                err_tail = b.decode("utf-8", errors="replace").strip()
            except Exception:
                err_tail = ""
            hint = (err_tail[-800:] if err_tail else "").strip()
            raise RuntimeError(f"codex 执行失败（exit={rc}）：{hint or 'unknown error'}")

        try:
            with open(out_path, "r", encoding="utf-8", errors="replace") as f:
                text = (f.read() or "").strip()
            ok = True
        except Exception as e:
            raise RuntimeError(f"codex 输出读取失败：{type(e).__name__}") from e
    finally:
        # Best-effort cleanup
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass
        if ok:
            try:
                if os.path.exists(log_path):
                    os.remove(log_path)
            except Exception:
                pass
    t1 = time.time()

    meta = {
        "source": "codex",
        "format": "md",
        "codex": {
            "cwd": codex_cwd,
            "sandbox": sandbox,
            "model": ASSISTANT_CODEX_MODEL or None,
            "duration_ms": int(round((t1 - t0) * 1000)),
            "task_id": None,
            "elapsed_s": round(float(t1 - t0), 3),
        },
    }
    return text, meta


async def _assistant_generate_reply_via_openai(
    *,
    session: Session,
    job: AssistantJob,
    messages: list[dict[str, str]],
) -> tuple[str, dict[str, Any]]:
    """Generate assistant reply via OpenAI-compatible /v1/responses."""
    AIError, chat_completion = _get_ai_client()

    job_id = int(getattr(job, "id", 0) or 0)
    attempt = int(getattr(job, "attempts", 0) or 0)

    t0 = time.time()
    try:
        text = await chat_completion(
            messages,
            model=ASSISTANT_MODEL,
            reasoning_effort=ASSISTANT_REASONING_EFFORT,
            timeout_s=float(ASSISTANT_TIMEOUT_S),
            retries=0,
            request_id=f"assistantjob-{job_id}",
            idempotency_key=f"assistantjob-{job_id}-attempt-{attempt}",
            session=session,
        )
    except AIError:
        raise
    t1 = time.time()

    meta = {
        "source": "openai",
        "format": "md",
        "openai": {
            "model": ASSISTANT_MODEL,
            "duration_ms": int(round((t1 - t0) * 1000)),
        },
    }
    return (text or "").strip(), meta


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
        now = datetime.utcnow()
        hard_cutoff = now - timedelta(seconds=max(60, stale_seconds))

        # Heartbeat-based recovery is intentionally more aggressive than hard_cutoff:
        # if a worker restarts mid-flight, the job stays "running" and blocks the UI.
        # Heartbeat should update every ~10s; if it's silent for ~90s, we can safely recover.
        hb_seconds = 90
        try:
            hb_seconds = int((os.getenv("ASSISTANT_HEARTBEAT_STALE_SECONDS") or "90").strip())
        except Exception:
            hb_seconds = 90
        hb_seconds = max(30, min(600, hb_seconds))
        hb_cutoff = now - timedelta(seconds=hb_seconds)

        # Candidate set: running jobs old enough that they should have a heartbeat.
        stuck = session.exec(
            select(AssistantJob)
            .where(AssistantJob.status == "running")
            .where(AssistantJob.started_at.is_not(None))
            .where(AssistantJob.started_at < hb_cutoff)
        ).all()
        stuck_no_started = session.exec(
            select(AssistantJob)
            .where(AssistantJob.status == "running")
            .where(AssistantJob.started_at.is_(None))
            .where(AssistantJob.created_at < hb_cutoff)
        ).all()
        if stuck_no_started:
            stuck = list(stuck or []) + list(stuck_no_started or [])
        if stuck:
            for j in stuck:
                hb = _job_heartbeat_dt(j)
                started_at = getattr(j, "started_at", None)
                created_at = getattr(j, "created_at", None)

                # Hard timeout: even with heartbeat, we should not keep a job running forever.
                if started_at is not None and started_at < hard_cutoff:
                    pass
                elif started_at is None and created_at is not None and created_at < hard_cutoff:
                    pass
                else:
                    # Multi-worker safety: only recover "running" when heartbeat is missing/stale.
                    if hb and hb >= hb_cutoff:
                        continue

                j.status = "error"  # type: ignore
                j.finished_at = datetime.utcnow()
                if getattr(j, "started_at", None) is None:
                    j.last_error = "任务异常恢复：运行中但缺少 started_at（可能 worker 重启或旧版本遗留）。已自动终止，请重试。"
                else:
                    # Prefer heartbeat wording (more accurate) unless it truly exceeded hard timeout.
                    if getattr(j, "started_at", None) is not None and getattr(j, "started_at", None) < hard_cutoff:
                        mins = max(1, int(round(float(stale_seconds) / 60.0)))
                        j.last_error = f"任务超时自动恢复：已运行超过 {mins} 分钟仍未返回（可能 worker 重启或上游 AI 卡住）。已自动终止，请重试。"
                    else:
                        j.last_error = f"任务异常恢复：超过 {hb_seconds} 秒未收到 worker 心跳（可能 worker 重启或崩溃）。已自动终止，请重试。"
                session.add(j)
            session.commit()
            # Make the failure visible in the chat history.
            for j in stuck:
                try:
                    if _job_status_str(j) == "error":
                        _ensure_assistant_job_error_message(session, j, getattr(j, "last_error", "") or "")
                except Exception:
                    pass

        # Backfill: historical error rows may have no visible assistant message,
        # which makes the UI look like "nothing happened" after refresh.
        try:
            missing_err_msgs = session.exec(
                select(AssistantJob)
                .where(AssistantJob.status == "error")
                .where(AssistantJob.assistant_message_id.is_(None))
                .where(AssistantJob.finished_at.is_not(None))
                .order_by(AssistantJob.finished_at.desc(), AssistantJob.id.desc())
                .limit(20)
            ).all()
            for j in (missing_err_msgs or []):
                if not (getattr(j, "last_error", None) or "").strip():
                    continue
                _ensure_assistant_job_error_message(session, j, getattr(j, "last_error", "") or "")
        except Exception:
            pass
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
    try:
        extra = dict(job.extra or {}) if isinstance(job.extra, dict) else {}
        extra["heartbeat_at"] = _utc_iso_z()
        job.extra = extra
    except Exception:
        pass
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
    # Always leave a visible trace for the user (refresh/close shouldn't lose it).
    _ensure_assistant_job_error_message(session, j, j.last_error or "")


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

    upstream = (ASSISTANT_UPSTREAM or "openai").strip().lower()
    if upstream == "codex":
        system_prompt = build_codex_system_prompt(user=user)
        msgs = _build_messages(system_prompt)
        text, meta0 = await _assistant_generate_reply_via_codex(session=session, job=job, messages=msgs)
    else:
        system_prompt = build_system_prompt(session, user=user, question_hint=question_hint)
        msgs = _build_messages(system_prompt)
        text, meta0 = await _assistant_generate_reply_via_openai(session=session, job=job, messages=msgs)
    if not (text or "").strip():
        raise RuntimeError("assistant 返回为空")

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
        if isinstance(meta0, dict) and meta0.get("source") == "codex":
            job.extra = {**(job.extra or {}), "codex": (meta0 or {}).get("codex")}
        session.add(job)
        session.commit()
    except Exception:
        try:
            session.rollback()
        except Exception:
            pass

    cleaned, charts = _extract_charts(text)
    meta: dict[str, Any] = {**(meta0 or {}), "reasoning_effort": ASSISTANT_REASONING_EFFORT}
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
