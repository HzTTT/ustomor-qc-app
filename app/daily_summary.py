from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib

from sqlmodel import Session, select
from sqlalchemy.exc import ProgrammingError
from sqlalchemy import func

from ai_client import chat_completion, responses_text, get_ai_settings
from models import Conversation, Message, User, DailyAISummaryReport, DailyAISummaryJob, JobStatus


DEFAULT_DAILY_SUMMARY_PROMPT = """你是我们的客服质检负责人，请基于下面“当天全部对话数据”，输出一份【管理者可直接转发】的《每日对话AI总结》：

必须包含：
1) 今日概览：对话量、涉及平台、主要问题分布（用大白话）
2) 客户侧高频问题Top5（每条给 1-2 句原因 + 解决建议）
3) 客服侧需要改进Top5（每条给 1-2 句具体怎么改）
4) VOC：商品/服务优缺点（把“客人真实表达”总结出来）
5) 需要重点跟进的对话清单：请按重要性排序，且每条必须引用对话主键信息（例如：CID=123 / external_id=xxx）方便我一键打开定位

输出风格：
- 先结论后细节
- 用项目符号，短句为主
- 不要写长篇论文
"""


@dataclass
class DailyInputBuildResult:
    run_date: str
    threshold_messages: int
    input_text: str
    input_chars: int
    included_conversations: int
    included_messages: int
    meta: dict
    blocks: list[str]


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _fmt_dt_seconds(dt: datetime | None) -> str:
    """Format datetime as 'YYYY-MM-DD HH:MM:SS' (no microseconds)."""
    if not dt:
        return ""
    try:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        # best-effort fallback
        return str(dt)


def _format_attachments_short(atts: object) -> str:
    """Render attachments in a compact, human-friendly way.

    We intentionally avoid the old placeholder like "[attachments]". If we
    have meaningful metadata (type/url/name/summary), we surface it; otherwise
    we keep a simple count.
    """
    if not atts:
        return ""

    items = []
    if isinstance(atts, list):
        items = atts
    elif isinstance(atts, dict):
        items = [atts]
    else:
        return ""

    # We only keep *meaningful* attachments for the daily input.
    # In Taobao/Leyan exports, it's common to see "meta" attachments
    # (e.g. type=meta) which are not useful for summarization.
    # We suppress those entirely.
    images_count = 0
    files_count = 0
    system_cards: list[str] = []
    other_types: dict[str, int] = {}

    for it in items:
        if not isinstance(it, dict):
            continue
        # Skip noisy meta-only blobs
        t_raw = str(it.get("type") or "").strip().lower()
        if (not t_raw) and any(str(k).lower().startswith("meta") for k in it.keys()):
            # Examples: {"meta": "meta1"} / {"meta_id": "..."}
            continue

        t = t_raw or "unknown"

        if t in ("meta", "metadata") or t.startswith("meta_"):
            continue

        if t == "image":
            # The daily AI input is plain text; image URLs are not useful.
            # We only keep a count.
            images_count += 1
            continue
        if t in ("file", "doc", "document"):
            # Same reason as images: links are not useful in a pure-text AI prompt.
            files_count += 1
            continue
        if t == "system_card":
            summary = str(it.get("summary") or "").strip()
            card_type = str(it.get("card_type") or "").strip()
            if summary and card_type:
                system_cards.append(f"{card_type}: {summary}")
            elif summary:
                system_cards.append(summary)
            elif card_type:
                system_cards.append(card_type)
            else:
                system_cards.append("")
            continue

        other_types[t] = other_types.get(t, 0) + 1

    parts: list[str] = []
    if images_count:
        parts.append(f"图片{images_count}")

    if files_count:
        parts.append(f"文件{files_count}")

    if system_cards:
        summaries = [s for s in system_cards if s]
        if summaries and len(summaries) <= 2:
            parts.append("系统卡片: " + " | ".join(summaries))
        else:
            parts.append(f"系统卡片{len(system_cards)}")

    if other_types:
        for k in sorted(other_types.keys()):
            parts.append(f"{k}{other_types[k]}")

    if not parts:
        # attachments exist, but after filtering meta/noise we have nothing meaningful
        return ""

    out = "附件：" + "；".join(parts)
    # prevent extremely long lines (urls etc.)
    if len(out) > 500:
        out = out[:497] + "..."
    return out



def _stable_key(prefix: str, parts: list[str]) -> str:
    raw = prefix + "||" + "||".join([p or "" for p in parts])
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{prefix}:{h[:32]}"


def build_daily_input(session: Session, *, run_date: str, threshold_messages: int) -> DailyInputBuildResult:
    """Collect all conversations for the given day and build a single text payload.

    Note: We use Conversation.started_at when available, otherwise Conversation.uploaded_at.
    """

    day_start = _parse_date(run_date)
    day_end = day_start + timedelta(days=1)

    # Find candidate conversations by day.
    convs = session.exec(
        select(Conversation)
        .where(
            ((Conversation.started_at >= day_start) & (Conversation.started_at < day_end))
            | (
                (Conversation.started_at.is_(None))
                & (Conversation.uploaded_at >= day_start)
                & (Conversation.uploaded_at < day_end)
            )
        )
        .order_by(Conversation.id.asc())
    ).all()

    if not convs:
        return DailyInputBuildResult(
            run_date=run_date,
            threshold_messages=threshold_messages,
            input_text="",
            input_chars=0,
            included_conversations=0,
            included_messages=0,
            meta={"conversation_ids": []},
            blocks=[],
        )

    conv_ids = [c.id for c in convs if c.id]

    # Message counts per conversation
    rows = session.exec(
        select(Message.conversation_id, func.count(Message.id))
        .where(Message.conversation_id.in_(conv_ids))
        .group_by(Message.conversation_id)
    ).all()
    msg_count_map = {int(cid): int(cnt) for (cid, cnt) in rows}

    included = [c for c in convs if c.id and msg_count_map.get(int(c.id), 0) >= int(threshold_messages)]

    blocks: list[str] = []
    included_message_total = 0
    truncated_conversation_ids: list[int] = []

    # === Preload internal agent users involved in each conversation ===
    # Prefer per-message agent_user_id (more accurate for multi-agent conversations),
    # fall back to Conversation.agent_user_id when message-level mapping is missing.
    included_conv_ids = [int(c.id) for c in included if c.id]
    conv_agent_user_ids_map: dict[int, set[int]] = {cid: set() for cid in included_conv_ids}

    if included_conv_ids:
        rows_uid = session.exec(
            select(Message.conversation_id, Message.agent_user_id)
            .where(
                Message.conversation_id.in_(included_conv_ids),
                Message.sender == "agent",
                Message.agent_user_id.is_not(None),
            )
            .distinct()
        ).all()
        for r in rows_uid or []:
            try:
                conv_id = int(r[0])
                uid = int(r[1])
            except Exception:
                continue
            if conv_id in conv_agent_user_ids_map:
                conv_agent_user_ids_map[conv_id].add(uid)

    # Add conversation-level agent_user_id as a fallback/union
    for c in included:
        if c.id and c.agent_user_id:
            conv_agent_user_ids_map.setdefault(int(c.id), set()).add(int(c.agent_user_id))

    all_uids: list[int] = sorted({uid for s in conv_agent_user_ids_map.values() for uid in s})
    users = session.exec(select(User).where(User.id.in_(all_uids))).all() if all_uids else []
    user_map = {int(u.id): u for u in users if u and u.id}

    for c in included:
        cid = int(c.id or 0)
        msg_count = msg_count_map.get(cid, 0)
        included_message_total += msg_count

        # Multi-agent: show ALL internal agents involved (if any).
        agent_info = "-"
        agent_uids = sorted(list(conv_agent_user_ids_map.get(cid, set())))
        if agent_uids:
            parts: list[str] = []
            for uid in agent_uids:
                u = user_map.get(uid)
                if not u:
                    continue
                parts.append(f"{u.name} ({u.username or u.email})")
            agent_info = ", ".join(parts) if parts else "-"

        started = _fmt_dt_seconds(c.started_at)
        ended = _fmt_dt_seconds(c.ended_at)

        # Multi-agent: collect all agent accounts seen in this conversation
        agent_participants = session.exec(
            select(Message.agent_account).where(
                Message.conversation_id == cid,
                Message.sender == "agent",
                Message.agent_account.is_not(None),
                Message.agent_account != "",
            ).distinct()
        ).all()
        agent_participants_list = sorted([str(r[0]).strip() for r in (agent_participants or []) if r and str(r[0]).strip()])
        agent_participants_str = ", ".join(agent_participants_list) if agent_participants_list else "-"
        header = (
            "=" * 32
            + f"\nCID={cid} | external_id={c.external_id} | platform={c.platform}"
            + f"\n客服账号(主)={c.agent_account or '-'} | 参与客服={agent_participants_str} | 站内客服={agent_info}"
            + f"\n对话时间: {started} ~ {ended}\n"
            + f"消息数: {msg_count}\n"
            + "=" * 32
        )

        msgs = session.exec(
            select(Message)
            .where(Message.conversation_id == cid)
            .order_by(Message.id.asc())
        ).all()

        lines: list[str] = []
        for m in msgs:
            ts = _fmt_dt_seconds(m.ts)
            sender = m.sender or ""
            if sender == "agent":
                nick = str(getattr(m, "agent_nick", "") or "").strip()
                acc = str(getattr(m, "agent_account", "") or "").strip()
                sender = f"agent({nick})" if nick else (f"agent({acc})" if acc else "agent")
            text = (m.text or "").strip()
            att = _format_attachments_short(getattr(m, "attachments", None))
            if att:
                text = (text + "\n" + att) if text else att
            lines.append(f"[{ts}] {sender}: {text}")

        body = "\n".join(lines).strip() + "\n"

        # Safety: if one conversation is extremely large, keep head+tail.
        if len(body) > 40000:
            truncated_conversation_ids.append(cid)
            head = body[:20000]
            tail = body[-20000:]
            body = head + "\n...（对话过长，已截断中间内容）...\n" + tail

        blocks.append(header + "\n" + body)

    input_text = "\n\n".join(blocks).strip()
    meta = {
        "conversation_ids": [int(c.id) for c in included if c.id],
        "truncated_conversation_ids": truncated_conversation_ids,
        "candidates_conversations": len(convs),
        "candidates_conversation_ids": conv_ids,
    }

    return DailyInputBuildResult(
        run_date=run_date,
        threshold_messages=int(threshold_messages),
        input_text=input_text,
        input_chars=len(input_text),
        included_conversations=len(included),
        included_messages=included_message_total,
        meta=meta,
        blocks=blocks,
    )


def get_or_create_daily_report(session: Session, *, run_date: str) -> DailyAISummaryReport:
    existing = session.exec(select(DailyAISummaryReport).where(DailyAISummaryReport.run_date == run_date)).first()
    if existing:
        return existing
    r = DailyAISummaryReport(run_date=run_date)
    session.add(r)
    session.commit()
    session.refresh(r)
    return r



def get_latest_daily_job(session: Session, *, run_date: str) -> DailyAISummaryJob | None:
    try:
        return session.exec(
            select(DailyAISummaryJob)
            .where(DailyAISummaryJob.run_date == run_date)
            .order_by(DailyAISummaryJob.created_at.desc())
            .limit(1)
        ).first()
    except Exception:
        return None


def enqueue_daily_job(
    session: Session,
    *,
    run_date: str,
    threshold_messages: int,
    prompt: str,
    model: str,
    created_by_user_id: int | None,
    estimate: dict | None = None,
) -> DailyAISummaryJob:
    """Create a pending job unless there is already a pending/running one for that day."""

    latest = get_latest_daily_job(session, run_date=run_date)
    if latest and str(latest.status) in ("pending", "running"):
        return latest

    # Ensure a report row exists (so UI can link to it even before generation finishes).
    report = get_or_create_daily_report(session, run_date=run_date)
    session.add(report)
    session.commit()
    session.refresh(report)

    job = DailyAISummaryJob(
        run_date=run_date,
        status=JobStatus.pending,  # type: ignore
        report_id=int(report.id or 0) if report.id else None,
        extra={
            "threshold_messages": int(threshold_messages),
            "prompt": prompt or "",
            "model": (model or "").strip(),
            "created_by_user_id": created_by_user_id,
            "estimate": estimate or {},
        },
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


def claim_one_daily_job(session: Session) -> DailyAISummaryJob | None:
    """Pick one pending daily job and mark running."""
    try:
        base_stmt = (
            select(DailyAISummaryJob)
            .where(DailyAISummaryJob.status == "pending")
            .order_by(DailyAISummaryJob.created_at.asc())
            .limit(1)
        )
        try:
            job = session.exec(base_stmt.with_for_update(skip_locked=True)).first()
        except Exception:
            job = session.exec(base_stmt).first()
    except ProgrammingError:
        # The table may not exist yet (startup race / first deploy).
        # Treat as "no job" and let polling retry later.
        return None
    if not job:
        return None

    job.status = "running"  # type: ignore
    job.started_at = datetime.utcnow()
    job.attempts = (job.attempts or 0) + 1
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


def mark_daily_job_done(session: Session, job: DailyAISummaryJob) -> None:
    job.status = "done"  # type: ignore
    job.finished_at = datetime.utcnow()
    job.last_error = ""
    session.add(job)
    session.commit()


def mark_daily_job_error(session: Session, job: DailyAISummaryJob, err: str) -> None:
    job.status = "error"  # type: ignore
    job.finished_at = datetime.utcnow()
    job.last_error = (err or "")[:2000]
    session.add(job)
    session.commit()




async def generate_daily_summary(
    session: Session,
    *,
    run_date: str,
    threshold_messages: int,
    prompt: str,
    model: str,
    created_by_user_id: int | None,
    max_chars_per_chunk: int = 120000,
) -> DailyAISummaryReport:
    """Generate (or overwrite) a daily report and persist it."""

    built = build_daily_input(session, run_date=run_date, threshold_messages=threshold_messages)

    ai_settings = get_ai_settings(session)

    report = get_or_create_daily_report(session, run_date=run_date)

    report.threshold_messages = int(threshold_messages)
    report.model = (model or "").strip()
    report.prompt = prompt or ""
    report.input_chars = int(built.input_chars)
    report.included_conversations = int(built.included_conversations)
    report.included_messages = int(built.included_messages)
    report.created_by_user_id = created_by_user_id

    # If no conversations meet the threshold, save an empty report (with a friendly note).
    if built.included_conversations == 0:
        report.report_text = "当日没有达到阈值的对话（可尝试把阈值调低）。"
        report.meta = {**(built.meta or {}), "chunks": 0, "ai_base_url": ai_settings.get("base_url"), "ai_retries": ai_settings.get("retries"), "ai_timeout": ai_settings.get("timeout_s")}
        session.add(report)
        session.commit()
        session.refresh(report)
        return report

    # Split into chunks if too big (chunk by conversation blocks).
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for b in (built.blocks or [built.input_text]):
        b_len = len(b)
        if current and (current_len + b_len) > max_chars_per_chunk:
            chunks.append("\n\n".join(current))
            current = [b]
            current_len = b_len
        else:
            current.append(b)
            current_len += b_len
    if current:
        chunks.append("\n\n".join(current))

    system = "你是经验丰富的客服质检负责人，擅长把大量对话汇总成管理者可直接转发的日报。"

    # 1) Chunk summaries
    if len(chunks) == 1:
        user_content = (
            (prompt or DEFAULT_DAILY_SUMMARY_PROMPT).strip()
            + "\n\n下面是当天对话数据（请严格引用 CID/external_id 以便定位）：\n"
            + built.input_text
        )
        final_text = await responses_text(
            input_text=user_content,
            instructions=system,
            model=model,
            request_id=_stable_key("dai-req", [run_date, "final", str(threshold_messages), str(built.input_chars)]),
            idempotency_key=_stable_key("dai", [run_date, "final", str(threshold_messages), str(built.input_chars)]),
            session=session,
        )
        report.report_text = (final_text or "").strip()
        report.meta = {**(built.meta or {}), "chunks": 1, "ai_base_url": ai_settings.get("base_url"), "ai_retries": ai_settings.get("retries"), "ai_timeout": ai_settings.get("timeout_s")}
        session.add(report)
        session.commit()
        session.refresh(report)
        return report

    partials: list[str] = []
    for idx, ch in enumerate(chunks, start=1):
        user_content = (
            "请先对下面这些对话做一个【局部总结】（这是当天对话的一部分）。\n"
            "输出要求：\n"
            "- 用项目符号\n"
            "- 必须引用 CID/external_id\n"
            "- 不要写长篇\n\n"
            f"【第 {idx}/{len(chunks)} 段数据】\n" + ch
        )
        part = await responses_text(
            input_text=user_content,
            instructions=system,
            model=model,
            request_id=_stable_key("dai-req", [run_date, f"part-{idx}", str(threshold_messages), str(len(ch))]),
            idempotency_key=_stable_key("dai", [run_date, f"part-{idx}", str(threshold_messages), str(len(ch))]),
            session=session,
        )
        partials.append((part or "").strip())

    # 2) Final merge
    merge_user = (
        (prompt or DEFAULT_DAILY_SUMMARY_PROMPT).strip()
        + "\n\n下面是同一天分段生成的局部总结，请合并成一份最终《每日对话AI总结》，去重、归类、按重要性排序。\n\n"
        + "\n\n".join([f"【局部总结 {i+1}】\n{t}" for i, t in enumerate(partials)])
    )
    final_text = await responses_text(
        input_text=merge_user,
        instructions=system,
        model=model,
        request_id=_stable_key("dai-req", [run_date, "merge", str(threshold_messages), str(len(partials))]),
        idempotency_key=_stable_key("dai", [run_date, "merge", str(threshold_messages), str(len(partials))]),
        session=session,
    )

    report.report_text = (final_text or "").strip()
    report.meta = {**(built.meta or {}), "chunks": len(chunks), "partial_chars": [len(p) for p in partials], "ai_base_url": ai_settings.get("base_url"), "ai_retries": ai_settings.get("retries"), "ai_timeout": ai_settings.get("timeout_s")}
    session.add(report)
    session.commit()
    session.refresh(report)
    return report
