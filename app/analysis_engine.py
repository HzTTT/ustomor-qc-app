from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Any

from sqlmodel import Session, select

from ai_client import chat_completion
from models import (
    Conversation,
    Message,
    ConversationAnalysis,
    AnalysisBatch,
    AIAnalysisJob,
    TagCategory,
    TagDefinition,
    ConversationTagHit,
    TagSuggestion,
)
from prompts import analysis_system_prompt, analysis_user_prompt, build_tag_catalog_for_prompt


def _tags_to_str(x: Any) -> str:
    """Excel规范：多选用 ';' 分隔。"""
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, (list, tuple, set)):
        return ";".join([str(i).strip() for i in x if str(i).strip()])
    return str(x).strip()


def _tag_parsing_to_str(x: Any) -> str:
    """Convert tag parsing to 'tag$$$reason&&&...'"""
    if not x:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, list):
        parts: list[str] = []
        for it in x:
            if not isinstance(it, dict):
                continue
            tag = str(it.get("tag", "")).strip()
            reason = str(it.get("reason", "")).strip()
            if not tag and not reason:
                continue
            parts.append(f"{tag}$$${reason}")
        return "&&&".join(parts)
    if isinstance(x, dict):
        parts: list[str] = []
        for k, v in x.items():
            tag = str(k).strip()
            if not tag:
                continue
            parts.append(f"{tag}$$${str(v).strip()}")
        return "&&&".join(parts)
    return str(x).strip()


def _safe_json_loads(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        # Trim triple-backticks fence; model sometimes wraps json in code fence.
        text = text.strip("`")
        text = text.replace("json\n", "", 1).strip()
    return json.loads(text)


def _pick_list(parsed: dict[str, Any], keys: list[str]) -> list:
    for k in keys:
        v = parsed.get(k)
        if isinstance(v, list):
            return v
    return []


def get_or_create_auto_batch(session: Session, *, batch_key: str) -> AnalysisBatch:
    existing = session.exec(select(AnalysisBatch).where(AnalysisBatch.source_filename == batch_key)).first()
    if existing:
        return existing
    b = AnalysisBatch(source_filename=batch_key, imported_by_user_id=None, batch_meta={"mode": "auto-ai"}, raw_json={})
    session.add(b)
    session.commit()
    session.refresh(b)
    return b


async def analyze_conversation(session: Session, *, conversation_id: int, batch_key: str) -> ConversationAnalysis:
    conv = session.get(Conversation, conversation_id)
    if not conv:
        raise RuntimeError("conversation not found")

    msgs = session.exec(
        select(Message).where(Message.conversation_id == conv.id).order_by(Message.id.asc())
    ).all()
    msg_payload = [
        {
            "sender": m.sender,
            "text": m.text,
            "attachments": m.attachments or [],
            "agent_account": getattr(m, "agent_account", "") or "",
            "agent_nick": getattr(m, "agent_nick", "") or "",
        }
        for m in msgs
    ]

    # Get previous conversation for historical context
    previous_conv = None
    previous_msgs_payload = []
    if conv.user_thread_id:
        # Find the most recent conversation before current one in the same thread
        prev_stmt = (
            select(Conversation)
            .where(Conversation.user_thread_id == conv.user_thread_id)
            .where(Conversation.id != conv.id)
        )
        # Order by started_at, fallback to uploaded_at, then id
        if conv.started_at:
            prev_stmt = prev_stmt.where(
                (Conversation.started_at < conv.started_at) | (Conversation.started_at == None)  # noqa: E711
            )
        prev_stmt = prev_stmt.order_by(
            Conversation.started_at.desc().nulls_last(),
            Conversation.uploaded_at.desc().nulls_last(),
            Conversation.id.desc()
        ).limit(1)
        
        prev_convs = session.exec(prev_stmt).all()
        if prev_convs:
            previous_conv = prev_convs[0]
            # Get messages from previous conversation (limit to 50 to control token usage)
            prev_msgs = session.exec(
                select(Message)
                .where(Message.conversation_id == previous_conv.id)
                .order_by(Message.id.asc())
                .limit(50)
            ).all()
            previous_msgs_payload = [
                {
                    "sender": m.sender,
                    "text": m.text[:500] if len(m.text) > 500 else m.text,  # Truncate long messages
                    "agent_account": getattr(m, "agent_account", "") or "",
                }
                for m in prev_msgs
            ]

    meta = {
        "platform": conv.platform,
        "agent_account": conv.agent_account,
        "buyer_id": conv.buyer_id,
        "started_at": conv.started_at.isoformat() if conv.started_at else "",
        "ended_at": conv.ended_at.isoformat() if conv.ended_at else "",
        "external_id": conv.external_id,
    }

    # Build tag catalog snapshot (active categories + active tags)
    categories = session.exec(
        select(TagCategory)
        .where(TagCategory.is_active == True)  # noqa: E712
        .order_by(TagCategory.sort_order.asc(), TagCategory.id.asc())
    ).all()

    tag_rows = session.exec(
        select(TagDefinition)
        .where(TagDefinition.is_active == True)  # noqa: E712
        .order_by(TagDefinition.category_id.asc(), TagDefinition.sort_order.asc(), TagDefinition.id.asc())
    ).all()

    tags_by_cat: dict[int, list[TagDefinition]] = {}
    for t in tag_rows:
        tags_by_cat.setdefault(int(t.category_id), []).append(t)

    catalog_payload: list[dict[str, Any]] = []
    for c in categories:
        catalog_payload.append(
            {
                "name": c.name,
                "tags": [
                    {
                        "id": t.id,
                        "name": t.name,
                        "standard": t.standard,
                        "description": t.description,
                    }
                    for t in (tags_by_cat.get(int(c.id or 0), []) if c.id else [])
                    if t.id is not None
                ],
            }
        )

    tag_catalog = build_tag_catalog_for_prompt(catalog_payload)

    # Build previous conversation context if available
    prev_conv_data = None
    if previous_conv and previous_msgs_payload:
        prev_conv_data = {
            "meta": {
                "started_at": previous_conv.started_at.isoformat() if previous_conv.started_at else "",
                "ended_at": previous_conv.ended_at.isoformat() if previous_conv.ended_at else "",
            },
            "messages": previous_msgs_payload,
        }

    messages = [
        {"role": "system", "content": analysis_system_prompt(session)},
        {"role": "user", "content": analysis_user_prompt(meta, msg_payload, tag_catalog=tag_catalog, previous_conversation=prev_conv_data)},
    ]

    raw = await chat_completion(messages, temperature=0.2, session=session)

    parsed: dict[str, Any]
    try:
        parsed = _safe_json_loads(raw)
    except Exception:
        parsed = {"raw_text": raw}

    yellow = _pick_list(parsed, ["customer_issue_highlights", "客服问题对话识别", "yellow_highlights", "highlights_yellow"])  # type: ignore
    green = _pick_list(parsed, ["must_read_highlights", "需要阅读的地方", "green_highlights", "highlights_green"])  # type: ignore
    highlights = _pick_list(parsed, ["highlights", "问题定位", "问题定位（对话片段）"])  # type: ignore

    batch = get_or_create_auto_batch(session, batch_key=batch_key)

    # Reset previous tag hits for this conversation (to ensure clean state)
    # Get all previous analyses for this conversation
    prev_analyses = session.exec(
        select(ConversationAnalysis).where(ConversationAnalysis.conversation_id == conv.id)
    ).all()
    for prev_analysis in prev_analyses:
        # Delete all ConversationTagHit records for this analysis
        session.exec(
            select(ConversationTagHit).where(ConversationTagHit.analysis_id == prev_analysis.id)
        ).all()
        for hit in session.exec(
            select(ConversationTagHit).where(ConversationTagHit.analysis_id == prev_analysis.id)
        ).all():
            session.delete(hit)
    session.commit()

    reception = str(parsed.get("reception_scenario", "") or parsed.get("dialog_type", "") or "").strip()
    satisfaction = str(parsed.get("satisfaction_change", "") or "").strip()
    dialog_type = str(parsed.get("dialog_type", "") or "").strip() or (reception if reception else "")

    day_summary = str(parsed.get("day_summary", "")) or ""
    # Some legacy DBs require NOT NULL `summary` column. Keep it aligned.
    summary = str(parsed.get("summary", "")) or day_summary

    analysis = ConversationAnalysis(
        batch_id=batch.id,
        conversation_id=conv.id,
        reception_scenario=reception,
        satisfaction_change=satisfaction,
        dialog_type=dialog_type,
        pre_positive_tags=_tags_to_str(parsed.get("pre_positive_tags")),
        after_positive_tags=_tags_to_str(parsed.get("after_positive_tags")),
        pre_negative_tags=_tags_to_str(parsed.get("pre_negative_tags")),
        after_negative_tags=_tags_to_str(parsed.get("after_negative_tags")),
        day_summary=day_summary,
        summary=summary,
        tag_parsing=_tag_parsing_to_str(parsed.get("tag_parsing")),
        product_suggestion=str(parsed.get("product_suggestion", "")) or "",
        service_suggestion=str(parsed.get("service_suggestion", "")) or "",
        pre_rule_update=str(parsed.get("pre_rule_update", "")) or "",
        after_rule_update=str(parsed.get("after_rule_update", "")) or "",
        tag_update_suggestion=_tags_to_str(parsed.get("tag_update_suggestion")),
        overall_score=(parsed.get("overall_score") if isinstance(parsed.get("overall_score"), int) else None),
        sentiment=str(parsed.get("sentiment", "")) or "",
        issue_level=str(parsed.get("issue_level", "")) or "",
        problem_types=(parsed.get("problem_types") if isinstance(parsed.get("problem_types"), list) else []),
        flag_for_review=bool(parsed.get("flag_for_review", False)),
        evidence={
            "highlights": highlights,
            "customer_issue_highlights": yellow,
            "must_read_highlights": green,
            "raw": parsed,
        },
        extra={},
    )

    # 默认规则：存在任一负面标签/规则更新建议/商品或服务建议 => 需复核
    if (
        analysis.pre_negative_tags
        or analysis.after_negative_tags
        or analysis.pre_rule_update
        or analysis.after_rule_update
        or analysis.product_suggestion
        or analysis.service_suggestion
    ):
        analysis.flag_for_review = True

    session.add(analysis)
    session.commit()
    session.refresh(analysis)

    # Persist tag hits (best-effort)
    try:
        tag_hits = parsed.get("tag_hits")
        if isinstance(tag_hits, list) and analysis.id is not None:
            # De-dup by tag_id
            seen: set[int] = set()
            for it in tag_hits:
                if not isinstance(it, dict):
                    continue
                tid = it.get("tag_id")
                try:
                    tid_int = int(tid)
                except Exception:
                    continue
                if tid_int <= 0 or tid_int in seen:
                    continue
                seen.add(tid_int)

                reason = str(it.get("reason", "") or "")
                evidence_it = it.get("evidence")
                if not isinstance(evidence_it, (list, dict)):
                    evidence_it = {}

                hit = ConversationTagHit(
                    analysis_id=int(analysis.id),
                    tag_id=tid_int,
                    reason=reason[:4000],
                    evidence=(evidence_it if isinstance(evidence_it, (dict, list)) else {}),
                )
                session.add(hit)
            session.commit()
    except Exception:
        pass

    # Create TagSuggestion rows for new_tag_suggestions (manager review workflow)
    try:
        suggestions = parsed.get("new_tag_suggestions")
        if isinstance(suggestions, list) and analysis.id is not None:
            for s in suggestions:
                if not isinstance(s, dict):
                    continue
                cat = str(s.get("category") or "").strip()
                tname = str(s.get("tag_name") or "").strip()
                if not cat and not tname:
                    continue
                ts = TagSuggestion(
                    analysis_id=int(analysis.id),
                    conversation_id=int(conv.id),
                    suggested_category=cat,
                    suggested_tag_name=tname,
                    suggested_standard=str(s.get("standard") or "")[:4000],
                    suggested_description=str(s.get("description") or "")[:2000],
                    ai_reason=str(s.get("reason") or "")[:2000],
                    status="pending",
                )
                session.add(ts)
            session.commit()
    except Exception:
        pass

    return analysis


def enqueue_missing_analyses(session: Session) -> int:
    """Create AIAnalysisJob for conversations that have no analyses and no job."""

    conv_ids = [
        row[0]
        for row in session.exec(
            select(Conversation.id).where(~Conversation.analyses.any())
        ).all()
    ]
    if not conv_ids:
        return 0

    existing = set(
        [
            row[0]
            for row in session.exec(
                select(AIAnalysisJob.conversation_id).where(AIAnalysisJob.conversation_id.in_(conv_ids))
            ).all()
        ]
    )

    created = 0
    for cid in conv_ids:
        if cid in existing:
            continue
        session.add(AIAnalysisJob(conversation_id=cid))
        created += 1

    if created:
        session.commit()
    return created


def claim_one_job(session: Session) -> AIAnalysisJob | None:
    """Best-effort claim: pick one pending job and mark running."""

    # Recovery: if worker crashed mid-job, the row can be stuck in `running`.
    # Mark it as error after a grace period so it can be retried.
    try:
        stale_seconds = int(os.getenv("AI_JOB_STALE_SECONDS", "1800"))  # 30min
        cutoff = datetime.utcnow() - timedelta(seconds=max(60, stale_seconds))
        stuck = session.exec(
            select(AIAnalysisJob)
            .where(AIAnalysisJob.status == "running")
            .where(AIAnalysisJob.started_at.is_not(None))
            .where(AIAnalysisJob.started_at < cutoff)
        ).all()
        if stuck:
            for j in stuck:
                j.status = "error"  # type: ignore
                j.finished_at = datetime.utcnow()
                j.last_error = "job stale: worker crash/timeout recovered"
                session.add(j)
            session.commit()
    except Exception:
        pass

    job = session.exec(
        select(AIAnalysisJob)
        .where(AIAnalysisJob.status == "pending")
        .order_by(AIAnalysisJob.created_at.asc())
        .limit(1)
    ).first()
    if not job:
        return None

    job.status = "running"  # type: ignore
    job.started_at = datetime.utcnow()
    job.attempts = (job.attempts or 0) + 1
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


def mark_job_done(session: Session, job: AIAnalysisJob, *, batch_id: int) -> None:
    job.status = "done"  # type: ignore
    job.finished_at = datetime.utcnow()
    job.last_error = ""
    job.batch_id = batch_id
    session.add(job)
    session.commit()


def mark_job_error(session: Session, job: AIAnalysisJob, err: str) -> None:
    # If the previous transaction failed (e.g. INSERT constraint error), the
    # Session is in a "pending rollback" state. We must rollback before writing
    # the job status, otherwise worker will crash and the UI will look stuck.
    try:
        session.rollback()
    except Exception:
        pass

    j = None
    try:
        if getattr(job, "id", None) is not None:
            j = session.get(AIAnalysisJob, int(job.id))
    except Exception:
        j = None
    if j is None:
        j = job

    j.status = "error"  # type: ignore
    j.finished_at = datetime.utcnow()
    j.last_error = (err or "")[:2000]
    session.add(j)
    session.commit()
