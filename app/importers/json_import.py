from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Tuple

from sqlmodel import Session, select

from auth import get_user_by_email, get_user_by_username, hash_password
from models import AgentBinding, AnalysisBatch, Conversation, ConversationAnalysis, Message, Role, User, AIAnalysisJob


def _parse_dt(v: Any) -> datetime | None:
    if not v:
        return None
    if isinstance(v, datetime):
        return v
    if isinstance(v, (int, float)):
        # assume unix seconds
        try:
            return datetime.utcfromtimestamp(float(v))
        except Exception:
            return None
    if isinstance(v, str):
        s = v.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(s)
        except Exception:
            return None
    return None


def _to_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    return str(v).strip()


def _normalize_multi(v: Any) -> str:
    """Excel规范：多选用 ';' 分隔。"""
    if not v:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, (list, tuple, set)):
        return ";".join([str(x).strip() for x in v if str(x).strip()])
    return str(v).strip()


def _format_tag_parsing(v: Any) -> str:
    """Excel规范：标签A$$$原因A&&&标签B$$$原因B"""
    if not v:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, dict):
        parts = []
        for k, reason in v.items():
            tag = str(k).strip()
            if not tag:
                continue
            parts.append(f"{tag}$$${_to_text(reason)}")
        return "&&&".join(parts)
    if isinstance(v, list):
        parts = []
        for item in v:
            if isinstance(item, dict):
                tag = _to_text(item.get("tag") or item.get("标签") or "")
                reason = _to_text(item.get("reason") or item.get("解析") or item.get("原因") or "")
                if tag:
                    parts.append(f"{tag}$$${reason}")
        return "&&&".join(parts)
    return str(v).strip()


def _resolve_agent_user_id(session: Session, platform: str, agent_account: str) -> int | None:
    # 只返回“仍然有效”的绑定：绑定存在 + 目标站内账号未被删除/停用
    if not platform or not agent_account:
        return None
    platform_norm = (platform or "").strip().lower()
    agent_account_norm = (agent_account or "").strip()
    b = session.exec(
        select(AgentBinding).where(AgentBinding.platform == platform_norm, AgentBinding.agent_account == agent_account_norm)
    ).first()
    if not b:
        return None
    u = session.get(User, b.user_id)
    if not u or getattr(u, "is_active", True) is False:
        return None
    return b.user_id



def _get_or_create_agent(session: Session, agent_email: str, agent_name: str | None = None) -> User | None:
    """兼容老数据：如果 JSON 里只有 agent_email，则创建/获取一个站内客服账号。"""
    if not agent_email:
        return None
    u = get_user_by_email(session, agent_email)
    if u:
        return u

    # Generate a username from email local part, keep it unique.
    base = (agent_email.split("@")[0] or "agent").strip() or "agent"
    cand = base
    i = 1
    while get_user_by_username(session, cand):
        i += 1
        cand = f"{base}{i}"
    u = User(
        username=cand,
        email=agent_email,
        name=agent_name or agent_email.split("@")[0],
        role=Role.agent,
        password_hash=hash_password("temp123"),
    )
    session.add(u)
    session.flush()
    return u


def _get_or_create_conversation(session: Session, external_id: str, platform: str = "unknown") -> Conversation:
    c = session.exec(select(Conversation).where(Conversation.external_id == external_id)).first()
    if c:
        return c
    c = Conversation(external_id=external_id, platform=platform)
    session.add(c)
    session.flush()
    return c


def import_json_batch(
    session: Session,
    raw_text: str,
    source_filename: str,
    imported_by_user_id: int | None,
    create_jobs_if_missing: bool = True,
) -> Tuple[AnalysisBatch, Dict[str, Any]]:
    """导入 AI 输出 JSON(txt)。

    目标：完全对齐《AI输出.xlsx》14列（在 analysis 字段里）。

    兼容输入：
    - conversations[].analysis.{对话类型/售前正面评价标签/.../标签更新建议}
    - 也兼容老字段 overall_score/sentiment/issue_level 等

    对话基础字段：
    - conversations[].buyer_id / agent_account / uploaded_at
    """

    try:
        payload = json.loads(raw_text)
    except Exception as e:
        return None, {"ok": False, "error": f"Invalid JSON: {e}"}  # type: ignore

    batch = AnalysisBatch(
        source_filename=source_filename,
        imported_by_user_id=imported_by_user_id,
        batch_meta=payload.get("batch_meta") or {},
        raw_json=payload,
    )
    session.add(batch)
    session.flush()

    convs = payload.get("conversations") or []
    imported = 0
    skipped = 0

    for item in convs:
        ext_id = _to_text(item.get("conversation_id") or item.get("聊天记录编号") or item.get("id"))
        if not ext_id:
            skipped += 1
            continue

        platform = _to_text(item.get("platform") or item.get("平台") or batch.batch_meta.get("platform") or "unknown")

        conv = _get_or_create_conversation(session, ext_id, platform)

        # 对话库（基础字段）
        conv.platform = platform or conv.platform
        conv.uploaded_at = _parse_dt(item.get("uploaded_at") or item.get("上传日期")) or conv.uploaded_at
        conv.buyer_id = _to_text(item.get("buyer_id") or item.get("买家ID")) or conv.buyer_id
        conv.agent_account = _to_text(item.get("agent_account") or item.get("客服账号") or item.get("当日涉及客服账号")) or conv.agent_account
        conv.started_at = _parse_dt(item.get("started_at")) or conv.started_at
        conv.ended_at = _parse_dt(item.get("ended_at")) or conv.ended_at
        if isinstance(item.get("meta"), dict):
            conv.meta = {**(conv.meta or {}), **item.get("meta")}

        # 绑定站内客服（优先使用“主管绑定的外部账号映射”）
        resolved_uid = _resolve_agent_user_id(session, conv.platform, conv.agent_account)
        if resolved_uid:
            conv.agent_user_id = resolved_uid
        else:
            # 兼容旧字段 agent_email
            agent_email = _to_text(item.get("agent_email"))
            agent_name = _to_text(item.get("agent_name")) or None
            agent = _get_or_create_agent(session, agent_email, agent_name) if agent_email else None
            if agent and not conv.agent_user_id:
                conv.agent_user_id = agent.id

        # Messages（可选）
        existing_mids: set[str] = set()
        try:
            rows = session.exec(
                select(Message.external_message_id).where(
                    Message.conversation_id == int(conv.id),  # type: ignore[arg-type]
                    Message.external_message_id.is_not(None),
                )
            ).all()
            for x in rows:
                sx = str(x or "").strip()
                if sx:
                    existing_mids.add(sx)
        except Exception:
            existing_mids = set()

        for m in (item.get("messages") or []):
            attachments = m.get("attachments")
            if not attachments:
                # compat: images/files arrays
                images = m.get("images") or []
                files = m.get("files") or []
                attachments = []
                for it in images:
                    attachments.append({"type": "image", "url": _to_text(it)})
                for it in files:
                    attachments.append({"type": "file", "url": _to_text(it)})
            sender = _to_text(m.get("sender") or "unknown")
            ext_mid = _to_text(m.get("external_message_id") or m.get("message_id"))
            if ext_mid and ext_mid in existing_mids:
                continue
            agent_acc = _to_text(m.get("agent_account") or m.get("assistant_id"))
            agent_nick = _to_text(m.get("agent_nick") or m.get("assistant_nick"))

            agent_user_id = None
            if sender == "agent" and agent_acc:
                b = session.exec(
                    select(AgentBinding).where(
                        AgentBinding.platform == _to_text(conv.platform).lower(),
                        AgentBinding.agent_account == agent_acc,
                    )
                ).first()
                if b:
                    agent_user_id = int(b.user_id)

            session.add(
                Message(
                    conversation_id=conv.id,
                    sender=sender,
                    ts=_parse_dt(m.get("ts")),
                    text=_to_text(m.get("text")),
                    attachments=attachments or [],
                    external_message_id=ext_mid or None,
                    agent_account=agent_acc if sender == "agent" else "",
                    agent_nick=agent_nick if sender == "agent" else "",
                    agent_user_id=agent_user_id if sender == "agent" else None,
                )
            )
            if ext_mid:
                existing_mids.add(ext_mid)

        analysis = item.get("analysis") or item.get("AI分析") or {}

        # If analysis is missing, optionally create a background job and continue.
        if not analysis:
            if create_jobs_if_missing:
                exists_job = session.exec(select(AIAnalysisJob).where(AIAnalysisJob.conversation_id == conv.id, AIAnalysisJob.status.in_(["pending","running"])) ).first()
                if not exists_job:
                    session.add(AIAnalysisJob(conversation_id=conv.id))
            imported += 1
            continue

        ca = ConversationAnalysis(
            batch_id=batch.id,
            conversation_id=conv.id,
            # === 14列对齐 ===
            dialog_type=_to_text(analysis.get("对话类型") or analysis.get("dialog_type") or ""),
            pre_positive_tags=_normalize_multi(analysis.get("售前正面评价标签") or analysis.get("pre_positive_tags")),
            after_positive_tags=_normalize_multi(analysis.get("售后正面评价标签") or analysis.get("after_positive_tags")),
            pre_negative_tags=_normalize_multi(analysis.get("售前负面评价标签") or analysis.get("pre_negative_tags")),
            after_negative_tags=_normalize_multi(analysis.get("售后负面评价标签") or analysis.get("after_negative_tags")),
            day_summary=_to_text(analysis.get("当日对话summary") or analysis.get("day_summary")),
            tag_parsing=_format_tag_parsing(analysis.get("评价解析") or analysis.get("tag_parsing")),
            product_suggestion=_to_text(analysis.get("商品提升建议") or analysis.get("product_suggestion")),
            service_suggestion=_to_text(analysis.get("服务提升建议") or analysis.get("service_suggestion")),
            pre_rule_update=_to_text(analysis.get("售前规则更新建议") or analysis.get("pre_rule_update")),
            after_rule_update=_to_text(analysis.get("售后规则更新建议") or analysis.get("after_rule_update")),
            tag_update_suggestion=_normalize_multi(analysis.get("标签更新建议") or analysis.get("tag_update_suggestion")),
            # === 兼容字段 ===
            overall_score=analysis.get("overall_score"),
            sentiment=_to_text(analysis.get("sentiment")),
            issue_level=_to_text(analysis.get("issue_level")),
            problem_types=analysis.get("problem_types") or [],
            flag_for_review=bool(analysis.get("flag_for_review") or False),
            evidence={
                "highlights": analysis.get("highlights") or analysis.get("问题定位") or [],
                "customer_issue_highlights": analysis.get("客服问题对话识别") or analysis.get("customer_issue_highlights") or [],
                "must_read_highlights": analysis.get("需要阅读的地方") or analysis.get("must_read_highlights") or [],
                "raw": analysis,
            },
            extra={k: v for k, v in analysis.items() if k not in {
                "对话类型","dialog_type",
                "售前正面评价标签","pre_positive_tags",
                "售后正面评价标签","after_positive_tags",
                "售前负面评价标签","pre_negative_tags",
                "售后负面评价标签","after_negative_tags",
                "当日对话summary","day_summary",
                "评价解析","tag_parsing",
                "商品提升建议","product_suggestion",
                "服务提升建议","service_suggestion",
                "售前规则更新建议","pre_rule_update",
                "售后规则更新建议","after_rule_update",
                "标签更新建议","tag_update_suggestion",
                "overall_score","sentiment","issue_level","problem_types","flag_for_review",
            }},
        )

        # “有问题的质检”默认规则（可后续配置）：
        # - 有任一负面标签 / 规则更新建议 / 商品/服务建议 / 或 flag_for_review
        if (
            ca.pre_negative_tags
            or ca.after_negative_tags
            or ca.pre_rule_update
            or ca.after_rule_update
            or ca.product_suggestion
            or ca.service_suggestion
        ):
            ca.flag_for_review = True

        session.add(ca)
        imported += 1

    session.commit()
    return batch, {"ok": True, "batch_id": batch.id, "imported": imported, "skipped": skipped}
