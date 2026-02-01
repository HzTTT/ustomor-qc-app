from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Tuple

from sqlmodel import Session, select

from models import Conversation, Message, AgentBinding, User


_TZ_SH = timezone(timedelta(hours=8))


def _to_shanghai_naive(dt: datetime | None) -> datetime | None:
    """Normalize any datetime into *naive* Shanghai local time.

    Why: our templates render time with strftime directly (no timezone conversion).
    Postgres columns here are timestamp WITHOUT time zone, so passing an aware
    datetime will be converted to UTC then stored as naive, causing an -8h shift.
    """
    if not dt:
        return None
    try:
        return dt.astimezone(_TZ_SH).replace(tzinfo=None)
    except Exception:
        # best-effort
        return dt.replace(tzinfo=None)


def _is_agent_account_bound(session: Session, platform: str, agent_account: str) -> bool:
    """Whether an external agent_account is already bound to an active in-app user."""
    platform_norm = (platform or "").strip().lower()
    agent_account_norm = (agent_account or "").strip()
    if not platform_norm or not agent_account_norm:
        return False

    b = session.exec(
        select(AgentBinding).where(
            AgentBinding.platform == platform_norm,
            AgentBinding.agent_account == agent_account_norm,
        )
    ).first()
    if not b:
        return False
    u = session.get(User, b.user_id)
    if not u or getattr(u, "is_active", True) is False:
        return False
    return True



def _lookup_bound_user_id(session: Session, platform: str, agent_account: str) -> int | None:
    platform_norm = (platform or "").strip().lower()
    agent_account_norm = (agent_account or "").strip()
    if not platform_norm or not agent_account_norm:
        return None
    b = session.exec(
        select(AgentBinding).where(
            AgentBinding.platform == platform_norm,
            AgentBinding.agent_account == agent_account_norm,
        )
    ).first()
    if not b:
        return None
    u = session.get(User, b.user_id)
    if not u or getattr(u, "is_active", True) is False:
        return None
    return int(b.user_id)




def _parse_dt(v: Any) -> datetime | None:
    if not v:
        return None
    if isinstance(v, datetime):
        return v
    if isinstance(v, (int, float)):
        try:
            return datetime.fromtimestamp(float(v), tz=timezone.utc)
        except Exception:
            return None
    if isinstance(v, str):
        s = v.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dtv = datetime.fromisoformat(s)
            # normalize naive -> Shanghai
            if dtv.tzinfo is None:
                dtv = dtv.replace(tzinfo=_TZ_SH)
            return dtv
        except Exception:
            return None
    return None


def _safe_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return str(v)


def _sender_map(sent_by: str) -> str:
    s = (sent_by or "").upper()
    if s == "BUYER":
        return "buyer"
    if s == "ASSISTANT":
        return "agent"
    return "system"


def _parse_body(body: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Return (text, attachments)."""
    if not body:
        return "", []
    try:
        data = json.loads(body)
    except Exception:
        # sometimes body might be plain text
        return _safe_text(body).strip(), []

    fmt = (data.get("format") or "").upper()
    if fmt == "TEXT":
        return (_safe_text(data.get("content")).strip(), [])
    if fmt == "IMAGE":
        url = _safe_text(data.get("image_url")).strip()
        if url:
            return ("", [{"type": "image", "url": url}])
        return ("", [])
    if fmt == "SYSTEM_CARD":
        # keep as attachment so frontend can show it later if needed
        att = {
            "type": "system_card",
            "card_type": _safe_text(data.get("card_type")).strip(),
            "summary": _safe_text(data.get("summary")).strip(),
        }
        # keep any other keys
        for k, v in data.items():
            if k in ("format", "card_type", "summary"):
                continue
            att[k] = v
        return ("", [att])

    # unknown formats: keep best-effort
    text = _safe_text(data.get("content")).strip() if isinstance(data, dict) else ""
    atts: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        url = _safe_text(data.get("image_url")).strip()
        if url:
            atts.append({"type": "image", "url": url})
    return (text, atts)


def _get_or_create_conversation(
    session: Session,
    *,
    external_id: str,
    platform: str,
    buyer_id: str,
    agent_account: str,
    started_at: datetime | None,
    ended_at: datetime | None,
    meta: Dict[str, Any],
) -> Conversation:
    c = session.exec(select(Conversation).where(Conversation.external_id == external_id)).first()
    if c:
        # update window if we got wider range
        changed = False
        if started_at and (c.started_at is None or started_at < c.started_at):
            c.started_at = started_at
            changed = True
        if ended_at and (c.ended_at is None or ended_at > c.ended_at):
            c.ended_at = ended_at
            changed = True
        if meta:
            # merge shallowly
            c.meta = {**(c.meta or {}), **meta}
            changed = True
        if changed:
            session.add(c)
            session.flush()
        return c

    c = Conversation(
        external_id=external_id,
        platform=platform,
        buyer_id=buyer_id or "",
        agent_account=agent_account or "",
        started_at=started_at,
        ended_at=ended_at,
        meta=meta or {},
    )
    session.add(c)
    session.flush()
    return c


def import_leyan_jsonl(
    session: Session,
    raw_text: str,
    source_filename: str,
    platform: str = "taobao",
) -> Dict[str, Any]:
    """Import 乐言导出的聊天消息(JSON Lines) into Conversation/Message.

    We create one Conversation per (buyer_id, date) by default:
      external_id = f"{platform}:{seller_id}:{buyer_id}:{YYYY-MM-DD}"

    This keeps daily files small and matches your "daily fetch" workflow.
    """
    lines = [ln for ln in (raw_text or "").splitlines() if ln.strip()]
    if not lines:
        return {"ok": False, "error": "empty file", "source": source_filename}

    imported_msgs = 0
    imported_convs = 0
    errors = 0

    agent_accounts_seen: set[str] = set()
    unbound_agent_accounts: set[str] = set()
    # 记录未绑定客服账号的昵称信息：{account: nick}
    unbound_agent_nicks: Dict[str, str] = {}

    counts_by_date: Dict[str, Dict[str, int]] = {}

    # group messages by (seller_id, buyer_id, date)
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for ln in lines:
        try:
            obj = json.loads(ln)
        except Exception:
            errors += 1
            continue

        buyer_id = _safe_text(obj.get("leyan_buyer_id") or "").strip()
        seller_id = _safe_text(obj.get("seller_id") or obj.get("leyan_seller_id") or "").strip()
        sent_at = _parse_dt(obj.get("sent_at"))
        if not sent_at:
            errors += 1
            continue
        date_str = sent_at.astimezone(_TZ_SH).date().isoformat()

        k = (seller_id, buyer_id, date_str)
        groups.setdefault(k, []).append(obj)
        c = counts_by_date.setdefault(date_str, {"conversations": 0, "messages": 0})
        c["messages"] += 1

    for (seller_id, buyer_id, date_str), msgs in groups.items():
        # compute started/ended
        dts = []
        for m in msgs:
            ts = _parse_dt(m.get("sent_at"))
            if ts:
                dts.append(ts)
        started_at = _to_shanghai_naive(min(dts)) if dts else None
        ended_at = _to_shanghai_naive(max(dts)) if dts else None

        # choose a *primary* agent account (same conversation may involve multiple agents)
        agent_counts: Dict[str, int] = {}
        agent_nicks: Dict[str, str] = {}
        for mm in msgs:
            aid = _safe_text(mm.get("assistant_id") or "").strip()
            if not aid:
                continue
            agent_counts[aid] = agent_counts.get(aid, 0) + 1
            an = _safe_text(mm.get("assistant_nick") or "").strip()
            if an and aid not in agent_nicks:
                agent_nicks[aid] = an

        if agent_counts:
            # primary = most messages; tie-break by account string for determinism
            agent_account = sorted(agent_counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
        else:
            agent_account = _safe_text((msgs[0].get("assistant_id") or "")).strip()

        assistant_nick = _safe_text(agent_nicks.get(agent_account) or (msgs[0].get("assistant_nick") or "")).strip()
        buyer_nick = _safe_text((msgs[0].get("buyer_nick") or "")).strip()

        if agent_counts:
            for acc in agent_counts.keys():
                agent_accounts_seen.add(acc)
                if not _is_agent_account_bound(session, platform, acc):
                    unbound_agent_accounts.add(acc)
                    # 记录昵称信息
                    nick = agent_nicks.get(acc, "")
                    if nick:
                        unbound_agent_nicks[acc] = nick
        elif agent_account:
            agent_accounts_seen.add(agent_account)
            if not _is_agent_account_bound(session, platform, agent_account):
                unbound_agent_accounts.add(agent_account)
                # 记录昵称信息
                if assistant_nick:
                    unbound_agent_nicks[agent_account] = assistant_nick

        external_id = f"{platform}:{seller_id}:{buyer_id}:{date_str}"
        counts_by_date.setdefault(date_str, {"conversations": 0, "messages": 0})
        counts_by_date[date_str]["conversations"] += 1
        conv = _get_or_create_conversation(
            session,
            external_id=external_id,
            platform=platform,
            buyer_id=buyer_id,
            agent_account=agent_account,
            started_at=started_at,
            ended_at=ended_at,
            meta={
                "source": "leyan",
                "source_filename": source_filename,
                "seller_id": seller_id,
                "assistant_id": agent_account,
                "assistant_nick": assistant_nick,
                "agent_accounts": sorted(list(agent_counts.keys())) if agent_counts else ([agent_account] if agent_account else []),
                "buyer_nick": buyer_nick,
                "date": date_str,
            },
        )
        imported_convs += 1

        # deterministic ordering
        msgs.sort(key=lambda x: (_parse_dt(x.get("sent_at")) or datetime.min.replace(tzinfo=_TZ_SH), _safe_text(x.get("message_id"))))

        for m in msgs:
            ts = _parse_dt(m.get("sent_at"))
            sender = _sender_map(_safe_text(m.get("sent_by")))
            text, atts = _parse_body(_safe_text(m.get("body")))

            # include message_id in attachments meta for traceability (no schema change)
            mid = _safe_text(m.get("message_id")).strip()
            if mid:
                atts = (atts or []) + [{"type": "meta", "message_id": mid}]

            # carry refs (order sku etc.) into attachments meta
            refs = {
                "ref_trade_tid": m.get("ref_trade_tid"),
                "ref_order_oid": m.get("ref_order_oid"),
                "ref_num_iid": m.get("ref_num_iid"),
                "ref_sku_id": m.get("ref_sku_id"),
            }
            if any(v for v in refs.values()):
                atts = (atts or []) + [{"type": "refs", **refs}]

            mid = _safe_text(m.get("message_id")).strip()
            agent_acc = _safe_text(m.get("assistant_id")).strip()
            agent_n = _safe_text(m.get("assistant_nick")).strip()
            bound_uid = _lookup_bound_user_id(session, platform, agent_acc) if (sender == 'agent' and agent_acc) else None

            msg_row = Message(
                conversation_id=conv.id,  # type: ignore[arg-type]
                sender=sender,
                ts=_to_shanghai_naive(ts),
                text=text or "",
                external_message_id=mid or None,
                agent_account=(agent_acc if sender == 'agent' else ""),
                agent_nick=(agent_n if sender == 'agent' else ""),
                agent_user_id=bound_uid,
                attachments=atts or [],
            )
            session.add(msg_row)
            imported_msgs += 1

    session.commit()

    dates = sorted(counts_by_date.keys())
    return {
        "ok": True,
        "source": source_filename,
        "platform": platform,
        "conversations": len(groups),
        "messages": imported_msgs,
        "errors": errors,
        "dates": dates,
        "counts_by_date": counts_by_date,
        "agent_accounts": sorted([x for x in agent_accounts_seen if x]),
        "unbound_agent_accounts": sorted([x for x in unbound_agent_accounts if x]),
        "unbound_agent_nicks": unbound_agent_nicks,  # 新增：未绑定账号的昵称映射
    }
