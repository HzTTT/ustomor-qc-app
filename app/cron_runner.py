from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

from sqlmodel import Session, select

from db import engine
from bucket_import import sync_bucket_once, get_bucket_config, BucketConfig
from analysis_engine import enqueue_missing_analyses
from app_config import is_auto_analysis_enabled, get_app_config
from models import BucketObject, CronState, AIAnalysisJob
from notify import send_feishu_webhook, get_feishu_webhook_url
from qc_checker import get_conversations_eligible_for_qc


_TZ_SH = timezone(timedelta(hours=8))


def _now() -> datetime:
    return datetime.now(tz=_TZ_SH)


def _parse_hhmm(v: str, default: str = "10:15") -> tuple[int, int]:
    s = (v or default).strip()
    if ":" not in s:
        return (10, 15)
    a, b = s.split(":", 1)
    try:
        return (max(0, min(23, int(a))), max(0, min(59, int(b))))
    except Exception:
        return (10, 15)


def _date_prefix(base_prefix: str, date_str: str) -> str:
    # 乐言目录结构：/leyan/YYYY-MM-DD/xxx.jsonl.gz
    bp = (base_prefix or "").strip()
    bp = bp.strip("/")
    if not bp:
        return f"{date_str}/"
    return f"{bp}/{date_str}/"


def _get_state(session: Session, name: str) -> CronState:
    st = session.exec(select(CronState).where(CronState.name == name)).first()
    if st:
        return st
    st = CronState(name=name, state={})
    session.add(st)
    session.commit()
    session.refresh(st)
    return st


def _set_state(session: Session, st: CronState, state: dict) -> None:
    st.state = state
    st.updated_at = datetime.utcnow()
    session.add(st)
    session.commit()


def _should_run_hourly_qc_check(s: dict) -> bool:
    """检查是否应该运行每小时的自动质检检查."""
    last_qc_check = s.get("last_qc_check_at")
    if not last_qc_check:
        return True
    try:
        last_check_dt = datetime.fromisoformat(str(last_qc_check))
        if last_check_dt.tzinfo is None:
            last_check_dt = last_check_dt.replace(tzinfo=_TZ_SH)
        now = _now()
        # 每小时检查一次
        return (now - last_check_dt).total_seconds() >= 3600
    except Exception:
        return True


def _run_auto_qc_check(session: Session, state_dict: dict, log_keep: int = 800) -> dict:
    """执行自动质检检查：查找符合条件的对话并入队.
    
    返回更新后的 state_dict
    """
    try:
        # 获取符合条件的对话（>5条消息且所有客服已绑定）
        eligible = get_conversations_eligible_for_qc(session, min_messages=5, limit=None)
        
        if not eligible:
            state_dict = _append_log_state(
                state_dict,
                "INFO",
                f"自动质检检查：无符合条件的对话",
                keep=log_keep
            )
            state_dict["last_qc_check_at"] = _now().isoformat()
            return state_dict
        
        # 为符合条件的对话创建分析任务
        enqueued = 0
        for item in eligible:
            cid = item["conversation_id"]
            # 检查是否已有任务（pending 或 running）
            existing = session.exec(
                select(AIAnalysisJob).where(
                    AIAnalysisJob.conversation_id == cid,
                    AIAnalysisJob.status.in_(["pending", "running"])
                )
            ).first()
            
            if not existing:
                session.add(AIAnalysisJob(conversation_id=cid))
                enqueued += 1
        
        if enqueued > 0:
            session.commit()
            state_dict = _append_log_state(
                state_dict,
                "INFO",
                f"自动质检检查：发现 {len(eligible)} 个符合条件的对话，入队 {enqueued} 个新任务",
                keep=log_keep
            )
            # 发送飞书通知
            try:
                _notify(
                    session,
                    "✅ 自动质检已启动",
                    f"发现 {len(eligible)} 个符合质检条件的对话\n已为 {enqueued} 个对话创建AI质检任务\n\n条件：消息数>5 且 所有客服已绑定"
                )
            except Exception:
                pass
        else:
            state_dict = _append_log_state(
                state_dict,
                "INFO",
                f"自动质检检查：发现 {len(eligible)} 个符合条件的对话，但都已有任务在队列中",
                keep=log_keep
            )
        
        state_dict["last_qc_check_at"] = _now().isoformat()
        state_dict["last_qc_check_eligible"] = len(eligible)
        state_dict["last_qc_check_enqueued"] = enqueued
        
    except Exception as e:
        state_dict = _append_log_state(
            state_dict,
            "ERROR",
            f"自动质检检查失败：{str(e)}",
            keep=log_keep
        )
        state_dict["last_qc_check_at"] = _now().isoformat()
    
    return state_dict


def _is_date_imported(session: Session, bucket: str, date_prefix: str) -> bool:
    # any successfully imported object under that prefix counts as "fetched"
    row = session.exec(
        select(BucketObject).where(
            BucketObject.bucket == bucket,
            BucketObject.key.like(f"{date_prefix}%"),
            BucketObject.status == "imported",
        )
    ).first()
    return row is not None


def _notify(session: Session, title: str, text: str) -> None:
    url = get_feishu_webhook_url(session)
    if not url:
        return
    send_feishu_webhook(url, title=title, text=text)


def _append_log_state(s: dict, level: str, msg: str, keep: int = 800) -> dict:
    """Append one log line into a cron state dict (pure function)."""
    s = dict(s or {})
    logs = list(s.get("logs") or [])
    logs.append({"ts": _now().isoformat(), "level": (level or "INFO").upper(), "msg": str(msg or "")})
    if len(logs) > keep:
        logs = logs[-keep:]
    s["logs"] = logs
    return s


def _attempt_fetch_one(session: Session, *, source: str, cfg: BucketConfig, platform: str, date_str: str) -> dict:
    date_pref = _date_prefix(cfg.prefix, date_str)
    # If already imported, we treat as success and avoid re-scan
    if _is_date_imported(session, cfg.bucket, date_pref):
        return {"ok": True, "already": True, "date": date_str, "prefix": date_pref}

    res = sync_bucket_once(session, cfg=cfg, prefix=date_pref, platform=platform, imported_by_user_id=None, create_jobs_if_missing=False)
    res["date"] = date_str
    res["date_prefix"] = date_pref
    res["source"] = source
    res["platform"] = platform
    return res


def main() -> None:
    state_name = "bucket_poll_v1"
    cron_sleep = 10  # default if session fails

    while True:
        try:
            with Session(engine) as session:
                cfg_row = get_app_config(session)
                # runtime settings from AppConfig
                (hh, mm) = _parse_hhmm(cfg_row.bucket_daily_check_time or "10:15", default="10:15")
                retry_minutes = max(5, int(getattr(cfg_row, "bucket_retry_interval_minutes", 60) or 60))
                log_keep = max(100, min(5000, int(getattr(cfg_row, "bucket_log_keep", 800) or 800)))
                cron_sleep = max(10, int(getattr(cfg_row, "cron_interval_seconds", 600) or 600))

                # which sources to import (from AppConfig / settings UI)
                sources: list[tuple[str, str]] = []  # (SOURCE_PREFIX, platform)
                if bool(getattr(cfg_row, "taobao_bucket_import_enabled", True)):
                    sources.append(("TAOBAO", "taobao"))
                if bool(getattr(cfg_row, "douyin_bucket_import_enabled", False)):
                    sources.append(("DOUYIN", "douyin"))

                st = _get_state(session, state_name)
                s = dict(st.state or {})

                now = _now()
                yesterday = (now - timedelta(days=1)).date().isoformat()

                pending_date = s.get("pending_date") or ""
                next_attempt_at = s.get("next_attempt_at")  # iso str
                last_daily_mark = s.get("last_daily_mark") or ""  # YYYY-MM-DD of the "yesterday" chosen that morning

                # if we have no pending date, decide whether it's time to create one
                today_run = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
                if not pending_date:
                    # allow disabling cron polling from UI
                    if not bool(getattr(cfg_row, "bucket_fetch_enabled", True)):
                        # sleep a bit; still keep process alive (manual triggers happen in web app)
                        time.sleep(10)
                        continue

                    if now >= today_run and last_daily_mark != yesterday:
                        # create a new pending date (yesterday)
                        pending_date = yesterday
                        s["pending_date"] = pending_date
                        s["last_daily_mark"] = yesterday
                        # immediate attempt
                        s["next_attempt_at"] = now.isoformat()
                        next_attempt_at = s["next_attempt_at"]
                        s = _append_log_state(s, "INFO", f"开始轮巡：目标日期 {pending_date}，来源 {', '.join([x[0] for x in sources]) or '（未启用）'}", keep=log_keep)
                        _set_state(session, st, s)
                        _notify(
                            session,
                            "聊天记录抓取：开始轮巡",
                            f"目标日期：{pending_date}\n来源：{', '.join([x[0] for x in sources]) or '（未启用）'}",
                        )

                # if still no pending_date (e.g. before daily time), sleep until daily time
                if not pending_date:
                    sleep_to = today_run if now < today_run else (today_run + timedelta(days=1))
                    delta = max(5, int((sleep_to - now).total_seconds()))
                    time.sleep(min(delta, 300))  # wake up at least every 5min
                    continue

                # respect next_attempt_at if set
                if next_attempt_at:
                    try:
                        nta = datetime.fromisoformat(str(next_attempt_at))
                        if nta.tzinfo is None:
                            nta = nta.replace(tzinfo=_TZ_SH)
                    except Exception:
                        nta = now
                else:
                    nta = now

                if now < nta:
                    time.sleep(min(int((nta - now).total_seconds()), 300))
                    continue

                # run import attempts
                if not sources:
                    # nothing enabled; back off 1h
                    s = _append_log_state(s, "WARN", "未启用任何来源（淘宝/抖音），1小时后再试", keep=log_keep)
                    s["next_attempt_at"] = (_now() + timedelta(hours=1)).isoformat()
                    _set_state(session, st, s)
                    time.sleep(60)
                    continue

                all_ok = True
                any_imported = False
                any_seen = False
                summaries = []

                for (src, platform) in sources:
                    cfg = get_bucket_config(src, session)
                    if not cfg.bucket:
                        all_ok = False
                        summaries.append(f"{src}: bucket 未配置")
                        continue

                    try:
                        res = _attempt_fetch_one(session, source=src, cfg=cfg, platform=platform, date_str=pending_date)

                        # Notify when new (previously unseen) unbound external agent accounts appear.
                        try:
                            unbound = [str(x) for x in (res.get("unbound_agent_accounts") or []) if str(x).strip()]
                            unbound_nicks = res.get("unbound_agent_nicks") or {}
                            if unbound:
                                known_map = dict(s.get("known_unbound_agent_accounts") or {})
                                known_list = list(known_map.get(platform) or [])
                                known_set = set([str(x) for x in known_list if str(x).strip()])
                                new_set = set(unbound) - known_set
                                if new_set:
                                    # record first, so even if webhook fails we won't spam every retry
                                    known_set = known_set.union(new_set)
                                    known_map[platform] = sorted(list(known_set))
                                    s["known_unbound_agent_accounts"] = known_map
                                    # 构建带昵称的通知消息
                                    msg_lines = []
                                    for acc in sorted(list(new_set)):
                                        nick = unbound_nicks.get(acc, "")
                                        if nick:
                                            msg_lines.append(f"• {acc} (昵称: {nick})")
                                        else:
                                            msg_lines.append(f"• {acc}")
                                    _notify(
                                        session,
                                        "⚠️ 发现未绑定客服账号",
                                        f"平台：{platform}\n日期：{pending_date}\n\n未绑定账号（新增）：\n" + "\n".join(msg_lines) + "\n\n请到【设置 > 客服账号绑定】页面进行配置。",
                                    )
                        except Exception:
                            pass
                        if res.get("already"):
                            summaries.append(f"{src}: 已抓取过（{pending_date}）")
                            any_imported = True
                            continue

                        seen = int(res.get("seen") or 0)
                        imported = int(res.get("imported") or 0)
                        errors = int(res.get("errors") or 0)
                        any_seen = any_seen or (seen > 0)
                        any_imported = any_imported or (imported > 0)

                        if seen <= 0:
                            all_ok = False
                            summaries.append(f"{src}: 未发现文件（{pending_date}）")
                        elif errors > 0 and imported <= 0:
                            all_ok = False
                            summaries.append(f"{src}: 发现 {seen} 个文件，但导入失败 {errors} 次")
                        else:
                            summaries.append(f"{src}: 发现 {seen} 个文件，导入成功 {imported}，失败 {errors}")

                    except Exception as e:
                        all_ok = False
                        summaries.append(f"{src}: 异常 {e}")

                # notify + schedule next
                if any_imported:
                    s = _append_log_state(s, "INFO", "抓取成功：" + " | ".join(summaries), keep=log_keep)
                    _notify(session, "📥 聊天记录抓取：成功", "\n".join(summaries))
                    # clear pending
                    s["pending_date"] = ""
                    s["next_attempt_at"] = ""
                    _set_state(session, st, s)
                    
                    # 导入成功后，立即检查是否有符合条件的对话可以质检
                    # （不需要等待下一个小时检查）
                    try:
                        st = _get_state(session, state_name)
                        s = dict(st.state or {})
                        s = _run_auto_qc_check(session, s, log_keep=log_keep)
                        _set_state(session, st, s)
                    except Exception:
                        # 即使检查失败，也不影响导入流程
                        pass
                else:
                    # missing or failed — retry hourly
                    s = _append_log_state(s, "WARN", "未抓到/导入失败（将重试）：" + " | ".join(summaries), keep=log_keep)
                    _notify(session, "聊天记录抓取：未抓到（将每小时重试）", "\n".join(summaries))
                    s["next_attempt_at"] = (_now() + timedelta(minutes=retry_minutes)).isoformat()
                    _set_state(session, st, s)

                # enqueue analysis if enabled
                if bool(getattr(cfg_row, "enable_enqueue_analysis", True)):
                    if is_auto_analysis_enabled(session):
                        enqueue_missing_analyses(session)
                
                # 每小时检查一次：符合条件的对话自动触发AI质检
                # 条件：>5消息 且 所有客服已绑定
                st = _get_state(session, state_name)
                s = dict(st.state or {})
                if _should_run_hourly_qc_check(s):
                    s = _run_auto_qc_check(session, s, log_keep=log_keep)
                    _set_state(session, st, s)

        except Exception:
            # best-effort cron
            pass

        time.sleep(cron_sleep)


if __name__ == "__main__":
    main()
