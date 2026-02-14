from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime

from sqlmodel import Session, select
from sqlalchemy import text

from db import engine
from analysis_engine import (
    claim_one_job,
    analyze_conversation,
    mark_job_done,
    mark_job_error,
)
from daily_summary import (
    claim_one_daily_job,
    mark_daily_job_done,
    mark_daily_job_error,
    generate_daily_summary,
)
from marllen_assistant import (
    claim_one_assistant_job,
    generate_assistant_reply,
    mark_assistant_job_done,
    mark_assistant_job_error,
)
from models import DailyAISummaryReport


WAIT_SCHEMA_DEADLINE = int(os.getenv("WORKER_WAIT_SCHEMA_SECONDS", "120"))


def _wait_for_schema() -> None:
    """Poll until app has run create_all + ensure_schema. Avoids duplicate enum race."""
    deadline = time.time() + WAIT_SCHEMA_DEADLINE
    while time.time() < deadline:
        try:
            with Session(engine) as session:
                session.execute(text('SELECT 1 FROM "user" LIMIT 1'))
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError("worker: schema not ready within deadline (is app running?)")


async def _assistant_job_heartbeat_loop(job_id: int, *, interval_s: float = 10.0) -> None:
    """Keep a DB heartbeat while an assistant job is running.

    Why: we may run multiple worker replicas. Without a heartbeat, a "stale recovery"
    pass can mistakenly mark a long-running but active job as error, which causes
    the reply to be dropped and the UI to look like "没有显示".
    """
    if not job_id:
        return
    interval_s = max(3.0, float(interval_s or 10.0))
    ident = (os.getenv("HOSTNAME") or "").strip() or f"pid:{os.getpid()}"

    while True:
        try:
            with Session(engine) as s:
                # Local imports to avoid circular deps at import time.
                from models import AssistantJob  # type: ignore
                from marllen_assistant import _utc_iso_z

                j = s.get(AssistantJob, int(job_id))
                if not j:
                    return
                st = str(getattr(getattr(j, "status", None), "value", None) or getattr(j, "status", None) or "").strip().lower()
                if st != "running":
                    return

                extra = dict(j.extra or {}) if isinstance(getattr(j, "extra", None), dict) else {}
                extra["heartbeat_at"] = _utc_iso_z()
                extra["claimed_by"] = ident
                j.extra = extra
                s.add(j)
                s.commit()
        except Exception:
            # Best-effort only; never break the worker loop.
            pass

        await asyncio.sleep(interval_s)


async def _run_daily_job(session: Session, job) -> None:
    extra = job.extra or {}
    run_date = (job.run_date or "").strip()
    threshold_messages = int(extra.get("threshold_messages") or 8)
    prompt = str(extra.get("prompt") or "")
    model = str(extra.get("model") or "").strip()
    created_by_user_id = extra.get("created_by_user_id")
    created_by_user_id = int(created_by_user_id) if created_by_user_id not in (None, "") else None

    # Call generator; it will overwrite/update DailyAISummaryReport for that day.
    await generate_daily_summary(
        session,
        run_date=run_date,
        threshold_messages=threshold_messages,
        prompt=prompt,
        model=model,
        created_by_user_id=created_by_user_id,
    )
    # Link report_id if missing (best-effort)
    if not getattr(job, "report_id", None):
        r = session.exec(
            select(DailyAISummaryReport).where(DailyAISummaryReport.run_date == run_date)
        ).first()
        if r and r.id:
            job.report_id = int(r.id)
            session.add(job)
            session.commit()


async def main() -> None:
    _wait_for_schema()
    poll_seconds = 5  # default if session fails

    while True:
        with Session(engine) as session:
            from app_config import get_app_config
            try:
                cfg = get_app_config(session)
                poll_seconds = max(1, int(getattr(cfg, "worker_poll_seconds", 5) or 5))
            except Exception:
                poll_seconds = 5
            # 1) Marllen assistant jobs (floating AI butler)
            ajob = None
            try:
                ajob = claim_one_assistant_job(session)
            except Exception:
                ajob = None

            if ajob:
                hb_task = None
                try:
                    hb_task = asyncio.create_task(_assistant_job_heartbeat_loop(int(ajob.id or 0)))
                    mid = await generate_assistant_reply(session, job=ajob)
                    mark_assistant_job_done(session, ajob, assistant_message_id=mid)
                except Exception as e:
                    mark_assistant_job_error(session, ajob, str(e))
                finally:
                    if hb_task:
                        try:
                            hb_task.cancel()
                            await hb_task
                        except Exception:
                            pass

                await asyncio.sleep(0.2)
                continue

            # 2) Daily summary jobs
            djob = None
            try:
                djob = claim_one_daily_job(session)
            except Exception:
                djob = None

            if djob:
                try:
                    await _run_daily_job(session, djob)
                    mark_daily_job_done(session, djob)
                except Exception as e:
                    mark_daily_job_error(session, djob, str(e))

                await asyncio.sleep(0.2)
                continue

            # 3) Conversation analysis jobs (can be heavy; keep lowest priority)
            job = None
            try:
                job = claim_one_job(session)
            except Exception:
                job = None

            if job:
                overwrite = bool((job.extra or {}).get("overwrite")) if getattr(job, "extra", None) is not None else False
                batch_prefix = "manual-ai" if overwrite else "auto-ai"
                batch_key = f"{batch_prefix}-{datetime.utcnow().date().isoformat()}"

                try:
                    analysis = await analyze_conversation(session, conversation_id=job.conversation_id, batch_key=batch_key)
                    mark_job_done(session, job, batch_id=analysis.batch_id)
                except Exception as e:
                    # After a DB flush error, the session is in a failed state;
                    # rollback before trying to persist the job status.
                    try:
                        session.rollback()
                    except Exception:
                        pass

                    try:
                        mark_job_error(session, job, str(e))
                    except Exception:
                        # Never crash the worker loop; otherwise UI looks "stuck".
                        try:
                            print(f"[worker] failed to mark job error for job_id={getattr(job, 'id', None)}")
                        except Exception:
                            pass

                await asyncio.sleep(0.2)
                continue

        await asyncio.sleep(poll_seconds)


if __name__ == "__main__":
    asyncio.run(main())
