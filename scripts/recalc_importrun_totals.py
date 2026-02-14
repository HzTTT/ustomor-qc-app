#!/usr/bin/env python3
"""
Recalculate ImportRun daily totals from DB truth (Conversation/Message tables).

Why:
- ImportRun.details may be inflated if the same day is re-imported (e.g. bucket object ETag changed),
  or if older importer versions reported non-idempotent per-file counts.
- The admin "抓取缺失看板" should reflect DB totals for that day.

Usage:
  python scripts/recalc_importrun_totals.py --from 2026-02-01 --to 2026-02-08
  python scripts/recalc_importrun_totals.py --from 2026-02-01 --to 2026-02-08 --platform taobao,douyin
  python scripts/recalc_importrun_totals.py --from 2026-02-01 --to 2026-02-08 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add app directory to path (兼容：宿主机 repo 根目录运行 / 容器内 /app/scripts 运行)
root_dir = Path(__file__).resolve().parent.parent
app_dir = root_dir / "app"
if app_dir.exists():
    sys.path.insert(0, str(app_dir))
else:
    sys.path.insert(0, str(root_dir))

from sqlalchemy import func  # noqa: E402
from sqlmodel import Session, select  # noqa: E402

from db import engine  # noqa: E402
from models import Conversation, Message, ImportRun  # noqa: E402


def _parse_date(s: str) -> datetime:
    return datetime.fromisoformat(s.strip())


def _daterange(start: datetime, end_inclusive: datetime):
    cur = start
    while cur.date() <= end_inclusive.date():
        yield cur.date().isoformat()
        cur += timedelta(days=1)


def _totals_for_day(session: Session, *, platform: str, run_date: str) -> tuple[int, int]:
    plat = (platform or "").strip().lower() or "unknown"
    day_start = _parse_date(run_date)
    day_end = day_start + timedelta(days=1)
    ts_expr = func.coalesce(Conversation.started_at, Conversation.uploaded_at)

    conv_total = session.exec(
        select(func.count(Conversation.id)).where(
            Conversation.platform == plat,
            ts_expr >= day_start,
            ts_expr < day_end,
        )
    ).one()
    msg_total = session.exec(
        select(func.count(Message.id))
        .select_from(Message)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .where(
            Conversation.platform == plat,
            ts_expr >= day_start,
            ts_expr < day_end,
        )
    ).one()
    return (int(conv_total or 0), int(msg_total or 0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_date", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--to", dest="to_date", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--platform", default="taobao,douyin", help="comma-separated, default: taobao,douyin")
    ap.add_argument("--source", default="bucket", help="ImportRun.source to update, default: bucket")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    start = _parse_date(args.from_date)
    end = _parse_date(args.to_date)
    if end < start:
        start, end = end, start

    platforms = [p.strip().lower() for p in (args.platform or "").split(",") if p.strip()]
    if not platforms:
        platforms = ["taobao", "douyin"]

    updated = 0
    created = 0

    with Session(engine) as session:
        for ds in _daterange(start, end):
            for plat in platforms:
                conv_total, msg_total = _totals_for_day(session, platform=plat, run_date=ds)

                ir = session.exec(
                    select(ImportRun).where(
                        ImportRun.platform == plat,
                        ImportRun.run_date == ds,
                        ImportRun.source == args.source,
                    )
                ).first()

                if not ir:
                    # Only create when there's actual data; otherwise leave it missing.
                    if conv_total <= 0 and msg_total <= 0:
                        continue
                    ir = ImportRun(
                        platform=plat,
                        run_date=ds,
                        source=args.source,
                        status="done",
                        details={"files_imported": 0},
                    )
                    session.add(ir)
                    created += 1

                det = dict(ir.details or {})
                det["conversations"] = int(conv_total)
                det["messages"] = int(msg_total)
                det["recalc_at"] = datetime.utcnow().isoformat()
                ir.details = det
                if ir.status != "error":
                    ir.status = "done" if conv_total > 0 else ir.status
                session.add(ir)
                updated += 1

        if args.dry_run:
            session.rollback()
        else:
            session.commit()

    print(f"[recalc] platforms={platforms} range={start.date().isoformat()}..{end.date().isoformat()} source={args.source}")
    print(f"[recalc] updated={updated} created={created} dry_run={bool(args.dry_run)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

