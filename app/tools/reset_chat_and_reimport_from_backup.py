from __future__ import annotations

import argparse
import gzip
import os
from pathlib import Path
from typing import Iterable, Tuple

from sqlalchemy import text
from sqlmodel import Session, select

from db import engine, create_db_and_tables
from importers.leyan_import import import_leyan_jsonl
from migrate import _backfill_message_agent_user_id, _refresh_conversation_primary_agent
from models import Conversation, Message


LEYAN_SUFFIXES = (
    ".jsonl",
    ".jsonl.gz",
    ".txt",
    ".txt.gz",
)


def _decode_file(p: Path) -> str:
    data = p.read_bytes()
    low = p.name.lower()
    if low.endswith(".gz"):
        try:
            return gzip.decompress(data).decode("utf-8", errors="replace")
        except Exception:
            return data.decode("utf-8", errors="replace")
    return data.decode("utf-8", errors="replace")


def _iter_backup_files(backup_dir: Path) -> Iterable[Path]:
    if not backup_dir.exists():
        return []
    for p in backup_dir.rglob("*"):
        if not p.is_file():
            continue
        low = p.name.lower()
        # 乐言导出通常是 jsonl(.gz) 或 txt(.gz)。
        if any(low.endswith(suf) for suf in LEYAN_SUFFIXES) and ("leyan" in str(p).lower() or low.endswith(".jsonl") or low.endswith(".jsonl.gz")):
            yield p


def _guess_platform(p: Path, default_platform: str) -> str:
    low = str(p).lower()
    if "douyin" in low or "抖音" in low:
        return "douyin"
    if "taobao" in low or "淘宝" in low or "leyan" in low:
        return "taobao"
    return default_platform


def _truncate_chat_tables() -> None:
    # 只清聊天相关数据；不动 user / agentbinding / bucketobject / appconfig
    sql = """
    TRUNCATE TABLE
      conversation,
      message,
      conversationanalysis,
      analysisbatch,
      aianalysisjob,
      trainingtask,
      trainingreflection,
      trainingsimulationmessage,
      trainingsimulation,
      dailyaisummaryjob,
      dailyaisummaryreport
    RESTART IDENTITY CASCADE;
    """
    with engine.begin() as conn:
        conn.execute(text(sql))


def _count_before_after(session: Session) -> Tuple[int, int]:
    conv_cnt = session.exec(select(Conversation.id)).all()
    msg_cnt = session.exec(select(Message.id)).all()
    return (len(conv_cnt), len(msg_cnt))


def main() -> int:
    ap = argparse.ArgumentParser(description="Reset all chat records and re-import from local bucket backups.")
    ap.add_argument("--backup-dir", default=os.getenv("BUCKET_BACKUP_DIR") or "/data/bucket_backup", help="Backup dir that stores downloaded bucket objects")
    ap.add_argument("--default-platform", default="taobao", help="Fallback platform when cannot infer")
    ap.add_argument("--dry-run", action="store_true", help="Only print what would happen")
    args = ap.parse_args()

    create_db_and_tables()

    backup_dir = Path(args.backup_dir)
    files = sorted(list(_iter_backup_files(backup_dir)), key=lambda x: str(x))

    with Session(engine) as session:
        before_conv, before_msg = _count_before_after(session)

    print(f"[reset] current DB: conversations={before_conv} messages={before_msg}")
    print(f"[reset] backup dir: {backup_dir} | files found: {len(files)}")

    if args.dry_run:
        for p in files[:30]:
            plat = _guess_platform(p, args.default_platform)
            print(f"[dry-run] {plat} <- {p}")
        if len(files) > 30:
            print(f"[dry-run] ... +{len(files) - 30} more")
        return 0

    # 1) wipe
    print("[reset] truncating chat tables...")
    _truncate_chat_tables()

    # 2) reimport
    imported_files = 0
    total_msgs = 0
    total_convs = 0
    errors = 0

    with Session(engine) as session:
        for p in files:
            platform = _guess_platform(p, args.default_platform)
            try:
                raw = _decode_file(p)
                res = import_leyan_jsonl(session, raw_text=raw, source_filename=str(p), platform=platform)
                if res.get("ok"):
                    imported_files += 1
                    total_msgs += int(res.get("imported_messages") or 0)
                    total_convs += int(res.get("imported_conversations") or 0)
                else:
                    errors += 1
            except Exception as e:
                errors += 1
                print(f"[reset] ERROR importing {p}: {type(e).__name__}: {e}")

        # 3) ensure agent_user_id is backfilled for bound agent accounts
        try:
            _backfill_message_agent_user_id()
            _refresh_conversation_primary_agent()
        except Exception:
            pass

        after_conv, after_msg = _count_before_after(session)

    print("[reset] done")
    print(f"[reset] imported files={imported_files} errors={errors}")
    print(f"[reset] imported convs={total_convs} msgs={total_msgs}")
    print(f"[reset] DB after: conversations={after_conv} messages={after_msg}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
