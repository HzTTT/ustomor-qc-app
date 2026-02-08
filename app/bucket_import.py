from __future__ import annotations

import gzip
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from sqlmodel import Session, select

from importers.json_import import import_json_batch
from importers.leyan_import import import_leyan_jsonl
from models import BucketObject, ImportRun

if TYPE_CHECKING:
    from sqlmodel import Session

try:  # pragma: no cover
    from platform_detect import detect_platform_from_leyan_jsonl  # type: ignore
except Exception:  # pragma: no cover
    from app.platform_detect import detect_platform_from_leyan_jsonl  # type: ignore


@dataclass
class BucketItem:
    key: str
    etag: str
    size: int
    last_modified: Optional[datetime]


@dataclass
class BucketConfig:
    """S3-compatible bucket config.

    Works for:
    - Aliyun OSS (S3-compatible endpoint)
    - AWS S3 (and other S3-compatible providers)
    """
    bucket: str
    prefix: str = ""
    endpoint: str = ""
    region: str = ""
    access_key: str = ""
    secret_key: str = ""


def get_bucket_config(source: str = "DEFAULT", session: "Session | None" = None) -> BucketConfig:
    """Read bucket config. AppConfig (when session given) overrides env."""
    from app_config import get_bucket_config_dict

    d = get_bucket_config_dict(source, session)
    return BucketConfig(
        bucket=d.get("bucket") or "",
        prefix=d.get("prefix") or "",
        endpoint=d.get("endpoint") or "",
        region=d.get("region") or "",
        access_key=d.get("access_key") or "",
        secret_key=d.get("secret_key") or "",
    )


def _get_s3_client(cfg: BucketConfig) -> Any:
    # Aliyun OSS (S3-compatible) often requires "virtual hosted" style:
    #   https://<bucket>.<endpoint>/<key>
    # Path-style (https://<endpoint>/<bucket>/<key>) may be rejected with:
    #   SecondLevelDomainForbidden / Please use virtual hosted style to access.
    if not cfg.endpoint and not (cfg.access_key and cfg.secret_key):
        # allow AWS default chain if user runs on AWS
        return boto3.client(
            "s3",
            region_name=cfg.region or None,
            config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
        )

    if not (cfg.endpoint and cfg.access_key and cfg.secret_key):
        raise ValueError("Missing S3 config: endpoint/access_key/secret_key")

    return boto3.client(
        "s3",
        endpoint_url=cfg.endpoint,
        region_name=cfg.region or None,
        aws_access_key_id=cfg.access_key,
        aws_secret_access_key=cfg.secret_key,
        config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
    )


def _accept_key(key: str) -> bool:
    low = (key or "").lower()
    if low.endswith("/"):
        return False
    # 乐言通常是 jsonl.gz；也兼容 json/jsonl/txt 以及 gzip 压缩
    return any(
        low.endswith(suf)
        for suf in (
            ".json",
            ".jsonl",
            ".txt",
            ".json.gz",
            ".jsonl.gz",
            ".txt.gz",
            ".gz",  # last resort, we'll try to decode
        )
    )


def _list_bucket_items(s3: Any, bucket: str, prefix: str) -> List[BucketItem]:
    items: List[BucketItem] = []
    token: Optional[str] = None
    while True:
        kwargs: Dict[str, Any] = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents") or []:
            key = obj.get("Key") or ""
            if not key or not _accept_key(key):
                continue
            size = int(obj.get("Size") or 0)
            if size <= 0:
                continue
            etag = (obj.get("ETag") or "").strip('"')
            lm = obj.get("LastModified")
            last_modified = lm if isinstance(lm, datetime) else None
            items.append(BucketItem(key=key, etag=etag, size=size, last_modified=last_modified))

        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
            continue
        break

    items.sort(key=lambda x: (x.last_modified or datetime.min, x.key))
    return items


def _safe_join_backup(backup_root: Path, key: str) -> Path:
    key = (key or "").lstrip("/")
    key = key.replace("..", "_")
    return backup_root / key


def _safe_dir_name(v: str, default: str) -> str:
    """Sanitize user-provided strings for filesystem directory names."""
    s = (v or "").strip().strip("/")
    if not s:
        return default
    s = s.replace("..", "_")
    s = s.replace("/", "_").replace("\\", "_")
    return s or default


def _decode_object_bytes(key: str, body: bytes) -> str:
    low = (key or "").lower()
    if low.endswith(".gz"):
        try:
            return gzip.decompress(body).decode("utf-8", errors="replace")
        except Exception:
            # some .gz might still be plain text (rare)
            return body.decode("utf-8", errors="replace")
    return body.decode("utf-8", errors="replace")


def _choose_importer(key: str) -> str:
    """Return 'leyan' or 'ai'."""
    low = (key or "").lower()
    if low.endswith(".jsonl.gz") or low.endswith(".jsonl"):
        return "leyan"
    # Heuristic: anything under /leyan/ is treated as 乐言聊天日志
    if "/leyan/" in low or low.startswith("leyan/"):
        return "leyan"
    return "ai"

def sync_bucket_once(
    session: Session,
    *,
    cfg: Optional[BucketConfig] = None,
    bucket: Optional[str] = None,
    prefix: Optional[str] = None,
    imported_by_user_id: Optional[int] = None,
    create_jobs_if_missing: bool = False,
    platform: str = "taobao",
) -> Dict[str, Any]:
    """Scan bucket for new/updated objects, download+backup, import into DB.

    De-dup strategy: if we already have a row with same (key, etag) we skip.
    """

    if cfg is None:
        cfg = get_bucket_config("DEFAULT")
        # allow override by args for backward compatibility
        if bucket is not None:
            cfg.bucket = bucket
        if prefix is not None:
            cfg.prefix = prefix
    else:
        if bucket is not None:
            cfg.bucket = bucket
        if prefix is not None:
            cfg.prefix = prefix

    from app_config import get_bucket_backup_dir
    backup_dir = Path(get_bucket_backup_dir(session) or "/data/bucket_backup")
    backup_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.bucket:
        return {"ok": False, "error": "bucket name is empty"}

    s3 = _get_s3_client(cfg)
    items = _list_bucket_items(s3, bucket=cfg.bucket, prefix=cfg.prefix or "")

    processed = 0
    imported = 0
    skipped = 0
    errors = 0

    # For notifications (e.g., "new unbound agent accounts") we aggregate
    # importer-level signals across all processed objects in this run.
    agent_accounts_seen: set[str] = set()
    unbound_agent_accounts: set[str] = set()
    unbound_agent_nicks: Dict[str, str] = {}  # account -> nick mapping

    # mixed-platform support (same bucket/prefix may contain multiple platforms)
    agent_accounts_seen_by_platform: Dict[str, set[str]] = {}
    unbound_agent_accounts_by_platform: Dict[str, set[str]] = {}
    unbound_agent_nicks_by_platform: Dict[str, Dict[str, str]] = {}
    imported_files_by_platform: Dict[str, int] = {}
    platform_overrides: List[Dict[str, str]] = []

    for it in items:
        exists = session.exec(
            select(BucketObject).where(BucketObject.bucket == cfg.bucket, BucketObject.key == it.key, BucketObject.etag == it.etag)
        ).first()
        if exists:
            skipped += 1
            continue

        row = BucketObject(
            bucket=cfg.bucket,
            key=it.key,
            etag=it.etag,
            size=it.size,
            last_modified=it.last_modified,
            status="new",
        )
        session.add(row)
        session.commit()
        session.refresh(row)

        try:
            resp = s3.get_object(Bucket=cfg.bucket, Key=it.key)
            body = resp["Body"].read()

            raw_text = _decode_object_bytes(it.key, body)
            importer = _choose_importer(it.key)

            object_platform = platform
            if importer == "leyan":
                detected = detect_platform_from_leyan_jsonl(raw_text, key=it.key)
                if detected and detected != object_platform:
                    object_platform = detected
                    if len(platform_overrides) < 50:
                        platform_overrides.append({"key": it.key, "from": platform, "to": object_platform})

            # backup (raw bytes)
            # Put each platform/bucket under its own folder to avoid collisions when
            # different buckets share the same key layout (e.g. both use leyan/YYYY-MM-DD/*.jsonl.gz).
            backup_base = backup_dir / _safe_dir_name(object_platform, "unknown") / _safe_dir_name(cfg.bucket, "bucket")
            backup_path = _safe_join_backup(backup_base, it.key)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            backup_path.write_bytes(body)

            row.backup_path = str(backup_path)
            row.status = "downloaded"
            row.downloaded_at = datetime.utcnow()
            session.add(row)
            session.commit()

            if importer == "leyan":
                result = import_leyan_jsonl(
                    session=session,
                    raw_text=raw_text,
                    source_filename=f"bucket:{cfg.bucket}/{it.key}",
                    platform=object_platform,
                )
                try:
                    agent_accounts_seen_by_platform.setdefault(object_platform, set())
                    unbound_agent_accounts_by_platform.setdefault(object_platform, set())
                    unbound_agent_nicks_by_platform.setdefault(object_platform, {})

                    for x in (result.get("agent_accounts") or []):
                        sx = str(x)
                        agent_accounts_seen.add(sx)
                        agent_accounts_seen_by_platform[object_platform].add(sx)
                    for x in (result.get("unbound_agent_accounts") or []):
                        sx = str(x)
                        unbound_agent_accounts.add(sx)
                        unbound_agent_accounts_by_platform[object_platform].add(sx)
                    # 收集未绑定账号的昵称信息
                    nicks = result.get("unbound_agent_nicks") or {}
                    if isinstance(nicks, dict):
                        for acc, nick in nicks.items():
                            if str(acc).strip() and str(nick).strip():
                                acc_s = str(acc).strip()
                                nick_s = str(nick).strip()
                                unbound_agent_nicks[acc_s] = nick_s
                                unbound_agent_nicks_by_platform[object_platform][acc_s] = nick_s
                except Exception:
                    pass
            else:
                _batch, result = import_json_batch(
                    session=session,
                    raw_text=raw_text,
                    source_filename=f"bucket:{cfg.bucket}/{it.key}",
                    imported_by_user_id=imported_by_user_id,
                    create_jobs_if_missing=create_jobs_if_missing,
                )

            row.status = "imported" if result.get("ok") else "error"
            row.import_result = result
            row.imported_at = datetime.utcnow()
            session.add(row)
            session.commit()

            # Record daily import status for the admin board
            try:
                dates = result.get("dates") or []
                counts_by_date = result.get("counts_by_date") or {}
                if not dates:
                    # fallback: infer a single date from requested prefix (YYYY-MM-DD/)
                    pfx = (cfg.prefix or "").strip("/").split("/")
                    if pfx and len(pfx[-1]) == 10 and pfx[-1][4] == "-" and pfx[-1][7] == "-":
                        dates = [pfx[-1]]
                for ds in dates:
                    plat = str(result.get("platform") or object_platform or platform or "unknown").strip().lower() or "unknown"
                    ir = session.exec(
                        select(ImportRun).where(
                            ImportRun.platform == plat,
                            ImportRun.run_date == ds,
                            ImportRun.source == "bucket",
                        )
                    ).first()
                    details = counts_by_date.get(ds) or {}
                    conv_inc = int(details.get("conversations", 0)) if details else 0
                    msg_inc = int(details.get("messages", 0)) if details else 0
                    ok = bool(result.get("ok"))
                    files_inc = int(result.get("imported_files") or 0) if isinstance(result, dict) else 0
                    if files_inc <= 0:
                        files_inc = 1 if ok else 0
                    payload = {
                        "bucket": cfg.bucket,
                        "prefix": cfg.prefix or "",
                        "files_imported": files_inc,
                        "conversations": conv_inc,
                        "messages": msg_inc,
                    }
                    if ir:
                        # accumulate counts across multiple files for same (platform, date)
                        prev = ir.details or {}
                        payload["files_imported"] = int(prev.get("files_imported") or 0) + files_inc
                        payload["conversations"] = int(prev.get("conversations") or 0) + conv_inc
                        payload["messages"] = int(prev.get("messages") or 0) + msg_inc

                        if ir.status != "error":
                            ir.status = "done" if ok else "error"
                        elif not ok:
                            ir.status = "error"
                        ir.details = {**prev, **payload}
                        session.add(ir)
                    else:
                        ir = ImportRun(
                            platform=plat,
                            run_date=ds,
                            source="bucket",
                            status="done" if ok else "error",
                            details=payload,
                        )
                        session.add(ir)
                session.commit()
            except Exception:
                # never break the import pipeline because of status board
                session.rollback()

            processed += 1
            if result.get("ok"):
                imported += 1
                imported_files_by_platform[object_platform] = int(imported_files_by_platform.get(object_platform) or 0) + 1
            else:
                errors += 1

        except ClientError as e:
            row.status = "error"
            row.error = str(e)
            row.imported_at = datetime.utcnow()
            session.add(row)
            session.commit()
            processed += 1
            errors += 1
        except Exception as e:
            row.status = "error"
            row.error = str(e)
            row.imported_at = datetime.utcnow()
            session.add(row)
            session.commit()
            processed += 1
            errors += 1

    return {
        "ok": True,
        "bucket": cfg.bucket,
        "prefix": cfg.prefix or "",
        "seen": len(items),
        "processed": processed,
        "imported": imported,
        "skipped": skipped,
        "errors": errors,
        "agent_accounts": sorted([x for x in agent_accounts_seen if x]),
        "unbound_agent_accounts": sorted([x for x in unbound_agent_accounts if x]),
        "unbound_agent_nicks": unbound_agent_nicks,  # 新增：未绑定账号的昵称映射
        # New: mixed-platform support
        "agent_accounts_by_platform": {k: sorted([x for x in v if x]) for k, v in agent_accounts_seen_by_platform.items()},
        "unbound_agent_accounts_by_platform": {k: sorted([x for x in v if x]) for k, v in unbound_agent_accounts_by_platform.items()},
        "unbound_agent_nicks_by_platform": unbound_agent_nicks_by_platform,
        "imported_files_by_platform": imported_files_by_platform,
        "platform_overrides": platform_overrides,
    }
