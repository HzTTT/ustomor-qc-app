from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlmodel import Session, select

from models import AppConfig


def get_app_config(session: Session) -> AppConfig:
    """Get or create the single AppConfig row."""
    cfg = session.exec(select(AppConfig).order_by(AppConfig.id.asc()).limit(1)).first()
    if cfg:
        return cfg
    cfg = AppConfig(auto_analysis_enabled=False)
    session.add(cfg)
    session.commit()
    session.refresh(cfg)
    return cfg


def _env(key: str, default: str = "") -> str:
    v = (os.getenv(key) or "").strip()
    return v if v else default


def get_ai_settings(session: Session | None) -> dict[str, Any]:
    """AI settings (OpenAI-compatible). AppConfig first, then env fallback."""
    base_url = _env("OPENAI_BASE_URL", "http://www.marllen.com:4000/v1")
    api_key = _env("OPENAI_API_KEY")
    model = _env("OPENAI_MODEL", "gpt-5.2")
    fallback_model = _env("OPENAI_FALLBACK_MODEL")
    timeout_s = 1800
    try:
        timeout_s = max(60, int(_env("OPENAI_TIMEOUT", "1800") or "1800"))
    except Exception:
        pass
    reasoning_effort = (_env("OPENAI_REASONING_EFFORT", "high") or "high").strip().lower()
    retries = 0
    try:
        retries = max(0, int(_env("OPENAI_RETRIES", "0") or "0"))
    except Exception:
        pass

    if session is not None:
        try:
            cfg = get_app_config(session)
            if (cfg.openai_base_url or "").strip():
                base_url = (cfg.openai_base_url or "").strip().rstrip("/")
            if (cfg.openai_api_key or "").strip():
                api_key = (cfg.openai_api_key or "").strip()
            if (cfg.openai_model or "").strip():
                model = (cfg.openai_model or "").strip()
            if (cfg.openai_fallback_model or "").strip():
                fallback_model = (cfg.openai_fallback_model or "").strip()
            if getattr(cfg, "openai_timeout", None) not in (None, 0):
                timeout_s = max(60, int(cfg.openai_timeout or 1800))
            if getattr(cfg, "openai_retries", None) is not None:
                retries = max(0, int(cfg.openai_retries or 0))
            if (getattr(cfg, "openai_reasoning_effort", None) or "").strip():
                reasoning_effort = (cfg.openai_reasoning_effort or "high").strip().lower()
        except Exception:
            pass

    return {
        "base_url": base_url.rstrip("/") if base_url else "",
        "api_key": api_key or "",
        "model": model or "gpt-5.2",
        "fallback_model": fallback_model or "",
        "timeout_s": float(timeout_s),
        "reasoning_effort": reasoning_effort or "high",
        "retries": retries,
    }


def set_ai_settings(
    session: Session,
    *,
    openai_base_url: str | None = None,
    openai_api_key: str | None = None,
    openai_model: str | None = None,
    openai_fallback_model: str | None = None,
    openai_timeout: int | None = None,
    openai_retries: int | None = None,
    openai_reasoning_effort: str | None = None,
) -> AppConfig:
    cfg = get_app_config(session)
    if openai_base_url is not None:
        cfg.openai_base_url = (openai_base_url or "").strip().rstrip("/")
    if openai_api_key is not None:
        cfg.openai_api_key = (openai_api_key or "").strip()
    if openai_model is not None:
        cfg.openai_model = (openai_model or "").strip() or "gpt-5.2"
    if openai_fallback_model is not None:
        cfg.openai_fallback_model = (openai_fallback_model or "").strip()
    if openai_timeout is not None:
        try:
            cfg.openai_timeout = max(60, int(openai_timeout))
        except Exception:
            pass
    if openai_retries is not None:
        try:
            cfg.openai_retries = max(0, int(openai_retries))
        except Exception:
            pass
    if openai_reasoning_effort is not None:
        cfg.openai_reasoning_effort = (openai_reasoning_effort or "high").strip().lower() or "high"
    cfg.updated_at = datetime.utcnow()
    session.add(cfg)
    session.commit()
    session.refresh(cfg)
    return cfg


def get_feishu_webhook_url(session: Session | None) -> str:
    if session is not None:
        try:
            cfg = get_app_config(session)
            if (getattr(cfg, "feishu_webhook_url", None) or "").strip():
                return (cfg.feishu_webhook_url or "").strip()
        except Exception:
            pass
    return _env("FEISHU_WEBHOOK_URL")


def get_open_registration(session: Session | None) -> bool:
    if session is not None:
        try:
            cfg = get_app_config(session)
            return bool(getattr(cfg, "open_registration", False))
        except Exception:
            pass
    return _env("OPEN_REGISTRATION", "0") == "1"


def set_open_registration(session: Session, enabled: bool) -> AppConfig:
    cfg = get_app_config(session)
    cfg.open_registration = bool(enabled)
    cfg.updated_at = datetime.utcnow()
    session.add(cfg)
    session.commit()
    session.refresh(cfg)
    return cfg


def set_auto_analysis(session: Session, enabled: bool) -> AppConfig:
    cfg = get_app_config(session)
    cfg.auto_analysis_enabled = bool(enabled)
    cfg.updated_at = datetime.utcnow()
    session.add(cfg)
    session.commit()
    session.refresh(cfg)
    return cfg


def is_auto_analysis_enabled(session: Session) -> bool:
    return bool(get_app_config(session).auto_analysis_enabled)


def _bucket_config_from_cfg(cfg: AppConfig, source: str) -> dict[str, str]:
    raw = getattr(cfg, "taobao_bucket_config", None) if source.upper() == "TAOBAO" else getattr(cfg, "douyin_bucket_config", None)
    if not isinstance(raw, dict):
        return {}
    return {k: str(v) for k, v in raw.items() if isinstance(v, (str, int)) and str(v).strip()}


def get_bucket_config_dict(source: str, session: Session | None) -> dict[str, str]:
    """Get bucket config as dict. AppConfig (taobao/douyin) overrides env."""
    p = (source or "DEFAULT").upper().strip()
    if p == "DEFAULT":
        out = {
            "bucket": _env("BUCKET_NAME"),
            "prefix": _env("BUCKET_PREFIX"),
            "endpoint": _env("S3_ENDPOINT") or _env("BUCKET_ENDPOINT"),
            "region": _env("S3_REGION") or _env("BUCKET_REGION"),
            "access_key": _env("S3_ACCESS_KEY") or _env("BUCKET_ACCESS_KEY"),
            "secret_key": _env("S3_SECRET_KEY") or _env("BUCKET_SECRET_KEY"),
        }
    else:
        out = {
            "bucket": _env(f"{p}_BUCKET_NAME"),
            "prefix": _env(f"{p}_BUCKET_PREFIX"),
            "endpoint": _env(f"{p}_S3_ENDPOINT"),
            "region": _env(f"{p}_S3_REGION"),
            "access_key": _env(f"{p}_S3_ACCESS_KEY"),
            "secret_key": _env(f"{p}_S3_SECRET_KEY"),
        }
    if session is not None and p in ("TAOBAO", "DOUYIN"):
        try:
            cfg = get_app_config(session)
            stored = _bucket_config_from_cfg(cfg, p)
            for k, v in stored.items():
                if v:
                    out[k] = v
        except Exception:
            pass
    return out


def get_bucket_backup_dir(session: Session | None) -> str:
    if session is not None:
        try:
            cfg = get_app_config(session)
            if (getattr(cfg, "bucket_backup_dir", None) or "").strip():
                return (cfg.bucket_backup_dir or "").strip()
        except Exception:
            pass
    return _env("BUCKET_BACKUP_DIR") or "/data/bucket_backup"


def set_bucket_settings(
    session: Session,
    *,
    bucket_fetch_enabled: bool | None = None,
    taobao_bucket_import_enabled: bool | None = None,
    douyin_bucket_import_enabled: bool | None = None,
    bucket_daily_check_time: str | None = None,
    bucket_retry_interval_minutes: int | None = None,
    bucket_log_keep: int | None = None,
    bucket_backup_dir: str | None = None,
    feishu_webhook_url: str | None = None,
    taobao_bucket_config: dict | None = None,
    douyin_bucket_config: dict | None = None,
) -> AppConfig:
    cfg = get_app_config(session)
    if bucket_fetch_enabled is not None:
        cfg.bucket_fetch_enabled = bool(bucket_fetch_enabled)
    if taobao_bucket_import_enabled is not None:
        cfg.taobao_bucket_import_enabled = bool(taobao_bucket_import_enabled)
    if douyin_bucket_import_enabled is not None:
        cfg.douyin_bucket_import_enabled = bool(douyin_bucket_import_enabled)
    if bucket_daily_check_time is not None:
        v = (bucket_daily_check_time or "").strip()
        if v:
            cfg.bucket_daily_check_time = v
    if bucket_retry_interval_minutes is not None:
        try:
            cfg.bucket_retry_interval_minutes = max(5, int(bucket_retry_interval_minutes))
        except Exception:
            pass
    if bucket_log_keep is not None:
        try:
            cfg.bucket_log_keep = max(100, min(5000, int(bucket_log_keep)))
        except Exception:
            pass
    if bucket_backup_dir is not None:
        cfg.bucket_backup_dir = (bucket_backup_dir or "").strip()
    if feishu_webhook_url is not None:
        cfg.feishu_webhook_url = (feishu_webhook_url or "").strip()
    if taobao_bucket_config is not None and isinstance(taobao_bucket_config, dict):
        cfg.taobao_bucket_config = {k: str(v) for k, v in taobao_bucket_config.items()}
    if douyin_bucket_config is not None and isinstance(douyin_bucket_config, dict):
        cfg.douyin_bucket_config = {k: str(v) for k, v in douyin_bucket_config.items()}

    cfg.updated_at = datetime.utcnow()
    session.add(cfg)
    session.commit()
    session.refresh(cfg)
    return cfg


def bucket_settings(session: Session) -> dict:
    cfg = get_app_config(session)
    tc = getattr(cfg, "taobao_bucket_config", None) or {}
    dc = getattr(cfg, "douyin_bucket_config", None) or {}
    return {
        "bucket_fetch_enabled": bool(cfg.bucket_fetch_enabled),
        "taobao_bucket_import_enabled": bool(cfg.taobao_bucket_import_enabled),
        "douyin_bucket_import_enabled": bool(cfg.douyin_bucket_import_enabled),
        "bucket_daily_check_time": cfg.bucket_daily_check_time or "10:15",
        "bucket_retry_interval_minutes": int(cfg.bucket_retry_interval_minutes or 60),
        "bucket_log_keep": int(cfg.bucket_log_keep or 800),
        "bucket_backup_dir": (cfg.bucket_backup_dir or "").strip() or "/data/bucket_backup",
        "feishu_webhook_url": (getattr(cfg, "feishu_webhook_url", None) or "").strip(),
        "taobao_bucket_config": tc if isinstance(tc, dict) else {},
        "douyin_bucket_config": dc if isinstance(dc, dict) else {},
    }


SETTINGS_FILE = Path("/data/settings.json")


def _mask_secrets(data: dict) -> dict:
    out = {}
    for k, v in data.items():
        if isinstance(v, dict):
            out[k] = _mask_secrets(v)
        elif k.lower() in ("api_key", "secret_key", "access_key", "openai_api_key"):
            out[k] = "****" if (v and str(v).strip()) else ""
        else:
            out[k] = v
    return out


def export_settings_to_file(session: Session) -> Path:
    """Export AppConfig-backed settings to data/settings.json. Secrets masked."""
    cfg = get_app_config(session)
    data: dict[str, Any] = {
        "openai_base_url": (cfg.openai_base_url or "").strip(),
        "openai_api_key": "****" if (cfg.openai_api_key or "").strip() else "",
        "openai_model": (cfg.openai_model or "").strip() or "gpt-5.2",
        "openai_fallback_model": (cfg.openai_fallback_model or "").strip(),
        "openai_timeout": int(getattr(cfg, "openai_timeout", 1800) or 1800),
        "openai_retries": int(getattr(cfg, "openai_retries", 0) or 0),
        "openai_reasoning_effort": (getattr(cfg, "openai_reasoning_effort", None) or "high").strip(),
        "open_registration": bool(getattr(cfg, "open_registration", False)),
        "feishu_webhook_url": (getattr(cfg, "feishu_webhook_url", None) or "").strip(),
        "bucket_backup_dir": (getattr(cfg, "bucket_backup_dir", None) or "").strip(),
        "bucket_fetch_enabled": bool(cfg.bucket_fetch_enabled),
        "taobao_bucket_import_enabled": bool(cfg.taobao_bucket_import_enabled),
        "douyin_bucket_import_enabled": bool(cfg.douyin_bucket_import_enabled),
        "bucket_daily_check_time": (cfg.bucket_daily_check_time or "10:15").strip(),
        "bucket_retry_interval_minutes": int(cfg.bucket_retry_interval_minutes or 60),
        "bucket_log_keep": int(cfg.bucket_log_keep or 800),
        "cron_interval_seconds": int(getattr(cfg, "cron_interval_seconds", 600) or 600),
        "worker_poll_seconds": int(getattr(cfg, "worker_poll_seconds", 5) or 5),
        "enable_enqueue_analysis": bool(getattr(cfg, "enable_enqueue_analysis", True)),
        "taobao_bucket_config": _mask_secrets(getattr(cfg, "taobao_bucket_config", None) or {}),
        "douyin_bucket_config": _mask_secrets(getattr(cfg, "douyin_bucket_config", None) or {}),
    }
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return SETTINGS_FILE


def import_settings_from_file(session: Session, path: Path | None = None) -> dict[str, Any]:
    """Import from JSON file (e.g. data/settings.json). Updates AppConfig. Returns applied keys."""
    path = path or SETTINGS_FILE
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        return {}

    applied: list[str] = []
    cfg = get_app_config(session)

    def set_if(k: str, v: Any, skip_secret_mask: bool = False) -> None:
        if not skip_secret_mask and isinstance(v, str) and v == "****":
            return
        if not hasattr(cfg, k):
            return
        applied.append(k)
        setattr(cfg, k, v)

    for key in ("openai_base_url", "openai_model", "openai_fallback_model", "openai_reasoning_effort",
                "feishu_webhook_url", "bucket_backup_dir", "bucket_daily_check_time"):
        if key in data and data[key] is not None:
            set_if(key, str(data[key]).strip() if isinstance(data[key], str) else data[key])
    if "openai_api_key" in data and isinstance(data["openai_api_key"], str) and data["openai_api_key"] != "****":
        set_if("openai_api_key", data["openai_api_key"].strip())
    for key in ("openai_timeout", "openai_retries", "bucket_retry_interval_minutes", "bucket_log_keep", "cron_interval_seconds", "worker_poll_seconds"):
        if key in data and data[key] is not None:
            try:
                set_if(key, int(data[key]))
            except Exception:
                pass
    for key in ("open_registration", "bucket_fetch_enabled", "taobao_bucket_import_enabled", "douyin_bucket_import_enabled", "enable_enqueue_analysis"):
        if key in data and data[key] is not None:
            set_if(key, bool(data[key]))

    for key in ("taobao_bucket_config", "douyin_bucket_config"):
        if key in data and isinstance(data[key], dict):
            cleaned = {str(k): str(v) for k, v in data[key].items() if v != "****" and str(v).strip()}
            set_if(key, cleaned)

    cfg.updated_at = datetime.utcnow()
    session.add(cfg)
    session.commit()
    return {k: True for k in applied}
