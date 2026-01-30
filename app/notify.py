from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Optional

import httpx


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


async def send_feishu_webhook_async(webhook_url: str, *, title: str, text: str) -> Dict[str, Any]:
    """Send a simple Feishu bot message (webhook).

    We use the "text" message type for maximum compatibility.
    """
    if not webhook_url:
        return {"ok": False, "error": "empty webhook_url"}

    payload = {"msg_type": "text", "content": {"text": f"[{_now_str()}] {title}\n{text}".strip()}}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(webhook_url, json=payload)
            return {"ok": r.status_code >= 200 and r.status_code < 300, "status_code": r.status_code, "body": r.text}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def send_feishu_webhook(webhook_url: str, *, title: str, text: str) -> Dict[str, Any]:
    """Sync wrapper (cron uses sync code)."""
    if not webhook_url:
        return {"ok": False, "error": "empty webhook_url"}

    payload = {"msg_type": "text", "content": {"text": f"[{_now_str()}] {title}\n{text}".strip()}}
    try:
        with httpx.Client(timeout=10) as client:
            r = client.post(webhook_url, json=payload)
            return {"ok": r.status_code >= 200 and r.status_code < 300, "status_code": r.status_code, "body": r.text}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def get_feishu_webhook_url(session=None) -> str:
    """AppConfig when session given, else env."""
    if session is not None:
        try:
            from app_config import get_feishu_webhook_url as _get
            return _get(session)
        except Exception:
            pass
    return os.getenv("FEISHU_WEBHOOK_URL") or ""
