from __future__ import annotations

import io
import json
import re
from typing import Optional
from urllib.parse import urlparse

_URL_RE = re.compile(r"https?://[^\s\]\)\}\>\"']+")


def _url_host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _host_matches(host: str, hint: str) -> bool:
    host = (host or "").strip(".").lower()
    hint = (hint or "").strip(".").lower()
    if not host or not hint:
        return False
    return host == hint or host.endswith("." + hint) or host.endswith(hint)


def detect_platform_from_leyan_jsonl(raw_text: str, *, key: str = "") -> Optional[str]:
    """Best-effort platform detection from a 乐言 JSONL export.

    Some upstream pipelines may upload multiple platforms into the same bucket/prefix.
    The JSONL itself does not always carry an explicit platform field, but message bodies
    usually include platform-specific URLs (item links / image hosts), which is a robust signal.
    """
    low_key = (key or "").lower()
    if "douyin" in low_key or "抖音" in low_key:
        return "douyin"
    if "taobao" in low_key or "淘宝" in low_key:
        return "taobao"

    # Host hints (based on observed exports)
    taobao_hints = ("taobao.com", "tmall.com", "alicdn.com", "tb.cn")
    douyin_hints = ("jinritemai.com", "ecombdimg.com", "douyin.com", "iesdouyin.com", "snssdk.com")

    taobao_score = 0
    douyin_score = 0

    try:
        for i, ln in enumerate(io.StringIO(raw_text or "")):
            if i >= 200:
                break
            ln = (ln or "").strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            body = obj.get("body")
            if not body:
                continue
            try:
                payload = json.loads(body)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue

            urls: list[str] = []
            for v in payload.values():
                if isinstance(v, str) and v.startswith("http"):
                    urls.append(v)
            content = payload.get("content")
            if isinstance(content, str) and ("http://" in content or "https://" in content):
                urls.extend(_URL_RE.findall(content))

            for u in urls:
                h = _url_host(u)
                if not h:
                    continue
                if any(_host_matches(h, x) for x in douyin_hints):
                    douyin_score += 1
                if any(_host_matches(h, x) for x in taobao_hints):
                    taobao_score += 1

            # fast-path when signal is already strong
            if douyin_score >= 3 and taobao_score == 0:
                return "douyin"
            if taobao_score >= 3 and douyin_score == 0:
                return "taobao"

    except Exception:
        return None

    if douyin_score > taobao_score and douyin_score > 0:
        return "douyin"
    if taobao_score > douyin_score and taobao_score > 0:
        return "taobao"
    return None

