from __future__ import annotations

import os
from typing import Any

import httpx


class ClawdRelayError(RuntimeError):
    pass


def _env(key: str, default: str = "") -> str:
    v = (os.getenv(key) or "").strip()
    return v if v else default


def get_clawd_relay_settings() -> dict[str, str]:
    """Read clawd-relay settings from environment.

    Intended to be used inside Docker containers:
      - Use host.docker.internal to reach the macOS host relay.
      - Token must be provided via env (do NOT commit it).
    """
    base_url = _env("CLAWD_RELAY_BASE_URL", "http://host.docker.internal:333").rstrip("/")
    exec_token = _env("CLAWD_RELAY_EXEC_TOKEN", "")
    return {"base_url": base_url, "exec_token": exec_token}


def _timeout_obj(timeout_s: float) -> httpx.Timeout:
    t = float(timeout_s or 30)
    return httpx.Timeout(connect=min(5.0, t), read=t, write=t, pool=t)


async def exec_via_clawd_relay(
    *,
    command: str,
    cwd: str | None = None,
    timeout_s: float = 1800.0,
    timeout_ms: int | None = None,
    include_output: bool = False,
    max_output_bytes: int | None = None,
) -> dict[str, Any]:
    s = get_clawd_relay_settings()
    if not s["exec_token"]:
        raise ClawdRelayError("CLAWD_RELAY_EXEC_TOKEN 未配置（无法通过 clawd-relay 调用本机 codex）")

    url = f"{s['base_url']}/api/exec"
    headers = {"Authorization": f"Bearer {s['exec_token']}"}
    payload: dict[str, Any] = {"command": str(command or "")}
    if cwd:
        payload["cwd"] = str(cwd)
    if timeout_ms is not None:
        payload["timeoutMs"] = int(timeout_ms)
    if include_output:
        payload["includeOutput"] = True
    if max_output_bytes is not None:
        payload["maxOutputBytes"] = int(max_output_bytes)

    try:
        async with httpx.AsyncClient(timeout=_timeout_obj(timeout_s)) as client:
            r = await client.post(url, headers=headers, json=payload)
    except httpx.TimeoutException as e:
        raise ClawdRelayError(f"clawd-relay 调用超时：{type(e).__name__}") from e
    except httpx.RequestError as e:
        raise ClawdRelayError(f"clawd-relay 网络错误：{type(e).__name__}: {str(e)[:200]}") from e

    if r.status_code >= 400:
        raise ClawdRelayError(f"clawd-relay 返回 HTTP {r.status_code}: {(r.text or '')[:500]}")

    try:
        data = r.json()
    except Exception as e:
        raise ClawdRelayError(f"clawd-relay 返回非 JSON：{(r.text or '')[:500]}") from e

    if not isinstance(data, dict):
        raise ClawdRelayError(f"clawd-relay 返回格式异常：{str(data)[:200]}")

    if not data.get("ok"):
        err = data.get("error") or data.get("message") or ""
        stdout = str(data.get("stdout") or "")
        stderr = str(data.get("stderr") or "")
        exit_code = data.get("exitCode")
        timed_out = bool(data.get("timedOut"))
        hint_parts = []
        if err:
            hint_parts.append(str(err)[:300])
        if timed_out:
            hint_parts.append("timed_out=true")
        if exit_code not in (None, "", 0, "0"):
            hint_parts.append(f"exitCode={exit_code}")
        if stderr.strip():
            hint_parts.append("stderr: " + stderr.strip()[-400:])
        elif stdout.strip():
            hint_parts.append("stdout: " + stdout.strip()[-400:])
        hint = "；".join([p for p in hint_parts if p]) or "unknown error"
        raise ClawdRelayError(f"clawd-relay 执行失败：{hint}")

    return data
