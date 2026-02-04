from __future__ import annotations

import time
import asyncio
from typing import Any, TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from sqlmodel import Session


class AIError(RuntimeError):
    pass


def _timeout_obj(timeout_s: float) -> httpx.Timeout:
    """Build a slightly more forgiving timeout config.

    httpx's single-number timeout applies to multiple phases; for long-running
    reasoning calls we want a longer read timeout without making connect hang.
    """
    t = float(timeout_s or 300)
    return httpx.Timeout(connect=min(10.0, t), read=t, write=t, pool=t)


async def _post_with_retries(
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_s: float,
    retries: int = 0,
    request_id: str | None = None,
    idempotency_key: str | None = None,
) -> httpx.Response:
    """POST with a small retry on transient network timeouts.

    Notes:
      - This function **raises AIError** on network/timeout failures so that
        callers can surface a friendly message in the UI instead of 500.
    """


    # Copy headers so callers can reuse dicts safely
    headers = dict(headers or {})
    if request_id:
        headers["X-Request-ID"] = str(request_id)
    if idempotency_key:
        # OpenAI-compatible idempotency key (if upstream/relay supports it)
        headers["Idempotency-Key"] = str(idempotency_key)

    last_exc: Exception | None = None
    for i in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=_timeout_obj(timeout_s)) as client:
                return await client.post(url, headers=headers, json=payload)

        # --- transient timeouts: retry a couple times ---
        except httpx.TimeoutException as e:
            last_exc = e
            if i < retries:
                await asyncio.sleep(1.0 * (2**i))
                continue
            raise AIError(
                "AI请求超时：一直没等到上游返回。\\n"
                f"- 请求地址: {url}\\n"
                "建议检查：OPENAI_BASE_URL / Relay 是否在线、网络/VPN是否通畅、以及 OPENAI_TIMEOUT 是否足够。"
            ) from e

        # --- other request-level network errors ---
        except httpx.RequestError as e:
            last_exc = e
            if i < retries:
                await asyncio.sleep(0.5 * (2**i))
                continue
            raise AIError(
                "AI网络错误：无法连到上游服务。\\n"
                f"- 请求地址: {url}\\n"
                f"- 错误: {type(e).__name__}: {str(e)[:200]}\\n"
                "建议检查：OPENAI_BASE_URL 是否写对、Relay是否启动、以及容器到上游的网络是否可达。"
            ) from e

        except AIError:
            raise
        except Exception as e:
            last_exc = e
            raise AIError(f"AI请求失败：{type(e).__name__}: {str(e)[:300]}") from e

    # Should never reach here
    if last_exc is not None:
        raise AIError(f"AI请求失败：{type(last_exc).__name__}: {str(last_exc)[:300]}")
    raise AIError("AI请求失败：未知网络错误")


def get_ai_settings(session: Session | None = None) -> dict[str, Any]:
    """AI settings (OpenAI-compatible). AppConfig when session given, else env."""
    from app_config import get_ai_settings as _get
    return _get(session)


_models_cache: dict[str, Any] = {"ts": 0.0, "base_url": "", "ids": []}


def list_model_ids_sync(*, session: Session | None = None, force: bool = False) -> list[str]:
    """Try to fetch available model IDs from GET /v1/models (OpenAI compatible).

    If the endpoint is unavailable (some relays don't expose it), returns [].
    """
    s = get_ai_settings(session)
    if not s["api_key"]:
        return []
    base_url = s["base_url"]
    now = time.time()
    if (not force) and _models_cache["ids"] and _models_cache["base_url"] == base_url and (now - float(_models_cache["ts"])) < 300:
        return list(_models_cache["ids"])

    url = f"{base_url}/models"
    headers = {"Authorization": f"Bearer {s['api_key']}"}

    try:
        with httpx.Client(timeout=min(10.0, float(s["timeout_s"]) or 60.0)) as client:
            r = client.get(url, headers=headers)
        if r.status_code >= 400:
            return []
        data = r.json()
        ids = []
        for item in (data.get("data") or []):
            mid = (item.get("id") or "").strip()
            if mid:
                ids.append(mid)
        ids = sorted(set(ids))
        _models_cache.update({"ts": now, "base_url": base_url, "ids": ids})
        return ids
    except Exception:
        return []


async def list_model_ids_async(*, session: Session | None = None, force: bool = False) -> list[str]:
    s = get_ai_settings(session)
    if not s["api_key"]:
        return []
    base_url = s["base_url"]
    now = time.time()
    if (not force) and _models_cache["ids"] and _models_cache["base_url"] == base_url and (now - float(_models_cache["ts"])) < 300:
        return list(_models_cache["ids"])

    url = f"{base_url}/models"
    headers = {"Authorization": f"Bearer {s['api_key']}"}

    try:
        async with httpx.AsyncClient(timeout=min(10.0, float(s["timeout_s"]) or 60.0)) as client:
            r = await client.get(url, headers=headers)
        if r.status_code >= 400:
            return []
        data = r.json()
        ids = []
        for item in (data.get("data") or []):
            mid = (item.get("id") or "").strip()
            if mid:
                ids.append(mid)
        ids = sorted(set(ids))
        _models_cache.update({"ts": now, "base_url": base_url, "ids": ids})
        return ids
    except Exception:
        return []


def _pick_fallback(preferred: str | None, session: Session | None = None) -> str | None:
    """Pick a reasonable fallback model when the preferred model is not found."""
    s = get_ai_settings(session)

    # 1) Explicit fallback
    fb = (s.get("fallback_model") or "").strip()
    if fb:
        return fb

    # 2) Env default model
    env_default = (s.get("model") or "").strip()
    if env_default and env_default != (preferred or "").strip():
        return env_default

    # 3) First available model from /models
    ids = list_model_ids_sync(session=session)
    if ids:
        # Prefer GPT-5.1 if present, else first.
        for k in ["gpt-5.2", "gpt-5.2-pro", "gpt-5.1", "gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini"]:
            if k in ids:
                return k
        return ids[0]

    return None


def _extract_responses_output_text(data: dict[str, Any]) -> str:
    # Some SDKs expose `output_text`; be tolerant.
    if isinstance(data, dict) and isinstance(data.get("output_text"), str):
        return data.get("output_text") or ""

    out: list[str] = []
    for item in (data.get("output") or []):
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            for c in (item.get("content") or []):
                if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                    t = c.get("text")
                    if isinstance(t, str) and t.strip():
                        out.append(t)
    return "\n".join(out).strip()

def _messages_to_instructions_and_input(messages: list[dict[str, str]]) -> tuple[str, str]:
    """Convert ChatCompletions-style messages into Responses-style (instructions + input_text).

    - Merge all `system` messages into `instructions`
    - Convert the rest into a simple transcript for `input_text`
    """
    instructions_parts: list[str] = []
    transcript_parts: list[str] = []

    for m in (messages or []):
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "").strip().lower() or "user"
        content = m.get("content")
        if not isinstance(content, str):
            content = "" if content is None else str(content)

        if role == "system":
            if content.strip():
                instructions_parts.append(content.strip())
            continue

        label = "USER"
        if role == "assistant":
            label = "ASSISTANT"
        elif role == "developer":
            label = "DEVELOPER"
        elif role == "tool":
            label = "TOOL"
        transcript_parts.append(f"{label}: {content}".rstrip())

    instructions = "\n\n".join(instructions_parts).strip()
    input_text = "\n\n".join(transcript_parts).strip()
    return instructions, input_text


async def responses_text(
    *,
    input_text: str,
    instructions: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    max_output_tokens: int | None = None,
    retries: int | None = None,
    request_id: str | None = None,
    idempotency_key: str | None = None,
    session: Session | None = None,
) -> str:
    """Call OpenAI-compatible Responses API (/v1/responses).

    - `reasoning.effort` controls how much the model "thinks" before answering.
    """

    s = get_ai_settings(session)
    if not s["api_key"]:
        raise AIError("OPENAI_API_KEY 未配置")

    url = f"{s['base_url']}/responses"
    headers = {"Authorization": f"Bearer {s['api_key']}"}

    preferred = (model or s["model"] or "").strip() or "gpt-5.2"
    eff = (reasoning_effort or s.get("reasoning_effort") or "high").strip().lower() or "high"

    async def _call(chosen_model: str) -> httpx.Response:
        payload: dict[str, Any] = {"model": chosen_model, "input": input_text}
        if instructions:
            payload["instructions"] = instructions
        if eff:
            payload["reasoning"] = {"effort": eff}
        if max_output_tokens is not None:
            payload["max_output_tokens"] = int(max_output_tokens)

        return await _post_with_retries(
            url=url,
            headers=headers,
            payload=payload,
            timeout_s=float(s["timeout_s"]),
            retries=int(s.get("retries") or 0) if retries is None else int(retries),
            request_id=request_id,
            idempotency_key=idempotency_key,
        )

    r = await _call(preferred)

    # IMPORTANT: do NOT fall back to /chat/completions. Keep all upstream calls on /responses.
    if r.status_code in (404, 405) and "model" not in (r.text or "").lower():
        raise AIError(
            "上游不支持 /v1/responses（或被禁用）。当前系统已禁用 /chat/completions 回退。\n"
            f"- 请求地址: {url}\n"
            "请检查：Relay 是否已支持并开启 Responses API，或切换到支持 /v1/responses 的上游。"
        )

    # Auto-fallback only for model_not_found
    if r.status_code == 404 and "model" in (r.text or "").lower():
        fallback = _pick_fallback(preferred, session)
        if fallback and fallback != preferred:
            r2 = await _call(fallback)
            if r2.status_code < 400:
                data = r2.json()
                text = _extract_responses_output_text(data)
                if text:
                    return text
                raise AIError(f"AI返回格式异常: {str(data)[:500]}")
            raise AIError(
                "AI请求失败(404): 模型不可用。\n"
                f"- 首选模型: {preferred}\n"
                f"- 回退模型: {fallback}\n"
                f"- 回退报错: {r2.text[:500]}"
            )

    if r.status_code >= 400:
        hint = ""
        if r.status_code == 404 and "model" in (r.text or "").lower():
            ids = await list_model_ids_async(session=session)
            if ids:
                hint = "\n可用模型示例：" + ", ".join(ids[:12]) + ("..." if len(ids) > 12 else "")
            hint += "\n建议：把 OPENAI_MODEL / 每日总结里的模型改成你 relay 支持的模型 ID。"
        raise AIError(f"AI请求失败({r.status_code}): {r.text[:500]}{hint}")

    data = r.json()
    text = _extract_responses_output_text(data)
    if text:
        return text
    raise AIError(f"AI返回格式异常: {str(data)[:500]}")


async def chat_completion(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.2,
    model: str | None = None,
    reasoning_effort: str | None = None,
    retries: int | None = None,
    request_id: str | None = None,
    idempotency_key: str | None = None,
    session: Session | None = None,
) -> str:
    """Compatibility wrapper for legacy callsites.

    This project now routes all upstream calls through /v1/responses.
    NOTE: `temperature` is ignored in Responses mode.
    """

    _ = temperature  # ignored (Responses API)

    instructions, input_text = _messages_to_instructions_and_input(messages)
    return await responses_text(
        input_text=input_text or "",
        instructions=instructions or None,
        model=model,
        reasoning_effort=reasoning_effort,
        retries=retries,
        request_id=request_id,
        idempotency_key=idempotency_key,
        session=session,
    )
