from __future__ import annotations

import html
from typing import Optional

# Lightweight markdown rendering with basic sanitization.
# - Today: AI output is mostly plain text / bullet points.
# - Future: can support richer markdown + links, while avoiding unsafe HTML injection.

_ALLOWED_TAGS = [
    "p","br","hr",
    "strong","b","em","i","code","pre",
    "ul","ol","li",
    "blockquote",
    "h1","h2","h3","h4","h5","h6",
    "a",
]
_ALLOWED_ATTRS = {
    "a": ["href", "title", "target", "rel"],
}
_ALLOWED_PROTOCOLS = ["http", "https", "mailto"]


def render_markdown_safe(text: str, *, title: Optional[str] = None) -> str:
    raw = (text or "").strip()
    if not raw:
        return "<p>（空）</p>"

    try:
        import markdown  # type: ignore
        import bleach  # type: ignore

        html_out = markdown.markdown(
            raw,
            extensions=[
                "extra",
                "sane_lists",
                "nl2br",
            ],
            output_format="html5",
        )

        cleaned = bleach.clean(
            html_out,
            tags=_ALLOWED_TAGS,
            attributes=_ALLOWED_ATTRS,
            protocols=_ALLOWED_PROTOCOLS,
            strip=True,
        )
        # Make links open in a new tab
        cleaned = bleach.linkify(
            cleaned,
            callbacks=[bleach.callbacks.nofollow, bleach.callbacks.target_blank],
        )
        return cleaned
    except Exception:
        # Fallback: plain text -> <pre>
        return "<pre>" + html.escape(raw) + "</pre>"
