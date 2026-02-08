from __future__ import annotations

from http.cookies import SimpleCookie

from fastapi import BackgroundTasks
from starlette.requests import Request

from auth import auth_settings, hash_password
from main import login_action
from models import Role, User


def _make_request() -> Request:
    scope = {
        "type": "http",
        "asgi": {"spec_version": "2.3", "version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/login",
        "raw_path": b"/login",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    return Request(scope)


def _get_cookie(resp) -> SimpleCookie:
    cookies = resp.headers.getlist("set-cookie")
    assert cookies, "missing Set-Cookie header"
    c = SimpleCookie()
    c.load(cookies[0])
    return c


def test_login_sets_persistent_cookie_when_remember_on(session, monkeypatch):
    monkeypatch.setattr(auth_settings, "REMEMBER_EXPIRE_DAYS", 7, raising=False)
    monkeypatch.setattr(auth_settings, "COOKIE_SECURE", False, raising=False)

    u = User(
        username="u1",
        email="u1@example.com",
        name="U1",
        role=Role.admin,
        password_hash=hash_password("pw"),
    )
    session.add(u)
    session.commit()

    resp = login_action(
        _make_request(),
        BackgroundTasks(),
        username="u1",
        password="pw",
        remember="on",
        next=None,
        session=session,
    )
    assert resp.status_code == 302

    c = _get_cookie(resp)
    morsel = c[auth_settings.COOKIE_NAME]
    assert morsel.value
    assert morsel["max-age"] == str(7 * 24 * 3600)
    assert morsel["expires"]


def test_login_sets_session_cookie_when_remember_off(session, monkeypatch):
    monkeypatch.setattr(auth_settings, "REMEMBER_EXPIRE_DAYS", 7, raising=False)
    monkeypatch.setattr(auth_settings, "COOKIE_SECURE", False, raising=False)

    u = User(
        username="u2",
        email="u2@example.com",
        name="U2",
        role=Role.admin,
        password_hash=hash_password("pw"),
    )
    session.add(u)
    session.commit()

    resp = login_action(
        _make_request(),
        BackgroundTasks(),
        username="u2",
        password="pw",
        remember="off",
        next=None,
        session=session,
    )
    assert resp.status_code == 302

    c = _get_cookie(resp)
    morsel = c[auth_settings.COOKIE_NAME]
    assert morsel.value
    assert morsel["max-age"] == ""
    assert morsel["expires"] == ""

