from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, Request
from jose import jwt, JWTError
from passlib.context import CryptContext
from pydantic_settings import BaseSettings
from sqlmodel import Session, select

from db import get_session
from models import User


class AuthSettings(BaseSettings):
    APP_SECRET_KEY: str = "change-me"
    TOKEN_EXPIRE_HOURS: int = 72
    # “保持登录”时的有效期（基于设备 cookie 免登录）
    REMEMBER_EXPIRE_DAYS: int = 7
    COOKIE_NAME: str = "qc_token"
    # 生产环境(HTTPS)建议设为 True；本地 http 开发默认 False，避免 cookie 不生效
    COOKIE_SECURE: bool = False
    # Cookie SameSite: lax/strict/none. If set to "none", Secure will be forced on.
    COOKIE_SAMESITE: str = "lax"
    # Optional cookie domain (e.g. "marllen.com" to cover www/non-www).
    COOKIE_DOMAIN: str = ""

    class Config:
        case_sensitive = False


auth_settings = AuthSettings()
# Use PBKDF2-SHA256 for password hashing to avoid bcrypt backend/version issues
# and the 72-byte bcrypt input limit.
_pwd = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def hash_password(password: str) -> str:
    return _pwd.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return _pwd.verify(password, password_hash)


def create_token(user: User, *, expire_hours: int | None = None) -> str:
    now = datetime.utcnow()
    hours = auth_settings.TOKEN_EXPIRE_HOURS if expire_hours is None else int(expire_hours)
    exp = now + timedelta(hours=hours)
    payload = {
        "sub": str(user.id),
        "email": user.email,
        "username": getattr(user, "username", None),
        "role": user.role,
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
    }
    return jwt.encode(payload, auth_settings.APP_SECRET_KEY, algorithm="HS256")


def get_token_from_request(request: Request) -> Optional[str]:
    return request.cookies.get(auth_settings.COOKIE_NAME)


def get_current_user(
    request: Request,
    session: Session = Depends(get_session),
) -> User:
    token = get_token_from_request(request)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(token, auth_settings.APP_SECRET_KEY, algorithms=["HS256"])
        user_id = int(payload.get("sub"))
    except (JWTError, TypeError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid token")

    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    # soft-delete / disabled user
    if getattr(user, "is_active", True) is False:
        raise HTTPException(status_code=401, detail="User disabled")
    return user


def require_role(*roles: str):
    def _dep(user: User = Depends(get_current_user)) -> User:
        role_val = getattr(user.role, "value", None) or str(user.role)
        # Role is a StrEnum in most cases, but older DB rows / casts can behave differently.
        if role_val not in set(roles):
            raise HTTPException(status_code=403, detail="Forbidden")
        return user

    return _dep


def get_user_by_email(session: Session, email: str) -> Optional[User]:
    stmt = select(User).where(User.email == email)
    return session.exec(stmt).first()


def get_user_by_username(session: Session, username: str) -> Optional[User]:
    username = (username or "").strip()
    if not username:
        return None
    stmt = select(User).where(User.username == username)
    u = session.exec(stmt).first()
    if u and getattr(u, "is_active", True) is False:
        return None
    return u
