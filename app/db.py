from __future__ import annotations

import os
import time
from urllib.parse import urlparse

from pydantic_settings import BaseSettings
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlmodel import SQLModel, Session, create_engine


class Settings(BaseSettings):
    """Database settings.

    This project is run under Docker Compose where Postgres is persisted in a
    named volume. When the volume already exists, changing POSTGRES_USER in
    docker-compose.yml will NOT recreate roles.

    To avoid the common "role postgres does not exist" pitfall (caused by an
    old DATABASE_URL in .env), we prefer building the DB URL from POSTGRES_*
    variables by default.
    """

    # Legacy override. Only used when USE_POSTGRES_ENV=0.
    DATABASE_URL: str | None = None

    # Default: build url from POSTGRES_* env.
    USE_POSTGRES_ENV: bool = True

    DB_HOST: str = "db"
    DB_PORT: int = 5432
    POSTGRES_USER: str = "qc"
    POSTGRES_PASSWORD: str = "qc"
    POSTGRES_DB: str = "qc"

    class Config:
        case_sensitive = False


settings = Settings()


def build_database_url() -> str:
    if settings.USE_POSTGRES_ENV:
        return (
            f"postgresql+psycopg://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
            f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.POSTGRES_DB}"
        )
    if settings.DATABASE_URL:
        return settings.DATABASE_URL
    # last resort for local dev
    return "sqlite:///./dev.db"


DATABASE_URL = build_database_url()


def _safe_db_hint(url: str) -> str:
    try:
        p = urlparse(url)
        user = p.username or ""
        host = p.hostname or ""
        db = (p.path or "").lstrip("/")
        return f"user={user} host={host} db={db}"
    except Exception:
        return "(unparsed)"


engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    # Avoid hanging forever if DNS/network is flaky inside Docker.
    connect_args={"connect_timeout": int(os.getenv("DB_CONNECT_TIMEOUT", "5"))},
)


def create_db_and_tables() -> None:
    """Create tables with retry (db container may need a few seconds).
    Ignores IntegrityError for 'role' enum already exists (restart with existing DB).
    """
    print(f"[db] connecting: {_safe_db_hint(DATABASE_URL)}")
    deadline = time.time() + 60
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with engine.connect() as conn:  # noqa: F841
                pass
            SQLModel.metadata.create_all(engine)
            return
        except IntegrityError as e:
            msg = str(e).lower()
            if "pg_type_typname_nsp_index" in msg and "already exists" in msg:
                print("[db] enum/types already exist, skipping create_all")
                return
            raise
        except OperationalError as e:
            last_err = e
            print(f"[db] retrying in 2s: {type(e).__name__}: {e}")
            time.sleep(2)

    if last_err:
        raise last_err
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
