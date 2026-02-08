import os
import sys
from pathlib import Path

import pytest
from sqlmodel import SQLModel, Session, create_engine

# Force runtime DB settings to SQLite during tests, so importing `db`/`auth`
# never requires local Postgres/libpq.
os.environ.setdefault("USE_POSTGRES_ENV", "0")
os.environ.setdefault("DATABASE_URL", "sqlite://")

# Ensure imports like `from models import ...` (used by runtime code under /app)
# work when running pytest from repo root.
ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Import models to register them with SQLModel.metadata
import models  # noqa: F401, E402


@pytest.fixture()
def session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
