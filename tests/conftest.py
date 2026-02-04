import pytest
from sqlmodel import SQLModel, Session, create_engine

# Import models to register them with SQLModel.metadata
from app import models  # noqa: F401


@pytest.fixture()
def session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
