from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from bailiff.core.config import settings
from bailiff.features.memory.models import Base


def init_db():
    if "engine" in globals() and "SessionLocal" in globals():
        return
    data_dir = Path(settings.app.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    database_url = f"sqlite:///{(data_dir / 'bailiff.db').as_posix()}"
    eng = create_engine(database_url, connect_args={"check_same_thread": False})
    sl = sessionmaker(autocommit=False, autoflush=False, bind=eng)
    Base.metadata.create_all(bind=eng)
    globals()["engine"] = eng
    globals()["SessionLocal"] = sl


def get_session() -> Session:
    init_db()
    return globals()["SessionLocal"]()


# PEP 562: trigger lazy init when callers do `from bailiff.core.db import SessionLocal`.
def __getattr__(name):
    if name in {"SessionLocal", "engine"}:
        init_db()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@contextmanager
def get_db():
    db = get_session()
    try:
        yield db
    finally:
        db.close()
