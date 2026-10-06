import os

import pytest

import app.models  # noqa: F401  (registers tables on Base.metadata)
from app.config import Settings
from app.db import Base, make_session_factory

TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL", "postgresql+psycopg://photos:photos@localhost:5432/photos_test"
)


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(
        _env_file=None,
        database_url=TEST_DATABASE_URL,
        s3_endpoint_url="http://localhost:4566",
        s3_bucket_originals="test-photos-originals",
        s3_bucket_thumbs="test-photos-thumbs",
        chroma_collection="photos_test",
        max_upload_mb=1,
    )


@pytest.fixture(scope="session")
def _session_factory(settings):
    return make_session_factory(settings.database_url)


@pytest.fixture
def sessions(_session_factory):
    engine = _session_factory.kw["bind"]
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    return _session_factory
