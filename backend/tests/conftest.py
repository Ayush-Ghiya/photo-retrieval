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


from app.services.storage import Storage


def _empty_bucket(storage: Storage, kind: str) -> None:
    bucket = storage.bucket(kind)
    for obj in storage.client.list_objects_v2(Bucket=bucket).get("Contents", []):
        storage.client.delete_object(Bucket=bucket, Key=obj["Key"])


@pytest.fixture
def storage(settings) -> Storage:
    s = Storage(settings)
    s.ensure_buckets()
    _empty_bucket(s, "originals")
    _empty_bucket(s, "thumbs")
    return s


from tests.fakes import FakeEncoder


@pytest.fixture
def encoder() -> FakeEncoder:
    return FakeEncoder()


from app.services.vector_index import VectorIndex


@pytest.fixture
def index(settings, encoder) -> VectorIndex:
    idx = VectorIndex(settings.chroma_host, settings.chroma_port, settings.chroma_collection, encoder.model_name)
    idx.recreate()
    return idx


from app.services.images import ImageService


@pytest.fixture
def image_service(sessions, storage, index, encoder, settings) -> ImageService:
    return ImageService(sessions, storage, index, encoder, settings)


from app.services.search import SearchService


@pytest.fixture
def search_service(image_service, index, encoder, settings) -> SearchService:
    return SearchService(image_service, index, encoder, settings)


from fastapi.testclient import TestClient

from app.main import create_app
from app.services.container import Services


@pytest.fixture
def services(settings, sessions, storage, index, encoder, image_service, search_service) -> Services:
    return Services(settings, sessions, storage, index, encoder, image_service, search_service)


@pytest.fixture
def client(services):
    with TestClient(create_app(lambda: services)) as c:
        yield c
