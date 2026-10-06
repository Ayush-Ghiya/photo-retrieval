from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.db import make_session_factory
from app.services.clip_model import ClipEncoder, Encoder
from app.services.images import ImageService
from app.services.search import SearchService
from app.services.storage import Storage
from app.services.vector_index import ModelMismatchError, VectorIndex


@dataclass
class Services:
    settings: Settings
    sessions: sessionmaker[Session]
    storage: Storage
    index: VectorIndex
    encoder: Encoder
    images: ImageService
    search: SearchService

    def ping_db(self) -> None:
        with self.sessions() as s:
            s.execute(text("select 1"))

    def startup(self, *, recreate_index: bool = False) -> None:
        try:
            self.ping_db()
            self.storage.ensure_buckets()
            if recreate_index:
                self.index.recreate()
            else:
                self.index.ensure_collection()
        except ModelMismatchError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Startup check failed: {e}. Is `docker compose up -d` running and .env configured?"
            ) from e


def build_services(settings: Settings, encoder: Encoder | None = None) -> Services:
    encoder = encoder or ClipEncoder(settings.clip_model)
    sessions = make_session_factory(settings.database_url)
    storage = Storage(settings)
    index = VectorIndex(settings.chroma_host, settings.chroma_port, settings.chroma_collection, encoder.model_name)
    images = ImageService(sessions, storage, index, encoder, settings)
    search = SearchService(images, index, encoder, settings)
    return Services(settings, sessions, storage, index, encoder, images, search)
