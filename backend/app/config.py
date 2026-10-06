from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration, read from env vars / .env (repo root or backend/)."""

    model_config = SettingsConfigDict(env_file=("../.env", ".env"), extra="ignore")

    database_url: str = "postgresql+psycopg://photos:photos@localhost:5432/photos"

    s3_endpoint_url: str | None = None
    s3_public_endpoint_url: str | None = None
    aws_access_key_id: str = "test"
    aws_secret_access_key: str = "test"
    aws_region: str = "us-east-1"
    s3_bucket_originals: str = "photos-originals"
    s3_bucket_thumbs: str = "photos-thumbs"
    presign_ttl_seconds: int = 3600

    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection: str = "photos_v1"

    clip_model: str = "ViT-B/32"
    search_img_weight: float = Field(0.7, ge=0.0, le=1.0)

    max_upload_mb: int = 25
    max_batch_files: int = 50
    cors_origins: str = "http://localhost:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def s3_public_endpoint(self) -> str | None:
        return self.s3_public_endpoint_url or self.s3_endpoint_url


@lru_cache
def get_settings() -> Settings:
    return Settings()
