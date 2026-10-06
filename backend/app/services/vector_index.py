import functools
from dataclasses import dataclass
from uuid import UUID

import chromadb
from chromadb.errors import NotFoundError


@dataclass(frozen=True)
class Hit:
    image_id: UUID
    similarity: float


class ModelMismatchError(RuntimeError):
    pass


def build_where(tags: list[str], source: str | None) -> dict | None:
    conditions: list[dict] = [{f"tag_{t}": True} for t in tags]
    if source:
        conditions.append({"source": source})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _refreshing(method):
    """Retry once with a fresh collection handle if the collection was recreated elsewhere
    (e.g. `python -m app.cli reindex --all` while the API is running)."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except NotFoundError:
            self._col = None
            return method(self, *args, **kwargs)

    return wrapper


def _id(image_id: UUID) -> str:
    return f"{image_id}:img"


class VectorIndex:
    """Derived CLIP index: one ':img' vector per image, with tag/source metadata for filtering."""

    def __init__(self, host: str, port: int, collection_name: str, model_name: str):
        self._host, self._port = host, port
        self._name = collection_name
        self._model = model_name
        self._client = None
        self._col = None

    @property
    def client(self):
        if self._client is None:
            self._client = chromadb.HttpClient(host=self._host, port=self._port)
        return self._client

    @property
    def collection(self):
        if self._col is None:
            self.ensure_collection()
        return self._col

    def _metadata(self) -> dict:
        return {"hnsw:space": "cosine", "clip_model": self._model}

    def ensure_collection(self) -> None:
        col = self.client.get_or_create_collection(self._name, metadata=self._metadata())
        recorded = (col.metadata or {}).get("clip_model")
        if recorded != self._model:
            raise ModelMismatchError(
                f"Collection '{self._name}' was built with CLIP model '{recorded}' but "
                f"CLIP_MODEL is '{self._model}'. Run: python -m app.cli reindex --all"
            )
        self._col = col

    def recreate(self) -> None:
        try:
            self.client.delete_collection(self._name)
        except Exception:
            pass  # did not exist
        self._col = self.client.create_collection(self._name, metadata=self._metadata())

    def ping(self) -> None:
        self.client.heartbeat()

    @_refreshing
    def ids(self) -> list[str]:
        return list(self.collection.get(include=[])["ids"])

    @_refreshing
    def upsert(
        self,
        image_id: UUID,
        *,
        image_vector: list[float],
        tags: list[str],
        source: str,
    ) -> None:
        # Delete + add (rather than update) so removed tags never linger in the metadata.
        self.delete(image_id)
        metadata = {"image_id": str(image_id), "source": source, **{f"tag_{t}": True for t in tags}}
        self.collection.add(ids=[_id(image_id)], embeddings=[image_vector], metadatas=[metadata])

    @_refreshing
    def get_image_vector(self, image_id: UUID) -> list[float] | None:
        res = self.collection.get(ids=[_id(image_id)], include=["embeddings"])
        if not res["ids"]:
            return None
        return [float(x) for x in res["embeddings"][0]]

    @_refreshing
    def delete(self, image_id: UUID) -> None:
        self.collection.delete(ids=[_id(image_id)])

    @_refreshing
    def query(
        self, vector: list[float], n: int, tags: list[str] = (), source: str | None = None
    ) -> list[Hit]:
        total = self.collection.count()
        if total == 0:
            return []
        res = self.collection.query(
            query_embeddings=[vector],
            n_results=min(n, total),
            where=build_where(list(tags), source),
            include=["metadatas", "distances"],
        )
        return [
            Hit(UUID(meta["image_id"]), 1.0 - dist / 2.0)
            for meta, dist in zip(res["metadatas"][0], res["distances"][0])
        ]

    @_refreshing
    def image_similarities(self, image_ids: set[UUID], vector: list[float]) -> dict[UUID, float]:
        if not image_ids:
            return {}
        res = self.collection.get(
            ids=[_id(i) for i in image_ids], include=["embeddings", "metadatas"]
        )
        sims = {}
        for meta, emb in zip(res["metadatas"], res["embeddings"]):
            cos = sum(float(a) * b for a, b in zip(emb, vector))
            sims[UUID(meta["image_id"])] = (1.0 + cos) / 2.0  # == 1 - (1 - cos) / 2
        return sims
