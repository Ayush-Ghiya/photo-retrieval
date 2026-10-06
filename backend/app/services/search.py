from uuid import UUID

from app.config import Settings
from app.models import Image
from app.services.clip_model import Encoder
from app.services.images import ImageService
from app.services.vector_index import Hit, VectorIndex


def blend(hits: list[Hit], img_sims: dict[UUID, float], img_weight: float) -> list[tuple[UUID, float]]:
    """Combine per-image 'img' and 'txt' similarities into one score, best first."""
    img: dict[UUID, float] = {}
    txt: dict[UUID, float] = {}
    for h in hits:
        bucket = img if h.kind == "img" else txt
        bucket[h.image_id] = max(bucket.get(h.image_id, 0.0), h.similarity)

    scored = []
    for image_id in img.keys() | txt.keys():
        i = img.get(image_id, img_sims.get(image_id))
        t = txt.get(image_id)
        if t is None:
            score = i
        elif i is None:
            score = (1 - img_weight) * t
        else:
            score = img_weight * i + (1 - img_weight) * t
        scored.append((image_id, score))
    scored.sort(key=lambda pair: (-pair[1], str(pair[0])))
    return scored


class SearchService:
    def __init__(self, images: ImageService, index: VectorIndex, encoder: Encoder, settings: Settings):
        self.images = images
        self.index = index
        self.encoder = encoder
        self.settings = settings

    def search(
        self, q: str, *, tags: list[str], source: str | None, limit: int
    ) -> list[tuple[Image, float]]:
        vector = self.encoder.encode_text([q])[0]
        hits = self.index.query(vector, n=limit * 3, tags=tags, source=source)
        with_img = {h.image_id for h in hits if h.kind == "img"}
        txt_only = {h.image_id for h in hits if h.kind == "txt"} - with_img
        img_sims = self.index.image_similarities(txt_only, vector)
        ranked = blend(hits, img_sims, self.settings.search_img_weight)
        rows = self.images.get_many([image_id for image_id, _ in ranked])
        return [(rows[i], score) for i, score in ranked if i in rows][:limit]
