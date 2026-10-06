from uuid import UUID

from app.config import Settings
from app.models import Image
from app.services.clip_model import Encoder
from app.services.images import ImageService
from app.services.text_match import coverage, metadata_terms, query_terms
from app.services.vector_index import VectorIndex

# Upper bound on photos pulled in by a metadata keyword match (on top of CLIP's nearest neighbours).
METADATA_CANDIDATES = 500


def blend(
    img_sims: dict[UUID, float], txt_scores: dict[UUID, float], img_weight: float
) -> list[tuple[UUID, float]]:
    """score = w * visual similarity + (1 - w) * metadata keyword coverage, best first."""
    scored = [
        (image_id, img_weight * img_sims.get(image_id, 0.0) + (1 - img_weight) * txt_scores.get(image_id, 0.0))
        for image_id in img_sims.keys() | txt_scores.keys()
    ]
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
        img_sims = {h.image_id: h.similarity for h in self.index.query(vector, n=limit * 3, tags=tags, source=source)}

        terms = query_terms(q)
        matched = self.images.find_by_metadata(terms, tags=tags, source=source, limit=METADATA_CANDIDATES)
        txt_scores = {}
        for image in matched:
            score = coverage(terms, metadata_terms(image.title, image.description, [t.name for t in image.tags]))
            if score > 0:
                txt_scores[image.id] = score
        missing = set(txt_scores) - set(img_sims)
        img_sims.update(self.index.image_similarities(missing, vector))

        ranked = blend(img_sims, txt_scores, self.settings.search_img_weight)
        rows = self.images.get_many([image_id for image_id, _ in ranked])
        return [(rows[i], score) for i, score in ranked if i in rows][:limit]
