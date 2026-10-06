import hashlib
import math

from PIL import Image

DIM = 8


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class FakeEncoder:
    """Deterministic stand-in for CLIP: colour-based image vectors, hash-based text vectors."""

    model_name = "fake"

    def __init__(self, text_vectors: dict[str, list[float]] | None = None):
        self.text_vectors = text_vectors or {}

    def encode_images(self, images: list[Image.Image]) -> list[list[float]]:
        out = []
        for im in images:
            r, g, b = im.convert("RGB").resize((1, 1)).getpixel((0, 0))
            out.append(_normalize([r / 255, g / 255, b / 255, 0, 0, 0, 0, 0.01]))
        return out

    def encode_text(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            if t in self.text_vectors:
                out.append(_normalize(self.text_vectors[t]))
            else:
                digest = hashlib.sha256(t.encode()).digest()
                out.append(_normalize([b - 128 for b in digest[:DIM]]))
        return out


class BrokenIndex:
    """VectorIndex stand-in whose every call fails, simulating ChromaDB being down."""

    def __getattr__(self, name):
        def fail(*args, **kwargs):
            raise ConnectionError("chroma is down")

        return fail
