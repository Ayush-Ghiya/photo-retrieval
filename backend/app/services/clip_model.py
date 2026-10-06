from typing import Protocol

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image


class Encoder(Protocol):
    model_name: str

    def encode_images(self, images: list[Image.Image]) -> list[list[float]]: ...

    def encode_text(self, texts: list[str]) -> list[list[float]]: ...


class ClipEncoder:
    """OpenAI CLIP weights via open_clip. Load once per process."""

    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        arch = model_name.replace("/", "-")  # "ViT-B/32" -> "ViT-B-32"
        model, _, preprocess = open_clip.create_model_and_transforms(
            arch, pretrained="openai", device=self.device, force_quick_gelu=True
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(arch)  # truncates to 77 tokens

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image]) -> list[list[float]]:
        batch = torch.stack([self._preprocess(im.convert("RGB")) for im in images]).to(self.device)
        feats = self._model.encode_image(batch).float()
        return F.normalize(feats, dim=-1).cpu().tolist()

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> list[list[float]]:
        tokens = self._tokenizer(texts).to(self.device)
        feats = self._model.encode_text(tokens).float()
        return F.normalize(feats, dim=-1).cpu().tolist()
