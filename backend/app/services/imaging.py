import hashlib
import io
from dataclasses import dataclass
from datetime import datetime, timezone

from PIL import Image, ImageOps, UnidentifiedImageError

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # HEIC support is optional
    pillow_heif = None

# Pillow format name -> (mime type, file extension)
ALLOWED_FORMATS = {
    "JPEG": ("image/jpeg", "jpg"),
    "PNG": ("image/png", "png"),
    "WEBP": ("image/webp", "webp"),
    "HEIF": ("image/heic", "heic"),
}
_EXIF_IFD = 0x8769
_DATETIME_ORIGINAL = 36867
_DATETIME = 306


class UnsupportedImage(Exception):
    pass


@dataclass(frozen=True)
class ProcessedImage:
    content_hash: str
    mime_type: str
    ext: str
    width: int
    height: int
    taken_at: datetime | None
    thumbnail: bytes
    rgb: Image.Image


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _open(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        raise UnsupportedImage("File is not a readable image") from e
    return img


def open_rgb(data: bytes) -> Image.Image:
    """Decode bytes into an EXIF-oriented RGB image (used for CLIP encoding)."""
    return ImageOps.exif_transpose(_open(data)).convert("RGB")


def _taken_at(img: Image.Image) -> datetime | None:
    exif = img.getexif()
    raw = exif.get_ifd(_EXIF_IFD).get(_DATETIME_ORIGINAL) or exif.get(_DATETIME)
    if not isinstance(raw, str):
        return None
    try:
        return datetime.strptime(raw.strip(), "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def process_image(data: bytes, thumb_size: int = 512) -> ProcessedImage:
    img = _open(data)
    if img.format not in ALLOWED_FORMATS:
        raise UnsupportedImage(f"Unsupported image type {img.format}; use JPEG, PNG, WebP or HEIC")
    mime_type, ext = ALLOWED_FORMATS[img.format]
    taken_at = _taken_at(img)

    rgb = ImageOps.exif_transpose(img).convert("RGB")
    thumb = rgb.copy()
    thumb.thumbnail((thumb_size, thumb_size))
    buf = io.BytesIO()
    thumb.save(buf, format="WEBP", quality=80)

    return ProcessedImage(
        content_hash=sha256_hex(data),
        mime_type=mime_type,
        ext=ext,
        width=rgb.width,
        height=rgb.height,
        taken_at=taken_at,
        thumbnail=buf.getvalue(),
        rgb=rgb,
    )
