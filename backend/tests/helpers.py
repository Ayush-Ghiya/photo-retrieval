import io

from PIL import Image


def png_bytes(color=(220, 20, 20), size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def jpeg_with_exif(size=(200, 100), orientation: int = 1, taken: str | None = None) -> bytes:
    img = Image.new("RGB", size, (10, 120, 200))
    exif = Image.Exif()
    exif[0x0112] = orientation
    if taken:
        exif.get_ifd(0x8769)[36867] = taken  # DateTimeOriginal
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()
