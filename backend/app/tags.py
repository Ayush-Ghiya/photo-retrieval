import re

_VALID = re.compile(r"[a-z0-9-]{1,40}")


class InvalidTag(ValueError):
    pass


def normalize_tag(raw: str) -> str:
    name = re.sub(r"\s+", "-", raw.strip().lower())
    if not _VALID.fullmatch(name):
        raise InvalidTag(f"Invalid tag {raw!r}: use 1-40 letters, digits or hyphens")
    return name


def parse_tags(raw: str | list[str] | None) -> list[str]:
    if raw is None:
        return []
    items = raw.split(",") if isinstance(raw, str) else raw
    return sorted({normalize_tag(t) for t in items if t.strip()})
