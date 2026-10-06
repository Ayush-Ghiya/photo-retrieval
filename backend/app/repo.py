from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Tag, image_tags


def get_or_create_tags(session: Session, names: list[str]) -> list[Tag]:
    """Return Tag rows for already-normalised names, creating missing ones."""
    if not names:
        return []
    existing = {t.name: t for t in session.scalars(select(Tag).where(Tag.name.in_(names)))}
    tags = []
    for name in names:
        tag = existing.get(name)
        if tag is None:
            tag = Tag(name=name)
            session.add(tag)
            existing[name] = tag
        tags.append(tag)
    return tags


def tag_counts(session: Session) -> list[tuple[str, int]]:
    """Tags used by at least one image, most used first."""
    count = func.count(image_tags.c.image_id)
    stmt = (
        select(Tag.name, count)
        .join(image_tags, image_tags.c.tag_id == Tag.id)
        .group_by(Tag.name)
        .order_by(count.desc(), Tag.name)
    )
    return [(name, n) for name, n in session.execute(stmt)]
