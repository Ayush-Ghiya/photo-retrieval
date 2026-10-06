import pytest

from app.tags import InvalidTag, normalize_tag, parse_tags


@pytest.mark.parametrize(
    "raw,expected",
    [("Goa", "goa"), ("  Goa Trip ", "goa-trip"), ("new\tyear  2024", "new-year-2024"), ("a-b", "a-b")],
)
def test_normalize_tag(raw, expected):
    assert normalize_tag(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "#fun!", "café", "x" * 41])
def test_normalize_tag_rejects_invalid(raw):
    with pytest.raises(InvalidTag):
        normalize_tag(raw)


def test_parse_tags_from_comma_string_dedupes_and_sorts():
    assert parse_tags("Family, goa trip,family,, ") == ["family", "goa-trip"]


def test_parse_tags_from_list_and_none():
    assert parse_tags(["B", "a"]) == ["a", "b"]
    assert parse_tags(None) == []
    assert parse_tags([]) == []


def test_parse_tags_raises_on_any_invalid():
    with pytest.raises(InvalidTag):
        parse_tags("ok, #bad")
