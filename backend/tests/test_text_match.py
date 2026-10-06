from app.services.text_match import coverage, metadata_terms, query_terms


def test_query_terms_drop_stopwords_punctuation_and_duplicates():
    assert query_terms("A photo of the Goa-Trip, goa!") == ["goa", "trip"]
    assert query_terms("   ") == []


def test_metadata_terms_include_title_description_and_tag_parts():
    terms = metadata_terms("Sunset drive", "With Mum", ["goa-trip"])
    assert {"sunset", "drive", "mum", "goa", "trip"} <= terms


def test_plurals_fold_to_singular():
    assert query_terms("red cars") == ["red", "car"]
    assert "car" in metadata_terms(None, None, ["cars"])
    assert query_terms("glass") == ["glass"]  # double-s words are left alone


def test_coverage_is_fraction_of_query_terms_found():
    doc = metadata_terms("Sunset drive", None, ["goa-trip"])
    assert coverage(["goa", "trip"], doc) == 1.0
    assert coverage(["goa", "beach"], doc) == 0.5
    assert coverage(["beach"], doc) == 0.0
    assert coverage([], doc) == 0.0
