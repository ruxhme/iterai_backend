from title_engine import TitleIndex, enforce_guidelines, sanitize_input


def test_sanitize_input_normalizes_and_romanizes():
    assert sanitize_input("Nam4skar") == "namaskar"
    assert "desh" in sanitize_input("देश की आवाज")


def test_exact_match_conflict():
    index = TitleIndex()
    index.add_title("Indian Express")

    reasons, score = index.detect_lexical_conflicts("Indian Express")
    assert score == 100.0
    assert "Exact match" in reasons[0]


def test_word_order_conflict():
    index = TitleIndex()
    index.add_title("Indian Express")

    reasons, score = index.detect_lexical_conflicts("Express Indian")
    assert score >= 99.0
    assert "Word-order variation" in reasons[0]


def test_periodicity_extension_conflict():
    index = TitleIndex()
    index.add_title("Morning Herald")

    reasons, score = index.detect_lexical_conflicts("Daily Morning Herald")
    assert score >= 90.0
    assert "Periodicity modifier" in reasons[0]


def test_title_combination_conflict():
    index = TitleIndex()
    index.add_title("Hindu")
    index.add_title("Indian Express")

    reasons, score = index.detect_lexical_conflicts("Hindu Indian Express")
    assert score >= 90.0
    assert "combine existing titles" in reasons[0]


def test_guidelines_disallowed_words():
    index = TitleIndex()
    reasons = enforce_guidelines("National Crime Bulletin", index)
    assert any("disallowed words" in reason for reason in reasons)


def test_guidelines_prefix_conflict():
    index = TitleIndex()
    index.add_title("Awaz")

    reasons = enforce_guidelines("The Awaz", index)
    assert any("Disallowed prefix" in reason for reason in reasons)
