from agentci_privacy_filter.redactor import RegexRedactor, extract_labels


def test_extract_labels_deduplicates_and_sorts():
    labels = extract_labels("[SECRET] <PRIVATE_EMAIL> [SECRET]")

    assert labels == ["PRIVATE_EMAIL", "SECRET"]


def test_regex_redactor_masks_common_demo_secrets():
    result = RegexRedactor().redact(
        "Email maya@example.com, call +1 (415) 555-0124, and rotate sk-test-1234567890."
    )

    assert "[PRIVATE_EMAIL]" in result.redacted_text
    assert "[PRIVATE_PHONE]" in result.redacted_text
    assert "[SECRET]" in result.redacted_text
    assert result.labels == ["PRIVATE_EMAIL", "PRIVATE_PHONE", "SECRET"]
