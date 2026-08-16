from app.ingestion.cleaner import clean_text


def test_cleaner_preserves_structure_and_legal_symbols() -> None:
    source = "  Договор   № 42\r\nот 12.03.2025   г.\r\n\r\n\r\nЦена: 1 000 ₽; § 5. "
    cleaned = clean_text(source)
    assert cleaned == "Договор № 42\nот 12.03.2025 г.\n\nЦена: 1 000 ₽; § 5."
