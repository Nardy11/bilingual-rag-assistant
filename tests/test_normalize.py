from rag.normalize import (
    clean_text,
    detect_language,
    normalize_arabic,
    normalize_for_match,
    strip_tashkeel,
    tokenize,
)


def test_strip_tashkeel_and_tatweel():
    assert strip_tashkeel("مُرَكَّبٌ") == "مركب"
    assert strip_tashkeel("كـــتاب") == "كتاب"


def test_alef_yaa_taa_marbuta_hamza():
    assert normalize_arabic("أحمد إبراهيم آمنة") == "احمد ابراهيم امنه"
    assert normalize_arabic("مستشفى") == "مستشفي"
    assert normalize_arabic("مسؤول قائمة") == "مسءول قاءمه"


def test_match_normalisation_digits_and_case():
    assert normalize_for_match("١٤ حرفاً") == "14 حرفا"
    assert normalize_for_match("KWD Per Night") == "kwd per night"


def test_spelling_variants_tokenise_identically():
    assert tokenize("الإجازة السنوية") == tokenize("الاجازه السنويه")
    assert tokenize("وَالمُوَظَّفِين") == tokenize("الموظفين")


def test_tokenize_drops_stopwords_and_article():
    assert tokenize("ما هي مدة الضمان؟") == ["مده", "ضمان"]
    assert tokenize("How many days of leave?") == ["many", "day", "leave"]


def test_clean_text_keeps_readable_text():
    assert clean_text("a​  b\n\n\n\nc") == "a b\n\nc"


def test_detect_language():
    assert detect_language("ما هي مدة الضمان؟") == "ar"
    assert detect_language("What is the warranty?") == "en"
    assert detect_language("هل يمكنني استخدام USB؟") == "ar"
