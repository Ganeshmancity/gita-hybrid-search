"""
🕉️ Fixed prepare_documents.py

PROBLEM FIXED:
    Old: field_value.isascii() → strips ALL Hindi, Sanskrit, Devanagari commentary
    New: Uses all commentators in priority order, keeps multilingual content
"""

# Priority order for best English translation
EN_PRIORITY = [
    ("prabhu", "et"),   # Prabhupada — most detailed
    ("siva",   "et"),   # Sivananda
    ("gambir", "et"),   # Gambirananda
    ("adi",    "et"),   # Adidevananda
    ("purohit","et"),   # Purohit Swami
    ("san",    "et"),   # Sankaranarayan
]

# Priority order for Hindi translation
HI_PRIORITY = [
    ("tej",  "ht"),     # Tejomayananda
    ("rams", "ht"),     # Ramsukhdas
]

# Priority for commentary
COMMENTARY_PRIORITY = [
    ("prabhu", "ec"),
    ("siva",   "ec"),
    ("chinmay","hc"),
    ("rams",   "hc"),
]


# ── Colophon marker — all chapter-ending verses contain this ─────────────────
COLOPHON_MARKER = "तत्सदिति श्रीमद्भगवद्गीता"


def is_colophon(doc: dict) -> bool:
    """Returns True if verse is a chapter closing line — not a teaching."""
    return COLOPHON_MARKER in doc.get("slok", "")


def get_best(doc, priority_list):
    """Get best available field from commentator priority list."""
    for author_key, field_key in priority_list:
        author_data = doc.get(author_key, {})
        if isinstance(author_data, dict):
            val = author_data.get(field_key, "").strip()
            if val and len(val) > 10:
                return val
    return ""


def prepare_documents(raw_documents):
    """
    Fixed version — keeps Sanskrit, Hindi, all commentator translations.
    Old version used isascii() which stripped everything non-English.
    """
    structured_docs = []

    skipped = 0
    for doc in raw_documents:

        # ── Skip colophon verses (chapter endings — not teachings) ────────────
        if is_colophon(doc):
            skipped += 1
            continue

        chapter = doc.get("chapter")
        verse   = doc.get("verse")
        speaker = doc.get("speaker", "")
        slok    = doc.get("slok", "")
        translit= doc.get("transliteration", "")

        # Get best translations using priority lists
        en_translation = get_best(doc, EN_PRIORITY)
        hi_translation = get_best(doc, HI_PRIORITY)
        commentary     = get_best(doc, COMMENTARY_PRIORITY)

        # ── Build enriched content string (what gets embedded) ────────────
        # KEY FIX: Include Sanskrit + transliteration + English + Hindi + commentary
        # The more context embedded, the better semantic retrieval
        content_parts = []

        if slok:
            content_parts.append(f"Sanskrit shloka:\n{slok}")

        if translit:
            content_parts.append(f"Transliteration:\n{translit}")

        if en_translation:
            content_parts.append(f"English meaning:\n{en_translation}")

        if hi_translation:
            content_parts.append(f"Hindi meaning:\n{hi_translation}")

        # Add commentary — trim to 400 chars to avoid noise
        if commentary:
            content_parts.append(f"Commentary:\n{commentary[:400]}")

        content = "\n\n".join(content_parts)

        structured_docs.append({
            "content": content,
            "metadata": {
                "chapter":        chapter,
                "verse":          verse,
                "speaker":        speaker,
                "slok":           slok,
                "transliteration":translit,
                "translation_en": en_translation,
                "translation_hi": hi_translation,
            }
        })

    print(f"✅ Prepared {len(structured_docs)} documents (skipped {skipped} colophons)")
    return structured_docs