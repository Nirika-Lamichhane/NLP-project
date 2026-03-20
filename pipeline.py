from registry import registry
from comment_extractor import get_comments

def run_pipeline(url: str):
    comments = get_comments(url)
    results = []

    for comment in comments:

        # Step 1 — identify language
        lang_result = registry.language_identifier.predict(comment)
        language = lang_result["language"]

        # Step 2 — decide what to do based on language
        if language == "NE_ROM":
            # Roman Nepali → transliterate to Devanagari
            devanagari_comment = registry.transliterator.predict(comment)

        elif language == "NE_DEV":
            # Already Devanagari → skip transliteration entirely
            devanagari_comment = comment

        elif language == "EN":
            # English → skip, not relevant for our model
            # add to results with a note and continue
            results.append({
                "original_comment":   comment,
                "language":           "EN",
                "devanagari_comment": None,
                "aspect":             None,
                "sentiment":          None,
                "skipped":            True,
            })
            continue

        else:
            # UNKNOWN — skip as well
            results.append({
                "original_comment":   comment,
                "language":           "UNKNOWN",
                "devanagari_comment": None,
                "aspect":             None,
                "sentiment":          None,
                "skipped":            True,
            })
            continue

        # Step 3 — predict aspect and sentiment
        # only reaches here for NE_ROM and NE_DEV
        sentiment, aspect = registry.devanagari.predict(devanagari_comment)

        results.append({
            "original_comment":   comment,
            "language":           language,
            "devanagari_comment": devanagari_comment,
            "aspect":             aspect,
            "sentiment":          sentiment,
            "skipped":            False,
        })

    return results
