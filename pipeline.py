from registry import registry
from comment_extractor import get_comments

def run_pipeline(url: str):
    comments = get_comments(url)
    results = []

    for comment in comments:

        # Step 1 — identify language
        lang_result = registry.language_identifier.predict(comment)
        language = lang_result["language"]

        # Step 2 — route based on language
        if language == "NE_DEV":
            # already Devanagari — skip transliteration
            devanagari_comment = comment

        elif language == "EN":
            # English — translate to Devanagari using IndicTrans
            devanagari_comment = registry.transliterator.predict(comment)

        else:
            # NE_ROM or UNKNOWN — skip entirely
            results.append({
                "original_comment":   comment,
                "language":           language,
                "devanagari_comment": None,
                "aspect":             None,
                "sentiment":          None,
                "skipped":            True,
            })
            continue

        # Step 3 — predict aspect and sentiment
        # only NE_DEV and EN reach here
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
