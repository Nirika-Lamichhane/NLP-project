from registry import registry
from comment_extractor import get_comments

def run_pipeline(url: str):
    comments = get_comments(url)
    results = []

    for comment in comments:
        # transliterate
        devanagari_comment = registry.transliterator.predict(comment)

        # predict aspect and sentiment
        sentiment, aspect = registry.devanagari.predict(devanagari_comment)

        results.append({
            "original_comment":   comment,
            "devanagari_comment": devanagari_comment,
            "aspect":             aspect,
            "sentiment":          sentiment,
        })

    return results