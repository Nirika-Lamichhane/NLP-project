import pandas as pd
from registry import registry
from comment_extractor import get_comments_batch


def compute_stats(stat_rows: list) -> dict:
    if not stat_rows:
        return {
            "sentiment_counts": {},
            "aspect_counts": {},
            "per_target": []
        }

    df = pd.DataFrame(stat_rows)

    sentiment_counts = df["sentiment"].value_counts().to_dict()
    aspect_counts    = df["aspect"].value_counts().to_dict()

    top_targets = df["target"].value_counts().head(2).index.tolist()

    per_target = []
    for target in top_targets:
        group = df[df["target"] == target]
        top_aspects = group["aspect"].value_counts().head(3).index.tolist()

        breakdown = []
        for aspect in top_aspects:
            ag = group[group["aspect"] == aspect]
            breakdown.append({
                "aspect":   aspect,
                "positive": int((ag["sentiment"] == "positive").sum()),
                "negative": int((ag["sentiment"] == "negative").sum()),
                "neutral":  int((ag["sentiment"] == "neutral").sum()),
            })

        per_target.append({
            "target":   target,
            "mentions": int(len(group)),
            "breakdown": breakdown,
        })

    return {
        "sentiment_counts": sentiment_counts,
        "aspect_counts":    aspect_counts,
        "per_target":       per_target,
    }


def fetch_processable_comments(url: str, target: int = 10, max_fetch: int = 200):
    """
    Fetches comments in batches from YouTube.
    Language-detects each comment on the fly.
    Returns only NE_DEV and EN comments with 5+ words.
    Stops as soon as target number of processable comments are found.
    """
    processable   = []
    page_token    = None
    total_fetched = 0

    while total_fetched < max_fetch:

        # fetch one batch from YouTube
        batch, page_token = get_comments_batch(url, page_token, batch_size=20)

        if not batch:
            break

        for comment in batch:
            total_fetched += 1

            # pre-filter: must have 5+ words
            if len(comment.split()) < 5:
                continue

            # language detect — fast, ~0.1s
            lang_result = registry.language_identifier.predict(comment)
            language    = lang_result["language"]

            # only keep NE_DEV and EN
            if language == "NE_DEV":
                processable.append({
                    "comment":  comment,
                    "language": language,
                })

            elif language == "EN":
                processable.append({
                    "comment":  comment,
                    "language": language,
                })

            # stop the moment we have enough
            if len(processable) >= target:
                return processable

        # no more pages on YouTube
        if not page_token:
            break

    return processable


def run_pipeline(url: str):

    # Step 1 — fetch only processable comments
    processable = fetch_processable_comments(url, target=10)

    result_cards = []
    stat_rows    = []
    processed    = 0

    for item in processable:
        comment  = item["comment"]
        language = item["language"]

        # Step 2 — transliterate only English
        if language == "NE_DEV":
            devanagari_comment = comment
        else:
            # EN — transliterate to Devanagari
            devanagari_comment = registry.transliterator.predict(comment)

        # Step 3 — target identification
        targets = registry.target_model.predict(devanagari_comment)
        if not targets:
            targets = ["General"]

        # Step 4 — aspect + sentiment per target
        comment_targets = []
        for target in targets:
            sentiment, aspect = registry.devanagari.predict(devanagari_comment)
            comment_targets.append({
                "target":    target,
                "aspect":    aspect,
                "sentiment": sentiment,
            })
            stat_rows.append({
                "target":    target,
                "aspect":    aspect,
                "sentiment": sentiment,
            })

        processed += 1
        result_cards.append({
            "original_comment":   comment,
            "language":           language,
            "devanagari_comment": devanagari_comment,
            "targets":            comment_targets,
            "skipped":            False,
        })

    # Step 5 — compute stats
    stats = compute_stats(stat_rows)
    stats["total_comments"] = processed
    stats["processed"]      = processed
    stats["skipped"]        = 0

    return {
        "results": result_cards,
        "stats":   stats,
    }