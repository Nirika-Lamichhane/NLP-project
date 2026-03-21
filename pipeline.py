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
    Fetches comments in batches.
    Language-detects each comment on the fly.
    Stops when target number of NE_DEV or EN comments found.
    Returns two lists:
      - processable: NE_DEV and EN comments with 5+ words
      - skipped: everything else with their language label
    """
    processable   = []
    skipped       = []
    page_token    = None
    total_fetched = 0

    while total_fetched < max_fetch:

        batch, page_token = get_comments_batch(url, page_token, batch_size=20)

        if not batch:
            break

        for comment in batch:
            total_fetched += 1

            # pre-filter: less than 5 words → skip immediately
            if len(comment.split()) < 5:
                skipped.append({
                    "comment":  comment,
                    "language": "TOO_SHORT",
                })
                continue

            # language detect
            lang_result = registry.language_identifier.predict(comment)
            language    = lang_result["language"]

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

            else:
                # NE_ROM, CODE_MIXED, UNKNOWN → skipped
                skipped.append({
                    "comment":  comment,
                    "language": language,
                })

            # stop fetching when we have enough processable
            if len(processable) >= target:
                return processable, skipped

        if not page_token:
            break

    return processable, skipped


def run_pipeline(url: str):

    # Step 1 — fetch and filter
    processable, skipped_comments = fetch_processable_comments(url, target=10)

    result_cards = []
    stat_rows    = []
    processed    = 0

    # Step 2 — process good comments
    for item in processable:
        comment  = item["comment"]
        language = item["language"]

        # transliterate only English
        if language == "NE_DEV":
            devanagari_comment = comment
        else:
            devanagari_comment = registry.transliterator.predict(comment)

        # target identification
        targets = registry.target_model.predict(devanagari_comment)
        if not targets:
            targets = ["General"]

        # aspect + sentiment per target
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

    # Step 3 — add skipped comments at the end
    for item in skipped_comments:
        result_cards.append({
            "original_comment": item["comment"],
            "language":         item["language"],
            "targets":          [],
            "skipped":          True,
        })

    # Step 4 — compute stats
    stats = compute_stats(stat_rows)
    stats["total_comments"] = len(processable) + len(skipped_comments)
    stats["processed"]      = processed
    stats["skipped"]        = len(skipped_comments)

    return {
        "results": result_cards,
        "stats":   stats,
    }
