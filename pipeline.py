import pandas as pd
from registry import registry
from comment_extractor import get_comments

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


def run_pipeline(url: str):
    comments = get_comments(url)
    result_cards = []
    stat_rows    = []
    skipped      = 0
    processed    = 0

    for comment in comments:

        # Step 1 — language identification
        lang_result = registry.language_identifier.predict(comment)
        language    = lang_result["language"]

        # Step 2 — route by language
        if language == "NE_DEV":
            devanagari_comment = comment

        elif language == "EN":
            devanagari_comment = registry.transliterator.predict(comment)

        else:
            skipped += 1
            result_cards.append({
                "original_comment": comment,
                "language":         language,
                "targets":          [],
                "skipped":          True,
            })
            continue

        # Step 3 — target identification (real model)
        targets = registry.target_model.predict(devanagari_comment)

        # if no targets found, use "General" as fallback
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

    stats = compute_stats(stat_rows)
    stats["total_comments"] = len(comments)
    stats["processed"]      = processed
    stats["skipped"]        = skipped

    return {
        "results": result_cards,
        "stats":   stats,
    }