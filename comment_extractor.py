import os
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

load_dotenv()

API_KEYS = os.getenv("YOUTUBE_API_KEYS").split(",")


def extract_video_id(url):
    parsed_url = urlparse(url)

    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        return parse_qs(parsed_url.query).get("v", [None])[0]

    elif parsed_url.hostname == "youtu.be":
        return parsed_url.path[1:]

    return None


def is_processable(comment: str) -> bool:
    """
    A comment is processable if it has more than 5 words.
    Filters out emojis, single words, hashtags, short reactions.
    """
    words = comment.split()
    return len(words) >= 5


def get_comments(youtube_url, target_processed=10, batch_size=10, max_fetched=150):
    """
    Fetches comments in batches.
    Keeps fetching until target_processed comments with 5+ words are collected.
    Hard stops at max_fetched to avoid infinite loop.
    """
    video_id = extract_video_id(youtube_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")

    all_comments    = []
    processable_count = 0
    next_page_token = None
    key_index       = 0
    total_fetched   = 0

    youtube = build("youtube", "v3", developerKey=API_KEYS[key_index])

    while total_fetched < max_fetched:
        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=batch_size,
                pageToken=next_page_token,
                textFormat="plainText"
            )
            response = request.execute()

            for item in response["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                all_comments.append(comment)
                total_fetched += 1

                if is_processable(comment):
                    processable_count += 1

                # stop as soon as we have enough processable comments
                if processable_count >= target_processed:
                    return all_comments

            # no more pages
            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

        except Exception as e:
            key_index += 1
            if key_index >= len(API_KEYS):
                print("All API keys exhausted.")
                break
            print("Switching API key...")
            youtube = build("youtube", "v3", developerKey=API_KEYS[key_index])

    return all_comments