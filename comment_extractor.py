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


def get_comments_batch(youtube_url, page_token=None, batch_size=20):
    """
    Fetches ONE batch of comments from YouTube.
    Returns (comments_list, next_page_token).
    next_page_token is None if no more pages exist.
    """
    video_id = extract_video_id(youtube_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")

    key_index = 0
    youtube   = build("youtube", "v3", developerKey=API_KEYS[key_index])

    while key_index < len(API_KEYS):
        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=batch_size,
                pageToken=page_token,
                textFormat="plainText"
            )
            response = request.execute()

            comments = [
                item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                for item in response["items"]
            ]

            next_page_token = response.get("nextPageToken")
            return comments, next_page_token

        except Exception as e:
            print(f"API key {key_index} failed: {e}")
            key_index += 1
            if key_index >= len(API_KEYS):
                print("All API keys exhausted.")
                return [], None
            print("Switching API key...")
            youtube = build("youtube", "v3", developerKey=API_KEYS[key_index])

    return [], None
