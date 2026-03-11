# comment_extractor.py

import os
import re
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs

# -------------------------
# List of YouTube API keys
# -------------------------
API_KEYS = [
    "AIzaSyDnbLrIPxhZC2p6g--KoaRaVRh2brj8RqI",
    "YOUR_API_KEY_2",
    "YOUR_API_KEY_3",
    "YOUR_API_KEY_4"
]

# -------------------------
# Helper: extract video ID from URL
# -------------------------
def extract_video_id(url):
    """
    Extracts the video ID from a YouTube URL.
    """
    parsed_url = urlparse(url)
    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        query = parse_qs(parsed_url.query)
        return query.get("v", [None])[0]
    elif parsed_url.hostname in ["youtu.be"]:
        return parsed_url.path[1:]
    return None

# -------------------------
# Fetch comments using API keys
# -------------------------
def get_comments(youtube_url, max_comments=500):
    """
    Takes a YouTube URL and returns a list of comments.
    Rotates through multiple API keys if quota exceeded.
    """
    video_id = extract_video_id(youtube_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")

    comments = []
    next_page_token = None

    # Try each API key until quota is enough
    for api_key in API_KEYS:
        youtube = build("youtube", "v3", developerKey=api_key)
        try:
            while len(comments) < max_comments:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=100,
                    pageToken=next_page_token,
                    textFormat="plainText"
                )
                response = request.execute()

                for item in response.get("items", []):
                    comment_snippet = item["snippet"]["topLevelComment"]["snippet"]
                    comments.append(comment_snippet["textDisplay"])
                    if len(comments) >= max_comments:
                        break

                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break

            # If we got enough comments, stop
            if comments:
                break

        except Exception as e:
            print(f"API key {api_key} failed or quota exceeded: {e}")
            continue  # Try next key

    return comments