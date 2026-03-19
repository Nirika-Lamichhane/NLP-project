# comment_extractor.py

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


def get_comments(youtube_url, max_comments=10):

    video_id = extract_video_id(youtube_url)

    if not video_id:
        raise ValueError("Invalid YouTube URL")

    comments = []
    next_page_token = None
    key_index = 0

    youtube = build("youtube", "v3", developerKey=API_KEYS[key_index])

    while True:

        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=10,
                pageToken=next_page_token,
                textFormat="plainText"
            )

            response = request.execute()

            for item in response["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)
                if len(comments) >= max_comments:
                    return comments

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

    return comments