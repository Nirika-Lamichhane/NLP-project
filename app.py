# app.py

import streamlit as st
from comment_extractor import get_comments
from pipeline import nlp_pipeline

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="YouTube Comment Analysis",
    layout="wide"
)

st.title("YouTube Comment Sentiment & Aspect Analysis")
st.markdown(
    "Enter a YouTube video URL below, and the app will analyze all comments (Devanagari) "
    "for sentiment and aspects."
)

# -----------------------------
# Input URL
# -----------------------------
youtube_url = st.text_input("Enter YouTube Video URL")

max_comments = st.number_input(
    "Maximum number of comments to process",
    min_value=50,
    max_value=5000,
    value=500,
    step=50
)

# -----------------------------
# Run Pipeline Button
# -----------------------------
if st.button("Analyze Comments"):

    if not youtube_url:
        st.error("Please enter a valid YouTube URL")
    else:
        with st.spinner("Fetching comments and analyzing..."):
            try:
                # 1. Optional: fetch comments (if you want to visualize them)
                comments = get_comments(youtube_url, max_comments=max_comments)

                # 2. Run NLP pipeline
                results = nlp_pipeline(youtube_url)

                st.success(f"Analysis complete! Total comments processed: {len(results)}")

                # -----------------------------
                # Display results in a table
                # -----------------------------
                st.subheader("Sample of Analyzed Comments")
                st.dataframe(results[:20])  # show first 20

                # -----------------------------
                # Optional: Summary stats
                # -----------------------------
                sentiments = [r['sentiment'] for r in results]
                aspects = [r['aspect'] for r in results]

                st.subheader("Sentiment Summary")
                st.bar_chart(pd.Series(sentiments).value_counts())

                st.subheader("Aspect Summary")
                st.bar_chart(pd.Series(aspects).value_counts())

            except Exception as e:
                st.error(f"Error: {e}")