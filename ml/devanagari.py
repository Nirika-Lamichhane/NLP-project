# ml/devanagari.py

import tensorflow as tf
import pickle
import numpy as np
import fasttext

# -----------------------
# Paths
# -----------------------

DEVANAGARI_MODEL_PATH = "models/devanagari_models/devanagari_model.h5"
ASPECT_ENCODER_PATH = "models/devanagari_models/aspect_encoder.pkl"
SENTIMENT_ENCODER_PATH = "models/devanagari_models/sentiment_encoder.pkl"
FASTTEXT_MODEL_PATH = "embeddings/cc.ne.300.bin"

# -----------------------
# Load models
# -----------------------

model = tf.keras.models.load_model(DEVANAGARI_MODEL_PATH)

with open(ASPECT_ENCODER_PATH, "rb") as f:
    aspect_encoder = pickle.load(f)

with open(SENTIMENT_ENCODER_PATH, "rb") as f:
    sentiment_encoder = pickle.load(f)

ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)

MAX_LEN = 15


# -----------------------
# Preprocessing
# -----------------------

def preprocess(text, max_len=MAX_LEN):

    if isinstance(text, str):
        text = [text]

    embeddings = []

    for comment in text:
        tokens = comment.split()
        vectors = [ft_model.get_word_vector(tok) for tok in tokens]

        if len(vectors) < max_len:
            pad = [np.zeros(300)] * (max_len - len(vectors))
            vectors.extend(pad)
        else:
            vectors = vectors[:max_len]

        embeddings.append(np.array(vectors))

    return np.stack(embeddings)


# -----------------------
# Prediction function
# -----------------------

def predict_sentiment(text):

    X = preprocess(text)

    sentiment_probs, aspect_probs = model.predict(X)

    sentiment_label = sentiment_encoder.inverse_transform(
        [np.argmax(sentiment_probs[0])]
    )[0]

    aspect_label = aspect_encoder.inverse_transform(
        [np.argmax(aspect_probs[0])]
    )[0]

    return sentiment_label, aspect_label