import tensorflow as tf
import pickle
import numpy as np
import fasttext
from ml.base import BaseMLModel

DEVANAGARI_MODEL_PATH = "models/devanagari_models/devanagari_model.h5"
ASPECT_ENCODER_PATH   = "models/devanagari_models/aspect_encoder.pkl"
SENTIMENT_ENCODER_PATH = "models/devanagari_models/sentiment_encoder.pkl"
FASTTEXT_MODEL_PATH   = "embeddings/cc.ne.300.bin"

MAX_LEN = 15

class DevanagariModel(BaseMLModel):

    def load(self):
        self.model = tf.keras.models.load_model(DEVANAGARI_MODEL_PATH)

        with open(ASPECT_ENCODER_PATH, "rb") as f:
            self.aspect_encoder = pickle.load(f)

        with open(SENTIMENT_ENCODER_PATH, "rb") as f:
            self.sentiment_encoder = pickle.load(f)

        self.ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)

    def preprocess(self, text):
        if isinstance(text, str):
            text = [text]

        embeddings = []
        for comment in text:
            tokens = comment.split()
            vectors = [self.ft_model.get_word_vector(tok) for tok in tokens]

            if len(vectors) < MAX_LEN:
                pad = [np.zeros(300)] * (MAX_LEN - len(vectors))
                vectors.extend(pad)
            else:
                vectors = vectors[:MAX_LEN]

            embeddings.append(np.array(vectors))

        return np.stack(embeddings)

    def predict(self, text: str):
        X = self.preprocess(text)
        sentiment_probs, aspect_probs = self.model.predict(X)

        sentiment_label = self.sentiment_encoder.inverse_transform(
            [np.argmax(sentiment_probs[0])]
        )[0]

        aspect_label = self.aspect_encoder.inverse_transform(
            [np.argmax(aspect_probs[0])]
        )[0]

        return sentiment_label, aspect_label