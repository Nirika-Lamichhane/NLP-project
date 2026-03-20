import re
import numpy as np
import fasttext
from ml.base import BaseMLModel

MODEL_PATH = "models/language_identifier/fasttext_language_model.bin"
CONFIDENCE_THRESHOLD = 0.5

class LanguageIdentifier(BaseMLModel):

    def load(self):
        self.ft_model = fasttext.load_model(MODEL_PATH)

    def clean(self, text: str) -> str:
        text = str(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def predict(self, text: str) -> dict:
        text = self.clean(text)
        wc = len(text.split())

        if wc < 2:
            return {"language": "UNKNOWN", "confidence": 0.0}

        labels, probs = self.ft_model.predict(text)
        label = labels[0].replace("__label__", "")
        confidence = float(np.asarray(probs)[0])

        if confidence < CONFIDENCE_THRESHOLD:
            label = "UNKNOWN"

        return {
            "language": label,
            "confidence": round(confidence, 4)
        }