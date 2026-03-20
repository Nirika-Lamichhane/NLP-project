import json
import torch
import unicodedata
from transformers import AutoTokenizer, AutoModelForTokenClassification
from ml.base import BaseMLModel

MODEL_PATH          = "models/target_model"
GENERIC_TARGETS_PATH = "models/target_model/generic_targets.txt"
LABEL_MAPPINGS_PATH  = "models/target_model/label_mappings.json"

STOPWORDS = {
    "को", "का", "की",
    "ले", "लाई",
    "बाट", "देखि",
    "द्वारा",
    "मा", "सँग", "संग",
    "तर्फ", "माथि", "तल",
    "पछि", "अघि",
    "सम्म", "भर",
    "विरुद्ध", "लागि", "भित्र"
}

class TargetModel(BaseMLModel):

    def load(self):
        # load label mappings
        with open(LABEL_MAPPINGS_PATH, "r", encoding="utf-8") as f:
            mappings = json.load(f)
        self.label2id = mappings["label2id"]
        self.id2label = mappings["id2label"]

        # load generic targets fallback list
        self.generic_targets = []
        with open(GENERIC_TARGETS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                entry = line.strip()
                if entry:
                    self.generic_targets.append(entry)

        # load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, trust_remote_code=True
        )
        self.ner_model = AutoModelForTokenClassification.from_pretrained(
            MODEL_PATH, trust_remote_code=True
        )
        self.ner_model.eval()

    def preprocess_target(self, target: str) -> str:
        target = target.strip()
        target = unicodedata.normalize("NFC", target)
        target = target.replace(" ", "")
        return target

    def remove_stopwords(self, sentence: str) -> str:
        words = sentence.split()
        return " ".join(w for w in words if w not in STOPWORDS)

    def clean_target(self, target: str) -> str:
        words = target.split()
        cleaned = []
        for w in words:
            for sw in STOPWORDS:
                if w.endswith(sw) and len(w) > len(sw):
                    if w.startswith("एमाले"):
                        w = "एमाले"
                        break
                    w = w[:-len(sw)]
                    break
            cleaned.append(w)
        return " ".join(cleaned).strip()

    def predict_ner(self, comment: str):
        words = comment.split()
        inputs = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
        )
        with torch.no_grad():
            outputs = self.ner_model(**inputs).logits

        predictions = torch.argmax(outputs, dim=2)[0].numpy()
        word_ids = inputs.word_ids()

        predicted_tags = []
        previous_word_idx = None
        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx != previous_word_idx:
                predicted_tags.append(self.id2label[str(predictions[idx])])
            previous_word_idx = word_idx

        return words, predicted_tags

    def extract_targets(self, words, tags):
        targets = []
        current = []
        for word, tag in zip(words, tags):
            if tag.startswith("B-"):
                if current:
                    targets.append(" ".join(current))
                    current = []
                current.append(word)
            elif tag.startswith("I-") and current:
                current.append(word)
            else:
                if current:
                    targets.append(" ".join(current))
                    current = []
        if current:
            targets.append(" ".join(current))
        return targets

    def fallback_targets(self, comment: str, existing: list) -> list:
        return [
            g for g in self.generic_targets
            if g in comment and g not in existing
        ]

    def predict(self, comment: str) -> list:
        # Step 1 — NER
        words, tags = self.predict_ner(comment)

        # Step 2 — extract
        ner_targets = self.extract_targets(words, tags)

        # Step 3 — clean
        ner_targets = [self.clean_target(t) for t in ner_targets if t.strip()]

        # Step 4 — fallback if nothing found
        if len(ner_targets) == 0:
            fallback = self.fallback_targets(comment, ner_targets)
            final_targets = list(set(
                self.clean_target(t) for t in fallback
            ))
        else:
            final_targets = list(set(ner_targets))

        return final_targets  # list of target strings