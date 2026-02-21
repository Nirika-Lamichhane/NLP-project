import pandas as pd
import numpy as np
import unicodedata
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from collections import defaultdict


# =========================
# 1. TEXT NORMALIZATION
# =========================

def unicode_normalize(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower().strip()
    text = " ".join(text.split())
    return text


def devanagari_lexical_normalization(text):
    # Nukta normalization
    NUKTA_MAP = {
        'क़': 'क', 'ख़': 'ख', 'ग़': 'ग', 'ज़': 'ज',
        'ड़': 'ड', 'ढ़': 'ढ', 'फ़': 'फ', 'ऱ': 'र', 'य़': 'य'
    }
    for k, v in NUKTA_MAP.items():
        text = text.replace(k, v)

    # Nasal normalization
    NASAL_MAP = {
        'ङ्ग': 'ंग','ङ्घ': 'ंघ','ङ्क': 'ंक','ङ्ख': 'ंख',
        'ञ्च': 'ंच','ञ्छ': 'ंछ','ञ्ज': 'ंज','ञ्झ': 'ंझ',
        'ण्ड': 'ंड','ण्ठ': 'ंठ','ण्ट': 'ंट','ण्ढ': 'ंढ',
        'न्द': 'ंद','न्ध': 'ंध','न्त': 'ंत','न्थ': 'ंथ',
        'म्प': 'ंप','म्भ': 'ंभ'
    }
    for k, v in NASAL_MAP.items():
        text = text.replace(k, v)

    # Collapse repeated letters
    text = re.sub(r'(.)\1+', r'\1', text)

    # Remove modifiers
    modifiers = ['जी', 'दाइ', 'चोर', 'भाइ']
    for mod in modifiers:
        text = re.sub(r'\b' + mod + r'\b', '', text)

    return " ".join(text.split()).strip()


def full_normalize(text):
    text = unicode_normalize(text)
    text = devanagari_lexical_normalization(text)
    return text


# =========================
# 2. LOAD DATA


dataset_path = 'dataset/training_dataset.csv'
df = pd.read_csv(dataset_path, header=None, names=["comment","target","aspect","sentiment"])

# Normalize entire comment before NER
comments = df["comment"].astype(str).apply(full_normalize).tolist()
print("Total comments:", len(comments))


# =========================
# 3. LOAD NER MODEL


ner_model_name = "Davlan/xlm-roberta-base-ner-hrl"
tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
model = AutoModelForTokenClassification.from_pretrained(ner_model_name)

ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)


# =========================
# 4. ENTITY EXTRACTION + MULTI-WORD MERGING
# =========================


from collections import defaultdict

CONFIDENCE_THRESHOLD = 0.65
entity_frequency = defaultdict(int)

for comment in comments:
    detected = ner_pipeline(comment)

    i = 0
    while i < len(detected):
        ent = detected[i]

        # keep only confident PERSON entities
        if ent["entity_group"] not in ["PER","ORG"] or ent["score"] < CONFIDENCE_THRESHOLD:
            i += 1
            continue

        # normalize detected token
        merged_name = full_normalize(ent["word"])

        # merge consecutive PER tokens (multi-word names)
        j = i + 1
        count = 1
    
        while j < len(detected) and detected[j]["entity_group"] == "PER" and count < 2:
            merged_name += " " + full_normalize(detected[j]["word"])
            j += 1
            count += 1

        merged_name = merged_name.strip()

        # count frequency
        if len(merged_name) > 1:
            entity_frequency[merged_name] += 1

        i = j


sorted_entities = sorted(
    entity_frequency.items(),
    key=lambda x: x[1],
    reverse=True
)

print("\nTop detected person names:\n")
for name, freq in sorted_entities[:50]:
    print(name, "->", freq)
