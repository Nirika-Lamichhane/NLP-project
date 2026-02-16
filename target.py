import pandas as pd
import numpy as np
import unicodedata
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict


# =========================================================
# 1. TEXT NORMALIZATION FUNCTIONS
# =========================================================

def unicode_normalize(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower().strip()
    text = " ".join(text.split())
    return text


def devanagari_lexical_normalization(text):

    # -------------------------
    # Nukta normalization
    # -------------------------
    NUKTA_MAP = {
        'क़': 'क', 'ख़': 'ख', 'ग़': 'ग', 'ज़': 'ज',
        'ड़': 'ड', 'ढ़': 'ढ', 'फ़': 'फ', 'ऱ': 'र', 'य़': 'य'
    }
    for k, v in NUKTA_MAP.items():
        text = text.replace(k, v)

    # -------------------------
    # Nasal normalization
    # -------------------------
    NASAL_MAP = {
        'ङ्ग': 'ंग','ङ्घ': 'ंघ','ङ्क': 'ंक','ङ्ख': 'ंख',
        'ञ्च': 'ंच','ञ्छ': 'ंछ','ञ्ज': 'ंज','ञ्झ': 'ंझ',
        'ण्ड': 'ंड','ण्ठ': 'ंठ','ण्ट': 'ंट','ण्ढ': 'ंढ',
        'न्द': 'ंद','न्ध': 'ंध','न्त': 'ंत','न्थ': 'ंथ',
        'म्प': 'ंप','म्भ': 'ंभ'
    }
    for k, v in NASAL_MAP.items():
        text = text.replace(k, v)

    # collapse repeated letters
    text = re.sub(r'(.)\1+', r'\1', text)

    # remove honorific modifiers
    modifiers = ['जी', 'दाइ', 'चोर', 'भाइ']
    for mod in modifiers:
        text = re.sub(r'\b' + mod + r'\b', '', text)

    text = " ".join(text.split())
    return text.strip()


def full_normalize(text):
    """
    MASTER NORMALIZATION PIPELINE
    """
    text = unicode_normalize(text)
    text = devanagari_lexical_normalization(text)
    return text


# =========================================================
# 2. LOAD DATA
# =========================================================

dataset_path = 'dataset/training_dataset.csv'

df = pd.read_csv(
    dataset_path,
    header=None,
    names=["comment", "target", "aspect", "sentiment"]
)

# normalize entire sentence BEFORE NER
comments = df["comment"].astype(str).apply(full_normalize).tolist()

print("Total comments:", len(comments))


# =========================================================
# 3. LOAD NER MODEL
# =========================================================

ner_model_name = "Davlan/xlm-roberta-base-ner-hrl"

tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
model = AutoModelForTokenClassification.from_pretrained(ner_model_name)

ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)


# =========================================================
# 4. ENTITY EXTRACTION + ENTITY NORMALIZATION
# =========================================================

CONFIDENCE_THRESHOLD = 0.65

comment_entities = []
entity_frequency = defaultdict(int)

for comment in comments:

    detected = ner_pipeline(comment)
    filtered = []

    for ent in detected:
        if ent["score"] >= CONFIDENCE_THRESHOLD:

            # normalize entity AGAIN at token level
            word = full_normalize(ent["word"])

            filtered.append({
                "word": word,
                "label": ent["entity_group"],
                "score": float(ent["score"])
            })

            entity_frequency[word] += 1

    comment_entities.append(filtered)

print("NER extraction complete")
print("Unique raw entities:", len(entity_frequency))


unique_entities = list(entity_frequency.keys())


# =========================================================
# 5. SEMANTIC NORMALIZATION (EMBEDDING CLUSTERING)
# =========================================================

embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

entity_embeddings = embedding_model.encode(unique_entities)

clustering = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=0.35,
    metric="cosine",
    linkage="average"
)

cluster_ids = clustering.fit_predict(entity_embeddings)

cluster_map = defaultdict(list)
for entity, cluster_id in zip(unique_entities, cluster_ids):
    cluster_map[cluster_id].append(entity)

print("Total semantic clusters:", len(cluster_map))


# =========================================================
# 6. CANONICAL TARGET SELECTION
# =========================================================

canonical_map = {}

for cluster_id, variants in cluster_map.items():

    # choose most frequent spelling as canonical
    canonical = max(variants, key=lambda v: entity_frequency[v])

    for v in variants:
        canonical_map[v] = canonical


# =========================================================
# 7. APPLY CANONICAL MAPPING
# =========================================================

normalized_comment_entities = []

for ents in comment_entities:
    normalized = []

    for e in ents:
        canonical = canonical_map.get(e["word"], e["word"])

        normalized.append({
            "canonical": canonical,
            "label": e["label"],
            "score": e["score"]
        })

    normalized_comment_entities.append(normalized)


# =========================================================
# 8. FINAL TARGET LIST
# =========================================================

dropdown_targets = sorted({
    e["canonical"]
    for comment in normalized_comment_entities
    for e in comment
})

print("\nFINAL DROPDOWN TARGETS:")
print(dropdown_targets)
print("Total dropdown targets:", len(dropdown_targets))
