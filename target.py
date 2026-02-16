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
# =========================

dataset_path = 'dataset/training_dataset.csv'
df = pd.read_csv(dataset_path, header=None, names=["comment","target","aspect","sentiment"])

# Normalize entire comment before NER
comments = df["comment"].astype(str).apply(full_normalize).tolist()
print("Total comments:", len(comments))


# =========================
# 3. LOAD NER MODEL
# =========================

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

CONFIDENCE_THRESHOLD = 0.65
comment_entities = []
entity_frequency = defaultdict(int)

for comment in comments:
    detected = ner_pipeline(comment)
    merged_entities = []

    # Merge consecutive entities with same label
    i = 0
    while i < len(detected):
        ent = detected[i]
        if ent["score"] < CONFIDENCE_THRESHOLD:
            i += 1
            continue

        word = full_normalize(ent["word"])
        label = ent["entity_group"]

        # Merge with next tokens of same label
        j = i + 1
        merged_word = word
        while j < len(detected) and detected[j]["entity_group"] == label:
            merged_word += " " + full_normalize(detected[j]["word"])
            j += 1

        merged_entities.append({
            "word": merged_word,
            "label": label,
            "score": float(ent["score"])
        })
        entity_frequency[merged_word] += 1
        i = j

    comment_entities.append(merged_entities)

print("NER extraction + multiword merge complete")
print("Unique raw entities:", len(entity_frequency))


unique_entities = list(entity_frequency.keys())


# =========================
# 5. SEMANTIC NORMALIZATION (EMBEDDINGS)
# =========================

embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
entity_embeddings = embedding_model.encode(unique_entities)

# DBSCAN clustering
dbscan = DBSCAN(eps=0.35, min_samples=1, metric='cosine')
cluster_ids = dbscan.fit_predict(entity_embeddings)

cluster_map = defaultdict(list)
for entity, cluster_id in zip(unique_entities, cluster_ids):
    cluster_map[cluster_id].append(entity)

print("Total semantic clusters:", len(cluster_map))


# =========================
# 6. CANONICAL TARGET SELECTION
# =========================

canonical_map = {}

for cluster_id, variants in cluster_map.items():
    # compute cluster centroid
    centroid = np.mean([entity_embeddings[unique_entities.index(v)] for v in variants], axis=0)

    # score each variant: combine frequency + distance to centroid
    scores = []
    for v in variants:
        freq_score = entity_frequency[v]
        dist_score = 1 - cosine_distances([entity_embeddings[unique_entities.index(v)]], [centroid])[0][0]
        total_score = freq_score + dist_score  # weight can be tuned
        scores.append((v, total_score))

    canonical = max(scores, key=lambda x: x[1])[0]

    for v in variants:
        canonical_map[v] = canonical


# =========================
# 7. APPLY CANONICAL MAPPING
# =========================

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


# =========================
# 8. FINAL DROPDOWN TARGETS
# =========================

dropdown_targets = sorted({
    e["canonical"]
    for comment in normalized_comment_entities
    for e in comment
})

print("\nFINAL DROPDOWN TARGETS:")
print(dropdown_targets)
print("Total dropdown targets:", len(dropdown_targets))
