import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sentence_transformers import SentenceTransformer

from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
dataset_path = 'dataset/training_dataset.csv'

df = pd.read_csv(dataset_path, header=None,
                 names=["comment","target","aspect","sentiment"])

comments = df["comment"].astype(str).tolist()



print("Total comments:", len(comments))
ner_model_name = "Davlan/xlm-roberta-base-ner-hrl"

tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
model = AutoModelForTokenClassification.from_pretrained(ner_model_name)

ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)
def normalize_text(text):
    text = text.lower().strip()
    text = " ".join(text.split())
    return text

comment_entities = []
entity_frequency = defaultdict(int)

CONFIDENCE_THRESHOLD = 0.65

for comment in comments:
    detected = ner_pipeline(comment)

    filtered = []
    for ent in detected:
        if ent["score"] >= CONFIDENCE_THRESHOLD:
            word = normalize_text(ent["word"])

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

print("Sample entities:")
print(unique_entities[:20])
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

print("Total clusters:", len(cluster_map))
canonical_map = {}

for cluster_id, variants in cluster_map.items():
    canonical = max(variants, key=lambda v: entity_frequency[v])

    for v in variants:
        canonical_map[v] = canonical
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
dropdown_targets = sorted(list({
    e["canonical"]
    for comment in normalized_comment_entities
    for e in comment
}))

print("\nFINAL DROPDOWN TARGETS:")
print(dropdown_targets)
print("Total dropdown targets:", len(dropdown_targets))

