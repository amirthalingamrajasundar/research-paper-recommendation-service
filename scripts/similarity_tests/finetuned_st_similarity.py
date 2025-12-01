# scripts/similarity_tests/finetuned_st_similarity.py
"""Quick script to compare two sentences using fine-tuned Sentence Transformer."""
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings

# Load fine-tuned Sentence Transformer
model_path = settings.model.sentence_transformer.fine_tuning.paths.model
model = SentenceTransformer(str(model_path))
print(f"Loaded model: {model_path}")

# Two sentences to compare
sentence1 = "Soil"
sentence2 = "Attention mechanism"

# Generate embeddings
emb1 = model.encode(sentence1)
emb2 = model.encode(sentence2)

# Cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

score = cosine_similarity(emb1, emb2)

print(f"Sentence 1: {sentence1}")
print(f"Sentence 2: {sentence2}")
print(f"Cosine Similarity (Fine-tuned ST): {score:.4f}")
