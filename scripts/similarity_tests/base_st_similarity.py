# scripts/similarity_tests/base_st_similarity.py
"""Quick script to compare two sentences using base Sentence Transformer."""
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings

# Load base Sentence Transformer
model_name = settings.model.sentence_transformer.base_model
model = SentenceTransformer(model_name)
print(f"Loaded model: {model_name}")

# Two sentences to compare
sentence1 = "Transformer model"
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
print(f"Cosine Similarity (Base ST): {score:.4f}")
