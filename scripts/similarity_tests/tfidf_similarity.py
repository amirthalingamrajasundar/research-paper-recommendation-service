# scripts/similarity_tests/tfidf_similarity.py
"""Quick script to compare two sentences using TF-IDF."""
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings

# Load trained TF-IDF vectorizer
vectorizer_path = settings.model.tfidf.paths.vectorizer
vectorizer = joblib.load(vectorizer_path)

# Two sentences to compare
sentence1 = "Transformer model"
sentence2 = "Attention mechanism"

# Generate TF-IDF vectors
emb1 = vectorizer.transform([sentence1]).toarray()[0]
emb2 = vectorizer.transform([sentence2]).toarray()[0]

# Cosine similarity
score = cosine_similarity([emb1], [emb2])[0][0]

print(f"Sentence 1: {sentence1}")
print(f"Sentence 2: {sentence2}")
print(f"Cosine Similarity (TF-IDF): {score:.4f}")
