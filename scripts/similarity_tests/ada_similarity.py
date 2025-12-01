# quick_ada_similarity.py
"""Quick script to compare two sentences using ada-002 embeddings."""
import numpy as np
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize ada-002
embeddings_model = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    api_version="2024-12-01-preview",
    model=os.getenv("EMBEDDING_MODEL_NAME"),
)

# Two sentences to compare
sentence1 = "Transformer model"
sentence2 = "Attention mechanism"

# Generate embeddings
emb1 = embeddings_model.embed_query(sentence1)
emb2 = embeddings_model.embed_query(sentence2)

# Cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

score = cosine_similarity(emb1, emb2)

print(f"Sentence 1: {sentence1}")
print(f"Sentence 2: {sentence2}")
print(f"Cosine Similarity: {score:.4f}")