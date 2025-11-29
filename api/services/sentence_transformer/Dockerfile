# Sentence Transformer service Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (sentence-transformers pre-installed to avoid cold start download)
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pandas \
    pyarrow \
    scikit-learn \
    sentence-transformers \
    torch --extra-index-url https://download.pytorch.org/whl/cpu

# Pre-download the base model (baked into image)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy shared code
COPY api/shared /app/api/shared

# Copy service code
COPY api/services/sentence_transformer /app/api/services/sentence_transformer

# Copy data artifacts (baked into image)
COPY data/processed/index_100k.parquet /app/data/processed/index_100k.parquet
COPY models/sentence_transformer/embeddings.npy /app/models/sentence_transformer/embeddings.npy

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV DATA_PATH=/app/data/processed/index_100k.parquet
ENV EMBEDDINGS_PATH=/app/models/sentence_transformer/embeddings.npy
ENV BASE_MODEL=all-MiniLM-L6-v2

# Expose port
EXPOSE 8080

# Run the service
CMD ["uvicorn", "api.services.sentence_transformer.main:app", "--host", "0.0.0.0", "--port", "8080"]
