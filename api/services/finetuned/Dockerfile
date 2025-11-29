# Fine-tuned Sentence Transformer service Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn[standard]==0.27.0 \
    pandas==2.1.4 \
    pyarrow==14.0.2 \
    scikit-learn==1.3.2 \
    sentence-transformers==2.2.2 \
    torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# Copy shared code
COPY api/shared /app/api/shared

# Copy service code
COPY api/services/finetuned /app/api/services/finetuned

# Copy data and model artifacts (baked into image)
COPY data/processed/index_100k.parquet /app/data/processed/index_100k.parquet
COPY models/finetuned_st/model /app/models/finetuned_st/model
COPY models/finetuned_st/embeddings.npy /app/models/finetuned_st/embeddings.npy

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV DATA_PATH=/app/data/processed/index_100k.parquet
ENV MODEL_PATH=/app/models/finetuned_st/model
ENV EMBEDDINGS_PATH=/app/models/finetuned_st/embeddings.npy

# Expose port
EXPOSE 8080

# Run the service
CMD ["uvicorn", "api.services.finetuned.main:app", "--host", "0.0.0.0", "--port", "8080"]
