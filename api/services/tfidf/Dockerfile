# TF-IDF service Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pandas \
    pyarrow \
    scikit-learn \
    joblib

# Copy shared code
COPY api/shared /app/api/shared

# Copy service code
COPY api/services/tfidf /app/api/services/tfidf

# Copy model artifacts (baked into image)
COPY data/processed/index_100k.parquet /app/data/processed/index_100k.parquet
COPY models/tfidf/vectorizer.joblib /app/models/tfidf/vectorizer.joblib
COPY models/tfidf/embeddings.joblib /app/models/tfidf/embeddings.joblib

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV DATA_PATH=/app/data/processed/index_100k.parquet
ENV VECTORIZER_PATH=/app/models/tfidf/vectorizer.joblib
ENV MATRIX_PATH=/app/models/tfidf/embeddings.joblib

# Expose port
EXPOSE 8080

# Run the service
CMD ["uvicorn", "api.services.tfidf.main:app", "--host", "0.0.0.0", "--port", "8080"]
