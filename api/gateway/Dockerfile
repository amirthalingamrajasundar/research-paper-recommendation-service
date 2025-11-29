# Gateway service Dockerfile - lightweight, no ML dependencies
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn[standard]==0.27.0 \
    httpx==0.26.0

# Copy shared code and gateway
COPY api/shared /app/api/shared
COPY api/gateway /app/api/gateway

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Expose port
EXPOSE 8080

# Run the gateway
CMD ["uvicorn", "api.gateway.main:app", "--host", "0.0.0.0", "--port", "8080"]
