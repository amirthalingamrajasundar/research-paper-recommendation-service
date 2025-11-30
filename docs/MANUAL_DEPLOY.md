# Manual Deployment Guide

## Prerequisites

```powershell
gcloud auth login
```

---

## 1. TF-IDF Service

```powershell
# Define the Image URL
$IMAGE_URL = "asia-south1-docker.pkg.dev/amirthalingam/scholar-stream/tfidf"

# Build the image locally (from project root)
docker build -t $IMAGE_URL -f api/services/tfidf/Dockerfile .

# Push the image to Google Artifact Registry
docker push $IMAGE_URL

# Deploy to Cloud Run
gcloud run deploy scholar-stream-tfidf `
  --image $IMAGE_URL `
  --platform managed `
  --region asia-south1 `
  --allow-unauthenticated `
  --memory 1Gi `
  --cpu 1 `
  --min-instances 0 `
  --max-instances 3 `
  --project amirthalingam

# Test
curl -i https://scholar-stream-tfidf-142896340374.asia-south1.run.app/health
```

---

## 2. Sentence Transformer Service (Base)

```powershell
# Define the Image URL
$IMAGE_URL = "asia-south1-docker.pkg.dev/amirthalingam/scholar-stream/sentence-transformer"

# Build the image locally (from project root)
docker build -t $IMAGE_URL -f api/services/sentence_transformer/Dockerfile .

# Push the image to Google Artifact Registry
docker push $IMAGE_URL

# Deploy to Cloud Run
gcloud run deploy scholar-stream-sentence-transformer `
  --image $IMAGE_URL `
  --platform managed `
  --region asia-south1 `
  --allow-unauthenticated `
  --memory 2Gi `
  --cpu 1 `
  --min-instances 0 `
  --max-instances 3 `
  --project amirthalingam

# Test
curl -i https://scholar-stream-sentence-transformer-142896340374.asia-south1.run.app/health
```

---

## 3. Fine-tuned Sentence Transformer Service

```powershell
# Define the Image URL
$IMAGE_URL = "asia-south1-docker.pkg.dev/amirthalingam/scholar-stream/finetuned"

# Build the image locally (from project root)
docker build -t $IMAGE_URL -f api/services/finetuned/Dockerfile .

# Push the image to Google Artifact Registry
docker push $IMAGE_URL

# Deploy to Cloud Run
gcloud run deploy scholar-stream-finetuned `
  --image $IMAGE_URL `
  --platform managed `
  --region asia-south1 `
  --allow-unauthenticated `
  --memory 2Gi `
  --cpu 1 `
  --min-instances 0 `
  --max-instances 3 `
  --project amirthalingam

# Test
curl -i https://scholar-stream-finetuned-142896340374.asia-south1.run.app/health
```

---

## 4. Gateway Service

> **Note:** Deploy the gateway **after** all backend services are running. You need the backend URLs for environment variables.

```powershell
# Get backend service URLs
$TFIDF_URL = gcloud run services describe scholar-stream-tfidf --platform managed --region asia-south1 --format "value(status.url)" --project amirthalingam
$ST_URL = gcloud run services describe scholar-stream-sentence-transformer --platform managed --region asia-south1 --format "value(status.url)" --project amirthalingam
$FT_URL = gcloud run services describe scholar-stream-finetuned --platform managed --region asia-south1 --format "value(status.url)" --project amirthalingam

Write-Host "Backend URLs:"
Write-Host "  TF-IDF: $TFIDF_URL"
Write-Host "  Sentence Transformer: $ST_URL"
Write-Host "  Fine-tuned: $FT_URL"

# Define the Image URL
$IMAGE_URL = "asia-south1-docker.pkg.dev/amirthalingam/scholar-stream/gateway"

# Build the image locally (from project root)
docker build -t $IMAGE_URL -f api/gateway/Dockerfile .

# Push the image to Google Artifact Registry
docker push $IMAGE_URL

# Deploy to Cloud Run with backend URLs
gcloud run deploy scholar-stream-gateway `
  --image $IMAGE_URL `
  --platform managed `
  --region asia-south1 `
  --allow-unauthenticated `
  --memory 512Mi `
  --cpu 1 `
  --min-instances 0 `
  --max-instances 5 `
  --set-env-vars "TFIDF_SERVICE_URL=$TFIDF_URL,ST_SERVICE_URL=$ST_URL,FT_SERVICE_URL=$FT_URL" `
  --project amirthalingam

# Test
curl -i https://scholar-stream-gateway-142896340374.asia-south1.run.app/health
```

---

## 5. Verify Deployment

```powershell
# Get Gateway URL
$GATEWAY_URL = gcloud run services describe scholar-stream-gateway --platform managed --region asia-south1 --format "value(status.url)" --project amirthalingam

# Test search endpoint
curl "$GATEWAY_URL/api/v1/scholar-stream/search?q=machine+learning&model=tfidf&limit=3"

# Test with different models
curl "$GATEWAY_URL/api/v1/scholar-stream/search?q=neural+networks&model=base_transformer&limit=3"
curl "$GATEWAY_URL/api/v1/scholar-stream/search?q=deep+learning&model=fine_tuned_transformer&limit=3"

# Test recommendations by paper ID
curl "$GATEWAY_URL/api/v1/scholar-stream/recommendations?paper_id=0704.0001&model=tfidf&limit=5"
curl "$GATEWAY_URL/api/v1/scholar-stream/recommendations?paper_id=0704.0001&model=base_transformer&limit=5"
curl "$GATEWAY_URL/api/v1/scholar-stream/recommendations?paper_id=0704.0001&model=fine_tuned_transformer&limit=5"

# Test get paper details by ID
curl "$GATEWAY_URL/api/v1/scholar-stream/paper/0704.0001"
curl "$GATEWAY_URL/api/v1/scholar-stream/paper/0704.0002"
```