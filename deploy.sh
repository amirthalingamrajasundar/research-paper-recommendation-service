#!/bin/bash
# Deployment script for Google Cloud Run

set -e

PROJECT_ID=${PROJECT_ID:-"your-gcp-project-id"}
REGION=${REGION:-"us-central1"}

echo "Deploying Scholar Stream services to Google Cloud Run"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"

# Function to deploy a service
deploy_service() {
    local service=$1
    local dockerfile=$2
    local memory=${3:-"2Gi"}
    
    echo ""
    echo "=========================================="
    echo "Deploying $service..."
    echo "=========================================="
    
    # Build and push
    gcloud builds submit \
        --tag gcr.io/$PROJECT_ID/scholar-stream-$service \
        --project $PROJECT_ID \
        -f $dockerfile \
        .
    
    # Deploy to Cloud Run
    gcloud run deploy scholar-stream-$service \
        --image gcr.io/$PROJECT_ID/scholar-stream-$service \
        --platform managed \
        --region $REGION \
        --allow-unauthenticated \
        --memory $memory \
        --cpu 1 \
        --min-instances 0 \
        --max-instances 3 \
        --project $PROJECT_ID
    
    # Get the service URL
    URL=$(gcloud run services describe scholar-stream-$service \
        --platform managed \
        --region $REGION \
        --format 'value(status.url)' \
        --project $PROJECT_ID)
    
    echo "$service deployed at: $URL"
}

# Deploy backend services first
deploy_service "tfidf" "api/services/tfidf/Dockerfile" "1Gi"
deploy_service "sentence-transformer" "api/services/sentence_transformer/Dockerfile" "2Gi"
deploy_service "finetuned" "api/services/finetuned/Dockerfile" "2Gi"

# Get backend service URLs
TFIDF_URL=$(gcloud run services describe scholar-stream-tfidf --platform managed --region $REGION --format 'value(status.url)' --project $PROJECT_ID)
ST_URL=$(gcloud run services describe scholar-stream-sentence-transformer --platform managed --region $REGION --format 'value(status.url)' --project $PROJECT_ID)
FT_URL=$(gcloud run services describe scholar-stream-finetuned --platform managed --region $REGION --format 'value(status.url)' --project $PROJECT_ID)

echo ""
echo "Backend services deployed:"
echo "  TF-IDF: $TFIDF_URL"
echo "  Sentence Transformer: $ST_URL"
echo "  Fine-tuned: $FT_URL"

# Deploy gateway with backend URLs
echo ""
echo "=========================================="
echo "Deploying gateway with backend URLs..."
echo "=========================================="

gcloud builds submit \
    --tag gcr.io/$PROJECT_ID/scholar-stream-gateway \
    --project $PROJECT_ID \
    -f api/gateway/Dockerfile \
    .

gcloud run deploy scholar-stream-gateway \
    --image gcr.io/$PROJECT_ID/scholar-stream-gateway \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 512Mi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 5 \
    --set-env-vars "TFIDF_SERVICE_URL=$TFIDF_URL,ST_SERVICE_URL=$ST_URL,FT_SERVICE_URL=$FT_URL" \
    --project $PROJECT_ID

GATEWAY_URL=$(gcloud run services describe scholar-stream-gateway --platform managed --region $REGION --format 'value(status.url)' --project $PROJECT_ID)

echo ""
echo "=========================================="
echo "Deployment complete!"
echo "=========================================="
echo ""
echo "Gateway URL: $GATEWAY_URL"
echo ""
echo "API Endpoints:"
echo "  Search: $GATEWAY_URL/api/v1/scholar-stream/search?q=<query>&model=<model>"
echo "  Recommendations: $GATEWAY_URL/api/v1/scholar-stream/recommendations?paper_id=<id>&model=<model>"
echo "  Paper Details: $GATEWAY_URL/api/v1/scholar-stream/paper/<paper_id>"
echo ""
echo "Models: tfidf, base_transformer, fine_tuned_transformer"
