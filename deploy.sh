#!/bin/bash
# Deployment script for Scholar Stream on Google Cloud Run
# Uses Artifact Registry and Cloud Build

set -e

# Configuration
PROJECT_ID=${PROJECT_ID:-"amirthalingam"}
REGION=${REGION:-"asia-south1"}
REPO_NAME="scholar-stream"
REPO_REGION="asia-south1"

echo "=========================================="
echo "Starting Deployment for Scholar Stream"
echo "   Project:  $PROJECT_ID"
echo "   Region:   $REGION"
echo "   Repo:     $REPO_NAME"
echo "=========================================="

# Ensure required APIs are enabled
gcloud services enable artifactregistry.googleapis.com cloudbuild.googleapis.com run.googleapis.com

# ---------------------------------------------------------
# Helper Function: Deploy a Service
# ---------------------------------------------------------
deploy_service() {
    local service_name=$1      # e.g., tfidf
    local dockerfile_path=$2   # e.g., api/services/tfidf/Dockerfile
    local memory=${3:-"1Gi"}   # Default to 1Gi if not set
    
    # Define Image URL for Artifact Registry
    local image_url="$REPO_REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$service_name"

    echo ""
    echo "------------------------------------------"
    echo "Building & Deploying: $service_name"
    echo "------------------------------------------"
    
    # 1. FIX CONTEXT: Temporarily copy Dockerfile to root
    # This allows Docker to see 'api/shared' and other root-level folders
    if [ ! -f "$dockerfile_path" ]; then
        echo "Error: Dockerfile not found at $dockerfile_path"
        exit 1
    fi
    cp "$dockerfile_path" ./Dockerfile

    # 2. Build and Push to Artifact Registry (using root context .)
    echo "   Building image..."
    gcloud builds submit --tag "$image_url" .
    
    # 3. Cleanup temporary Dockerfile
    rm ./Dockerfile
    
    # 4. Deploy to Cloud Run
    echo "   Deploying to Cloud Run..."
    gcloud run deploy scholar-stream-$service_name \
        --image "$image_url" \
        --platform managed \
        --region $REGION \
        --allow-unauthenticated \
        --memory $memory \
        --cpu 1 \
        --min-instances 0 \
        --max-instances 3 \
        --project $PROJECT_ID > /dev/null 2>&1

    # 5. Retrieve and print the URL
    local service_url=$(gcloud run services describe scholar-stream-$service_name \
        --platform managed \
        --region $REGION \
        --format 'value(status.url)' \
        --project $PROJECT_ID)
    
    echo "Success! $service_name is live at: $service_url"
    
    # Return the URL so we can use it later
    echo "$service_url"
}

# ---------------------------------------------------------
# Step 1: Deploy Backend Services
# ---------------------------------------------------------
echo ""
echo ">>> Phase 1: Deploying Backend Services..."

# Capture the output (URL) of each function call
TFIDF_URL=$(deploy_service "tfidf" "api/services/tfidf/Dockerfile" "1Gi")
ST_URL=$(deploy_service "sentence-transformer" "api/services/sentence_transformer/Dockerfile" "2Gi")
FT_URL=$(deploy_service "finetuned" "api/services/finetuned/Dockerfile" "2Gi")

echo ""
echo ">>> Backend Services URLs Captured:"
echo "   TF-IDF:      $TFIDF_URL"
echo "   Transformer: $ST_URL"
echo "   Fine-tuned:  $FT_URL"

# ---------------------------------------------------------
# Step 2: Deploy Gateway (Linked to Backends)
# ---------------------------------------------------------
echo ""
echo ">>> Phase 2: Deploying Gateway..."

GATEWAY_IMAGE="$REPO_REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/gateway"

# Prepare Gateway Dockerfile (Assuming it is at api/gateway/Dockerfile)
# We use the same copy-to-root trick in case Gateway uses shared code
cp "api/gateway/Dockerfile" ./Dockerfile

gcloud builds submit --tag "$GATEWAY_IMAGE" .
rm ./Dockerfile

gcloud run deploy scholar-stream-gateway \
    --image "$GATEWAY_IMAGE" \
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

# ---------------------------------------------------------
# Final Summary
# ---------------------------------------------------------
echo ""
echo "=========================================="
echo " DEPLOYMENT COMPLETE!"
echo "=========================================="
echo "Main Access Point (Gateway):"
echo " $GATEWAY_URL"
echo ""
echo "Service Endpoints:"
echo " - TF-IDF: $TFIDF_URL"
echo " - Sentence Transformer: $ST_URL"
echo " - Fine Tuned: $FT_URL"
echo "=========================================="