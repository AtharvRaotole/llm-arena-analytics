#!/bin/bash
# GCP Cloud Run + Cloud SQL Deployment Script for LLM Arena Analytics
# This script deploys the application to Google Cloud Platform

set -e

echo "=========================================="
echo "GCP Deployment Setup"
echo "=========================================="
echo ""

# Configuration
PROJECT_ID=${GCP_PROJECT_ID}
REGION=${GCP_REGION:-us-central1}
SERVICE_ACCOUNT=${SERVICE_ACCOUNT:-llm-arena-sa}
DB_INSTANCE_NAME=${DB_INSTANCE_NAME:-llm-arena-db}
DB_NAME=${DB_NAME:-llm_arena_analytics}
DB_USER=${DB_USER:-postgres}
DB_PASSWORD=${DB_PASSWORD:-$(openssl rand -base64 32)}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check gcloud CLI
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI not installed. Install from https://cloud.google.com/sdk${NC}"
    exit 1
fi

# Check if logged in
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
    echo -e "${RED}Error: Not logged in to gcloud. Run 'gcloud auth login'${NC}"
    exit 1
fi

# Set project
if [ -z "$PROJECT_ID" ]; then
    echo -e "${YELLOW}Enter GCP Project ID:${NC}"
    read PROJECT_ID
fi

gcloud config set project $PROJECT_ID

echo -e "${GREEN}Step 1: Enabling Required APIs...${NC}"
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    sqladmin.googleapis.com \
    vpcaccess.googleapis.com \
    secretmanager.googleapis.com \
    --project=$PROJECT_ID

echo ""
echo -e "${GREEN}Step 2: Creating Cloud SQL PostgreSQL Instance...${NC}"
# Create Cloud SQL instance
if ! gcloud sql instances describe $DB_INSTANCE_NAME --project=$PROJECT_ID &>/dev/null; then
    gcloud sql instances create $DB_INSTANCE_NAME \
        --database-version=POSTGRES_15 \
        --tier=db-f1-micro \
        --region=$REGION \
        --root-password=$DB_PASSWORD \
        --backup-start-time=03:00 \
        --enable-bin-log \
        --project=$PROJECT_ID

    echo -e "${YELLOW}Waiting for Cloud SQL instance to be ready...${NC}"
    gcloud sql instances wait $DB_INSTANCE_NAME --project=$PROJECT_ID
else
    echo -e "${YELLOW}Cloud SQL instance already exists${NC}"
fi

# Create database
gcloud sql databases create $DB_NAME \
    --instance=$DB_INSTANCE_NAME \
    --project=$PROJECT_ID 2>/dev/null || echo "Database may already exist"

# Create user
gcloud sql users create $DB_USER \
    --instance=$DB_INSTANCE_NAME \
    --password=$DB_PASSWORD \
    --project=$PROJECT_ID 2>/dev/null || echo "User may already exist"

# Get connection name
CONNECTION_NAME=$(gcloud sql instances describe $DB_INSTANCE_NAME \
    --project=$PROJECT_ID \
    --format="value(connectionName)")

echo -e "${GREEN}Cloud SQL Connection Name: $CONNECTION_NAME${NC}"
echo ""

echo -e "${GREEN}Step 3: Storing Secrets in Secret Manager...${NC}"
# Store database password in Secret Manager
echo -n "$DB_PASSWORD" | gcloud secrets create db-password \
    --data-file=- \
    --project=$PROJECT_ID 2>/dev/null || \
    echo -n "$DB_PASSWORD" | gcloud secrets versions add db-password \
    --data-file=- \
    --project=$PROJECT_ID

echo ""
echo -e "${GREEN}Step 4: Building Docker Images...${NC}"
echo "Building backend image...${NC}"
# Build and push backend image
cd ../backend
gcloud builds submit --tag gcr.io/$PROJECT_ID/llm-arena-backend:latest --project=$PROJECT_ID

echo ""
echo -e "${GREEN}Building frontend image...${NC}"
cd ../frontend
gcloud builds submit --tag gcr.io/$PROJECT_ID/llm-arena-frontend:latest --project=$PROJECT_ID

cd ..

echo ""
echo -e "${GREEN}Step 5: Creating VPC Connector...${NC}"
VPC_CONNECTOR_NAME="llm-arena-connector"
if ! gcloud compute networks vpc-access connectors describe $VPC_CONNECTOR_NAME \
    --region=$REGION --project=$PROJECT_ID &>/dev/null; then
    gcloud compute networks vpc-access connectors create $VPC_CONNECTOR_NAME \
        --region=$REGION \
        --network=default \
        --range=10.8.0.0/28 \
        --min-instances=2 \
        --max-instances=3 \
        --project=$PROJECT_ID
fi

echo ""
echo -e "${GREEN}Step 6: Deploying Backend to Cloud Run...${NC}"
gcloud run deploy llm-arena-backend \
    --image gcr.io/$PROJECT_ID/llm-arena-backend:latest \
    --platform managed \
    --region=$REGION \
    --allow-unauthenticated \
    --add-cloudsql-instances=$CONNECTION_NAME \
    --vpc-connector=$VPC_CONNECTOR_NAME \
    --set-env-vars="DB_HOST=/cloudsql/$CONNECTION_NAME,DB_NAME=$DB_NAME,DB_USER=$DB_USER" \
    --set-secrets="DB_PASSWORD=db-password:latest" \
    --memory=2Gi \
    --cpu=2 \
    --timeout=300 \
    --max-instances=10 \
    --project=$PROJECT_ID

BACKEND_URL=$(gcloud run services describe llm-arena-backend \
    --region=$REGION \
    --project=$PROJECT_ID \
    --format="value(status.url)")

echo -e "${GREEN}Backend URL: $BACKEND_URL${NC}"
echo ""

echo -e "${GREEN}Step 7: Deploying Frontend to Cloud Run...${NC}"
gcloud run deploy llm-arena-frontend \
    --image gcr.io/$PROJECT_ID/llm-arena-frontend:latest \
    --platform managed \
    --region=$REGION \
    --allow-unauthenticated \
    --set-env-vars="API_BASE_URL=$BACKEND_URL" \
    --memory=1Gi \
    --cpu=1 \
    --timeout=300 \
    --max-instances=5 \
    --project=$PROJECT_ID

FRONTEND_URL=$(gcloud run services describe llm-arena-frontend \
    --region=$REGION \
    --project=$PROJECT_ID \
    --format="value(status.url)")

echo -e "${GREEN}Frontend URL: $FRONTEND_URL${NC}"
echo ""

echo -e "${GREEN}Step 8: Setting up Monitoring...${NC}"
# Create alerting policy for high error rate
gcloud alpha monitoring policies create \
    --notification-channels=$(gcloud alpha monitoring channels list --format="value(name)" | head -1) \
    --display-name="High Error Rate" \
    --condition-display-name="Error rate > 5%" \
    --condition-threshold-value=0.05 \
    --condition-threshold-duration=300s \
    --project=$PROJECT_ID 2>/dev/null || echo "Monitoring policy creation skipped"

echo ""
echo "=========================================="
echo -e "${GREEN}Deployment Summary${NC}"
echo "=========================================="
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Backend URL: $BACKEND_URL"
echo "Frontend URL: $FRONTEND_URL"
echo "Cloud SQL Instance: $DB_INSTANCE_NAME"
echo "Database Name: $DB_NAME"
echo "Database User: $DB_USER"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Run database migrations:"
echo "   gcloud sql connect $DB_INSTANCE_NAME --user=$DB_USER"
echo "   Then run: \\i schema.sql"
echo "2. Set up custom domain (optional)"
echo "3. Configure load balancer with SSL (optional)"
echo ""
echo -e "${RED}IMPORTANT: Database password stored in Secret Manager${NC}"
echo "=========================================="

