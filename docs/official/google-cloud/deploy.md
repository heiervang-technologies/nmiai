# Cloud Run Deployment Guide - NM i AI 2026

## Overview

Cloud Run enables hosting HTTPS endpoints for the Tripletex and Astar Island competition tasks. The service handles scaling, TLS certificates, and infrastructure automatically.

## Key Steps

### Step 1: Create FastAPI Endpoint
Build a minimal `/solve` endpoint accepting POST requests with task prompts and credentials. The endpoint processes the request and returns `{"status": "completed"}`.

### Step 2: Dockerfile Setup
Use Python 3.11-slim base image with FastAPI and uvicorn dependencies. The container exposes port 8080 and runs the application with uvicorn.

### Step 3: Deploy via gcloud
Execute deployment from Cloud Shell:
```bash
gcloud run deploy SERVICE_NAME \
  --region europe-north1 \
  --allow-unauthenticated \
  --memory 1Gi
```

### Step 4: Submit URL
Copy the generated HTTPS URL from Cloud Run and paste it into the competition submission page at app.ainm.no.

## Optimization Tips

- **Region Selection**: Deploy in `europe-north1` for lower latency versus validators
- **Cold Starts**: Use `--min-instances 1` to maintain always-on capacity during competition
- **Memory**: Default 512 MB suffices for external API calls; increase to 2Gi for local models
- **Updates**: Rerun the same deploy command to automatically build and deploy revisions
- **Logging**: Access logs via `gcloud run services logs read` or Cloud Console

## Task Requirements

Only Tripletex and Astar Island require Cloud Run endpoints. NorgesGruppen Data submissions use ZIP file uploads instead.
