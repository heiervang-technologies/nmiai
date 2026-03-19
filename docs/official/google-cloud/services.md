# Google Cloud Services & Tools - NM i AI 2026

## Hosting Options

**Cloud Run** is the primary deployment choice for Tripletex and Astar Island tasks. Provides containerized API deployment with automatic scaling and HTTPS. Start here.

**Compute Engine** serves teams requiring GPU acceleration or persistent background processes - a secondary option when Cloud Run's serverless model proves insufficient.

## AI & Machine Learning Services

| Service | Purpose |
|---------|---------|
| **Vertex AI** | Managed platform with API access to Gemini and other models |
| **Model Garden** | Browse and deploy pre-trained models (Gemini, Llama, Mistral) |
| **AI Studio** | Interactive environment for rapid prototyping and prompt engineering |

Code integration:
```python
import vertexai
from vertexai.generative_models import GenerativeModel

# Call Gemini from Cloud Run
# pip install google-cloud-aiplatform
```

## Data & Storage

- **Cloud Storage** - Datasets, model weights, and logs via bucket-based file management
- **Cloud SQL** - Managed relational databases (PostgreSQL/MySQL)
- **BigQuery** - Large-scale dataset analysis using SQL queries

## Development Environment

- **Cloud Shell** - Free terminal with pre-installed Python, Docker, gcloud CLI
- **Cloud Shell Editor** - Browser-based VS Code
- **Gemini Code Assist** - AI coding companion
- **Cloud Build** - Automatic Docker image construction

## Team Collaboration

The @gcplab.me account integrates with Gmail, Google Docs, Google Chat, and NotebookLM for distributed team workflows.
