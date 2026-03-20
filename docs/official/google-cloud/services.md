# Services & Tools

A curated list of GCP services relevant to competing in NM i AI. You don't need all of these — pick what fits your approach.

## Hosting Your Endpoint

<div class="table-scroll-wrapper">

| Service | Use case | When to use |
|----|----|----|
| [Cloud Run](https://console.cloud.google.com/run) | Deploy containerized APIs | Tripletex & Astar Island tasks — this is the go-to |
| [Compute Engine](https://console.cloud.google.com/compute) | Full VM (any OS) | Need GPU or persistent server |

</div>

**Recommendation:** Start with Cloud Run. It's simpler and free with your account. Only use Compute Engine if you need a GPU or persistent background processes.

## AI & Machine Learning

<div class="table-scroll-wrapper">

| Service | Use case | When to use |
|----|----|----|
| [Vertex AI](https://console.cloud.google.com/vertex-ai) | Managed ML platform | Access Gemini and other models via API |
| [Model Garden](https://console.cloud.google.com/vertex-ai/model-garden) | Pre-trained model catalog | Browse and deploy models (Gemini, Llama, Mistral) |
| [AI Studio](https://aistudio.google.com) | Experiment with Gemini | Quick prototyping, prompt engineering |

</div>

### Using Vertex AI from Your Endpoint

Call Gemini from your Cloud Run endpoint:

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="python" data-theme="github-dark-default"><code>import vertexai
from vertexai.generative_models import GenerativeModel
 
vertexai.init(project=&quot;your-project-id&quot;, location=&quot;europe-north1&quot;)
model = GenerativeModel(&quot;gemini-2.0-flash&quot;)
 
response = model.generate_content(&quot;Parse this accounting task: ...&quot;)
print(response.text)</code></pre>
</figure>

Install with: `pip install google-cloud-aiplatform`

## Data & Storage

<div class="table-scroll-wrapper">

| Service | Use case | When to use |
|----|----|----|
| [Cloud Storage](https://console.cloud.google.com/storage) | File storage (buckets) | Store datasets, model weights, logs |
| [Cloud SQL](https://console.cloud.google.com/sql) | Managed PostgreSQL/MySQL | Need a relational database |
| [BigQuery](https://console.cloud.google.com/bigquery) | Data warehouse | Analyze large datasets with SQL |

</div>

## Development Tools

<div class="table-scroll-wrapper">

| Tool | How to access | What it does |
|----|----|----|
| Cloud Shell | Console top-right icon | Free terminal with everything pre-installed |
| Cloud Shell Editor | "Open Editor" button | VS Code in the browser |
| Gemini Code Assist | Cloud Shell Editor sidebar | AI coding companion |
| Gemini CLI | `gemini` in Cloud Shell | AI assistant in the terminal |
| Cloud Build | Automatic with `gcloud run deploy --source .` | Builds your Docker images |

</div>

## Collaboration

Your `@gcplab.me` account also works with:

- **Gmail** — communicate with teammates
- **Google Docs** — shared documentation
- **Google Chat** — team messaging
- **NotebookLM** — AI-powered research notebook
