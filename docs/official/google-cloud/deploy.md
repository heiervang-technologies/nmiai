# Deploy on Cloud Run

Two of the three competition tasks — **Tripletex** and **Astar Island** — require you to host a public HTTPS endpoint that our validators call. Cloud Run is the easiest way to deploy one.

## What is Cloud Run?

Cloud Run takes a Docker container and gives you a public HTTPS URL. You push your code, it handles scaling, TLS, and everything else. You only pay for actual requests (and with your GCP account, it's free).

## Step 1: Write Your Endpoint

Here's a minimal FastAPI endpoint that matches the competition format:

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="python" data-theme="github-dark-default"><code>from fastapi import FastAPI
 
app = FastAPI()
 
@app.get(&quot;/health&quot;)
def health():
    return {&quot;status&quot;: &quot;ok&quot;}
 
@app.post(&quot;/solve&quot;)
async def solve(request: dict):
    prompt = request.get(&quot;prompt&quot;, &quot;&quot;)
    credentials = request.get(&quot;tripletex_credentials&quot;, {})
 
    # Your AI agent logic here:
    # 1. Parse the prompt
    # 2. Call the Tripletex API using the provided credentials
    # 3. Complete the accounting task
 
    return {&quot;status&quot;: &quot;completed&quot;}</code></pre>
</figure>

## Step 2: Create a Dockerfile

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="dockerfile" data-theme="github-dark-default"><code>FROM python:3.11-slim
 
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
 
CMD [&quot;uvicorn&quot;, &quot;main:app&quot;, &quot;--host&quot;, &quot;0.0.0.0&quot;, &quot;--port&quot;, &quot;8080&quot;]</code></pre>
</figure>

And a `requirements.txt`:

    fastapi
    uvicorn[standard]
    requests

## Step 3: Deploy

Open Cloud Shell and run:

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="bash" data-theme="github-dark-default"><code># Clone your repo (or upload files via Cloud Shell Editor)
cd ~
git clone &lt;your-repo-url&gt;
cd your-project
 
# Deploy to Cloud Run (builds and deploys in one command)
gcloud run deploy my-agent \
  --source . \
  --region europe-north1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --timeout 300</code></pre>
</figure>

That's it. Cloud Run builds the Docker image, deploys it, and gives you a URL like:

    https://my-agent-xxxxx-lz.a.run.app

## Step 4: Submit Your URL

1.  Copy the Cloud Run URL
2.  Go to the submission page for your task at [app.ainm.no](https://app.ainm.no)
3.  Paste the URL and submit
4.  Our validators will start calling your endpoint

## Tips

### Use `europe-north1` Region

Deploy in `europe-north1` (Finland) — same region as our validators. Lower latency = faster scoring.

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="bash" data-theme="github-dark-default"><code>gcloud run deploy my-agent --region europe-north1 ...</code></pre>
</figure>

### Handle Cold Starts

Cloud Run scales to zero when idle. The first request after idle may take a few seconds. To keep it warm:

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="bash" data-theme="github-dark-default"><code>gcloud run deploy my-agent --min-instances 1 ...</code></pre>
</figure>

This keeps one instance always running — useful during active competition.

### Increase Memory for LLMs

If you're calling external LLM APIs (like Vertex AI), the default 512 MB is fine. If you're running a local model, increase memory:

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="bash" data-theme="github-dark-default"><code>gcloud run deploy my-agent --memory 2Gi --cpu 2 ...</code></pre>
</figure>

### Update Your Deployment

After making changes, just run the same deploy command again:

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="bash" data-theme="github-dark-default"><code>gcloud run deploy my-agent --source . --region europe-north1 --allow-unauthenticated</code></pre>
</figure>

Cloud Run builds and deploys a new revision automatically.

### View Logs

Check what your endpoint is doing:

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="bash" data-theme="github-dark-default"><code>gcloud run services logs read my-agent --region europe-north1 --limit 50</code></pre>
</figure>

Or view logs in the [Cloud Console](https://console.cloud.google.com/run) under your service → Logs tab.

## Which Tasks Need Cloud Run?

<div class="table-scroll-wrapper">

| Task               | Submission type           | Cloud Run? |
|--------------------|---------------------------|------------|
| Tripletex          | HTTPS endpoint (`/solve`) | Yes        |
| Astar Island       | HTTPS endpoint (`/solve`) | Yes        |
| NorgesGruppen Data | Code upload (`.zip`)      | No         |

</div>
