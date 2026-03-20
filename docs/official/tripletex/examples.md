# Tripletex — Examples

## Minimal `/solve` Endpoint

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="python" data-theme="github-dark-default"><code>import base64
from pathlib import Path
 
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
 
app = FastAPI()
 
@app.post(&quot;/solve&quot;)
async def solve(request: Request):
    body = await request.json()
    prompt = body[&quot;prompt&quot;]
    files = body.get(&quot;files&quot;, [])
    creds = body[&quot;tripletex_credentials&quot;]
 
    base_url = creds[&quot;base_url&quot;]
    token = creds[&quot;session_token&quot;]
    auth = (&quot;0&quot;, token)
 
    for f in files:
        data = base64.b64decode(f[&quot;content_base64&quot;])
        Path(f[&quot;filename&quot;]).write_bytes(data)
 
    # TODO: Use an LLM to interpret the prompt and execute
    # the appropriate Tripletex API calls
 
    return JSONResponse({&quot;status&quot;: &quot;completed&quot;})</code></pre>
</figure>

Run with:

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="bash" data-theme="github-dark-default"><code>pip install fastapi uvicorn requests
uvicorn main:app --host 0.0.0.0 --port 8000</code></pre>
</figure>

Expose locally via HTTPS for testing:

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="bash" data-theme="github-dark-default"><code>npx cloudflared tunnel --url http://localhost:8000</code></pre>
</figure>

## Tripletex API Examples

### List employees

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="python" data-theme="github-dark-default"><code>resp = requests.get(
    f&quot;{base_url}/employee&quot;,
    auth=auth,
    params={&quot;fields&quot;: &quot;id,firstName,lastName,email&quot;}
)
employees = resp.json()[&quot;values&quot;]</code></pre>
</figure>

### Create a customer

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="python" data-theme="github-dark-default"><code>resp = requests.post(
    f&quot;{base_url}/customer&quot;,
    auth=auth,
    json={
        &quot;name&quot;: &quot;Acme AS&quot;,
        &quot;email&quot;: &quot;post@acme.no&quot;,
        &quot;isCustomer&quot;: True
    }
)
customer_id = resp.json()[&quot;value&quot;][&quot;id&quot;]</code></pre>
</figure>

### Create an invoice

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="python" data-theme="github-dark-default"><code>today = &quot;2026-03-03&quot;
resp = requests.post(
    f&quot;{base_url}/invoice&quot;,
    auth=auth,
    json={
        &quot;invoiceDate&quot;: today,
        &quot;invoiceDueDate&quot;: today,
        &quot;customer&quot;: {&quot;id&quot;: customer_id},
        &quot;orders&quot;: [{&quot;id&quot;: order_id}]
    }
)</code></pre>
</figure>

### Search for a specific entity

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="python" data-theme="github-dark-default"><code>resp = requests.get(
    f&quot;{base_url}/customer&quot;,
    auth=auth,
    params={
        &quot;name&quot;: &quot;Acme&quot;,
        &quot;fields&quot;: &quot;id,name,email&quot;,
        &quot;count&quot;: 10
    }
)
matches = resp.json()[&quot;values&quot;]</code></pre>
</figure>

## Building an Effective Agent

1.  **Parse the prompt** — Use an LLM to extract the task type, entity names, field values, and relationships from the Norwegian prompt
2.  **Handle files** — Some tasks include PDFs with invoices, contracts, or expense reports. Decode from base64 and extract relevant data
3.  **Map to API calls** — Determine which Tripletex endpoints to call and in what order. Some tasks require creating prerequisites first
4.  **Verify your work** — After creating entities, query back to confirm they exist with correct values
5.  **Handle errors** — Tripletex returns detailed error messages. Parse them to retry with corrections

## Common Task Patterns

<div class="table-scroll-wrapper">

| Pattern | Example | API Flow |
|----|----|----|
| Create single entity | "Create employee Ola Nordmann" | POST /employee |
| Create with linking | "Create invoice for customer" | GET /customer → POST /order → POST /invoice |
| Modify existing | "Add phone to contact" | GET /customer → PUT /customer/{id} |
| Delete/reverse | "Delete travel expense" | GET /travelExpense → DELETE /travelExpense/{id} |
| Multi-step setup | "Register payment" | POST /customer → POST /invoice → POST /payment |

</div>

## Common Errors

<div class="table-scroll-wrapper">

| Error | Cause | Fix |
|----|----|----|
| 401 Unauthorized | Wrong auth format | Use Basic Auth with username `0` and session token as password |
| 404 Not Found | Wrong endpoint path | Check the Tripletex v2 API docs for correct paths |
| 422 Validation Error | Missing required fields | Read error message — it specifies which fields are required |
| Empty `values` array | No results found | Check search parameters, try broader search |
| Timeout (5 min) | Agent too slow | Optimize API calls, reduce unnecessary requests |

</div>

## Tips

- The Tripletex sandbox starts empty — you may need to create prerequisites (customer, product) before creating invoices
- Use `?fields=*` to see all available fields on an entity
- Some tasks require enabling modules first (e.g., department accounting)
- Norwegian characters (æ, ø, å) work fine in API requests — send as UTF-8
- All API calls through the proxy are logged — use them for debugging in the submissions view
- Prompts come in 7 languages (nb, en, es, pt, nn, de, fr) — your agent should handle all of them

## Optimizing for Efficiency

Your score can go above 1.0 if you achieve perfect correctness with minimal API calls and zero errors. Higher-tier tasks have higher score ceilings (up to 6.0 for Tier 3). Tips:

- **Plan before calling** — Parse the prompt fully before making API calls. Understand what needs to be created/modified before starting
- **Avoid trial-and-error** — Every 4xx error (400, 404, 422) reduces your efficiency bonus. Validate inputs before sending
- **Minimize GET calls** — Don't fetch entities you don't need. If you created something, you already know its ID from the response
- **Batch where possible** — Some Tripletex endpoints accept lists. Use them instead of multiple individual calls
- **Read error messages** — If a call fails, the Tripletex error message tells you exactly what's wrong. Fix it in one retry, not several
