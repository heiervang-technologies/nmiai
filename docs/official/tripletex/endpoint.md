# Tripletex — Endpoint Specification

Your agent must expose a single HTTPS endpoint that accepts POST requests.

## `/solve` Endpoint

**Method:** POST **Content-Type:** application/json **Timeout:** 300 seconds (5 minutes)

## Request Format

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="json" data-theme="github-dark-default"><code>{
  &quot;prompt&quot;: &quot;Opprett en ansatt med navn Ola Nordmann, ola@example.org. Han skal være kontoadministrator.&quot;,
  &quot;files&quot;: [
    {
      &quot;filename&quot;: &quot;faktura.pdf&quot;,
      &quot;content_base64&quot;: &quot;JVBERi0xLjQg...&quot;,
      &quot;mime_type&quot;: &quot;application/pdf&quot;
    }
  ],
  &quot;tripletex_credentials&quot;: {
    &quot;base_url&quot;: &quot;https://&lt;provided-per-submission&gt;/v2&quot;,
    &quot;session_token&quot;: &quot;abc123...&quot;
  }
}</code></pre>
</figure>

<div class="table-scroll-wrapper">

| Field | Type | Description |
|----|----|----|
| `prompt` | string | The task in Norwegian natural language |
| `files` | array | Attachments (PDFs, images) — may be empty |
| `files[].filename` | string | Original filename |
| `files[].content_base64` | string | Base64-encoded file content |
| `files[].mime_type` | string | MIME type (`application/pdf`, `image/png`, etc.) |
| `tripletex_credentials.base_url` | string | Proxy API URL — use this instead of the standard Tripletex URL |
| `tripletex_credentials.session_token` | string | Session token for authentication |

</div>

## Response Format

Return this JSON when your agent has finished executing the task:

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="json" data-theme="github-dark-default"><code>{
  &quot;status&quot;: &quot;completed&quot;
}</code></pre>
</figure>

## Authentication

Your agent authenticates with the Tripletex API using **Basic Auth**:

- **Username:** `0` (zero)
- **Password:** the `session_token` value from the request

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="python" data-theme="github-dark-default"><code>import requests
 
response = requests.get(
    f&quot;{base_url}/employee&quot;,
    auth=(&quot;0&quot;, session_token),
    params={&quot;fields&quot;: &quot;id,firstName,lastName,email&quot;}
)</code></pre>
</figure>

## API Key (Optional)

If you set an API key when submitting your endpoint, we send it as a Bearer token:

    Authorization: Bearer <your-api-key>

Use this to protect your endpoint from unauthorized access.

## Requirements

- Endpoint must be **HTTPS**
- Must respond within **5 minutes** (300 seconds)
- Must return `{"status": "completed"}` with HTTP 200
- All Tripletex API calls must go through the provided `base_url` (proxy)

## Tripletex API Reference

All standard Tripletex v2 endpoints are available through the proxy. Common endpoints:

<div class="table-scroll-wrapper">

| Endpoint          | Methods                | Description               |
|-------------------|------------------------|---------------------------|
| `/employee`       | GET, POST, PUT         | Manage employees          |
| `/customer`       | GET, POST, PUT         | Manage customers          |
| `/product`        | GET, POST              | Manage products           |
| `/invoice`        | GET, POST              | Create and query invoices |
| `/order`          | GET, POST              | Manage orders             |
| `/travelExpense`  | GET, POST, PUT, DELETE | Travel expense reports    |
| `/project`        | GET, POST              | Manage projects           |
| `/department`     | GET, POST              | Manage departments        |
| `/ledger/account` | GET                    | Query chart of accounts   |
| `/ledger/posting` | GET                    | Query ledger postings     |
| `/ledger/voucher` | GET, POST, DELETE      | Manage vouchers           |

</div>

## API Tips

- Use the `fields` parameter to select specific fields: `?fields=id,firstName,lastName,*`
- Use `count` and `from` for pagination: `?from=0&count=100`
- POST/PUT requests take JSON body
- DELETE requests use the ID in the URL path: `DELETE /employee/123`
- List responses are wrapped: `{"fullResultSize": N, "values": [...]}`
