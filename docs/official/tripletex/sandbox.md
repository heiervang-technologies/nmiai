# Tripletex — Sandbox Account

Every team gets a free Tripletex sandbox account to explore the API and web interface before submitting to the competition.

## Getting Your Sandbox

1.  Go to the **Tripletex submission page** on the platform
2.  Click **"Get Sandbox Account"**
3.  Your sandbox is provisioned instantly

You'll receive:

- **Tripletex UI URL** — log in and explore the accounting interface
- **API base URL** — call the Tripletex v2 REST API directly
- **Session token** — authenticate your API calls

## Logging Into the Web UI

1.  Go to `https://kkpqfuj-amager.tripletex.dev`
2.  Enter the email shown on your sandbox card
3.  Click **"Forgot password"** to set up your Visma Connect account (first time only)
4.  Set a password and log in

Once you've set up Visma Connect, the same credentials work for all Tripletex test accounts — including the ones created during competition submissions.

## Using the API

Authenticate with **Basic Auth** using `0` as username and the session token as password:

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="python" data-theme="github-dark-default"><code>import requests
 
BASE_URL = &quot;https://kkpqfuj-amager.tripletex.dev/v2&quot;
SESSION_TOKEN = &quot;your-session-token-here&quot;
 
# List employees
response = requests.get(
    f&quot;{BASE_URL}/employee&quot;,
    auth=(&quot;0&quot;, SESSION_TOKEN),
    params={&quot;fields&quot;: &quot;id,firstName,lastName,email&quot;}
)
print(response.json())
 
# Create a customer
response = requests.post(
    f&quot;{BASE_URL}/customer&quot;,
    auth=(&quot;0&quot;, SESSION_TOKEN),
    json={
        &quot;name&quot;: &quot;Test Customer AS&quot;,
        &quot;email&quot;: &quot;test@example.com&quot;,
        &quot;isCustomer&quot;: True,
    }
)
print(response.json())</code></pre>
</figure>

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="bash" data-theme="github-dark-default"><code># curl example
curl -u &quot;0:your-session-token-here&quot; \
  &quot;https://kkpqfuj-amager.tripletex.dev/v2/employee?fields=id,firstName,lastName&quot;</code></pre>
</figure>

## What You Can Do

The sandbox is a full Tripletex test environment. Use it to:

- **Explore the API** — try creating employees, customers, invoices, and more
- **See the UI** — understand what the accounting data looks like in the interface
- **Test your agent** — point your `/solve` endpoint at the sandbox to debug
- **Learn the data model** — see how resources relate to each other

## Key Differences from Competition

<div class="table-scroll-wrapper">

|            | Sandbox                   | Competition                  |
|------------|---------------------------|------------------------------|
| Account    | Persistent, yours to keep | Fresh account per submission |
| API access | Direct to Tripletex       | Via authenticated proxy      |
| Data       | Accumulates over time     | Starts empty each time       |
| Scoring    | None                      | Automated field-by-field     |

</div>

## Tips

- Create some test data manually in the UI, then query it via the API to understand the response format
- Try the same operations your agent will need: creating employees, invoices, products, etc.
- The sandbox token expires March 31, 2026
- Each team gets one sandbox — all team members share it
