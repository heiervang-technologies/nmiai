# Tripletex - Examples

## FastAPI Template

```python
from fastapi import FastAPI
import httpx

app = FastAPI()


@app.post("/solve")
async def solve(request: dict):
    prompt = request["prompt"]
    credentials = request["tripletex_credentials"]
    base_url = credentials["base_url"]
    token = credentials["session_token"]

    auth = httpx.BasicAuth(username="0", password=token)

    async with httpx.AsyncClient(base_url=base_url, auth=auth) as client:
        # Your solution logic here
        pass

    return {"status": "completed"}
```

## List Employees

```python
async def list_employees(client: httpx.AsyncClient):
    response = await client.get("/v2/employee", params={"count": 100})
    response.raise_for_status()
    data = response.json()
    return data["values"]
```

## Create Customer

```python
async def create_customer(client: httpx.AsyncClient, name: str, email: str):
    payload = {
        "name": name,
        "email": email,
    }
    response = await client.post("/v2/customer", json=payload)
    response.raise_for_status()
    return response.json()["value"]
```

## Create Invoice

```python
async def create_invoice(client: httpx.AsyncClient, customer_id: int, lines: list):
    payload = {
        "customer": {"id": customer_id},
        "invoiceDate": "2026-03-19",
        "dueDate": "2026-04-19",
        "orders": [],
    }
    response = await client.post("/v2/invoice", json=payload)
    response.raise_for_status()
    return response.json()["value"]
```

## Search Entity

```python
async def search_entity(client: httpx.AsyncClient, endpoint: str, query: str):
    response = await client.get(f"/v2/{endpoint}", params={"query": query, "count": 25})
    response.raise_for_status()
    return response.json()["values"]
```

## Deployment

### Using uvicorn + cloudflared tunnel

```bash
# Start the server
uvicorn main:app --host 0.0.0.0 --port 8000

# In another terminal, expose via cloudflared
cloudflared tunnel --url http://localhost:8000
```

This gives you a public HTTPS URL to register as your endpoint on the competition platform.

## Common Task Patterns

### Single Entity

Create or update a single entity (employee, customer, project). Parse the prompt to extract fields, make one API call.

### Multi-Step

Tasks that require creating multiple related entities. Example: create a customer, then create an invoice for that customer.

### Modification

Find an existing entity, update specific fields. Requires a search/list call followed by a PUT/PATCH.

### Deletion

Find and delete an entity. Requires identifying the entity first, then calling DELETE.

## Optimization Tips

- **Plan before calling**: Parse the prompt fully before making any API calls.
- **Avoid trial-and-error**: Each failed API call (4xx) hurts your efficiency score.
- **Minimize GETs**: Only fetch data you actually need.
- **Batch operations**: Where the API supports it, batch creates/updates.
- **Read error responses**: Tripletex error messages are descriptive and help fix issues.
- **Norwegian characters**: Work fine as UTF-8. No special encoding needed.
