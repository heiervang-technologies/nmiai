# Tripletex - Endpoint Specification

## Solve Endpoint

### POST /solve

Your service must expose a `/solve` endpoint that accepts task submissions.

### Request

```json
{
  "prompt": "Create an employee named Ola Nordmann with email ola@example.com",
  "files": [
    {
      "filename": "data.csv",
      "content": "base64-encoded-content"
    }
  ],
  "tripletex_credentials": {
    "base_url": "https://your-sandbox.tripletex.dev",
    "session_token": "your-session-token"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `prompt` | string | Natural language task description |
| `files` | array | Optional attached files (base64 encoded) |
| `tripletex_credentials` | object | Sandbox connection details |
| `tripletex_credentials.base_url` | string | Sandbox API base URL |
| `tripletex_credentials.session_token` | string | Authentication token |

### Response

```json
{
  "status": "completed"
}
```

### Constraints

- **Timeout**: 5 minutes per request
- **HTTPS**: Required for all connections

## Proxy Endpoints

The following Tripletex API proxy endpoints are available:

- `/employee`
- `/customer`
- `/invoice`
- `/project`
- `/ledger/voucher`

These endpoints proxy to the corresponding Tripletex API resources using the provided credentials.
