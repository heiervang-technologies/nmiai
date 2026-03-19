# Tripletex - Sandbox Environment

## Free Sandbox

A free Tripletex sandbox is provided through the competition platform. No separate Tripletex account is needed.

## Web UI

Access the sandbox web interface at the **tripletex.dev** domain.

## Authentication

The sandbox uses **Basic Authentication**:

| Field | Value |
|-------|-------|
| Username | `0` |
| Password | `session_token` (provided per submission) |

## Credential Lifecycle

- Credentials are provided with each submission via `tripletex_credentials` in the request payload.
- Sandbox credentials **expire March 31**.
- Each team shares **one account** across all team members.

## Notes

- A fresh sandbox is provisioned for each submission, so state does not persist between submissions.
- The sandbox mirrors production Tripletex functionality relevant to the competition tasks.
