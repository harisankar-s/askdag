# 22. One shared API key on /api/*, off by default; real authentication is out of scope

**Status:** accepted, 2026-11-23

## Context
`/api/ask` reads every node, owner and incident in the database and can spend LLM tokens on a paid key. On a laptop that is fine. Once the Docker stack (ADR 0021) runs on a shared VM, anyone who can reach port 8000 can do both. This build has no users, so a login system would be scope creep; but the repo should show where the check goes and how it fails.

## Decision
- **`API_KEY` unset: open.** The clean-clone quickstart keeps working with no setup.
- **`API_KEY` set: every `/api/*` request needs it in `X-API-Key`.** One FastAPI dependency (`api/auth.py`) on both routers, compared with `hmac.compare_digest`, so the check takes the same time however much of the key matches. A missing or wrong key gets `401` with `WWW-Authenticate: APIKey`, and the body never echoes the key.
- **A key shorter than 16 characters fails at startup.** `API_KEY=` left empty in `.env` would otherwise look like protection and be none, or be trivially guessable.
- **`/health/*`, `/` and `/docs` stay open.** Probes carry no key; the page and the OpenAPI schema hold no data. The OpenAPI schema declares the header, so `/docs` has an Authorize button.
- **The web page asks for the key once, on the first 401**, keeps it in `sessionStorage` (this tab only, gone when it closes, never `localStorage`) and sends it on every call.
- **Header, not query string.** Query strings end up in access logs, browser history and `Referer` headers.

## Not built, and what would replace this
- Per-user identity, scopes, rotation, rate limits: an OIDC proxy (oauth2-proxy, a cloud load balancer's auth) in front of the app, or a gateway. The app would then read the verified identity from a header and this key would go.
- Per-project authorisation (team A must not see team B's lineage): needs identity first; `project` is already a request field, so the check would sit in the same dependency.
- TLS: the proxy's job. A key over plain HTTP is readable on the network; the README says so.

No measurement decides this; the tests guard the promise (open while unset; 401 without the right key; health and page open; short keys refused).
