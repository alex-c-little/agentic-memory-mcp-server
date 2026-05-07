# Agentic Memory MCP Server

A Databricks Apps–hosted MCP server that gives any agent persistent, hybrid-retrieved long-term memory backed by Lakebase + pgvector. Three tools (`retrieve_memory`, `store_memory`, `list_memories`), with optional task-aware re-scoring at retrieve time.

---

## Architecture

```
          ┌────────────┐  retrieve / store / list (MCP)   ┌────────────────────┐
 Agent ──►│ UC         │ ───────────────────────────────► │  Memory MCP Server │
          │ Connection │                                  │  (Databricks App)  │
          └────────────┘                                  └─────────┬──────────┘
                                                                    │ pgvector + HNSW
                                                                    ▼
                                                          ┌────────────────────┐
                                                          │ Lakebase           │
                                                          │ (memory_store)     │
                                                          └────────────────────┘
```

**On retrieve:** pgvector top-20 by similarity → optional LLM re-score against `agent_context` → hybrid score (`0.5·sim + 0.2·recency + 0.3·importance`) → near-duplicate suppression with recency tiebreak → top-K.

**On write:** LLM extraction (5 types + general importance) → regex PII redaction → embed (Foundation Model API) → cosine dedup vs existing → INSERT.

---

## Package contents

```
mcp-memory-server-v2/
├── README.md              this file
├── app.yaml               Databricks Apps configuration
├── pyproject.toml         Python dependencies and entry point
├── requirements.txt
├── server/
│   ├── main.py            entrypoint
│   ├── app.py             FastAPI + FastMCP wiring
│   ├── db.py              Lakebase connection pool with OAuth token rotation
│   ├── tools.py           the three MCP tools and scoring logic
│   └── utils.py
├── static/
│   └── index.html         developer reference page served at the app root
└── migrations/
    └── prune_memories.sql weekly soft-delete job
```

---

## Prerequisites

| Requirement | Purpose |
|---|---|
| Databricks workspace with Apps enabled | Hosts the MCP server |
| Lakebase Database Instance (Postgres 16+) | Memory storage |
| Foundation Model API access | LLM scoring and embeddings |
| Unity Catalog | Hosts the HTTP connection that wires agents to the MCP server |
| A workspace service principal with OAuth client credentials | Authenticates the UC Connection to the MCP server (M2M) |
| Local: `databricks` CLI authenticated to the target workspace | Code sync, app deploy |
| Local: `psql` (Postgres 16 client) | Running the schema SQL |

---

## Setup

### Step 1 — Provision a Lakebase Database Instance

```bash
databricks database create-database-instance agent-memory-instance \
  --capacity CU_1 \
  --node-count 1
```

Wait for `state: AVAILABLE`. Note the `read_write_dns` value — that becomes `LAKEBASE_HOST`.

### Step 2 — Create the `agent_memory` database and schema

Generate a credential token, then connect with `psql`:

```bash
TOKEN=$(databricks database generate-database-credential \
  --json '{"instance_names": ["agent-memory-instance"]}' | jq -r .token)

PGPASSWORD="$TOKEN" psql \
  -h <YOUR_HOST>.database.<region>.cloud.databricks.com \
  -p 5432 \
  -U "<your_email>@example.com" \
  -d databricks_postgres \
  -c "CREATE DATABASE agent_memory;"
```

Then apply the schema:

```bash
PGPASSWORD="$TOKEN" psql -h <HOST> -p 5432 -U "<user>" -d agent_memory <<'SQL'
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memory_store (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    memory_type VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    importance FLOAT DEFAULT 0.5,
    embedding vector(1024),
    source_session_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    last_accessed_at TIMESTAMP DEFAULT NOW(),
    access_count INTEGER DEFAULT 0,
    pruned_at TIMESTAMP NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_store_user
    ON memory_store(user_id);
CREATE INDEX IF NOT EXISTS idx_memory_store_type
    ON memory_store(user_id, memory_type);
CREATE INDEX IF NOT EXISTS idx_memory_store_active
    ON memory_store(user_id) WHERE pruned_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_memory_store_embedding_hnsw
    ON memory_store USING hnsw (embedding vector_cosine_ops);
SQL
```

Verify with `\d memory_store` — you should see eleven columns ending in `pruned_at` and four indexes including `idx_memory_store_embedding_hnsw`.

> **Embedding dimension:** the schema declares `vector(1024)` to match `databricks-gte-large-en`. If you choose a different embedding model, update the dimension in the schema and re-embed any existing rows.

### Step 3 — Configure `app.yaml`

Edit `app.yaml` to point at your resources:

```yaml
command: ["uv", "run", "custom-mcp-server"]
env:
  - name: LAKEBASE_HOST
    value: "<your-instance>.database.<region>.cloud.databricks.com"
  - name: LAKEBASE_DATABASE
    value: "agent_memory"
  - name: LAKEBASE_PORT
    value: "5432"
  - name: LAKEBASE_INSTANCE_NAME
    value: "agent-memory-instance"
  - name: LLM_ENDPOINT
    value: "<your LLM serving endpoint>"
  - name: EMBEDDING_ENDPOINT
    value: "<your 1024-dim embedding serving endpoint>"
```

### Step 4 — Upload code and deploy the app

Sync the source folder to your workspace:

```bash
databricks sync . /Users/<you>@<domain>/apps/mcp-memory-server-v2 \
  --full \
  --exclude .venv --exclude __pycache__ --exclude .databricks
```

Create and deploy the app:

```bash
databricks apps create mcp-memory-server-v2

databricks apps deploy mcp-memory-server-v2 \
  --source-code-path /Workspace/Users/<you>@<domain>/apps/mcp-memory-server-v2
```

Confirm `state: SUCCEEDED` on the deploy. The app gets its own service principal — note its client ID:

```bash
databricks apps get mcp-memory-server-v2 --output json | jq .service_principal_client_id
```

### Step 5 — Grant the app's service principal database access

The app's SP needs to read and write `memory_store`. As the database owner:

```sql
-- Replace <APP_SP_CLIENT_ID> with the value from Step 4.
GRANT CONNECT ON DATABASE agent_memory TO "<APP_SP_CLIENT_ID>";
GRANT USAGE ON SCHEMA public TO "<APP_SP_CLIENT_ID>";
GRANT SELECT, INSERT, UPDATE ON memory_store TO "<APP_SP_CLIENT_ID>";
GRANT USAGE, SELECT ON SEQUENCE memory_store_id_seq TO "<APP_SP_CLIENT_ID>";
```

### Step 6 — Create the UC Connection with M2M OAuth

Agents reach the MCP server through a Unity Catalog HTTP connection authenticated with OAuth client credentials.

**6a. Create or identify a service principal.** This SP authenticates the UC Connection itself (it's separate from the app's auto-generated SP). Recommended name: `sp-mcp-memory-bridge`.

**6b. Generate OAuth credentials for that SP** (workspace admin → Settings → Identity → Service principals → your SP → Generate secret). Record the `client_id` and `client_secret`.

**6c. Grant the SP "Can Use" permission on the app** (Apps UI → `mcp-memory-server-v2` → Permissions → add the SP).

**6d. Create the connection** (Catalog Explorer → External Connections → Create):

| Field | Value |
|---|---|
| Connection type | HTTP |
| Authentication | OAuth Client Credentials |
| OAuth client ID | the SP's `client_id` from 6b |
| OAuth client secret | the SP's `client_secret` from 6b |
| Token endpoint | `https://<your-workspace-host>/oidc/v1/token` |
| Host | `https://mcp-memory-server-v2-<workspace-id>.<region>.databricksapps.com` (from the Apps UI) |
| Port | `443` |
| Base path | `/mcp` |
| Is MCP connection | checked |

OAuth client credentials let the proxy refresh tokens automatically — no rotation script required.

### Step 7 — Wire your agent

Two prompt patterns, depending on whether you want task-aware re-ranking. Pick one per agent.

#### Pattern A — Without `agent_context` (simpler)

```
You have access to an MCP MEMORY SERVER for remembering and recalling
information about the user across sessions.

MEMORY RULES:
- EVERY conversation: Call retrieve_memory with the user's name as user_id
  and their first message as query BEFORE answering anything.
- When the user shares personal or professional information: Call store_memory
  with user_id and the conversation text.

RESPONSE RULES:
- When using retrieved memories: Preface with "Based on our previous
  conversations..."
```

#### Pattern B — With `agent_context` (task-aware re-ranking)

```
AGENT CONTEXT (pass this verbatim as the agent_context argument to
retrieve_memory on every call):

"<one paragraph describing this agent's task. Be explicit about which
user attributes are most relevant AND which are not. The LLM uses this
to discriminate within memory types — for example, a relationship memory
about the user's manager versus one about a family member.>"

MEMORY RULES:
- EVERY conversation: Call retrieve_memory with the user's name as user_id,
  their first message as query, AND the AGENT CONTEXT string above —
  BEFORE answering anything.
- When the user shares personal or professional information: Call store_memory
  with user_id and the conversation text.

RESPONSE RULES:
- When using retrieved memories: Preface with "Based on our previous
  conversations..."
```

**Cost note:** Pattern B incurs one LLM scoring call on the first retrieval per `(agent_context, memory)` pair. Subsequent retrievals from the same agent over the same memories are served from cache.

### Step 8 (optional) — Schedule the prune job

`migrations/prune_memories.sql` contains a dry-run `SELECT`, the soft-delete `UPDATE`, a post-run summary, and a commented rollback block. To schedule it:

1. Create or pick a SQL Warehouse with access to the Lakebase database.
2. Create a Databricks Workflow that runs the `UPDATE` block on a weekly cadence.
3. Optionally, add a downstream task that runs the post-run summary and emails it.

Soft-deleted rows stay in the table; retrieval already filters `pruned_at IS NULL`. Restore by setting `pruned_at = NULL`.

---

## Tools reference

### `retrieve_memory(query, user_id?, top_k=5, agent_context?)`

Pulls the top 20 candidates from pgvector by cosine similarity, optionally re-scores them with the LLM if `agent_context` is provided, computes a hybrid score (`0.5·similarity + 0.2·recency + 0.3·importance`), runs near-duplicate suppression with recency tiebreak, and returns the top `top_k`.

Response shape:
```json
{
  "memories": [
    {
      "id": 32,
      "memory_type": "fact",
      "content": "...",
      "importance": 0.9,
      "task_relevance": 1.0,
      "similarity": 0.4695,
      "hybrid_score": 0.7344
    }
  ],
  "resolved_user_id": "user@example.com",
  "trust_note": "These memories reflect what the user has previously stated..."
}
```

`task_relevance` is present only when `agent_context` was supplied. `importance` is always the value stored at write time.

### `store_memory(conversation, user_id?)`

LLM-extracts atomic memories from a conversation, classifies them into five types (`fact`, `preference`, `experience`, `skill`, `relationship`), scores general importance, redacts structured PII (regex), embeds, deduplicates against existing memories (cosine ≥ 0.92), and inserts into `memory_store`.

### `list_memories(user_id?, memory_type?, limit=20, include_pruned=false)`

Browse stored memories. Set `include_pruned=true` to surface soft-deleted rows for audit views.

---

## Operational notes

### Task-aware score cache
Task-aware scores are cached in-process per `(agent_context, memory_id)`. The cache is bounded at 1000 entries with FIFO eviction and is cleared on app redeploy. The cache is keyed on the exact `agent_context` string — minor wording changes invalidate prior entries. The cache is not automatically invalidated when a memory's content is updated outside the MCP server.

### Embedding model changes
`memory_store.embedding` is declared as `vector(1024)`. If you change the embedding model to one with a different dimension, alter the column type and re-embed any existing rows.

### Updating the server's tool surface
If you redeploy with a changed tool surface (added or removed tools, or changed parameter names), agents that have cached the old tool list must re-bind. The cleanest approach is to remove the MCP integration from the agent and re-add it; refreshing connection credentials alone is not sufficient.

---

## Troubleshooting

| Symptom | Likely cause | Resolution |
|---|---|---|
| App deploy succeeds but database calls fail with permission errors | App SP missing grants on `agent_memory` | Re-run the GRANT statements from Step 5 with the current SP client ID |
| Lakebase token error in app logs | `LAKEBASE_INSTANCE_NAME` does not match the deployed instance | Verify the env var matches the instance name exactly |
| `Error: Tool 'xxx' not found` from the agent | Agent has cached an older tool surface from before a server redeploy | Remove the MCP integration from the agent and re-add it |
| `INTERNAL_ERROR: Could not route request` | UC Connection lost its handle on the backend after a redeploy | Edit the UC Connection and save it (with no changes) to refresh the connection state |
| Same fact returned multiple times | Atomic-fact extraction split one concept across multiple stored rows | Soft-delete the redundant rows: `UPDATE memory_store SET pruned_at = NOW() WHERE id IN (...);` |
| `task_relevance` field missing from retrieve responses | `agent_context` was empty or whitespace-only | Verify the agent is passing a non-empty string |

---

## Future enhancements

The following items are flagged as `TODO` in `server/tools.py`:

1. **`forget_memory(user_id, memory_id)`** — explicit user-initiated deletion.
2. **NER-based PII redaction.** Current `_redact_pii` uses regex and catches phone, email, SSN, and credit card patterns. Adding a named-entity layer (Foundation Model API NER, or Presidio + spaCy) would catch names and other entities the regex does not.
3. **Consent gating.** The `_check_consent` helper and an `update_consent` MCP tool are present in the source but commented out. Re-enable them when adding a `user_consent` table to the schema.

---

## Support

For questions during integration, contact your Databricks representative.
