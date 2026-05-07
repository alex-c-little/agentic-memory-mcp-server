"""MCP tools for agent memory: retrieve, store, list, session management.

v2 changes:
- Item 1: Auto-resolve user identity from request headers (falls back to user_id param)
- Item 2: Optional agent_context parameter on retrieve_memory for context-aware scoring (placeholder)
- Item 3: Trust note included in retrieve_memory responses
"""

import json
import os
import re
import math
from collections import OrderedDict
from datetime import datetime, timezone

from server.db import execute_sql, get_pool
from server.utils import header_store

# -- Configuration --
LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT", "databricks-claude-sonnet-4-6")
EMBEDDING_ENDPOINT = os.environ.get("EMBEDDING_ENDPOINT", "databricks-gte-large-en")
SIMILARITY_THRESHOLD = 0.92      # write-time dedup threshold (strict — wrong merge has consequences)
NEAR_DUP_THRESHOLD = 0.85        # retrieval-time near-dup suppression (looser — wrong hide is per-call)

# -- Agent-relative score cache --
# Keys: (agent_context_str, memory_id). Value: float score in [0.0, 1.0].
# Bounded FIFO; oldest entries evicted once at cap. In-process only — Apps
# containers persist across requests so this stays warm during a run.
_AGENT_SCORE_CACHE_MAX = 1000
_AGENT_SCORE_CACHE: OrderedDict = OrderedDict()


def _cache_get_agent_score(agent_context: str, memory_id):
    return _AGENT_SCORE_CACHE.get((agent_context, memory_id))


def _cache_put_agent_score(agent_context: str, memory_id, score: float):
    key = (agent_context, memory_id)
    if key not in _AGENT_SCORE_CACHE and len(_AGENT_SCORE_CACHE) >= _AGENT_SCORE_CACHE_MAX:
        _AGENT_SCORE_CACHE.popitem(last=False)  # FIFO evict oldest
    _AGENT_SCORE_CACHE[key] = score

# -- Trust note returned with every retrieval --
TRUST_NOTE = (
    "These memories reflect what the user has previously stated, not verified facts "
    "or official policy. If this agent has access to authoritative knowledge sources, "
    "defer to those over retrieved memories when they conflict."
)

# -- PII patterns (regex layer) --
PII_PATTERNS = [
    (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]"),
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),
    (r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "[CARD]"),
]


AGENT_RELEVANCE_PROMPT = """You are scoring memories about a user for relevance to a specific agent's task.

Agent's task / context: "{agent_context}"

For each numbered memory below, rate from 0.0 (irrelevant to this agent's task)
to 1.0 (highly relevant). Be discerning — within the same memory_type, individual
memories can have very different relevance to a given task. For example, a
relationship memory about the user's manager is highly relevant to an
overtime-routing agent, while a relationship memory about a family member is not.

Memories:
{items}

Return ONLY a JSON array of {n} floats in the same order as the memories above.
No prose, no code fences. Example: [0.8, 0.2, 0.9, 0.1, 0.7]"""


def _score_memories_for_agent(agent_context: str, memories: list) -> list:
    """Score each memory's relevance to the given agent context.

    Returns a list of floats in [0.0, 1.0], aligned 1:1 with the input list.
    Uses cache for previously-seen (agent_context, memory_id) pairs.
    On LLM error or empty agent_context, falls back to each memory's stored
    importance value, so the caller can use the result blindly.
    """
    n = len(memories)
    if n == 0:
        return []

    # Cache lookup pass
    scores = [None] * n
    to_score_idx = []
    for i, m in enumerate(memories):
        cached = _cache_get_agent_score(agent_context, m["id"])
        if cached is not None:
            scores[i] = cached
        else:
            to_score_idx.append(i)

    if not to_score_idx:
        return scores

    # Build the prompt for the un-cached subset
    items = "\n".join(
        f"{j+1}. [{memories[idx]['memory_type']}] {memories[idx]['content']}"
        for j, idx in enumerate(to_score_idx)
    )
    prompt = AGENT_RELEVANCE_PROMPT.format(
        agent_context=agent_context,
        items=items,
        n=len(to_score_idx),
    )

    try:
        response = _call_llm(
            prompt=prompt,
            system_prompt="You return only a JSON array of floats. No prose, no code fences.",
            max_tokens=400,
        )
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        new_scores = json.loads(cleaned)
        if not isinstance(new_scores, list) or len(new_scores) != len(to_score_idx):
            raise ValueError(f"Expected list of {len(to_score_idx)} scores, got {new_scores!r}")
        for j, idx in enumerate(to_score_idx):
            v = max(0.0, min(1.0, float(new_scores[j])))
            scores[idx] = v
            _cache_put_agent_score(agent_context, memories[idx]["id"], v)
    except Exception:
        # Degrade gracefully: any unscored slot falls back to stored importance.
        for idx in to_score_idx:
            if scores[idx] is None:
                scores[idx] = float(memories[idx].get("importance", 0.5))

    return scores


def _parse_embedding(raw):
    """Convert pgvector embedding (list, str '[...]', or memoryview) into a list of floats."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, (bytes, bytearray, memoryview)):
        raw = bytes(raw).decode("utf-8")
    if isinstance(raw, str):
        return [float(x) for x in raw.strip("[]").split(",") if x.strip()]
    return list(raw)


def _cosine_sim(a, b):
    """Cosine similarity between two equal-length float vectors."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _resolve_user_id(user_id_param: str = None) -> str:
    """Resolve the user's identity.

    Priority:
    1. Extract email from the forwarded auth headers (Databricks Apps on-behalf-of-user)
    2. Fall back to the user_id parameter passed by the calling agent
    3. Default to 'unknown' if neither is available

    This ensures consistent user_id regardless of whether the caller passes a name,
    email, or nothing — and works for both Databricks-hosted and external agents.
    """
    # Try to get identity from request headers (Databricks Apps flow)
    try:
        headers = header_store.get({})
        token = headers.get("x-forwarded-access-token")
        if token:
            from databricks.sdk import WorkspaceClient
            w = WorkspaceClient(token=token, auth_type="pat")
            me = w.current_user.me()
            if me and me.user_name:
                return me.user_name
    except Exception:
        pass  # Fall through to parameter-based ID

    # Fall back to the passed user_id (for external agents, Foundry, etc.)
    if user_id_param and user_id_param.strip():
        return user_id_param

    return "unknown"


def _call_llm(prompt: str, system_prompt: str = "You are a helpful assistant.", max_tokens: int = 1000) -> str:
    """Call the LLM endpoint via Databricks Model Serving."""
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

    w = WorkspaceClient()
    response = w.serving_endpoints.query(
        name=LLM_ENDPOINT,
        messages=[
            ChatMessage(role=ChatMessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=ChatMessageRole.USER, content=prompt),
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return response.choices[0].message.content


def _get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings from Databricks Foundation Model Embeddings API."""
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    response = w.serving_endpoints.query(
        name=EMBEDDING_ENDPOINT,
        input=texts,
    )
    return [item.embedding for item in response.data]


def _redact_pii(text: str) -> str:
    """Redact structured PII patterns from text."""
    redacted = text
    for pattern, replacement in PII_PATTERNS:
        redacted = re.sub(pattern, replacement, redacted)
    return redacted


# --------------------------------------------------------------
# Consent mechanism — DISABLED in this release. Re-enable once
# the user_consent table is added to the schema and the MCP tool
# surface for consent is finalized.
# --------------------------------------------------------------
# def _check_consent(user_id: str) -> bool:
#     """Check if user has consented to memory storage. Default: True (opt-out model)."""
#     cols, rows = execute_sql(
#         "SELECT consent FROM user_consent WHERE user_id = %s",
#         (user_id,),
#     )
#     if rows:
#         return rows[0][0]
#     return True  # Default: opted in


MEMORY_EXTRACTION_PROMPT = """Analyze this conversation and extract discrete, atomic memories about the user.

For each memory, classify it into exactly one type:
- fact: Objective information about the user or their organization
- preference: Stated or implied preferences
- experience: Past experiences or current situation
- skill: Technical skills or expertise
- relationship: People, teams, or organizational relationships

Rate importance from 0.0 to 1.0 based on how useful this memory would be in future conversations.
Only extract explicitly stated information — do not infer facts that were not directly said.

Return a JSON array of objects with fields: memory_type, content, importance.

CONVERSATION:
{conversation}

Return ONLY valid JSON array, no other text."""


def load_tools(mcp_server):
    """Register all memory tools on the MCP server."""

    @mcp_server.tool
    def retrieve_memory(
        user_id: str = "",
        query: str = "",
        top_k: int = 5,
        agent_context: str = None,
    ) -> dict:
        """
        Search for relevant long-term memories for a user using hybrid retrieval.
        Combines vector similarity, recency, and importance scoring.
        Use this at the START of a conversation to get context about the user.

        Args:
            user_id: Optional identifier for the user. If not provided, the server
                     resolves the user's identity from their authenticated session.
            query: Natural language query describing what context is needed.
            top_k: Maximum number of memories to return (default 5).
            agent_context: Optional description of the calling agent's purpose
                          (e.g., 'HR benefits enrollment assistant'). Used to
                          adjust relevance scoring in future versions.

        Returns:
            Dictionary with 'memories' list, 'trust_note' with guidance on
            how to treat retrieved memories, and 'resolved_user_id'.
        """
        resolved_id = _resolve_user_id(user_id)

        # When agent_context is provided, we re-score candidates against the
        # agent's task at retrieval time (cached per (agent_context, memory_id)).
        # When it's not provided, we use the stored general importance from
        # write-time extraction. Both scoring modes feed the same hybrid formula.
        use_agent_scoring = bool(agent_context and agent_context.strip())

        if not query:
            return {
                "memories": [],
                "resolved_user_id": resolved_id,
                "trust_note": TRUST_NOTE,
                "message": "No query provided.",
            }

        embedding = _get_embeddings([query])[0]
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

        cols, rows = execute_sql(
            f"""
            SELECT
                id, memory_type, content, importance,
                created_at, last_accessed_at, access_count,
                embedding,
                1 - (embedding <=> '{embedding_str}'::vector) as similarity
            FROM memory_store
            WHERE user_id = %s
              AND pruned_at IS NULL
            ORDER BY embedding <=> '{embedding_str}'::vector
            LIMIT 20
            """,
            (resolved_id,),
        )

        if not rows:
            return {
                "memories": [],
                "resolved_user_id": resolved_id,
                "trust_note": TRUST_NOTE,
                "message": "No memories found for this user.",
            }

        # First pass: shape rows into dicts the scorer can consume.
        candidates = []
        for mid, mtype, content, importance, created, last_accessed, access_count, mem_embedding, similarity in rows:
            if isinstance(last_accessed, str):
                last_accessed = datetime.fromisoformat(last_accessed)
            if last_accessed.tzinfo is None:
                last_accessed = last_accessed.replace(tzinfo=timezone.utc)
            if isinstance(created, str):
                created = datetime.fromisoformat(created)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            candidates.append({
                "id": mid,
                "memory_type": mtype,
                "content": content,
                "importance": float(importance),
                "similarity": float(similarity),
                "_created_at": created,
                "_last_accessed": last_accessed,
                "_embedding": mem_embedding,
            })

        # Optional agent-relative re-scoring (one batched LLM call, cached).
        if use_agent_scoring:
            relevance_scores = _score_memories_for_agent(agent_context, candidates)
        else:
            relevance_scores = [c["importance"] for c in candidates]

        # Second pass: compute hybrid using whichever importance signal applies.
        now = datetime.now(timezone.utc)
        scored = []
        for c, rel in zip(candidates, relevance_scores):
            days_since = max((now - c["_last_accessed"]).total_seconds() / 86400, 0.01)
            recency = math.exp(-0.1 * days_since)
            hybrid = 0.5 * c["similarity"] + 0.2 * recency + 0.3 * rel
            entry = {
                "id": c["id"],
                "memory_type": c["memory_type"],
                "content": c["content"],
                "importance": round(c["importance"], 2),         # always the stored value
                "similarity": round(c["similarity"], 4),
                "hybrid_score": round(hybrid, 4),
                "_created_at": c["_created_at"],
                "_embedding": c["_embedding"],
            }
            if use_agent_scoring:
                entry["task_relevance"] = round(rel, 2)          # only when agent_context provided
            scored.append(entry)

        scored.sort(key=lambda x: x["hybrid_score"], reverse=True)

        # --- Near-duplicate suppression with recency tiebreak ---
        # If two retrieved memories of the same type are highly similar to each
        # other (cosine > NEAR_DUP_THRESHOLD), assume they describe the same
        # concept. Drop the older one. This handles "user said X two months ago,
        # then said NOT X yesterday" — we surface the more recent claim.
        kept = []
        for cand in scored:
            cand_emb = _parse_embedding(cand["_embedding"])
            is_dup = False
            for k in kept:
                if k["memory_type"] != cand["memory_type"]:
                    continue
                k_emb = _parse_embedding(k["_embedding"])
                if _cosine_sim(cand_emb, k_emb) >= NEAR_DUP_THRESHOLD:
                    # Older candidate loses to the already-kept (higher-ranked) one.
                    if cand["_created_at"] <= k["_created_at"]:
                        is_dup = True
                        break
                    # Candidate is newer — replace the older kept entry.
                    kept.remove(k)
                    kept.append(cand)
                    is_dup = True  # already inserted
                    break
            if not is_dup:
                kept.append(cand)
            if len(kept) >= top_k:
                break

        # Strip internal fields before returning.
        top = [
            {k: v for k, v in m.items() if not k.startswith("_")}
            for m in kept[:top_k]
        ]

        # Update access timestamps
        top_ids = [m["id"] for m in top]
        if top_ids:
            ids_str = ",".join(str(i) for i in top_ids)
            execute_sql(
                f"UPDATE memory_store SET last_accessed_at = NOW(), access_count = access_count + 1 WHERE id IN ({ids_str})",
                fetch=False,
            )

        return {
            "memories": top,
            "resolved_user_id": resolved_id,
            "trust_note": TRUST_NOTE,
        }

    @mcp_server.tool
    def store_memory(user_id: str = "", conversation: str = "") -> dict:
        """
        Extract and store long-term memories from a conversation.
        Runs the full pipeline: extract with LLM, classify into 5 types,
        score importance, redact PII, embed, and deduplicate.
        Use this when a conversation contains information worth remembering.

        Args:
            user_id: Optional identifier for the user. If not provided, the server
                     resolves the user's identity from their authenticated session.
            conversation: The conversation text to extract memories from.

        Returns:
            Dictionary with count of memories stored, skipped (duplicates),
            resolved_user_id, and list of extracted memories.
        """
        resolved_id = _resolve_user_id(user_id)

        # Consent check disabled in this release — re-enable with _check_consent above.
        # if not _check_consent(resolved_id):
        #     return {
        #         "status": "skipped",
        #         "resolved_user_id": resolved_id,
        #         "reason": "User has not consented to memory storage.",
        #     }

        if not conversation:
            return {
                "status": "skipped",
                "resolved_user_id": resolved_id,
                "reason": "No conversation text provided.",
            }

        # Extract memories via LLM
        prompt = MEMORY_EXTRACTION_PROMPT.format(conversation=conversation)
        response = _call_llm(
            prompt=prompt,
            system_prompt="You are a memory extraction system. Extract structured memories from conversations. Return only valid JSON.",
            max_tokens=2000,
        )

        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]

        try:
            memories = json.loads(cleaned)
        except json.JSONDecodeError:
            return {
                "status": "error",
                "resolved_user_id": resolved_id,
                "reason": "Failed to parse LLM extraction response.",
            }

        stored = 0
        skipped = 0
        extracted = []

        for mem in memories:
            content = _redact_pii(mem.get("content", ""))
            memory_type = mem.get("memory_type", "fact")
            importance = mem.get("importance", 0.5)

            # Dedup check
            embedding = _get_embeddings([content])[0]
            embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

            cols, rows = execute_sql(
                f"""
                SELECT id, content,
                       1 - (embedding <=> '{embedding_str}'::vector) as similarity
                FROM memory_store
                WHERE user_id = %s
                ORDER BY embedding <=> '{embedding_str}'::vector
                LIMIT 1
                """,
                (resolved_id,),
            )

            is_duplicate = False
            if rows and rows[0][2] >= SIMILARITY_THRESHOLD:
                is_duplicate = True
                skipped += 1

            if not is_duplicate:
                execute_sql(
                    """
                    INSERT INTO memory_store
                        (user_id, memory_type, content, importance, embedding, source_session_id)
                    VALUES (%s, %s, %s, %s, %s::vector, %s)
                    """,
                    (resolved_id, memory_type, content, importance, embedding_str, "mcp"),
                    fetch=False,
                )
                stored += 1

            extracted.append({
                "memory_type": memory_type,
                "content": content,
                "importance": importance,
                "status": "duplicate_skipped" if is_duplicate else "stored",
            })

        return {
            "stored": stored,
            "skipped": skipped,
            "resolved_user_id": resolved_id,
            "memories": extracted,
        }

    @mcp_server.tool
    def list_memories(
        user_id: str = "",
        memory_type: str = None,
        limit: int = 20,
        include_pruned: bool = False,
    ) -> dict:
        """
        List all stored memories for a user, optionally filtered by type.
        Use this to audit or review what the agent knows about a user.

        Args:
            user_id: Optional identifier for the user. If not provided, the server
                     resolves the user's identity from their authenticated session.
            memory_type: Optional filter: fact, preference, experience, skill, relationship.
            limit: Maximum number of memories to return (default 20).
            include_pruned: If True, include soft-deleted (pruned) rows. Default False
                            mirrors the retrieval path — agents see only active memories.

        Returns:
            Dictionary with total count, resolved_user_id, and list of memories.
        """
        resolved_id = _resolve_user_id(user_id)

        type_filter = ""
        params = [resolved_id]
        if memory_type:
            type_filter = "AND memory_type = %s"
            params.append(memory_type)

        prune_filter = "" if include_pruned else "AND pruned_at IS NULL"

        cols, rows = execute_sql(
            f"""
            SELECT id, memory_type, content, importance,
                   created_at, last_accessed_at, access_count, pruned_at
            FROM memory_store
            WHERE user_id = %s {type_filter} {prune_filter}
            ORDER BY importance DESC, created_at DESC
            LIMIT %s
            """,
            (*params, limit),
        )

        if not rows:
            return {"total": 0, "resolved_user_id": resolved_id, "memories": []}

        return {
            "total": len(rows),
            "resolved_user_id": resolved_id,
            "memories": [
                {
                    "id": r[0],
                    "memory_type": r[1],
                    "content": r[2],
                    "importance": round(r[3], 2),
                    "created_at": str(r[4]),
                    "last_accessed_at": str(r[5]) if r[5] else None,
                    "access_count": r[6],
                    "pruned_at": str(r[7]) if r[7] else None,
                }
                for r in rows
            ],
        }

    # --------------------------------------------------------------
    # update_consent — DISABLED in this release. Re-enable once the
    # user_consent table is created in the schema and the MCP tool
    # surface for consent is finalized.
    # --------------------------------------------------------------
    # @mcp_server.tool
    # def update_consent(user_id: str = "", consent: bool = True) -> dict:
    #     """
    #     Update a user's consent preference for memory storage.
    #     When consent is False, no new memories will be stored for this user.
    #     Existing memories are not deleted (use a separate deletion process for that).
    #
    #     Args:
    #         user_id: Optional identifier for the user. If not provided, the server
    #                  resolves the user's identity from their authenticated session.
    #         consent: True to allow memory storage, False to opt out.
    #
    #     Returns:
    #         Confirmation of the updated consent status and resolved_user_id.
    #     """
    #     resolved_id = _resolve_user_id(user_id)
    #
    #     execute_sql(
    #         """
    #         INSERT INTO user_consent (user_id, consent, updated_at)
    #         VALUES (%s, %s, NOW())
    #         ON CONFLICT (user_id) DO UPDATE SET consent = %s, updated_at = NOW()
    #         """,
    #         (resolved_id, consent, consent),
    #         fetch=False,
    #     )
    #     return {
    #         "user_id": resolved_id,
    #         "consent": consent,
    #         "status": "updated",
    #     }

    # ================================================================
    # Tools intentionally not included in this release:
    #   - store_session_turn / get_session_context (short-term recall
    #     tier; out of scope when long-term memory is the focus).
    #   - core_memory_append / core_memory_replace / core_memory_rethink
    #     (always-in-context labeled-block tier; not part of this
    #     architecture).
    # ================================================================

    # --- Deferred / future work flags ---
    # TODO(v2): forget_memory(user_id, memory_id) — explicit user-driven
    #   deletion path (GDPR-style). Holding off until we settle on whether
    #   it does soft-delete (set pruned_at + reason) or hard-delete.
    # TODO(v2): NER-based PII redaction. Current _redact_pii is regex-only
    #   and does not catch named entities (people, orgs, locations).
    #   Consider Foundation Model API NER prompt or Presidio + spaCy.
    # TODO(v2): agent_context-driven importance reweighting. The
    #   agent_context parameter on retrieve_memory is accepted and logged
    #   but does not yet influence scoring. Plan: pass it to a scoring
    #   function that boosts memory_type weights based on calling agent's
    #   role (e.g. HR agent boosts 'relationship', ops agent boosts 'fact').
