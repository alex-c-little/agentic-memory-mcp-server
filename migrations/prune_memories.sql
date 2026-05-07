-- ============================================================
-- Memory Pruning Job (v1) — soft-delete stale archival memories
-- ============================================================
-- Run weekly against the agent_memory Lakebase database.
-- Soft-delete only: sets pruned_at = NOW(); rows remain in the
-- table and can be restored by setting pruned_at back to NULL.
--
-- Pruning rule (all conditions must hold):
--   1. Created more than 90 days ago.
--   2. Never retrieved, OR last retrieved more than 90 days ago.
--   3. Memory type is not durable (preference/skill/relationship).
--      Durable types are protected from automatic pruning.
--
-- Note: importance threshold intentionally NOT included — once
-- importance becomes agent-relative (different agents weight
-- the same memory differently), a global threshold has no
-- meaningful interpretation at prune time.
-- ============================================================

-- -------- DRY RUN: rows that WOULD be pruned --------
SELECT
    id,
    user_id,
    memory_type,
    LEFT(content, 60) AS content_preview,
    importance,
    created_at,
    last_accessed_at,
    access_count,
    EXTRACT(DAY FROM (NOW() - created_at))::INT       AS age_days,
    EXTRACT(DAY FROM (NOW() - last_accessed_at))::INT AS days_since_access
FROM memory_store
WHERE pruned_at IS NULL
  AND created_at < NOW() - INTERVAL '90 days'
  AND (last_accessed_at IS NULL OR last_accessed_at < NOW() - INTERVAL '90 days')
  AND memory_type NOT IN ('preference', 'skill', 'relationship')
ORDER BY user_id, created_at;


-- -------- EXECUTE: soft-delete matching rows --------
UPDATE memory_store
SET pruned_at = NOW()
WHERE pruned_at IS NULL
  AND created_at < NOW() - INTERVAL '90 days'
  AND (last_accessed_at IS NULL OR last_accessed_at < NOW() - INTERVAL '90 days')
  AND memory_type NOT IN ('preference', 'skill', 'relationship');


-- -------- POST-RUN: summary by memory_type --------
SELECT
    memory_type,
    COUNT(*) FILTER (WHERE pruned_at IS NULL)  AS active_count,
    COUNT(*) FILTER (WHERE pruned_at IS NOT NULL) AS pruned_count,
    COUNT(*) AS total_count
FROM memory_store
GROUP BY memory_type
ORDER BY memory_type;


-- -------- ROLLBACK (if a prune run was wrong) --------
-- Restores rows pruned in the last hour. Adjust window as needed.
--
-- UPDATE memory_store
-- SET pruned_at = NULL
-- WHERE pruned_at IS NOT NULL
--   AND pruned_at > NOW() - INTERVAL '1 hour';
