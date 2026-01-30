# Shift existing Taobao chat timestamps +8 hours in Postgres.
# Usage:
#   1) Open PowerShell
#   2) cd <repo-root>
#   3) .\scripts\shift_taobao_timestamps_plus8.ps1
#
# Notes:
# - Only affects Taobao conversations/messages (conversation.platform = 'taobao')
# - Updates: message.ts, conversation.started_at, conversation.ended_at

$ErrorActionPreference = "Stop"

Write-Host "[1/3] Checking Postgres connection in docker compose..."
docker compose exec -T db psql -U qc -d qc -c "SELECT 1;" | Out-Null

Write-Host "[2/3] Preview: rows that will be affected (taobao only)"
docker compose exec -T db psql -U qc -d qc -c @"
SELECT
  (SELECT COUNT(1)
   FROM message m
   JOIN conversation c ON c.id = m.conversation_id
   WHERE c.platform = 'taobao' AND m.ts IS NOT NULL) AS taobao_messages_with_ts,
  (SELECT COUNT(1)
   FROM conversation c
   WHERE c.platform = 'taobao' AND c.started_at IS NOT NULL) AS taobao_conversations_with_started_at,
  (SELECT COUNT(1)
   FROM conversation c
   WHERE c.platform = 'taobao' AND c.ended_at IS NOT NULL) AS taobao_conversations_with_ended_at;
"@

Write-Host "[3/3] Applying +8 hours shift..."
docker compose exec -T db psql -U qc -d qc -c @"
BEGIN;

UPDATE message m
SET ts = m.ts + INTERVAL '8 hours'
FROM conversation c
WHERE c.id = m.conversation_id
  AND c.platform = 'taobao'
  AND m.ts IS NOT NULL;

UPDATE conversation c
SET started_at = c.started_at + INTERVAL '8 hours'
WHERE c.platform = 'taobao'
  AND c.started_at IS NOT NULL;

UPDATE conversation c
SET ended_at = c.ended_at + INTERVAL '8 hours'
WHERE c.platform = 'taobao'
  AND c.ended_at IS NOT NULL;

COMMIT;
"@

Write-Host "Done. If you want to verify, open the site and check a known conversation time." -ForegroundColor Green
