COPY (
WITH quality_convs AS (
  SELECT DISTINCT c.id, c.user_thread_id, c.platform, c.buyer_id
  FROM conversation c
  JOIN conversationanalysis a ON a.conversation_id = c.id
  JOIN conversationtaghit h ON h.analysis_id = a.id
  JOIN tagdefinition t ON t.id = h.tag_id
  JOIN tagcategory cat ON cat.id = t.category_id
  WHERE cat.name = '产品质量投诉'
  UNION
  SELECT DISTINCT c.id, c.user_thread_id, c.platform, c.buyer_id
  FROM conversation c
  JOIN manualtagbinding mb ON mb.conversation_id = c.id
  JOIN tagdefinition t ON t.id = mb.tag_id
  JOIN tagcategory cat ON cat.id = t.category_id
  WHERE cat.name = '产品质量投诉'
),
quality_threads AS (
  SELECT DISTINCT user_thread_id, platform, buyer_id
  FROM quality_convs
),
target_convs AS (
  SELECT c.*
  FROM conversation c
  JOIN quality_threads qt
    ON (qt.user_thread_id IS NOT NULL AND c.user_thread_id = qt.user_thread_id)
    OR (qt.user_thread_id IS NULL AND c.user_thread_id IS NULL AND c.platform = qt.platform AND c.buyer_id = qt.buyer_id)
)
SELECT
  COALESCE(c.user_thread_id::text, c.platform || '|' || c.buyer_id) AS thread_key,
  c.platform,
  c.buyer_id,
  c.id as conversation_id,
  c.external_id,
  COALESCE(c.meta->>'date', to_char(c.started_at::date, 'YYYY-MM-DD'), '') AS conversation_date,
  m.id as message_id,
  m.ts,
  m.sender,
  m.text,
  m.attachments
FROM target_convs c
JOIN message m ON m.conversation_id = c.id
ORDER BY thread_key, c.started_at NULLS LAST, c.id, m.ts NULLS LAST, m.id
) TO STDOUT WITH CSV HEADER;
