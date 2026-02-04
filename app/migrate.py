from __future__ import annotations

from sqlalchemy import text

from db import engine


# NOTE: This project runs under Docker Compose and restarts frequently.
# Some schema patches (e.g. de-dup + index builds) can be slow on large tables.
# To avoid blocking every app boot, we record a one-time migration marker.
MIGRATION_ID = "2026-01-30-ensure_schema-v7-qc"
SETTINGS_COLUMNS_MIGRATION_ID = "2026-01-30-appconfig-settings-v2"
REJECTED_TAG_RULES_MIGRATION_ID = "2026-02-03-rejected-tag-rules-v1"
PRODUCT_QUALITY_SCOPE_MIGRATION_ID = "2026-02-04-product-quality-scope-v1"
BACKFILL_ID = "2026-01-27-backfill-multi-agent-v1"
USERTHREAD_BACKFILL_ID = "2026-01-28-backfill-user-thread-v1"


def _try_advisory_lock(conn, key: str) -> bool:
    """Avoid concurrent backfills across containers."""
    r = conn.execute(text("SELECT pg_try_advisory_lock(hashtext(:k));"), {"k": key}).first()
    return bool(r and r[0])


def _advisory_unlock(conn, key: str) -> None:
    conn.execute(text("SELECT pg_advisory_unlock(hashtext(:k));"), {"k": key})


def _ensure_migrations_table(conn) -> None:
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
              id TEXT PRIMARY KEY,
              applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
    )


def _already_applied(conn, migration_id: str) -> bool:
    r = conn.execute(
        text("SELECT 1 FROM schema_migrations WHERE id = :id LIMIT 1"),
        {"id": migration_id},
    ).first()
    return bool(r)


def _mark_applied(conn, migration_id: str) -> None:
    conn.execute(
        text(
            """
            INSERT INTO schema_migrations (id)
            VALUES (:id)
            ON CONFLICT (id) DO NOTHING;
            """
        ),
        {"id": migration_id},
    )


def ensure_schema() -> None:
    """Best-effort schema migration for MVP.

    - SQLModel's create_all() only creates missing tables, and won't add new columns.
    - In your environment you keep a persistent Postgres volume, so old tables remain.

    This function patches existing tables by adding missing columns idempotently.
    """

    if not engine.dialect.name.startswith("postgres"):
        return

    stmts = [
        'ALTER TABLE "user" ADD COLUMN IF NOT EXISTS username TEXT;',
        'ALTER TABLE "user" ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE;',
        'ALTER TABLE "user" ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP;',

        "ALTER TABLE conversation ADD COLUMN IF NOT EXISTS platform TEXT;",
        "ALTER TABLE conversation ADD COLUMN IF NOT EXISTS uploaded_at TIMESTAMP;",
        "ALTER TABLE conversation ADD COLUMN IF NOT EXISTS buyer_id TEXT;",
        "ALTER TABLE conversation ADD COLUMN IF NOT EXISTS agent_account TEXT;",
        "ALTER TABLE conversation ADD COLUMN IF NOT EXISTS agent_user_id INTEGER;",
        "ALTER TABLE conversation ADD COLUMN IF NOT EXISTS started_at TIMESTAMP;",
        "ALTER TABLE conversation ADD COLUMN IF NOT EXISTS ended_at TIMESTAMP;",
        "ALTER TABLE conversation ADD COLUMN IF NOT EXISTS meta JSONB DEFAULT '{}'::jsonb;",

        # Message attachments (images/files)
        "ALTER TABLE message ADD COLUMN IF NOT EXISTS attachments JSONB DEFAULT '[]'::jsonb;",

        # Multi-agent per-message identity
        "ALTER TABLE message ADD COLUMN IF NOT EXISTS external_message_id TEXT;",
        "ALTER TABLE message ADD COLUMN IF NOT EXISTS agent_account TEXT;",
        "ALTER TABLE message ADD COLUMN IF NOT EXISTS agent_nick TEXT;",
        "ALTER TABLE message ADD COLUMN IF NOT EXISTS agent_user_id INTEGER;",
        "CREATE INDEX IF NOT EXISTS idx_message_external_message_id ON message (external_message_id);",
        "CREATE INDEX IF NOT EXISTS idx_message_agent_account ON message (agent_account);",
        "CREATE INDEX IF NOT EXISTS idx_message_agent_user_id ON message (agent_user_id);",

        # ConversationAnalysis fields (aligned to your Excel)
        "ALTER TABLE conversationanalysis ADD COLUMN IF NOT EXISTS dialog_type TEXT;",
        "ALTER TABLE conversationanalysis ADD COLUMN IF NOT EXISTS pre_positive_tags TEXT;",
        "ALTER TABLE conversationanalysis ADD COLUMN IF NOT EXISTS after_positive_tags TEXT;",
        "ALTER TABLE conversationanalysis ADD COLUMN IF NOT EXISTS pre_negative_tags TEXT;",
        "ALTER TABLE conversationanalysis ADD COLUMN IF NOT EXISTS after_negative_tags TEXT;",
        "ALTER TABLE conversationanalysis ADD COLUMN IF NOT EXISTS day_summary TEXT;",
        # Legacy compatibility: older DBs may have a NOT NULL `summary` column.
        # Ensure it exists and is safe to write.
        "ALTER TABLE conversationanalysis ADD COLUMN IF NOT EXISTS summary TEXT DEFAULT '';",
        "UPDATE conversationanalysis SET summary = '' WHERE summary IS NULL;",
        "ALTER TABLE conversationanalysis ALTER COLUMN summary SET DEFAULT '';",
        "ALTER TABLE conversationanalysis ADD COLUMN IF NOT EXISTS tag_parsing TEXT;",
        "ALTER TABLE conversationanalysis ADD COLUMN IF NOT EXISTS evidence JSONB DEFAULT '{}'::jsonb;",
        "ALTER TABLE conversationanalysis ADD COLUMN IF NOT EXISTS product_suggestion TEXT;",
        "ALTER TABLE conversationanalysis ADD COLUMN IF NOT EXISTS service_suggestion TEXT;",
        "ALTER TABLE conversationanalysis ADD COLUMN IF NOT EXISTS pre_rule_update TEXT;",
        "ALTER TABLE conversationanalysis ADD COLUMN IF NOT EXISTS after_rule_update TEXT;",
        "ALTER TABLE conversationanalysis ADD COLUMN IF NOT EXISTS tag_update_suggestion TEXT;",

        # Compatibility fields
        "ALTER TABLE conversationanalysis ADD COLUMN IF NOT EXISTS overall_score INTEGER;",
        "ALTER TABLE conversationanalysis ADD COLUMN IF NOT EXISTS sentiment TEXT;",
        "ALTER TABLE conversationanalysis ADD COLUMN IF NOT EXISTS issue_level TEXT;",
        "ALTER TABLE conversationanalysis ADD COLUMN IF NOT EXISTS problem_types JSONB DEFAULT '[]'::jsonb;",
        "ALTER TABLE conversationanalysis ADD COLUMN IF NOT EXISTS flag_for_review BOOLEAN DEFAULT FALSE;",
        "ALTER TABLE conversationanalysis ADD COLUMN IF NOT EXISTS extra JSONB DEFAULT '{}'::jsonb;",
        "ALTER TABLE conversationanalysis ADD COLUMN IF NOT EXISTS reception_scenario TEXT DEFAULT '';",
        "ALTER TABLE conversationanalysis ADD COLUMN IF NOT EXISTS satisfaction_change TEXT DEFAULT '';",

        # TrainingTask additions
        "ALTER TABLE trainingtask ADD COLUMN IF NOT EXISTS focus_negative_tag TEXT;",
        "ALTER TABLE trainingtask ADD COLUMN IF NOT EXISTS reflection_question TEXT;",

        # AgentBinding meta
        "ALTER TABLE agentbinding ADD COLUMN IF NOT EXISTS meta JSONB DEFAULT '{}'::jsonb;",

        # ImportRun (daily status board)
        "ALTER TABLE importrun ADD COLUMN IF NOT EXISTS platform TEXT DEFAULT 'unknown';",
        "ALTER TABLE importrun ADD COLUMN IF NOT EXISTS source TEXT DEFAULT 'bucket';",
        "ALTER TABLE importrun ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'done';",
        "ALTER TABLE importrun ADD COLUMN IF NOT EXISTS details JSONB DEFAULT '{}'::jsonb;",

        # AppConfig (admin-tunable runtime settings)
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS bucket_fetch_enabled BOOLEAN DEFAULT TRUE;",
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS taobao_bucket_import_enabled BOOLEAN DEFAULT TRUE;",
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS douyin_bucket_import_enabled BOOLEAN DEFAULT FALSE;",
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS bucket_daily_check_time TEXT DEFAULT '10:15';",
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS bucket_retry_interval_minutes INTEGER DEFAULT 60;",
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS bucket_log_keep INTEGER DEFAULT 800;",

        # Daily AI summary settings (stored in AppConfig)
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS daily_summary_threshold INTEGER DEFAULT 8;",
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS daily_summary_model TEXT DEFAULT 'gpt-5.2';",
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS daily_summary_prompt TEXT DEFAULT '';",
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS qc_system_prompt TEXT DEFAULT '';",

        # App-configurable (env overridable via UI; stored in AppConfig)
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS bucket_backup_dir TEXT DEFAULT '';",
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS taobao_bucket_config JSONB DEFAULT '{}'::jsonb;",
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS douyin_bucket_config JSONB DEFAULT '{}'::jsonb;",
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS feishu_webhook_url TEXT DEFAULT '';",
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS openai_base_url TEXT DEFAULT '';",
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS openai_api_key TEXT DEFAULT '';",
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS openai_model TEXT DEFAULT 'gpt-5.2';",
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS openai_fallback_model TEXT DEFAULT '';",
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS openai_timeout INTEGER DEFAULT 1800;",
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS openai_retries INTEGER DEFAULT 0;",
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS openai_reasoning_effort TEXT DEFAULT 'high';",
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS open_registration BOOLEAN DEFAULT FALSE;",
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS cron_interval_seconds INTEGER DEFAULT 600;",
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS worker_poll_seconds INTEGER DEFAULT 5;",
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS enable_enqueue_analysis BOOLEAN DEFAULT TRUE;",

        # Constraints / indexes for binding & fast lookup
        # 目标：同一站内账号在同一平台允许绑定多个客服账号。
        # 因此：
        # - 仍然要求 (platform, agent_account) 唯一（一个平台账号只能绑定到一个站内账号）
        # - 不再要求 (platform, user_id) 唯一（一个站内账号可绑定多个平台账号）

        # De-duplicate AgentBinding rows before applying UNIQUE index (avoid startup failure)
        "DELETE FROM agentbinding a USING agentbinding b WHERE a.id < b.id AND a.platform = b.platform AND a.agent_account = b.agent_account;",

        # Drop legacy UNIQUE constraint/index on (platform, user_id) if exists.
        # 你当前线上报错的名字是 agentbinding_platform_user_uq（是 CONSTRAINT 不是 INDEX）。
        'ALTER TABLE agentbinding DROP CONSTRAINT IF EXISTS "agentbinding_platform_user_uq";',
        'DROP INDEX IF EXISTS "agentbinding_platform_user_uq";',
        'DROP INDEX IF EXISTS "agentbinding_platform_user_id_uniq";',

        # Safety net: if历史库里约束名字不一致，也把 (platform,user_id) 的 UNIQUE 约束删掉
        """
        DO $$
        DECLARE c_name text;
        BEGIN
          SELECT con.conname INTO c_name
          FROM pg_constraint con
          JOIN pg_class rel ON rel.oid = con.conrelid
          WHERE rel.relname = 'agentbinding'
            AND con.contype = 'u'
            AND (
              -- array_agg(attname) returns name[]; cast to text[] to compare safely
              SELECT array_agg(att.attname::text ORDER BY att.attname::text)
              FROM unnest(con.conkey) AS k(attnum)
              JOIN pg_attribute att ON att.attrelid = rel.oid AND att.attnum = k.attnum
            ) = ARRAY['platform','user_id'];

          IF c_name IS NOT NULL THEN
            EXECUTE format('ALTER TABLE agentbinding DROP CONSTRAINT IF EXISTS %I', c_name);
          END IF;
        END $$;
        """,

        'CREATE UNIQUE INDEX IF NOT EXISTS "agentbinding_platform_agent_account_uniq" ON agentbinding (platform, agent_account);',
        'CREATE INDEX IF NOT EXISTS "agentbinding_platform_user_id_idx" ON agentbinding (platform, user_id);',
        'CREATE INDEX IF NOT EXISTS "conversation_platform_agent_account_idx" ON conversation (platform, agent_account);',

        
        # UserThread: cross-day buyer history grouping
        "CREATE TABLE IF NOT EXISTS userthread (id SERIAL PRIMARY KEY, platform TEXT DEFAULT 'unknown', buyer_id TEXT NOT NULL, created_at TIMESTAMP DEFAULT NOW(), meta JSONB DEFAULT '{}'::jsonb);",
        "ALTER TABLE userthread ADD COLUMN IF NOT EXISTS platform TEXT;",
        "ALTER TABLE userthread ADD COLUMN IF NOT EXISTS buyer_id TEXT;",
        "ALTER TABLE userthread ADD COLUMN IF NOT EXISTS created_at TIMESTAMP;",
        "ALTER TABLE userthread ADD COLUMN IF NOT EXISTS meta JSONB DEFAULT '{}'::jsonb;",
        "CREATE UNIQUE INDEX IF NOT EXISTS userthread_platform_buyer_id_uniq ON userthread (platform, buyer_id);",
        "CREATE INDEX IF NOT EXISTS userthread_created_at_idx ON userthread (created_at);",
        "ALTER TABLE conversation ADD COLUMN IF NOT EXISTS user_thread_id INTEGER;",
        "CREATE INDEX IF NOT EXISTS conversation_user_thread_id_idx ON conversation (user_thread_id);",

# Daily AI summary report lookup
        'CREATE UNIQUE INDEX IF NOT EXISTS "dailyaisummaryreport_run_date_uniq" ON dailyaisummaryreport (run_date);',

        # =========================
        # Tag system (category + tag + hits)
        # =========================

        """
        CREATE TABLE IF NOT EXISTS tagcategory (
          id SERIAL PRIMARY KEY,
          name TEXT UNIQUE NOT NULL,
          description TEXT DEFAULT '',
          sort_order INTEGER DEFAULT 0,
          is_active BOOLEAN DEFAULT TRUE,
          created_at TIMESTAMP DEFAULT NOW(),
          updated_at TIMESTAMP DEFAULT NOW()
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS tagdefinition (
          id SERIAL PRIMARY KEY,
          category_id INTEGER NOT NULL REFERENCES tagcategory(id),
          name TEXT NOT NULL,
          standard TEXT DEFAULT '',
          description TEXT DEFAULT '',
          sort_order INTEGER DEFAULT 0,
          is_active BOOLEAN DEFAULT TRUE,
          created_at TIMESTAMP DEFAULT NOW(),
          updated_at TIMESTAMP DEFAULT NOW()
        );
        """,
        'CREATE UNIQUE INDEX IF NOT EXISTS tagdefinition_category_name_uniq ON tagdefinition (category_id, name);',
        'CREATE INDEX IF NOT EXISTS tagdefinition_category_idx ON tagdefinition (category_id);',
        'CREATE INDEX IF NOT EXISTS tagdefinition_active_idx ON tagdefinition (is_active);',

        """
        CREATE TABLE IF NOT EXISTS conversationtaghit (
          id SERIAL PRIMARY KEY,
          analysis_id INTEGER NOT NULL REFERENCES conversationanalysis(id),
          tag_id INTEGER NOT NULL REFERENCES tagdefinition(id),
          hit BOOLEAN DEFAULT TRUE,
          reason TEXT DEFAULT '',
          evidence JSONB DEFAULT '{}'::jsonb,
          created_at TIMESTAMP DEFAULT NOW()
        );
        """,
        'ALTER TABLE conversationtaghit ADD COLUMN IF NOT EXISTS hit BOOLEAN DEFAULT TRUE;',
        'ALTER TABLE conversationtaghit ADD COLUMN IF NOT EXISTS evidence JSONB DEFAULT\'{}\'::jsonb;',
        'CREATE UNIQUE INDEX IF NOT EXISTS conversationtaghit_analysis_tag_uniq ON conversationtaghit (analysis_id, tag_id);',
        'CREATE INDEX IF NOT EXISTS conversationtaghit_tag_idx ON conversationtaghit (tag_id);',
        'CREATE INDEX IF NOT EXISTS conversationtaghit_analysis_idx ON conversationtaghit (analysis_id);',

        # QC PDR: TagSuggestion (new-tag review workflow)
        """
        CREATE TABLE IF NOT EXISTS tagsuggestion (
            id SERIAL PRIMARY KEY,
            analysis_id INTEGER NOT NULL REFERENCES conversationanalysis(id),
            conversation_id INTEGER NOT NULL REFERENCES conversation(id),
            suggested_category TEXT DEFAULT '',
            suggested_tag_name TEXT DEFAULT '',
            suggested_standard TEXT DEFAULT '',
            suggested_description TEXT DEFAULT '',
            ai_reason TEXT DEFAULT '',
            status TEXT DEFAULT 'pending',
            reviewed_by_user_id INTEGER REFERENCES "user"(id),
            reviewed_at TIMESTAMP,
            review_notes TEXT DEFAULT '',
            created_tag_id INTEGER REFERENCES tagdefinition(id),
            created_at TIMESTAMP DEFAULT NOW()
        );
        """,
        'CREATE INDEX IF NOT EXISTS tagsuggestion_status_idx ON tagsuggestion (status);',
        'CREATE INDEX IF NOT EXISTS tagsuggestion_conversation_idx ON tagsuggestion (conversation_id);',
        'CREATE INDEX IF NOT EXISTS tagsuggestion_analysis_idx ON tagsuggestion (analysis_id);',

        # QC PDR: ManualTagBinding (admin/supervisor manually bind tag to conversation)
        """
        CREATE TABLE IF NOT EXISTS manualtagbinding (
            id SERIAL PRIMARY KEY,
            conversation_id INTEGER NOT NULL REFERENCES conversation(id),
            tag_id INTEGER NOT NULL REFERENCES tagdefinition(id),
            created_by_user_id INTEGER NOT NULL REFERENCES "user"(id),
            created_at TIMESTAMP DEFAULT NOW(),
            reason TEXT DEFAULT ''
        );
        """,
        'CREATE UNIQUE INDEX IF NOT EXISTS manualtagbinding_conv_tag_uniq ON manualtagbinding (conversation_id, tag_id);',
        'CREATE INDEX IF NOT EXISTS manualtagbinding_conversation_idx ON manualtagbinding (conversation_id);',
        'CREATE INDEX IF NOT EXISTS manualtagbinding_tag_idx ON manualtagbinding (tag_id);',
        'ALTER TABLE manualtagbinding ADD COLUMN IF NOT EXISTS evidence JSONB DEFAULT \'{}\'::jsonb;',
]

    with engine.begin() as conn:
        _ensure_migrations_table(conn)
        if not _already_applied(conn, MIGRATION_ID):
            for s in stmts:
                conn.execute(text(s))
            conn.execute(
                text(
                    """
                    UPDATE appconfig
                    SET daily_summary_model = 'gpt-5.2'
                    WHERE daily_summary_model = 'gpt-5.2-thinking';
                    """
                )
            )
            conn.execute(
                text(
                    """
                    UPDATE "user"
                    SET username = COALESCE(
                        NULLIF(username, ''),
                        NULLIF(split_part(email, '@', 1), ''),
                        NULLIF(name, ''),
                        'user_' || id::text
                    )
                    WHERE username IS NULL OR username = '';
                    """
                )
            )
            _mark_applied(conn, MIGRATION_ID)

    _apply_rejected_tag_rules(engine)
    # Separate migration for new AppConfig columns (runs even if v6 was already applied).
    _apply_appconfig_settings_columns(engine)
    _apply_product_quality_scope(engine)


def _apply_product_quality_scope(eng) -> None:
    """Force product-quality complaint tags to only hit on post-delivery feedback.

    用户诉求：避免售前“会不会起球/会不会掉色”等咨询也命中“产品质量投诉”二级标签。
    通过在该一级分类下的所有二级标签判定标准前追加明确的“命中判定范围”约束实现。
    """
    prefix = "命中判定范围：用户收到货后反馈（含试穿/使用/洗后）；售前/未收货仅咨询不命中。命中标准："
    with eng.begin() as conn:
        _ensure_migrations_table(conn)
        if _already_applied(conn, PRODUCT_QUALITY_SCOPE_MIGRATION_ID):
            return

        # Idempotent: only modify "产品质量投诉" category.
        # - If already correct, keep.
        # - If has an old "命中判定范围：...命中标准：" header, replace the header.
        # - Else, just prepend the header.
        conn.execute(
            text(
                """
                WITH cat AS (
                  SELECT id
                  FROM tagcategory
                  WHERE trim(name) = '产品质量投诉'
                  LIMIT 1
                )
                UPDATE tagdefinition t
                SET
                  standard = CASE
                    WHEN coalesce(t.standard, '') LIKE :ok_prefix THEN t.standard
                    WHEN coalesce(t.standard, '') LIKE '命中判定范围：%' AND position('命中标准：' in coalesce(t.standard, '')) > 0 THEN
                      :prefix || substring(
                        coalesce(t.standard, ''),
                        position('命中标准：' in coalesce(t.standard, '')) + char_length('命中标准：')
                      )
                    ELSE :prefix || coalesce(t.standard, '')
                  END,
                  updated_at = NOW()
                WHERE t.category_id = (SELECT id FROM cat);
                """
            ),
            {"prefix": prefix, "ok_prefix": "命中判定范围：用户收到货后反馈%"},
        )

        _mark_applied(conn, PRODUCT_QUALITY_SCOPE_MIGRATION_ID)


def _apply_rejected_tag_rules(eng) -> None:
    stmts = [
        """
        CREATE TABLE IF NOT EXISTS rejectedtagrule (
            id SERIAL PRIMARY KEY,
            category TEXT DEFAULT '',
            tag_name TEXT DEFAULT '',
            aliases TEXT DEFAULT '',
            notes TEXT DEFAULT '',
            norm_key TEXT DEFAULT '',
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW(),
            created_by_user_id INTEGER REFERENCES "user"(id),
            updated_by_user_id INTEGER REFERENCES "user"(id)
        );
        """,
        "CREATE UNIQUE INDEX IF NOT EXISTS rejectedtagrule_norm_key_uniq ON rejectedtagrule (norm_key);",
        "CREATE INDEX IF NOT EXISTS rejectedtagrule_active_idx ON rejectedtagrule (is_active);",
        "CREATE INDEX IF NOT EXISTS rejectedtagrule_category_idx ON rejectedtagrule (category);",
        "ALTER TABLE rejectedtagrule ALTER COLUMN is_active SET DEFAULT TRUE;",
    ]
    with eng.begin() as conn:
        _ensure_migrations_table(conn)
        if _already_applied(conn, REJECTED_TAG_RULES_MIGRATION_ID):
            return
        for s in stmts:
            conn.execute(text(s))
        conn.execute(text("UPDATE rejectedtagrule SET is_active = TRUE WHERE is_active IS NULL;"))
        # Backfill from historical rejected suggestions (best-effort).
        conn.execute(
            text(
                r"""
                INSERT INTO rejectedtagrule (category, tag_name, notes, norm_key, is_active, created_at, updated_at)
                SELECT DISTINCT ON (norm_key)
                    suggested_category,
                    suggested_tag_name,
                    review_notes,
                    norm_key,
                    TRUE,
                    NOW(),
                    NOW()
                FROM (
                    SELECT
                        suggested_category,
                        suggested_tag_name,
                        review_notes,
                        lower(regexp_replace(
                            coalesce(suggested_category, '') || '::' || coalesce(suggested_tag_name, ''),
                            '\s+',
                            '',
                            'g'
                        )) AS norm_key,
                        reviewed_at
                    FROM tagsuggestion
                    WHERE status = 'rejected'
                      AND (coalesce(suggested_category, '') <> '' OR coalesce(suggested_tag_name, '') <> '')
                ) t
                WHERE norm_key <> ''
                ORDER BY norm_key, reviewed_at DESC NULLS LAST
                ON CONFLICT (norm_key) DO NOTHING;
                """
            )
        )
        _mark_applied(conn, REJECTED_TAG_RULES_MIGRATION_ID)


def _apply_appconfig_settings_columns(eng) -> None:
    stmts = [
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS bucket_backup_dir TEXT DEFAULT '';",
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS taobao_bucket_config JSONB DEFAULT '{}'::jsonb;",
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS douyin_bucket_config JSONB DEFAULT '{}'::jsonb;",
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS feishu_webhook_url TEXT DEFAULT '';",
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS openai_base_url TEXT DEFAULT '';",
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS openai_api_key TEXT DEFAULT '';",
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS openai_model TEXT DEFAULT 'gpt-5.2';",
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS openai_fallback_model TEXT DEFAULT '';",
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS openai_timeout INTEGER DEFAULT 1800;",
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS openai_retries INTEGER DEFAULT 0;",
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS openai_reasoning_effort TEXT DEFAULT 'high';",
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS open_registration BOOLEAN DEFAULT FALSE;",
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS cron_interval_seconds INTEGER DEFAULT 600;",
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS worker_poll_seconds INTEGER DEFAULT 5;",
        "ALTER TABLE appconfig ADD COLUMN IF NOT EXISTS enable_enqueue_analysis BOOLEAN DEFAULT TRUE;",
    ]
    with eng.begin() as conn:
        _ensure_migrations_table(conn)
        if _already_applied(conn, SETTINGS_COLUMNS_MIGRATION_ID):
            return
        for s in stmts:
            conn.execute(text(s))
        _mark_applied(conn, SETTINGS_COLUMNS_MIGRATION_ID)


def run_multi_agent_backfill() -> None:
    """Backfill per-message agent identity for historical rows.

    Runs in small transactions to avoid holding locks for a long time.
    Safe to call multiple times; guarded by schema_migrations + advisory lock.
    """
    if not engine.dialect.name.startswith("postgres"):
        return

    # Quick check + global lock
    with engine.begin() as conn:
        _ensure_migrations_table(conn)
        if _already_applied(conn, BACKFILL_ID):
            return
        if not _try_advisory_lock(conn, BACKFILL_ID):
            # Another container is doing it.
            return

    try:
        _backfill_external_message_id()
        _backfill_agent_account_from_backups()
        _backfill_message_agent_user_id()
        _refresh_conversation_primary_agent()

        with engine.begin() as conn:
            _ensure_migrations_table(conn)
            _mark_applied(conn, BACKFILL_ID)
    finally:
        with engine.begin() as conn:
            try:
                _advisory_unlock(conn, BACKFILL_ID)
            except Exception:
                pass



def run_user_thread_backfill() -> None:
    """Backfill Conversation.user_thread_id and create UserThread rows.

    - Each external user is grouped by (platform, buyer_id).
    - Conversations remain day-scoped, but the UI can show the full history by thread.

    Safe to call multiple times; guarded by schema_migrations + advisory lock.
    """
    if not engine.dialect.name.startswith("postgres"):
        return

    with engine.begin() as conn:
        _ensure_migrations_table(conn)
        if _already_applied(conn, USERTHREAD_BACKFILL_ID):
            return
        if not _try_advisory_lock(conn, USERTHREAD_BACKFILL_ID):
            return

    try:
        # 1) Create threads for distinct buyers
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO userthread (platform, buyer_id, created_at)
                    SELECT DISTINCT lower(coalesce(platform,'unknown')) AS platform, buyer_id, NOW()
                    FROM conversation
                    WHERE buyer_id IS NOT NULL AND buyer_id <> ''
                    ON CONFLICT (platform, buyer_id) DO NOTHING;
                    """
                )
            )

        # 2) Attach conversations to their thread
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE conversation c
                    SET user_thread_id = ut.id
                    FROM userthread ut
                    WHERE (c.user_thread_id IS NULL OR c.user_thread_id = 0)
                      AND c.buyer_id IS NOT NULL AND c.buyer_id <> ''
                      AND lower(coalesce(c.platform,'unknown')) = ut.platform
                      AND c.buyer_id = ut.buyer_id;
                    """
                )
            )

        with engine.begin() as conn:
            _ensure_migrations_table(conn)
            _mark_applied(conn, USERTHREAD_BACKFILL_ID)
    finally:
        with engine.begin() as conn:
            try:
                _advisory_unlock(conn, USERTHREAD_BACKFILL_ID)
            except Exception:
                pass


def _backfill_external_message_id(batch: int = 5000) -> None:
    while True:
        with engine.begin() as conn:
            rows = conn.execute(
                text(
                    """
                    WITH todo AS (
                      SELECT id
                      FROM message
                      WHERE (external_message_id IS NULL OR external_message_id = '')
                        AND attachments IS NOT NULL
                        AND jsonb_typeof((attachments)::jsonb) = 'array'
                      LIMIT :lim
                    )
                    UPDATE message m
                    SET external_message_id = (
                      SELECT NULLIF(elem->>'message_id','')
                      FROM jsonb_array_elements(COALESCE((m.attachments)::jsonb, '[]'::jsonb)) elem
                      WHERE (elem->>'type') = 'meta' AND (elem ? 'message_id')
                      LIMIT 1
                    )
                    WHERE m.id IN (SELECT id FROM todo)
                    RETURNING m.id;
                    """
                ),
                {"lim": batch},
            ).fetchall()
        if not rows:
            break


def _backfill_agent_account_from_backups(max_needed: int = 300000) -> None:
    """Recover agent_account/agent_nick for historical agent messages.

    Best-effort: depends on bucketobject.backup_path existing on disk.
    """
    try:
        import os, json, gzip
    except Exception:
        return

    # Build the "needed" set first (read-only)
    with engine.begin() as conn:
        needed_rows = conn.execute(
            text(
                """
                SELECT DISTINCT external_message_id
                FROM message
                WHERE sender = 'agent'
                  AND external_message_id IS NOT NULL
                  AND external_message_id <> ''
                  AND (agent_account IS NULL OR agent_account = '')
                LIMIT :lim;
                """
            ),
            {"lim": max_needed},
        ).fetchall()
        needed = {r[0] for r in (needed_rows or []) if r and r[0]}
        if not needed:
            return

        paths = conn.execute(
            text(
                """
                SELECT backup_path
                FROM bucketobject
                WHERE backup_path IS NOT NULL AND backup_path <> ''
                ORDER BY imported_at DESC NULLS LAST, downloaded_at DESC NULLS LAST, id DESC;
                """
            )
        ).fetchall()

    mapping: dict[str, dict[str, str]] = {}
    for (p,) in (paths or []):
        if not p or not os.path.exists(p):
            continue
        try:
            opener = gzip.open if str(p).endswith(".gz") else open
            with opener(p, "rt", encoding="utf-8") as f:
                for line in f:
                    line = (line or "").strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    mid = str(obj.get("message_id") or "").strip()
                    if not mid or mid not in needed:
                        continue
                    aid = str(obj.get("assistant_id") or "").strip()
                    anick = str(obj.get("assistant_nick") or "").strip()
                    if aid or anick:
                        mapping[mid] = {"mid": mid, "aid": aid, "anick": anick}
                        needed.discard(mid)
                    if not needed:
                        break
        except Exception:
            continue
        if not needed:
            break

    if not mapping:
        return

    params = list(mapping.values())
    # Write back in a single short transaction
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE message
                SET agent_account = COALESCE(NULLIF(agent_account,''), :aid),
                    agent_nick = COALESCE(NULLIF(agent_nick,''), :anick)
                WHERE sender = 'agent'
                  AND external_message_id = :mid
                  AND (agent_account IS NULL OR agent_account = '');
                """
            ),
            params,
        )


def _backfill_message_agent_user_id(batch: int = 8000) -> None:
    while True:
        with engine.begin() as conn:
            rows = conn.execute(
                text(
                    """
                    WITH todo AS (
                      SELECT m.id
                      FROM message m
                      JOIN conversation c ON m.conversation_id = c.id
                      JOIN agentbinding b
                        ON lower(b.platform) = lower(c.platform)
                       AND b.agent_account = m.agent_account
                      WHERE m.sender = 'agent'
                        AND m.agent_user_id IS NULL
                        AND m.agent_account IS NOT NULL
                        AND m.agent_account <> ''
                      LIMIT :lim
                    )
                    UPDATE message m
                    SET agent_user_id = b.user_id
                    FROM conversation c
                    JOIN agentbinding b ON lower(b.platform) = lower(c.platform)
                    WHERE m.id IN (SELECT id FROM todo)
                      AND m.conversation_id = c.id
                      AND b.agent_account = m.agent_account
                    RETURNING m.id;
                    """
                ),
                {"lim": batch},
            ).fetchall()
        if not rows:
            break


def _refresh_conversation_primary_agent() -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                WITH ranked AS (
                  SELECT
                    conversation_id,
                    agent_account,
                    COUNT(*) AS cnt,
                    ROW_NUMBER() OVER (
                      PARTITION BY conversation_id
                      ORDER BY COUNT(*) DESC, agent_account ASC
                    ) AS rn
                  FROM message
                  WHERE sender = 'agent'
                    AND agent_account IS NOT NULL
                    AND agent_account <> ''
                  GROUP BY conversation_id, agent_account
                )
                UPDATE conversation c
                SET agent_account = r.agent_account
                FROM ranked r
                WHERE r.conversation_id = c.id
                  AND r.rn = 1;
                """
            )
        )

        conn.execute(
            text(
                """
                WITH ranked_u AS (
                  SELECT
                    conversation_id,
                    agent_user_id,
                    COUNT(*) AS cnt,
                    ROW_NUMBER() OVER (
                      PARTITION BY conversation_id
                      ORDER BY COUNT(*) DESC, agent_user_id ASC
                    ) AS rn
                  FROM message
                  WHERE sender = 'agent'
                    AND agent_user_id IS NOT NULL
                  GROUP BY conversation_id, agent_user_id
                )
                UPDATE conversation c
                SET agent_user_id = r.agent_user_id
                FROM ranked_u r
                WHERE r.conversation_id = c.id
                  AND r.rn = 1;
                """
            )
        )


def start_multi_agent_backfill_background() -> None:
    """Start backfill without blocking app startup."""
    try:
        import threading

        t = threading.Thread(target=run_multi_agent_backfill, name="qc-multi-agent-backfill", daemon=True)
        t.start()

        t2 = threading.Thread(target=run_user_thread_backfill, name="qc-user-thread-backfill", daemon=True)
        t2.start()
    except Exception:
        # best-effort; do not block startup
        pass
