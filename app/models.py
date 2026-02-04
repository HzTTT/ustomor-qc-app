from datetime import datetime
from enum import Enum
from typing import Optional

from sqlmodel import SQLModel, Field, Relationship, Column
from sqlalchemy import JSON, Text


class Role(str, Enum):
    admin = "admin"
    supervisor = "supervisor"
    agent = "agent"


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    # 登录名：用于“用户名+密码”登录。旧数据会在迁移时自动填充。
    username: Optional[str] = Field(default=None, index=True)

    # email 可保留用于找回/通知等（MVP 仍然必填，后续你也可以改为可空）
    email: str = Field(index=True, unique=True)
    name: str
    role: Role = Field(default=Role.agent)
    password_hash: str

    # 软删除/停用：删除账号时不会破坏历史数据
    is_active: bool = Field(default=True, index=True)
    deleted_at: Optional[datetime] = Field(default=None, index=True)

    created_at: datetime = Field(default_factory=datetime.utcnow)

    conversations: list["Conversation"] = Relationship(back_populates="agent")



class UserThread(SQLModel, table=True):
    """A cross-day thread for the same external user (platform + buyer_id).

    Think of it like a folder that groups the daily conversations of the same buyer.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    platform: str = Field(default="unknown", index=True)
    buyer_id: str = Field(default="", index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    meta: dict = Field(default_factory=dict, sa_column=Column(JSON))

    conversations: list["Conversation"] = Relationship(back_populates="user_thread")


class Conversation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    external_id: str = Field(index=True, unique=True)

    # === 对话库（基础导入端）===
    platform: str = Field(default="unknown", index=True)
    uploaded_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    buyer_id: str = Field(default="", index=True)
    user_thread_id: Optional[int] = Field(default=None, foreign_key="userthread.id", index=True)
    agent_account: str = Field(default="", index=True)  # 平台客服账号/工号等

    agent_user_id: Optional[int] = Field(default=None, foreign_key="user.id", index=True)
    started_at: Optional[datetime] = Field(default=None, index=True)
    ended_at: Optional[datetime] = Field(default=None, index=True)

    meta: dict = Field(default_factory=dict, sa_column=Column(JSON))

    user_thread: Optional["UserThread"] = Relationship(back_populates="conversations")
    agent: Optional["User"] = Relationship(back_populates="conversations")
    messages: list["Message"] = Relationship(back_populates="conversation")
    analyses: list["ConversationAnalysis"] = Relationship(back_populates="conversation")
    tasks: list["TrainingTask"] = Relationship(back_populates="conversation")


class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    conversation_id: int = Field(foreign_key="conversation.id", index=True)
    sender: str = Field(index=True)  # buyer/agent/system
    ts: Optional[datetime] = Field(default=None, index=True)
    text: str = Field(default="")

    # === Multi-agent support (same conversation may involve multiple agents) ===
    # Platform message id for backfill / traceability (e.g. leyan message_id)
    external_message_id: Optional[str] = Field(default=None, index=True)

    # For agent messages, store the actual external agent identity on EACH message
    agent_account: str = Field(default="", index=True)  # assistant_id / agent account
    agent_nick: str = Field(default="")
    agent_user_id: Optional[int] = Field(default=None, foreign_key="user.id", index=True)

    # Attachments for images/files (bucket object keys or URLs)
    # Example: [{"type":"image","url":"https://..."},{"type":"file","name":"xxx.pdf","url":"..."}]
    attachments: list = Field(default_factory=list, sa_column=Column(JSON))

    conversation: "Conversation" = Relationship(back_populates="messages")


class AnalysisBatch(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    source_filename: str = Field(default="", index=True)
    imported_by_user_id: Optional[int] = Field(default=None, foreign_key="user.id")
    imported_at: datetime = Field(default_factory=datetime.utcnow, index=True)

    batch_meta: dict = Field(default_factory=dict, sa_column=Column(JSON))
    raw_json: dict = Field(default_factory=dict, sa_column=Column(JSON))

    analyses: list["ConversationAnalysis"] = Relationship(back_populates="batch")


class ConversationAnalysis(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    batch_id: int = Field(foreign_key="analysisbatch.id", index=True)
    conversation_id: int = Field(foreign_key="conversation.id", index=True)

    # === 质量诊断（按你文档/Excel 对齐）===
    dialog_type: str = Field(default="", index=True)  # 售前/售后/售前售后/其他
    # 接待场景（5选项）：售前/售后/售前和售后/其他接待场景/无法判定
    reception_scenario: str = Field(default="", index=True)
    # 满意度变化（6选项）：大幅减少/小幅减少/无显著变化/小幅增加/大幅增加/无法判定
    satisfaction_change: str = Field(default="", index=True)

    # 接待正向结果（售前/售后各自正面标签）
    pre_positive_tags: str = Field(default="")
    after_positive_tags: str = Field(default="")

    # 接待负向结果（售前/售后各自负面标签）
    pre_negative_tags: str = Field(default="")
    after_negative_tags: str = Field(default="")

    day_summary: str = Field(default="")

    # Legacy compatibility: some existing DBs have a NOT NULL `summary` column.
    # Keep it in sync with day_summary so inserts never fail.
    summary: str = Field(default="")

    # 评价解析：`tag$$$reason&&&tag$$$reason`
    tag_parsing: str = Field(default="")

    # 问题定位：AI返回的“证据片段”，方便快速定位对话中相关部分
    # Example: {"highlights": [{"message_index": 12, "sender": "buyer", "quote": "...", "tag": "...", "why": "..."}]}
    evidence: dict = Field(default_factory=dict, sa_column=Column(JSON))

    # VOC: 商品/服务建议
    product_suggestion: str = Field(default="")
    service_suggestion: str = Field(default="")

    # 规则更新建议
    pre_rule_update: str = Field(default="")
    after_rule_update: str = Field(default="")
    tag_update_suggestion: str = Field(default="")

    # 兼容字段（可选保留，用于排序/阈值等）
    overall_score: Optional[int] = Field(default=None, index=True)
    sentiment: str = Field(default="", index=True)
    issue_level: str = Field(default="", index=True)
    problem_types: list = Field(default_factory=list, sa_column=Column(JSON))
    flag_for_review: bool = Field(default=False, index=True)

    extra: dict = Field(default_factory=dict, sa_column=Column(JSON))

    batch: "AnalysisBatch" = Relationship(back_populates="analyses")
    conversation: "Conversation" = Relationship(back_populates="analyses")


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    error = "error"


class AIAnalysisJob(SQLModel, table=True):
    """后台分析任务：为尚未有AI分析的对话生成 ConversationAnalysis."""

    id: Optional[int] = Field(default=None, primary_key=True)
    conversation_id: int = Field(foreign_key="conversation.id", index=True)
    status: JobStatus = Field(default=JobStatus.pending, index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    started_at: Optional[datetime] = Field(default=None, index=True)
    finished_at: Optional[datetime] = Field(default=None, index=True)
    attempts: int = Field(default=0)
    last_error: str = Field(default="")

    # 用于追踪：本次分析落到哪个batch（auto-ai）
    batch_id: Optional[int] = Field(default=None, foreign_key="analysisbatch.id", index=True)

    extra: dict = Field(default_factory=dict, sa_column=Column(JSON))



class DailyAISummaryJob(SQLModel, table=True):
    """后台日报生成任务：长耗时AI调用由 worker 异步执行，避免页面等待/刷新丢失。"""

    id: Optional[int] = Field(default=None, primary_key=True)
    run_date: str = Field(index=True)  # YYYY-MM-DD

    status: JobStatus = Field(default=JobStatus.pending, index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    started_at: Optional[datetime] = Field(default=None, index=True)
    finished_at: Optional[datetime] = Field(default=None, index=True)
    attempts: int = Field(default=0)
    last_error: str = Field(default="")

    # Link to report for convenience (optional). One report per run_date.
    report_id: Optional[int] = Field(default=None, foreign_key="dailyaisummaryreport.id", index=True)

    extra: dict = Field(default_factory=dict, sa_column=Column(JSON))


class ImportRun(SQLModel, table=True):
    """每日导入记录：用于“哪些天抓取/导入过”看板."""

    id: Optional[int] = Field(default=None, primary_key=True)
    platform: str = Field(default="unknown", index=True)
    run_date: str = Field(index=True)  # YYYY-MM-DD
    source: str = Field(default="bucket", index=True)
    status: str = Field(default="done", index=True)  # done/error
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    details: dict = Field(default_factory=dict, sa_column=Column(JSON))


class TaskStatus(str, Enum):
    open = "open"
    learning = "learning"         # 已创建，进入“自我学习/自我思考”阶段
    practicing = "practicing"     # 已进入模拟训练
    submitted = "submitted"       # 已提交改写/思考
    reviewed = "reviewed"         # 主管已点评
    closed = "closed"             # 结束


class TrainingTask(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    conversation_id: int = Field(foreign_key="conversation.id", index=True)
    created_by_user_id: int = Field(foreign_key="user.id", index=True)
    assigned_to_user_id: int = Field(foreign_key="user.id", index=True)

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    status: TaskStatus = Field(default=TaskStatus.open, index=True)

    # 主管/管理员留言（为何要做这个任务、关注点）
    notes: str = Field(default="")

    # 本任务聚焦的负面标签（可空；默认自动从最近一次诊断里取一个）
    focus_negative_tag: str = Field(default="", index=True)

    # AI自动出的“自我思考题目”缓存（避免每次刷新都重新请求）
    reflection_question: str = Field(default="")

    conversation: "Conversation" = Relationship(back_populates="tasks")
    attempts: list["TrainingAttempt"] = Relationship(back_populates="task")
    reflections: list["TrainingReflection"] = Relationship(back_populates="task")
    simulations: list["TrainingSimulation"] = Relationship(back_populates="task")


class TrainingAttempt(SQLModel, table=True):
    """客服提交的“改写练习”（也可用来收集主管点评）。"""

    id: Optional[int] = Field(default=None, primary_key=True)
    task_id: int = Field(foreign_key="trainingtask.id", index=True)
    attempt_by_user_id: int = Field(foreign_key="user.id", index=True)

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    attempt_text: str = Field(default="")

    # supervisor/admin review
    reviewed_at: Optional[datetime] = Field(default=None, index=True)
    reviewed_by_user_id: Optional[int] = Field(default=None, foreign_key="user.id", index=True)
    review_score: Optional[int] = Field(default=None)
    review_notes: str = Field(default="")

    task: "TrainingTask" = Relationship(back_populates="attempts")


class TrainingReflection(SQLModel, table=True):
    """自我思考：AI出题 -> 客服作答 -> AI审核是否过关。"""

    id: Optional[int] = Field(default=None, primary_key=True)
    task_id: int = Field(foreign_key="trainingtask.id", index=True)
    user_id: int = Field(foreign_key="user.id", index=True)

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)

    question: str = Field(default="")
    answer: str = Field(default="")

    ai_passed: Optional[bool] = Field(default=None, index=True)
    ai_feedback: str = Field(default="")
    ai_score: Optional[int] = Field(default=None)

    extra: dict = Field(default_factory=dict, sa_column=Column(JSON))

    task: "TrainingTask" = Relationship(back_populates="reflections")


class TrainingSimulation(SQLModel, table=True):
    """模拟训练：针对某个负面标签，和AI做对话式训练。"""

    id: Optional[int] = Field(default=None, primary_key=True)
    task_id: int = Field(foreign_key="trainingtask.id", index=True)
    user_id: int = Field(foreign_key="user.id", index=True)

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    focus_tag: str = Field(default="", index=True)

    extra: dict = Field(default_factory=dict, sa_column=Column(JSON))

    task: "TrainingTask" = Relationship(back_populates="simulations")
    messages: list["TrainingSimulationMessage"] = Relationship(back_populates="simulation")


class TrainingSimulationMessage(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    simulation_id: int = Field(foreign_key="trainingsimulation.id", index=True)
    role: str = Field(default="user", index=True)  # user/assistant/system
    content: str = Field(default="")
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)

    simulation: "TrainingSimulation" = Relationship(back_populates="messages")


class AgentBinding(SQLModel, table=True):
    """主管/管理员把聊天记录中的“客服账号(agent_account)”绑定到站内客服用户。"""

    id: Optional[int] = Field(default=None, primary_key=True)
    platform: str = Field(default="unknown", index=True)
    agent_account: str = Field(index=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    created_by_user_id: int = Field(foreign_key="user.id", index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)

    meta: dict = Field(default_factory=dict, sa_column=Column(JSON))


class AppConfig(SQLModel, table=True):
    """Runtime-tunable settings controlled from the admin UI.

    All .env-style configs (except DB / APP_SECRET_KEY / DEFAULT_ADMIN_*) can be
    configured here and optionally exported/imported via data/settings.json.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    auto_analysis_enabled: bool = Field(default=False, index=True)

    # Bucket fetch/import (daily polling + manual trigger)
    bucket_fetch_enabled: bool = Field(default=True, index=True)
    taobao_bucket_import_enabled: bool = Field(default=True, index=True)
    douyin_bucket_import_enabled: bool = Field(default=False, index=True)
    bucket_daily_check_time: str = Field(default="10:15")  # HH:MM (Shanghai)
    bucket_retry_interval_minutes: int = Field(default=60)  # hourly retry when missing
    bucket_log_keep: int = Field(default=800)
    bucket_backup_dir: str = Field(default="", sa_column=Column(Text))  # e.g. /data/bucket_backup

    # Taobao/Douyin S3-compatible: JSON {bucket, prefix, endpoint, region, access_key, secret_key}
    taobao_bucket_config: dict = Field(default_factory=dict, sa_column=Column(JSON))
    douyin_bucket_config: dict = Field(default_factory=dict, sa_column=Column(JSON))

    # Notify
    feishu_webhook_url: str = Field(default="", sa_column=Column(Text))

    # Daily AI summary (admin report)
    daily_summary_threshold: int = Field(default=8)
    daily_summary_model: str = Field(default="gpt-5.2")
    daily_summary_prompt: str = Field(default="", sa_column=Column(Text))

    # QC system prompt (editable part; system appends fixed block)
    qc_system_prompt: str = Field(default="", sa_column=Column(Text))

    # AI (OpenAI-compatible relay)
    openai_base_url: str = Field(default="", sa_column=Column(Text))
    openai_api_key: str = Field(default="", sa_column=Column(Text))
    openai_model: str = Field(default="gpt-5.2")
    openai_fallback_model: str = Field(default="")
    openai_timeout: int = Field(default=1800)  # seconds
    openai_retries: int = Field(default=0)
    openai_reasoning_effort: str = Field(default="high")

    # Application
    open_registration: bool = Field(default=False, index=True)  # allow self-register
    
    # Cron & Worker
    cron_interval_seconds: int = Field(default=600)  # main loop sleep (cron)
    worker_poll_seconds: int = Field(default=5)  # analysis worker poll interval
    enable_enqueue_analysis: bool = Field(default=True, index=True)  # auto-enqueue missing analyses

    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)


class BucketObject(SQLModel, table=True):
    """Tracks bucket objects we've seen/downloaded/imported.

    We keep one row per (key, etag) so updated files are re-processed.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    bucket: str = Field(default="", index=True)
    key: str = Field(default="", index=True)
    etag: str = Field(default="", index=True)
    size: int = Field(default=0)
    last_modified: Optional[datetime] = Field(default=None, index=True)

    status: str = Field(default="new", index=True)  # new|downloaded|imported|error

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    downloaded_at: Optional[datetime] = Field(default=None, index=True)
    imported_at: Optional[datetime] = Field(default=None, index=True)

    backup_path: str = Field(default="")
    import_result: dict = Field(default_factory=dict, sa_column=Column(JSON))
    error: str = Field(default="")


class CronState(SQLModel, table=True):
    """Persisted cron state (so retries survive container restarts)."""

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    state: dict = Field(default_factory=dict, sa_column=Column(JSON))
    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)


class DailyAISummaryReport(SQLModel, table=True):
    """A daily management report generated by the LLM.

    run_date: YYYY-MM-DD (business day chosen by admin).
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    run_date: str = Field(index=True, unique=True)

    threshold_messages: int = Field(default=8)
    model: str = Field(default="")
    prompt: str = Field(default="", sa_column=Column(Text))

    input_chars: int = Field(default=0)
    included_conversations: int = Field(default=0)
    included_messages: int = Field(default=0)

    report_text: str = Field(default="", sa_column=Column(Text))
    meta: dict = Field(default_factory=dict, sa_column=Column(JSON))

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    created_by_user_id: Optional[int] = Field(default=None, foreign_key="user.id", index=True)
class DailyAISummaryShare(SQLModel, table=True):
    """Public share link for a DailyAISummaryReport.

    token: random string used in the public URL.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    report_id: int = Field(foreign_key="dailyaisummaryreport.id", index=True)

    token: str = Field(index=True, unique=True)

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    created_by_user_id: Optional[int] = Field(default=None, foreign_key="user.id", index=True)

    # Allow soft revoke in the future without deleting rows
    revoked_at: Optional[datetime] = Field(default=None, index=True)


# =========================
# Tag System (Level-1 category + Level-2 tag)
# =========================


class TagCategory(SQLModel, table=True):
    """Level-1 category for management tags.

    Example: 客服接待 / 服务流程 / 产品质量投诉
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    description: str = Field(default="")

    sort_order: int = Field(default=0, index=True)
    is_active: bool = Field(default=True, index=True)

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)

    tags: list["TagDefinition"] = Relationship(back_populates="category")


class TagDefinition(SQLModel, table=True):
    """Level-2 tag definition.

    - standard: the judging rubric fed to AI (how to decide hit / not hit)
    - description: human-friendly explanation shown in UI
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    category_id: int = Field(foreign_key="tagcategory.id", index=True)

    name: str = Field(index=True)
    standard: str = Field(default="", sa_column=Column(Text))
    description: str = Field(default="", sa_column=Column(Text))

    sort_order: int = Field(default=0, index=True)
    is_active: bool = Field(default=True, index=True)

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)

    category: Optional[TagCategory] = Relationship(back_populates="tags")
    hits: list["ConversationTagHit"] = Relationship(back_populates="tag")


class ConversationTagHit(SQLModel, table=True):
    """AI tagging result per ConversationAnalysis.

    One row per (analysis, tag) when hit.
    evidence may include start_index, end_index for citation range.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    analysis_id: int = Field(foreign_key="conversationanalysis.id", index=True)
    tag_id: int = Field(foreign_key="tagdefinition.id", index=True)

    reason: str = Field(default="", sa_column=Column(Text))
    evidence: dict = Field(default_factory=dict, sa_column=Column(JSON))

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)

    tag: Optional[TagDefinition] = Relationship(back_populates="hits")


class TagSuggestion(SQLModel, table=True):
    """New-tag suggestion from AI; manager reviews approve/reject."""

    id: Optional[int] = Field(default=None, primary_key=True)
    analysis_id: int = Field(foreign_key="conversationanalysis.id", index=True)
    conversation_id: int = Field(foreign_key="conversation.id", index=True)

    suggested_category: str = Field(default="")
    suggested_tag_name: str = Field(default="")
    suggested_standard: str = Field(default="", sa_column=Column(Text))
    suggested_description: str = Field(default="", sa_column=Column(Text))
    ai_reason: str = Field(default="", sa_column=Column(Text))

    status: str = Field(default="pending", index=True)  # pending/approved/rejected
    reviewed_by_user_id: Optional[int] = Field(default=None, foreign_key="user.id", index=True)
    reviewed_at: Optional[datetime] = Field(default=None, index=True)
    review_notes: str = Field(default="")

    created_tag_id: Optional[int] = Field(default=None, foreign_key="tagdefinition.id", index=True)

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)


class RejectedTagRule(SQLModel, table=True):
    """Admin-maintained rejected tag rules for blocking similar suggestions."""

    id: Optional[int] = Field(default=None, primary_key=True)
    category: str = Field(default="", index=True)
    tag_name: str = Field(default="", index=True)
    aliases: str = Field(default="", sa_column=Column(Text))
    notes: str = Field(default="", sa_column=Column(Text))
    norm_key: str = Field(default="", index=True, unique=True)
    is_active: bool = Field(default=True, index=True)

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    created_by_user_id: Optional[int] = Field(default=None, foreign_key="user.id", index=True)
    updated_by_user_id: Optional[int] = Field(default=None, foreign_key="user.id", index=True)


class ManualTagBinding(SQLModel, table=True):
    """Admin/supervisor manually binds a tag to a conversation."""

    id: Optional[int] = Field(default=None, primary_key=True)
    conversation_id: int = Field(foreign_key="conversation.id", index=True)
    tag_id: int = Field(foreign_key="tagdefinition.id", index=True)
    created_by_user_id: int = Field(foreign_key="user.id", index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    reason: str = Field(default="")
    evidence: dict = Field(default_factory=dict, sa_column=Column(JSON))
