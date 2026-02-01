from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlmodel import Session

DEFAULT_QC_EDITABLE_PROMPT = """你是资深的女装电商客服质检与培训教练。
目标：提升接待与产品质量，让用户满意、产生信任与长期情感链接。
你要基于对话内容做客观诊断。"""


def get_editable_qc_prompt(session: "Session") -> str:
    """从数据库获取可编辑的质检 prompt 部分。"""
    from app_config import get_app_config

    cfg = get_app_config(session)
    raw = (cfg.qc_system_prompt or "").strip()
    return raw if raw else DEFAULT_QC_EDITABLE_PROMPT


def analysis_system_prompt_fixed_part(session: "Session | None" = None) -> str:
    """系统固定、不可编辑部分（含驳回标签列表）。"""
    # 获取驳回标签列表
    rejected_suggestions_block = ""
    if session is not None:
        rejected_suggestions_block = build_rejected_suggestions_block(session)
    
    base_prompt = """
输出要求（系统固定，不可编辑）：
1. 输出严格JSON，不要出现任何额外文本。
2. 字段缺失时用空字符串/空数组/false。
3. 不要编造对话里没有的信息；引用对话内容时要尽量短摘录并标明 message_index。
4. 重要：如果提供了历史对话上下文，请将其作为背景参考，但只对「当日对话」进行分析和标签判定。历史对话仅用于理解用户背景和上下文，不要对历史对话中的问题进行标签命中。

必须输出以下三大块：

【第一大块：接待场景】
reception_scenario: 从以下选项中选择最符合的一项（单选）
- "售前"
- "售后"
- "售前和售后"
- "其他接待场景"
- "无法判定"

【第二大块：满意度变化】
satisfaction_change: 评估用户从对话开始到结束的内心满意度变化（单选）
- "大幅减少"
- "小幅减少"
- "无显著变化"
- "小幅增加"
- "大幅增加"
- "无法判定"

【第三大块：标签命中】
tag_hits: 数组，每组包含 tag_id、reason、evidence。
evidence 每项: message_index, start_index, end_index, quote。
start_index/end_index 为引用对话的起始和结束消息索引（含）。
"""
    
    # 添加驳回标签提醒
    new_tag_block = """
【新标签建议】（可选）
若发现对话中有让客户不满意的问题，但现有标签库无法覆盖，输出：
new_tag_suggestions: [{
  "category": "建议的一级分类",
  "tag_name": "建议的二级标签名",
  "standard": "判定标准",
  "description": "说明",
  "reason": "为什么建议新增此标签"
}]
"""
    
    if rejected_suggestions_block:
        new_tag_block += f"""
{rejected_suggestions_block}
"""
    
    compatibility_block = """
兼容保留（已废弃字段可填空字符串）：
- 已废弃（填空字符串即可）：pre_positive_tags、after_positive_tags、pre_negative_tags、after_negative_tags、tag_parsing、product_suggestion、service_suggestion、pre_rule_update、after_rule_update
- 保留字段：dialog_type、day_summary、tag_update_suggestion、customer_issue_highlights、must_read_highlights、overall_score、sentiment、issue_level、problem_types、flag_for_review
"""
    
    return base_prompt + new_tag_block + compatibility_block


def analysis_system_prompt(session: "Session | None" = None) -> str:
    """组合：可编辑部分 + 系统固定部分。"""
    if session is not None:
        editable = get_editable_qc_prompt(session)
    else:
        editable = DEFAULT_QC_EDITABLE_PROMPT
    return editable + analysis_system_prompt_fixed_part(session)


def analysis_user_prompt(
    conversation_meta: dict[str, Any],
    messages: list[dict[str, Any]],
    *,
    tag_catalog: str = "",
    previous_conversation: dict[str, Any] | None = None,
) -> str:
    """Return the user prompt for analysis.

    messages item: {"sender":"buyer|agent|system","text":"...","attachments":[...]}
    previous_conversation: {
        "meta": {"started_at": "...", "ended_at": "..."},
        "messages": [{"sender": "...", "text": "...", "agent_account": "..."}]
    }
    """

    lines: list[str] = []
    for i, m in enumerate(messages):
        sender = m.get("sender") or "unknown"
        agent_account = (m.get("agent_account") or "").strip()
        agent_nick = (m.get("agent_nick") or "").strip()
        if sender == "agent":
            if agent_nick:
                sender = f"agent({agent_nick})"
            elif agent_account:
                sender = f"agent({agent_account})"
            else:
                sender = "agent"
        text = (m.get("text") or "").strip()
        if not text and m.get("attachments"):
            text = "[含附件]"
        text = text.replace("\n", " ")
        if len(text) > 300:
            text = text[:300] + "…"
        lines.append(f"#{i} {sender}: {text}")

    meta_str = "\n".join(
        [
            f"platform: {conversation_meta.get('platform','')} ",
            f"agent_account: {conversation_meta.get('agent_account','')} ",
            f"buyer_id: {conversation_meta.get('buyer_id','')} ",
            f"started_at: {conversation_meta.get('started_at','')} ",
            f"ended_at: {conversation_meta.get('ended_at','')} ",
        ]
    )

    # Build historical context if available
    history_block = ""
    if previous_conversation:
        prev_meta = previous_conversation.get("meta", {})
        prev_msgs = previous_conversation.get("messages", [])
        if prev_msgs:
            prev_lines: list[str] = []
            for i, m in enumerate(prev_msgs):
                sender = m.get("sender") or "unknown"
                agent_account = (m.get("agent_account") or "").strip()
                if sender == "agent" and agent_account:
                    sender = f"agent({agent_account})"
                text = (m.get("text") or "").strip().replace("\n", " ")
                if len(text) > 200:
                    text = text[:200] + "…"
                prev_lines.append(f"#{i} {sender}: {text}")
            
            prev_date = prev_meta.get("started_at", "")
            history_block = (
                "\n\n=== 历史对话上下文（仅供参考，不要对此进行分析）===\n"
                f"对话日期: {prev_date}\n"
                "【重要】此历史对话仅用于理解用户背景，不要对其进行标签判定！\n\n"
                + "\n".join(prev_lines)
                + "\n"
            )

    schema = {
        # === 第一大块：接待场景 ===
        "reception_scenario": "售前|售后|售前和售后|其他接待场景|无法判定",
        # === 第二大块：满意度变化 ===
        "satisfaction_change": "大幅减少|小幅减少|无显著变化|小幅增加|大幅增加|无法判定",
        # === 第三大块：标签命中（含引用起止）===
        "tag_hits": [
            {
                "tag_id": 0,
                "reason": "简述原因",
                "evidence": [
                    {"message_index": 0, "start_index": 0, "end_index": 0, "quote": "..."}
                ],
            }
        ],
        # === 新标签建议（可选）===
        "new_tag_suggestions": [
            {
                "category": "一级分类",
                "tag_name": "二级标签名",
                "standard": "判定标准",
                "description": "说明",
                "reason": "建议原因",
            }
        ],
        # === 兼容：与 AI输出.xlsx 对齐 ===
        "dialog_type": "售前|售后|售前售后|其他",
        # 已废弃字段（填空字符串）
        "pre_positive_tags": "",
        "after_positive_tags": "",
        "pre_negative_tags": "",
        "after_negative_tags": "",
        "tag_parsing": "",
        "product_suggestion": "",
        "service_suggestion": "",
        "pre_rule_update": "",
        "after_rule_update": "",
        # 保留字段
        "day_summary": "...",
        "tag_update_suggestion": "标签A;标签B",
        "customer_issue_highlights": [
            {"message_index": 0, "sender": "buyer|agent", "quote": "...", "tag": "...", "why": "..."}
        ],
        "must_read_highlights": [
            {"message_index": 0, "sender": "buyer|agent", "quote": "...", "tag": "...", "why": "..."}
        ],
        "overall_score": 0,
        "sentiment": "positive|neutral|negative",
        "issue_level": "low|mid|high",
        "problem_types": ["..."],
        "flag_for_review": False,
    }

    tag_block = ""
    if (tag_catalog or "").strip():
        tag_block = "\n\n=== 当前标签库（只能从这里选 tag_id；不要自造标签）===\n" + (tag_catalog.strip())

    return (
        "请根据以下「当日对话」做质检分析，并输出严格JSON。\n\n"
        + history_block
        + "\n=== 当日对话元信息 ===\n"
        f"{meta_str}\n\n"
        "=== 当日对话逐条记录（带索引）===\n"
        "【重要】请仅对以下当日对话进行分析和标签判定！\n"
        + "\n".join(lines)
        + "\n\n"
        + tag_block
        + "\n\n"
        "=== 输出JSON结构（必须严格遵守）===\n"
        + str(schema)
    )


def build_rejected_suggestions_block(session: "Session") -> str:
    """从 TagSuggestion 表读取已驳回的标签建议，构建提醒块，指导AI勿再建议类似标签。"""
    from models import TagSuggestion
    from sqlmodel import select

    rejected = session.exec(
        select(TagSuggestion)
        .where(TagSuggestion.status == "rejected")
        .order_by(TagSuggestion.reviewed_at.desc())
        .limit(50)
    ).all()

    if not rejected:
        return ""

    lines = ["【重要】以下标签建议已被管理员驳回，请勿再次建议类似标签："]
    for r in rejected:
        cat = r.suggested_category or "未分类"
        name = r.suggested_tag_name or "未命名"
        reason = (r.review_notes or "").strip()
        if reason:
            reason_short = reason.replace("\n", " ")
            if len(reason_short) > 80:
                reason_short = reason_short[:80] + "…"
            lines.append(f"- [{cat}] {name} - 驳回原因：{reason_short}")
        else:
            lines.append(f"- [{cat}] {name}")

    return "\n".join(lines)


def build_tag_catalog_for_prompt(categories: list[dict[str, Any]]) -> str:
    """Build a compact tag catalog string for the LLM prompt.

    categories: [{name, tags:[{id,name,standard,description}]}]
    """

    lines: list[str] = []
    for c in categories:
        cname = (c.get("name") or "").strip()
        if not cname:
            continue
        lines.append(f"[{cname}]")
        tags = c.get("tags") or []
        for t in tags:
            tid = t.get("id")
            tname = (t.get("name") or "").strip()
            std = (t.get("standard") or "").strip()
            desc = (t.get("description") or "").strip()
            if not tid or not tname:
                continue
            # Keep it short to avoid prompt bloat.
            std_short = std.replace("\n", " ").strip()
            if len(std_short) > 160:
                std_short = std_short[:160] + "…"
            desc_short = desc.replace("\n", " ").strip()
            if len(desc_short) > 80:
                desc_short = desc_short[:80] + "…"
            hint = ""
            if std_short:
                hint += f" 判定：{std_short}"
            if desc_short and not std_short:
                hint += f" 说明：{desc_short}"
            lines.append(f"- {int(tid)} | {tname}{hint}")
        lines.append("")
    return "\n".join([l for l in lines if l is not None])


def reflection_question_system_prompt() -> str:
    return (
        "你是客服主管的培训教练，擅长用一个问题逼迫对方把逻辑说清楚。\n"
        "只输出一句‘自我思考题’，不要解释。"
    )


def reflection_question_user_prompt(focus_tag: str, tag_reason: str, manager_note: str) -> str:
    return (
        f"负面标签：{focus_tag}\n"
        f"原因解析：{tag_reason}\n"
        f"管理者留言：{manager_note or '无'}\n\n"
        "请给客服出一道自我思考题，让他写出一段完整、可执行、以用户体验为中心的回复策略。"
    )


def reflection_grade_system_prompt() -> str:
    return (
        "你是严格的客服培训考官。\n"
        "只输出严格JSON：{passed:boolean, score:int(0-100), feedback:string}。\n"
        "不要出现任何额外文本。"
    )


def reflection_grade_user_prompt(focus_tag: str, question: str, answer: str) -> str:
    return (
        f"训练聚焦标签：{focus_tag}\n"
        f"题目：{question}\n"
        f"客服回答：{answer}\n\n"
        "按‘是否能显著降低该负面标签再次发生’来评分与给反馈。"
    )


def simulation_system_prompt() -> str:
    return (
        "你扮演对话中的顾客（buyer），风格真实自然。\n"
        "每次只回复一条顾客消息。\n"
        "并在末尾追加一段【教练点评】（告诉客服哪里做得好/不好，给一句可复制话术）。"
    )
