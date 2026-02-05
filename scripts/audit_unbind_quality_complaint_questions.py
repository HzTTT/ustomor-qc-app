"""
批量数据修复（按“引用范围 evidence”逐条检查）：
对数据库里一级分类=“产品质量投诉”的所有二级标签命中记录进行审计：

规则（按用户诉求）：
- 如果用户只是询问“是否/会不会/有没有”等质量问题（预测性/售前/未收货咨询），则把该标签与该对话解绑（删除命中记录/手动绑定）。
- 如果用户确实在收货后反馈了实际问题（含试穿/使用/洗后产生的问题），则保留该标签。

实现要点：
- AI 命中：ConversationTagHit（按 analysis_id 绑定到 ConversationAnalysis）
- 手动绑定：ManualTagBinding（对话级）
- 依据每条命中自带 evidence 的 start_index/end_index（引用范围）提取对话片段，优先只看该范围内的买家话术；
  evidence 缺失时回退到整段对话买家内容做判断，并在审计日志里标记。
- 为可回滚：输出备份 CSV + 全量 JSONL 审计日志；dry-run 默认不提交，--apply 才会写库。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import delete
from sqlmodel import Session, select

# 添加 app 目录到路径（兼容：在宿主机 repo 根目录运行 / 在容器 /app/scripts 运行）
root_dir = Path(__file__).resolve().parent.parent
app_dir = root_dir / "app"
if app_dir.exists():
    sys.path.insert(0, str(app_dir))
else:
    sys.path.insert(0, str(root_dir))

from db import engine  # noqa: E402
from models import (  # noqa: E402
    ConversationAnalysis,
    ConversationTagHit,
    ManualTagBinding,
    Message,
    TagCategory,
    TagDefinition,
)


CATEGORY_NAME = "产品质量投诉"

# 强信号：明确是“实际发生的问题”
_COMPLAINT_STRONG = [
    "质量问题",
    "瑕疵",
    "破损",
    "破了",
    "坏了",
    "开线",
    "脱线",
    "线头很多",
    "起球了",
    "掉色了",
    "褪色了",
    "缩水了",
    "变形了",
    "勾丝",
    "拉丝",
    "发霉",
    "异味",
    "污渍",
    "染色了",
    "沾色了",
    "退货",
    "退款",
    "换货",
    "补发",
    "赔偿",
]

# 强信号：典型“只是询问/担忧/听说”
_QUESTION_STRONG = [
    "会不会",
    "是否",
    "有没有",
    "会起球吗",
    "起球吗",
    "会掉色吗",
    "掉色吗",
    "褪色吗",
    "缩水吗",
    "变形吗",
    "评论说",
    "有人说",
    "听说",
    "担心",
    "怕",
]

# “收到货”相关词不等于投诉，但可用于辅助判断
_DELIVERY_HINTS = [
    "收到",
    "收货",
    "到货",
    "签收",
    "取件",
    "刚到",
    "拿到",
    "拆开",
    "打开",
    "试穿",
    "穿了",
    "洗了",
    "下水",
    "洗过",
    "用了",
    "使用",
]

_QUESTION_PUNCT_RE = re.compile(r"[?？]")
_TRAILING_MA_RE = re.compile(r"(吗|嘛)\s*$")
_WHY_Q_RE = re.compile(r"(为啥|为什么|怎么|咋|啥原因|什么原因|怎么办|怎么处理)")


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _norm_text(s: str) -> str:
    return (s or "").strip()


def _contains_any(text: str, needles: Iterable[str]) -> bool:
    t = text or ""
    for n in needles:
        if n and n in t:
            return True
    return False


def _split_issue_terms(tag_name: str) -> list[str]:
    name = (tag_name or "").strip()
    if not name:
        return []
    # 常见分隔：/、｜|、空格、顿号、逗号
    parts = re.split(r"[/|｜、,，\s]+", name)
    terms = []
    for p in parts:
        p = p.strip()
        if p and p not in terms:
            terms.append(p)
    return terms


def _is_question_like(text: str) -> bool:
    t = _norm_text(text)
    if not t:
        return False
    if _QUESTION_PUNCT_RE.search(t):
        return True
    if _TRAILING_MA_RE.search(t):
        return True
    if _contains_any(t, _QUESTION_STRONG):
        return True
    # 口语省略："起球不" / "掉色不" / "会起球不"
    if t.endswith("不") and _contains_any(t, ["起球", "掉色", "褪色", "缩水", "变形", "开线", "破损", "色差"]):
        return True
    return False


def _is_yesno_inquiry_like(text: str) -> bool:
    """是否/会不会/有没有/吗/不 等“询问是否存在问题”的语气（排除‘为啥/怎么/怎么办’类投诉追问）。"""
    t = _norm_text(text)
    if not t:
        return False
    if _WHY_Q_RE.search(t):
        return False
    if _contains_any(t, ["会不会", "是否", "有没有"]):
        return True
    if _TRAILING_MA_RE.search(t):
        return True
    if _QUESTION_PUNCT_RE.search(t) and _contains_any(t, ["会", "容易", "起球", "掉色", "褪色", "缩水", "变形", "色差", "开线", "破损", "刺", "痒", "扎", "闷"]):
        return True
    if t.endswith("不") and _contains_any(t, ["起球", "掉色", "褪色", "缩水", "变形", "开线", "破损", "色差"]):
        return True
    return False


def _is_actual_issue_like(text: str, issue_terms: list[str]) -> bool:
    t = _norm_text(text)
    if not t:
        return False
    # 退换等强信号，基本可视为实际问题已发生
    if _contains_any(t, _COMPLAINT_STRONG):
        return True

    # “收到/穿了/洗了/用了/才穿” 等体验语境 + 质量词：更偏投诉（即使带问号也可能是反问/追问）
    if _contains_any(t, ["才穿", "穿了", "试穿", "洗了", "下水", "洗过", "用了", "使用", "打开", "拆开", "收到", "收货", "到货", "签收", "取件"]):
        if issue_terms:
            for term in issue_terms:
                if term and term in t:
                    return True
        if _contains_any(t, ["色差", "不一样", "不符", "跟图不一样", "和图片不一样", "与图片不符", "和页面不一样", "模特图", "展示图", "描述不符", "粉色"]):
            return True

    # “发现/出现/就是/有点/已经/现在/结果/就/明显/很/太/严重/有” + 质量词：偏投诉
    if issue_terms:
        for term in issue_terms:
            if not term:
                continue
            if term in t:
                # “起球了/掉色了/缩水了”等
                if f"{term}了" in t:
                    return True
                if re.search(r"(发现|出现|存在|就是|确实|明显|很|特别|严重|有点|已经|现在|结果|就|太|有)\s*.*" + re.escape(term), t):
                    return True
                # “有点扎扎的/有点痒/有点闷” 常见表达（term 可能是“扎/痒/闷”）
                if re.search(r"有点.*" + re.escape(term), t):
                    return True
    return False


@dataclass
class EvidenceScope:
    indices: list[int]
    used_fallback_full_conversation: bool
    evidence_missing: bool
    evidence_out_of_range: bool


def _extract_evidence_indices(evidence: Any, msg_count: int) -> EvidenceScope:
    ev_list: list[dict[str, Any]] = []
    if isinstance(evidence, list):
        ev_list = [e for e in evidence if isinstance(e, dict)]
    elif isinstance(evidence, dict):
        raw = evidence.get("evidence")
        if isinstance(raw, list):
            ev_list = [e for e in raw if isinstance(e, dict)]

    if not ev_list:
        return EvidenceScope(indices=list(range(msg_count)), used_fallback_full_conversation=True, evidence_missing=True, evidence_out_of_range=False)

    idx_set: set[int] = set()
    out_of_range = False
    for e in ev_list:
        start_loc = e.get("start_index") if e.get("start_index") is not None else e.get("message_index")
        end_loc = e.get("end_index") if e.get("end_index") is not None else e.get("message_index")
        if start_loc is None and end_loc is None:
            continue
        try:
            si = int(start_loc) if start_loc is not None else 0
            ei = int(end_loc) if end_loc is not None else si
        except Exception:
            continue
        if si > ei:
            si, ei = ei, si
        if si < 0 or ei >= msg_count:
            out_of_range = True
        si = max(0, si)
        ei = min(msg_count - 1, ei) if msg_count > 0 else -1
        for i in range(si, ei + 1):
            if 0 <= i < msg_count:
                idx_set.add(i)

    if not idx_set:
        return EvidenceScope(indices=list(range(msg_count)), used_fallback_full_conversation=True, evidence_missing=False, evidence_out_of_range=out_of_range)

    indices = sorted(idx_set)
    return EvidenceScope(indices=indices, used_fallback_full_conversation=False, evidence_missing=False, evidence_out_of_range=out_of_range)


def _buyer_texts_in_scope(messages: list[Message], indices: list[int]) -> list[str]:
    res: list[str] = []
    for i in indices:
        if i < 0 or i >= len(messages):
            continue
        m = messages[i]
        if (m.sender or "").strip() != "buyer":
            continue
        t = (m.text or "").strip()
        if t:
            res.append(t)
    return res


def _classify_should_unbind(buyer_texts: list[str], *, tag_name: str) -> tuple[bool, dict[str, Any]]:
    """
    返回：
    - should_unbind: True 表示“只是询问/未发生实际问题” -> 解绑
    - meta: 用于审计落盘的信号与说明
    """
    issue_terms = _split_issue_terms(tag_name) or []
    combined = "\n".join([_norm_text(x) for x in buyer_texts if _norm_text(x)])

    has_issue_term = _contains_any(combined, issue_terms) if issue_terms else _contains_any(combined, ["起球", "掉色", "褪色", "色差", "破损", "开线", "缩水", "变形", "异味", "发霉", "污渍", "瑕疵"])
    inquiry_like = any(_is_yesno_inquiry_like(t) for t in buyer_texts)
    actual_issue_like = any(_is_actual_issue_like(t, issue_terms) for t in buyer_texts)

    delivery_hint = _contains_any(combined, _DELIVERY_HINTS)

    # 决策：保守策略 —— 只在“明确是询问且未出现实际问题”的情况下解绑
    should_unbind = bool(has_issue_term and inquiry_like and (not actual_issue_like))

    why = []
    if should_unbind:
        why.append("question_like_without_actual_issue")
    if actual_issue_like:
        why.append("actual_issue_like")
    if delivery_hint:
        why.append("delivery_hint_present")
    if not has_issue_term:
        why.append("no_issue_term_detected")

    meta = {
        "tag_name": tag_name,
        "issue_terms": issue_terms,
        "has_issue_term": has_issue_term,
        "inquiry_like": inquiry_like,
        "actual_issue_like": actual_issue_like,
        "delivery_hint": delivery_hint,
        "why_flags": why,
        "buyer_snippet": (combined[:220] + ("…" if len(combined) > 220 else "")) if combined else "",
    }
    return should_unbind, meta


def _chunked(xs: list[int], size: int) -> Iterable[list[int]]:
    for i in range(0, len(xs), size):
        yield xs[i : i + size]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="实际执行解绑并提交（默认 dry-run）")
    parser.add_argument("--out-dir", default="scripts/tmp", help="备份/审计输出目录（默认 scripts/tmp/）")
    parser.add_argument("--limit", type=int, default=0, help="限制处理条数（0=不限制）")
    parser.add_argument("--only", default="", help="仅处理 ai 或 manual（可选：ai/manual）")
    args = parser.parse_args()

    out_dir = str(args.out_dir or "scripts/tmp")
    os.makedirs(out_dir, exist_ok=True)

    audit_path = os.path.join(out_dir, f"audit_unbind_{CATEGORY_NAME}_{_ts()}.jsonl")
    backup_hits_path = os.path.join(out_dir, f"backup_unbind_{CATEGORY_NAME}_hits_{_ts()}.csv")
    backup_manual_path = os.path.join(out_dir, f"backup_unbind_{CATEGORY_NAME}_manual_{_ts()}.csv")

    to_delete_hit_ids: list[int] = []
    to_delete_manual_ids: list[int] = []

    processed = 0
    unbind_cnt = 0
    keep_cnt = 0

    conv_msg_cache: dict[int, list[Message]] = {}

    def _messages_for_conv(session: Session, conversation_id: int) -> list[Message]:
        cached = conv_msg_cache.get(int(conversation_id))
        if cached is not None:
            return cached
        msgs = session.exec(
            select(Message)
            .where(Message.conversation_id == int(conversation_id))
            .order_by(Message.ts.asc().nulls_last(), Message.id.asc())
        ).all()
        conv_msg_cache[int(conversation_id)] = msgs
        return msgs

    with Session(engine) as session:
        cat = session.exec(select(TagCategory).where(TagCategory.name == CATEGORY_NAME)).first()
        if not cat or cat.id is None:
            print(f"[ERR] 未找到一级分类：{CATEGORY_NAME}")
            return 2

        tag_ids = [int(tid) for tid in session.exec(select(TagDefinition.id).where(TagDefinition.category_id == int(cat.id))).all()]
        if not tag_ids:
            print(f"[OK] 一级分类“{CATEGORY_NAME}”下没有任何二级标签，无需处理。")
            return 0

        # 备份 CSV：只写入“将解绑”的行，便于回滚/复核
        hits_csv = open(backup_hits_path, "w", newline="", encoding="utf-8")
        manual_csv = open(backup_manual_path, "w", newline="", encoding="utf-8")
        hits_w = csv.writer(hits_csv)
        manual_w = csv.writer(manual_csv)
        hits_w.writerow(
            [
                "hit_id",
                "analysis_id",
                "conversation_id",
                "tag_id",
                "tag_name",
                "created_at",
                "reason",
                "evidence_json",
                "buyer_snippet",
                "why_flags",
            ]
        )
        manual_w.writerow(
            [
                "binding_id",
                "conversation_id",
                "tag_id",
                "tag_name",
                "created_at",
                "reason",
                "evidence_json",
                "buyer_snippet",
                "why_flags",
            ]
        )

        with open(audit_path, "w", encoding="utf-8") as audit_f:
            print(f"[INFO] 目标一级分类：{CATEGORY_NAME}（tag_ids={len(tag_ids)}）")
            print(f"[INFO] 审计日志：{audit_path}")

            # ========== AI 命中 ==========
            if args.only.strip() in ("", "ai"):
                rows = session.exec(
                    select(ConversationTagHit, ConversationAnalysis, TagDefinition)
                    .join(ConversationAnalysis, ConversationAnalysis.id == ConversationTagHit.analysis_id)
                    .join(TagDefinition, TagDefinition.id == ConversationTagHit.tag_id)
                    .where(ConversationTagHit.tag_id.in_(tag_ids))
                    .order_by(ConversationTagHit.id.asc())
                ).all()

                for hit, ana, tag in rows:
                    if args.limit and processed >= int(args.limit):
                        break
                    processed += 1

                    conv_id = int(ana.conversation_id or 0)
                    msgs = _messages_for_conv(session, conv_id) if conv_id else []
                    scope = _extract_evidence_indices(hit.evidence, len(msgs))
                    buyer_texts = _buyer_texts_in_scope(msgs, scope.indices) if msgs else []

                    should_unbind, meta = _classify_should_unbind(buyer_texts, tag_name=str(tag.name or ""))
                    meta.update(
                        {
                            "type": "ai_hit",
                            "hit_id": int(hit.id or 0),
                            "analysis_id": int(hit.analysis_id or 0),
                            "conversation_id": conv_id,
                            "reception_scenario": str(getattr(ana, "reception_scenario", "") or ""),
                            "created_at": str(getattr(hit, "created_at", "") or ""),
                            "evidence_scope": {
                                "indices": scope.indices,
                                "used_fallback_full_conversation": scope.used_fallback_full_conversation,
                                "evidence_missing": scope.evidence_missing,
                                "evidence_out_of_range": scope.evidence_out_of_range,
                            },
                            "reason": (hit.reason or "").strip(),
                        }
                    )

                    audit_f.write(json.dumps(meta, ensure_ascii=False) + "\n")

                    if should_unbind and hit.id is not None:
                        to_delete_hit_ids.append(int(hit.id))
                        unbind_cnt += 1
                        hits_w.writerow(
                            [
                                int(hit.id or 0),
                                int(hit.analysis_id or 0),
                                conv_id,
                                int(tag.id or 0),
                                str(tag.name or ""),
                                str(getattr(hit, "created_at", "") or ""),
                                (hit.reason or "").strip(),
                                json.dumps(hit.evidence, ensure_ascii=False),
                                meta.get("buyer_snippet", ""),
                                ",".join(meta.get("why_flags") or []),
                            ]
                        )
                    else:
                        keep_cnt += 1

            # ========== 手动绑定 ==========
            if args.only.strip() in ("", "manual"):
                rows = session.exec(
                    select(ManualTagBinding, TagDefinition)
                    .join(TagDefinition, TagDefinition.id == ManualTagBinding.tag_id)
                    .where(ManualTagBinding.tag_id.in_(tag_ids))
                    .order_by(ManualTagBinding.id.asc())
                ).all()

                for mb, tag in rows:
                    if args.limit and processed >= int(args.limit):
                        break
                    processed += 1

                    conv_id = int(mb.conversation_id or 0)
                    msgs = _messages_for_conv(session, conv_id) if conv_id else []
                    scope = _extract_evidence_indices(mb.evidence, len(msgs))
                    buyer_texts = _buyer_texts_in_scope(msgs, scope.indices) if msgs else []

                    should_unbind, meta = _classify_should_unbind(buyer_texts, tag_name=str(tag.name or ""))
                    meta.update(
                        {
                            "type": "manual_binding",
                            "binding_id": int(mb.id or 0),
                            "conversation_id": conv_id,
                            "created_at": str(getattr(mb, "created_at", "") or ""),
                            "evidence_scope": {
                                "indices": scope.indices,
                                "used_fallback_full_conversation": scope.used_fallback_full_conversation,
                                "evidence_missing": scope.evidence_missing,
                                "evidence_out_of_range": scope.evidence_out_of_range,
                            },
                            "reason": (mb.reason or "").strip(),
                        }
                    )
                    audit_f.write(json.dumps(meta, ensure_ascii=False) + "\n")

                    if should_unbind and mb.id is not None:
                        to_delete_manual_ids.append(int(mb.id))
                        unbind_cnt += 1
                        manual_w.writerow(
                            [
                                int(mb.id or 0),
                                conv_id,
                                int(tag.id or 0),
                                str(tag.name or ""),
                                str(getattr(mb, "created_at", "") or ""),
                                (mb.reason or "").strip(),
                                json.dumps(mb.evidence, ensure_ascii=False),
                                meta.get("buyer_snippet", ""),
                                ",".join(meta.get("why_flags") or []),
                            ]
                        )
                    else:
                        keep_cnt += 1

        hits_csv.close()
        manual_csv.close()

        print(f"[INFO] 扫描条数：{processed}；将解绑：{len(to_delete_hit_ids) + len(to_delete_manual_ids)}")
        print(f"[INFO] 备份(将解绑) AI 命中：{backup_hits_path}")
        print(f"[INFO] 备份(将解绑) 手动绑定：{backup_manual_path}")

        if not args.apply:
            print("[DRY-RUN] 未执行写库（加 --apply 才会提交）。")
            session.rollback()
            return 0

        # 批量删除（分批，避免 IN 过长）
        deleted_hits = 0
        for chunk in _chunked(to_delete_hit_ids, 1000):
            stmt = delete(ConversationTagHit).where(ConversationTagHit.id.in_(chunk))
            deleted_hits += int(session.exec(stmt).rowcount or 0)

        deleted_manual = 0
        for chunk in _chunked(to_delete_manual_ids, 1000):
            stmt = delete(ManualTagBinding).where(ManualTagBinding.id.in_(chunk))
            deleted_manual += int(session.exec(stmt).rowcount or 0)

        session.commit()
        print(f"[OK] 已解绑（删除）AI 命中：{deleted_hits} 行；手动绑定：{deleted_manual} 行。")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
