"""
批量数据修复：
删除“接待场景=售前”的对话中，一级分类为“产品质量投诉”的所有二级标签命中/手动绑定。

说明：
- AI 命中：ConversationTagHit（绑定在“最新一条 ConversationAnalysis”上）
- 手动绑定：ManualTagBinding（对话级）
- 仅处理“最新一条分析”的 reception_scenario == "售前" 的对话
- 为可回滚，默认会导出备份 CSV；dry-run 默认不提交
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy import func, delete
from sqlmodel import Session, select

# 添加 app 目录到路径（兼容：在宿主机 repo 根目录运行 / 在容器 /app/scripts 运行）
root_dir = Path(__file__).resolve().parent.parent
app_dir = root_dir / "app"
if app_dir.exists():
    sys.path.insert(0, str(app_dir))
else:
    sys.path.insert(0, str(root_dir))

from db import engine
from models import (
    ConversationAnalysis,
    ConversationTagHit,
    ManualTagBinding,
    TagCategory,
    TagDefinition,
)


CATEGORY_NAME = "产品质量投诉"
SCENARIO_NAME = "售前"


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="实际执行删除并提交（默认 dry-run）")
    parser.add_argument("--out-dir", default="tmp", help="备份 CSV 输出目录（默认 tmp/）")
    args = parser.parse_args()

    out_dir = str(args.out_dir or "tmp")
    os.makedirs(out_dir, exist_ok=True)

    backup_hits_path = os.path.join(out_dir, f"backup_presales_{CATEGORY_NAME}_hits_{_ts()}.csv")
    backup_manual_path = os.path.join(out_dir, f"backup_presales_{CATEGORY_NAME}_manual_{_ts()}.csv")

    with Session(engine) as session:
        cat = session.exec(select(TagCategory).where(TagCategory.name == CATEGORY_NAME)).first()
        if not cat or cat.id is None:
            print(f"[ERR] 未找到一级分类：{CATEGORY_NAME}")
            return 2

        tag_ids = [int(tid) for tid in session.exec(select(TagDefinition.id).where(TagDefinition.category_id == int(cat.id))).all()]
        if not tag_ids:
            print(f"[OK] 一级分类“{CATEGORY_NAME}”下没有任何二级标签，无需处理。")
            return 0

        # 最新分析（按对话取 max(id)）
        latest_subq = (
            select(
                ConversationAnalysis.conversation_id.label("cid"),
                func.max(ConversationAnalysis.id).label("analysis_id"),
            )
            .group_by(ConversationAnalysis.conversation_id)
        ).subquery()

        latest_presales_subq = (
            select(
                ConversationAnalysis.id.label("analysis_id"),
                ConversationAnalysis.conversation_id.label("cid"),
            )
            .select_from(ConversationAnalysis)
            .join(latest_subq, latest_subq.c.analysis_id == ConversationAnalysis.id)
            .where(ConversationAnalysis.reception_scenario == SCENARIO_NAME)
        ).subquery()

        # 备份：AI 命中
        hits_stmt = (
            select(
                ConversationTagHit.id.label("hit_id"),
                ConversationTagHit.analysis_id.label("analysis_id"),
                latest_presales_subq.c.cid.label("conversation_id"),
                TagDefinition.id.label("tag_id"),
                TagDefinition.name.label("tag_name"),
                ConversationTagHit.created_at.label("created_at"),
                ConversationTagHit.reason.label("reason"),
            )
            .select_from(ConversationTagHit)
            .join(latest_presales_subq, latest_presales_subq.c.analysis_id == ConversationTagHit.analysis_id)
            .join(TagDefinition, TagDefinition.id == ConversationTagHit.tag_id)
            .where(ConversationTagHit.tag_id.in_(tag_ids))
            .order_by(ConversationTagHit.id.asc())
        )

        backed_up_hit_rows = 0
        with open(backup_hits_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["hit_id", "analysis_id", "conversation_id", "tag_id", "tag_name", "created_at", "reason"])
            for row in session.exec(hits_stmt):
                w.writerow(
                    [
                        int(row.hit_id or 0),
                        int(row.analysis_id or 0),
                        int(row.conversation_id or 0),
                        int(row.tag_id or 0),
                        str(row.tag_name or ""),
                        str(row.created_at or ""),
                        str(row.reason or ""),
                    ]
                )
                backed_up_hit_rows += 1

        # 备份：手动绑定
        manual_stmt = (
            select(
                ManualTagBinding.id.label("binding_id"),
                ManualTagBinding.conversation_id.label("conversation_id"),
                TagDefinition.id.label("tag_id"),
                TagDefinition.name.label("tag_name"),
                ManualTagBinding.created_at.label("created_at"),
                ManualTagBinding.reason.label("reason"),
            )
            .select_from(ManualTagBinding)
            .join(TagDefinition, TagDefinition.id == ManualTagBinding.tag_id)
            .join(latest_presales_subq, latest_presales_subq.c.cid == ManualTagBinding.conversation_id)
            .where(ManualTagBinding.tag_id.in_(tag_ids))
            .order_by(ManualTagBinding.id.asc())
        )

        backed_up_manual_rows = 0
        with open(backup_manual_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["binding_id", "conversation_id", "tag_id", "tag_name", "created_at", "reason"])
            for row in session.exec(manual_stmt):
                w.writerow(
                    [
                        int(row.binding_id or 0),
                        int(row.conversation_id or 0),
                        int(row.tag_id or 0),
                        str(row.tag_name or ""),
                        str(row.created_at or ""),
                        str(row.reason or ""),
                    ]
                )
                backed_up_manual_rows += 1

        print(f"[INFO] 目标一级分类：{CATEGORY_NAME}（tag_ids={len(tag_ids)}）")
        print(f"[INFO] 备份 AI 命中行数：{backed_up_hit_rows} -> {backup_hits_path}")
        print(f"[INFO] 备份 手动绑定行数：{backed_up_manual_rows} -> {backup_manual_path}")

        if not args.apply:
            print("[DRY-RUN] 未执行删除（加 --apply 才会提交）。")
            session.rollback()
            return 0

        # 删除 AI 命中（仅最新分析 & 售前）
        del_hits_stmt = delete(ConversationTagHit).where(
            ConversationTagHit.analysis_id.in_(select(latest_presales_subq.c.analysis_id)),
            ConversationTagHit.tag_id.in_(tag_ids),
        )
        deleted_hits = int(session.exec(del_hits_stmt).rowcount or 0)

        # 删除手动绑定（对话级 & 售前对话）
        del_manual_stmt = delete(ManualTagBinding).where(
            ManualTagBinding.conversation_id.in_(select(latest_presales_subq.c.cid)),
            ManualTagBinding.tag_id.in_(tag_ids),
        )
        deleted_manual = int(session.exec(del_manual_stmt).rowcount or 0)

        session.commit()
        print(f"[OK] 已删除 AI 命中：{deleted_hits} 行；手动绑定：{deleted_manual} 行。")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
