#!/usr/bin/env python3
"""
测试对话列表时间筛选后计数是否正确

用法：
    python scripts/test_conversation_list_count.py
"""

import sys
from pathlib import Path

# Add app directory to path (兼容：宿主机 repo 根目录运行 / 容器内 /app/scripts 运行)
root_dir = Path(__file__).resolve().parent.parent
app_dir = root_dir / "app"
if app_dir.exists():
    sys.path.insert(0, str(app_dir))
else:
    sys.path.insert(0, str(root_dir))

from datetime import datetime, timedelta
from sqlalchemy import func, select, and_, or_, exists
from sqlmodel import Session
from db import engine
from models import Conversation, Message


def _as_int(v):
    try:
        if hasattr(v, "__len__") and len(v) == 1 and hasattr(v, "__getitem__"):
            v = v[0]
    except Exception:
        pass
    try:
        return int(v)
    except Exception:
        return v


def test_count_with_date_filter():
    """测试时间筛选后的计数是否正确"""
    
    with Session(engine) as session:
        # 1. 获取一些对话的时间范围
        print("=" * 60)
        print("测试对话列表时间筛选计数")
        print("=" * 60)
        
        # 获取所有对话的客户消息时间
        customer_ts_sq = (
            select(
                Message.conversation_id.label("cid"),
                func.max(Message.ts).label("customer_ts"),
            )
            .where(Message.sender == "buyer", Message.ts.is_not(None))
            .group_by(Message.conversation_id)
        ).subquery()
        
        display_ts = func.coalesce(
            customer_ts_sq.c.customer_ts,
            Conversation.started_at,
            Conversation.uploaded_at
        )
        
        # 获取时间范围
        result = session.exec(
            select(
                func.min(display_ts).label("min_ts"),
                func.max(display_ts).label("max_ts"),
                func.count(Conversation.id).label("total")
            )
            .outerjoin(customer_ts_sq, customer_ts_sq.c.cid == Conversation.id)
        ).first()
        
        if not result or not result[2]:
            print("❌ 数据库中没有对话数据")
            return False
        
        min_ts, max_ts, total_convs = result
        print(f"\n数据库统计：")
        print(f"  总对话数: {total_convs}")
        print(f"  最早时间: {min_ts}")
        print(f"  最晚时间: {max_ts}")
        
        if not min_ts or not max_ts:
            print("❌ 无法获取时间范围（可能所有对话都没有时间戳）")
            return False
        
        # 2. 测试一个具体的日期范围
        # 选择中间的某一天
        test_date = min_ts.date() if isinstance(min_ts, datetime) else min_ts
        start_dt = datetime.combine(test_date, datetime.min.time())
        end_dt = start_dt + timedelta(days=1)
        
        print(f"\n测试时间范围：{start_dt.date()} (包含该天的所有时间)")
        
        # 方法1：使用修复后的计数方式（带 JOIN）
        conv_where = [
            display_ts >= start_dt,
            display_ts < end_dt
        ]
        
        count_q = (
            select(func.count(Conversation.id))
            .outerjoin(customer_ts_sq, customer_ts_sq.c.cid == Conversation.id)
            .where(*conv_where)
        )
        count_with_join = _as_int(session.exec(count_q).one())
        
        # 方法2：获取实际的对话列表数量
        list_q = (
            select(Conversation.id)
            .outerjoin(customer_ts_sq, customer_ts_sq.c.cid == Conversation.id)
            .where(*conv_where)
        )
        actual_ids = session.exec(list_q).all()
        actual_count = len(actual_ids)
        
        # 方法3：错误的计数方式（不带 JOIN，模拟修复前的bug）
        count_without_join = _as_int(session.exec(select(func.count(Conversation.id)).where(*conv_where)).one())
        
        print(f"\n结果对比：")
        print(f"  修复后的计数（带 JOIN）: {count_with_join}")
        print(f"  实际对话数量: {actual_count}")
        print(f"  修复前的计数（不带 JOIN）: {count_without_join}")
        
        # 验证修复是否正确
        if count_with_join == actual_count:
            print(f"\n✅ 修复成功！计数正确：{count_with_join} == {actual_count}")
            success = True
        else:
            print(f"\n❌ 修复失败！计数不匹配：{count_with_join} != {actual_count}")
            success = False
        
        # 显示修复前的问题
        if count_without_join != actual_count:
            print(f"⚠️  修复前的bug：计数错误 {count_without_join} != {actual_count}")
        
        # 3. 测试多个日期范围
        print(f"\n" + "=" * 60)
        print("测试多个日期范围")
        print("=" * 60)
        
        test_cases = [
            ("最早一天", min_ts.date(), min_ts.date()),
            ("最晚一天", max_ts.date(), max_ts.date()),
        ]
        
        # 如果时间范围跨度 > 3天，测试中间的某一天
        if (max_ts.date() - min_ts.date()).days > 3:
            mid_date = min_ts.date() + timedelta(days=(max_ts.date() - min_ts.date()).days // 2)
            test_cases.append(("中间某天", mid_date, mid_date))
        
        all_passed = True
        for name, start_date, end_date in test_cases:
            start_dt = datetime.combine(start_date, datetime.min.time())
            end_dt = datetime.combine(end_date, datetime.max.time())
            
            conv_where = [
                display_ts >= start_dt,
                display_ts < (end_dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))
            ]
            
            count_with_join = _as_int(session.exec(
                select(func.count(Conversation.id))
                .outerjoin(customer_ts_sq, customer_ts_sq.c.cid == Conversation.id)
                .where(*conv_where)
            ).one())
            
            actual_count = len(session.exec(
                select(Conversation.id)
                .outerjoin(customer_ts_sq, customer_ts_sq.c.cid == Conversation.id)
                .where(*conv_where)
            ).all())
            
            status = "✅" if count_with_join == actual_count else "❌"
            print(f"{status} {name} ({start_date}): 计数={count_with_join}, 实际={actual_count}")
            
            if count_with_join != actual_count:
                all_passed = False
        
        print(f"\n" + "=" * 60)
        if all_passed and success:
            print("✅ 所有测试通过！时间筛选计数修复成功。")
            return True
        else:
            print("❌ 部分测试失败，请检查修复。")
            return False


if __name__ == "__main__":
    try:
        success = test_count_with_date_filter()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
