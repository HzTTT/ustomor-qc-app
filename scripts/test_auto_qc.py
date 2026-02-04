#!/usr/bin/env python3
"""测试自动质检功能的脚本.

使用方法：
    python scripts/test_auto_qc.py

功能：
    1. 测试未绑定客服账号检测
    2. 测试符合质检条件的对话查找
    3. 测试飞书通知发送
"""

import sys
from pathlib import Path

# 添加 app 目录到路径
root_dir = Path(__file__).resolve().parent.parent
app_dir = root_dir / "app"
if not app_dir.exists():
    app_dir = root_dir
sys.path.insert(0, str(app_dir))

from sqlmodel import Session, select
from db import engine
from qc_checker import get_unbound_agent_info, get_conversations_eligible_for_qc
from notify import send_feishu_webhook, get_feishu_webhook_url
from models import Conversation, Message, AgentBinding


def test_unbound_agent_detection():
    """测试未绑定客服账号检测."""
    print("\n=== 测试1: 未绑定客服账号检测 ===")
    
    with Session(engine) as session:
        # 获取所有平台的客服账号
        convs = session.exec(
            select(Conversation.platform, Conversation.agent_account)
            .where(Conversation.agent_account != "")
            .distinct()
        ).all()
        
        if not convs:
            print("❌ 数据库中没有对话记录")
            return False
        
        print(f"✓ 找到 {len(convs)} 个客服账号")
        
        # 按平台分组
        by_platform = {}
        for platform, acc in convs:
            by_platform.setdefault(platform, []).append(acc)
        
        # 检查每个平台
        for platform, accounts in by_platform.items():
            print(f"\n平台: {platform}")
            print(f"  客服账号总数: {len(accounts)}")
            
            unbound_info = get_unbound_agent_info(
                session,
                platform=platform,
                agent_accounts=accounts
            )
            
            if unbound_info:
                print(f"  ⚠️  未绑定账号: {len(unbound_info)}")
                for acc, info in list(unbound_info.items())[:5]:  # 只显示前5个
                    nick = info.get("nick", "")
                    print(f"    • {acc} (昵称: {nick or '无'})")
                if len(unbound_info) > 5:
                    print(f"    ... 还有 {len(unbound_info) - 5} 个未绑定账号")
            else:
                print(f"  ✅ 所有账号已绑定")
        
        return True


def test_eligible_conversations():
    """测试符合质检条件的对话查找."""
    print("\n=== 测试2: 符合质检条件的对话查找 ===")
    
    with Session(engine) as session:
        # 测试不同的消息数阈值
        for min_msgs in [5, 10, 20]:
            print(f"\n消息数阈值: >{min_msgs}")
            
            eligible = get_conversations_eligible_for_qc(
                session,
                min_messages=min_msgs,
                limit=10
            )
            
            if eligible:
                print(f"  ✓ 找到 {len(eligible)} 个符合条件的对话（显示前10个）")
                for item in eligible[:3]:  # 只显示前3个
                    cid = item["conversation_id"]
                    ext_id = item["external_id"]
                    msg_count = item["message_count"]
                    accounts = item["agent_accounts"]
                    print(f"    • CID={cid} | {ext_id} | {msg_count}条消息 | 客服: {', '.join(accounts)}")
                if len(eligible) > 3:
                    print(f"    ... 还有 {len(eligible) - 3} 个对话")
            else:
                print(f"  ℹ️  暂无符合条件的对话")
        
        return True


def test_feishu_notification():
    """测试飞书通知发送."""
    print("\n=== 测试3: 飞书通知发送 ===")
    
    with Session(engine) as session:
        webhook_url = get_feishu_webhook_url(session)
        
        if not webhook_url:
            print("❌ 未配置飞书 Webhook URL")
            print("   请在【设置 > 对象存储设置】页面配置")
            return False
        
        print(f"✓ Webhook URL: {webhook_url[:50]}...")
        
        # 发送测试通知
        print("\n发送测试通知...")
        result = send_feishu_webhook(
            webhook_url,
            title="🧪 自动质检功能测试",
            text="这是一条测试消息，用于验证飞书通知功能是否正常工作。\n\n如果收到此消息，说明配置正确。"
        )
        
        if result.get("ok"):
            print("✅ 通知发送成功")
            return True
        else:
            print(f"❌ 通知发送失败: {result.get('error')}")
            return False


def test_database_statistics():
    """显示数据库统计信息."""
    print("\n=== 数据库统计 ===")
    
    with Session(engine) as session:
        # 对话总数
        conv_count = session.exec(
            select(Conversation.id)
        ).all()
        print(f"对话总数: {len(conv_count)}")
        
        # 未质检对话数
        from models import ConversationAnalysis
        unanalyzed = session.exec(
            select(Conversation.id)
            .where(~Conversation.analyses.any())
        ).all()
        print(f"未质检对话: {len(unanalyzed)}")
        
        # 消息数>5的对话
        from sqlalchemy import func
        conv_ids_sample = [
            c[0] if isinstance(c, (list, tuple)) else c
            for c in conv_count[:1000]
        ]
        msg_counts = session.exec(
            select(func.count(Message.id))
            .where(Message.conversation_id.in_(conv_ids_sample))  # 限制查询范围
        ).one()
        print(f"抽样对话消息总数(前1000对话): {msg_counts}")
        
        # 已绑定的客服账号数
        bindings = session.exec(
            select(AgentBinding.id)
        ).all()
        print(f"已绑定客服账号: {len(bindings)}")
        
        # 待处理的质检任务
        from models import AIAnalysisJob
        pending_jobs = session.exec(
            select(AIAnalysisJob.id)
            .where(AIAnalysisJob.status == "pending")
        ).all()
        print(f"待处理质检任务: {len(pending_jobs)}")
        
        running_jobs = session.exec(
            select(AIAnalysisJob.id)
            .where(AIAnalysisJob.status == "running")
        ).all()
        print(f"正在处理质检任务: {len(running_jobs)}")


def main():
    """运行所有测试."""
    print("=" * 60)
    print("自动质检功能测试")
    print("=" * 60)
    
    try:
        # 显示统计信息
        test_database_statistics()
        
        # 运行测试
        test1 = test_unbound_agent_detection()
        test2 = test_eligible_conversations()
        test3 = test_feishu_notification()
        
        # 总结
        print("\n" + "=" * 60)
        print("测试总结")
        print("=" * 60)
        print(f"未绑定账号检测: {'✅ 通过' if test1 else '❌ 失败'}")
        print(f"符合条件对话查找: {'✅ 通过' if test2 else '❌ 失败'}")
        print(f"飞书通知发送: {'✅ 通过' if test3 else '❌ 失败'}")
        
        if all([test1, test2, test3]):
            print("\n🎉 所有测试通过！")
            return 0
        else:
            print("\n⚠️  部分测试失败，请检查配置")
            return 1
            
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
