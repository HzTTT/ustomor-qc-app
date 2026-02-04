"""质检检查器：负责检查对话是否符合自动质检条件."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Set, Any

from sqlmodel import Session, select, func
from sqlalchemy import and_, or_

from models import Conversation, Message, ConversationAnalysis, AgentBinding, User


def get_unbound_agent_info(
    session: Session,
    *,
    platform: str,
    agent_accounts: List[str]
) -> Dict[str, Dict[str, str]]:
    """获取未绑定客服账号及其详细信息.
    
    Args:
        session: 数据库会话
        platform: 平台标识
        agent_accounts: 需要检查的客服账号列表
    
    Returns:
        未绑定账号信息字典，格式：{account: {"nick": "昵称", "first_seen_in": "conversation_id"}}
    """
    if not agent_accounts:
        return {}
    
    platform_norm = (platform or "").strip().lower()
    unbound_info: Dict[str, Dict[str, str]] = {}
    
    for acc in agent_accounts:
        acc_norm = (acc or "").strip()
        if not acc_norm:
            continue
        
        # 检查是否已绑定
        binding = session.exec(
            select(AgentBinding).where(
                AgentBinding.platform == platform_norm,
                AgentBinding.agent_account == acc_norm,
            )
        ).first()
        
        if binding:
            # 检查绑定的用户是否有效
            user = session.get(User, binding.user_id)
            if user and getattr(user, "is_active", True):
                continue  # 已绑定且用户有效
        
        # 未绑定或绑定失效，收集信息
        # 从最近的对话中获取 assistant_nick
        conv = session.exec(
            select(Conversation)
            .where(
                Conversation.platform == platform_norm,
                Conversation.agent_account == acc_norm,
            )
            .order_by(Conversation.id.desc())
            .limit(1)
        ).first()
        
        nick = ""
        conv_id = ""
        if conv:
            conv_id = conv.external_id or str(conv.id)
            # 尝试从 meta 中获取 assistant_nick
            nick = str((conv.meta or {}).get("assistant_nick", ""))
            
            # 如果 meta 中没有，尝试从消息中获取
            if not nick:
                msg = session.exec(
                    select(Message)
                    .where(
                        Message.conversation_id == conv.id,
                        Message.sender == "agent",
                        Message.agent_account == acc_norm,
                        Message.agent_nick != "",
                    )
                    .limit(1)
                ).first()
                if msg:
                    nick = msg.agent_nick or ""
        
        unbound_info[acc_norm] = {
            "nick": nick,
            "first_seen_in": conv_id,
        }
    
    return unbound_info


def get_conversations_eligible_for_qc(
    session: Session,
    *,
    min_messages: int = 5,
    limit: int | None = 100
) -> List[Dict[str, Any]]:
    """获取符合自动质检条件的对话列表.
    
    条件：
    1. 未质检过（没有 ConversationAnalysis 记录）
    2. 消息数 > min_messages
    3. 对话中所有客服账号都已绑定
    
    Args:
        session: 数据库会话
        min_messages: 最少消息数阈值
        limit: 返回数量限制；传 None 或 <=0 表示不限制
    
    Returns:
        符合条件的对话列表，每项包含 {conversation_id, message_count, agent_accounts}
    """
    # 1. 找出未质检的对话（没有 analysis 记录）
    unanalyzed_convs = session.exec(
        select(Conversation.id)
        .where(~Conversation.analyses.any())
    ).all()
    
    if not unanalyzed_convs:
        return []
    
    # SQLModel may return scalars for single-column selects.
    conv_ids = [int(row) for row in unanalyzed_convs]
    
    # 2. 统计每个对话的消息数
    msg_counts = session.exec(
        select(Message.conversation_id, func.count(Message.id))
        .where(Message.conversation_id.in_(conv_ids))
        .group_by(Message.conversation_id)
        .having(func.count(Message.id) > min_messages)
    ).all()
    
    if not msg_counts:
        return []
    
    # 筛选出消息数满足条件的对话
    eligible_conv_ids = [int(cid) for cid, cnt in msg_counts if int(cnt) > min_messages]
    
    if not eligible_conv_ids:
        return []
    
    # 3. 检查每个对话中的客服账号是否全部绑定
    eligible_convs: List[Dict[str, Any]] = []
    
    max_take: int | None = None
    if limit is not None:
        try:
            max_take = int(limit)
        except Exception:
            max_take = None
    if max_take is not None and max_take <= 0:
        max_take = None

    conv_id_pool = eligible_conv_ids if max_take is None else eligible_conv_ids[:max_take]

    for cid in conv_id_pool:  # 可选限制检查数量
        conv = session.get(Conversation, cid)
        if not conv:
            continue
        
        # 获取该对话中涉及的所有客服账号（从消息中）
        agent_accounts = session.exec(
            select(Message.agent_account)
            .where(
                Message.conversation_id == cid,
                Message.sender == "agent",
                Message.agent_account != "",
            )
            .distinct()
        ).all()
        
        # SQLModel returns scalars for single-column selects.
        agent_accounts_list = [str(acc).strip() for acc in agent_accounts if acc and str(acc).strip()]
        
        # 如果没有客服消息，跳过（可能是纯买家咨询）
        if not agent_accounts_list:
            continue
        
        # 检查所有客服账号是否都已绑定
        all_bound = True
        for acc in agent_accounts_list:
            binding = session.exec(
                select(AgentBinding).where(
                    AgentBinding.platform == (conv.platform or "").lower(),
                    AgentBinding.agent_account == acc,
                )
            ).first()
            
            if not binding:
                all_bound = False
                break
            
            # 检查绑定的用户是否有效
            user = session.get(User, binding.user_id)
            if not user or not getattr(user, "is_active", True):
                all_bound = False
                break
        
        if all_bound:
            msg_count = next((cnt for conv_id, cnt in msg_counts if conv_id == cid), 0)
            eligible_convs.append({
                "conversation_id": cid,
                "external_id": conv.external_id,
                "platform": conv.platform,
                "message_count": int(msg_count),
                "agent_accounts": agent_accounts_list,
            })
    
    return eligible_convs
