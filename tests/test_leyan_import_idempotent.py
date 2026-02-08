import json

from sqlmodel import select

from importers.leyan_import import import_leyan_jsonl
from models import Conversation, Message


def test_import_leyan_jsonl_is_idempotent_by_external_message_id(session):
    lines = [
        {
            "leyan_buyer_id": "buyer_1",
            "seller_id": "seller_1",
            "sent_at": "2026-02-05T10:00:00+08:00",
            "sent_by": "BUYER",
            "message_id": "m1",
            "body": json.dumps({"format": "TEXT", "content": "hello"}),
        },
        {
            "leyan_buyer_id": "buyer_1",
            "seller_id": "seller_1",
            "sent_at": "2026-02-05T10:00:05+08:00",
            "sent_by": "ASSISTANT",
            "assistant_id": "agent_1",
            "assistant_nick": "Agent",
            "message_id": "m2",
            "body": json.dumps({"format": "TEXT", "content": "hi"}),
        },
    ]
    raw_text = "\n".join(json.dumps(x, ensure_ascii=False) for x in lines)

    r1 = import_leyan_jsonl(session, raw_text=raw_text, source_filename="t1.jsonl", platform="taobao")
    assert r1["ok"] is True
    assert r1["messages"] == 2
    assert len(session.exec(select(Conversation)).all()) == 1
    assert len(session.exec(select(Message)).all()) == 2

    # Re-import the same content: should not duplicate messages.
    r2 = import_leyan_jsonl(session, raw_text=raw_text, source_filename="t2.jsonl", platform="taobao")
    assert r2["ok"] is True
    assert r2["messages"] == 0
    assert len(session.exec(select(Conversation)).all()) == 1
    assert len(session.exec(select(Message)).all()) == 2
