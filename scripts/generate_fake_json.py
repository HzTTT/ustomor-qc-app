import json
from datetime import datetime, timedelta
import random

PROBLEMS = ["refund", "exchange", "logistics", "quality", "size", "invoice"]


def main(out_path: str = "fake_batch.json", n: int = 10):
    now = datetime.utcnow()
    convs = []
    for i in range(n):
        cid = f"demo_{now.strftime('%Y%m%d')}_{i:03d}"
        score = random.randint(50, 95)
        flag = score < 70
        types = random.sample(PROBLEMS, k=random.randint(1, 2))
        convs.append(
            {
                "conversation_id": cid,
                "platform": "demo",
                "agent_email": f"agent{random.randint(1,3)}@example.com",
                "started_at": (now - timedelta(minutes=30*i)).isoformat() + "Z",
                "messages": [
                    {"sender": "buyer", "ts": (now - timedelta(minutes=30*i) ).isoformat() + "Z", "text": "我有个问题..."},
                    {"sender": "agent", "ts": (now - timedelta(minutes=30*i) + timedelta(seconds=20)).isoformat() + "Z", "text": "好的我帮你看下"},
                ],
                "analysis": {
                    "overall_score": score,
                    "sentiment": "negative" if score < 70 else "neutral" if score < 85 else "positive",
                    "issue_level": "high" if score < 60 else "medium" if score < 75 else "low",
                    "problem_types": types,
                    "summary": "示例摘要",
                    "good": "示例优点",
                    "bad": "示例问题",
                    "suggestions": ["示例建议1", "示例建议2"],
                    "flag_for_review": flag,
                },
            }
        )

    payload = {"batch_meta": {"platform": "demo", "date": now.strftime('%Y-%m-%d')}, "conversations": convs}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--out", default="fake_batch.json")
    p.add_argument("--n", type=int, default=10)
    args = p.parse_args()
    main(args.out, args.n)
