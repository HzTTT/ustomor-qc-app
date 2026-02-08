import json


from platform_detect import detect_platform_from_leyan_jsonl


def _mk_line(body_obj: dict) -> str:
    return json.dumps({"body": json.dumps(body_obj, ensure_ascii=False)}, ensure_ascii=False)


def test_detect_platform_taobao_by_image_host():
    raw = _mk_line({"format": "IMAGE", "image_url": "https://img.alicdn.com/imgextra/xx.jpg"}) + "\n"
    assert detect_platform_from_leyan_jsonl(raw) == "taobao"


def test_detect_platform_douyin_by_image_host():
    raw = _mk_line({"format": "IMAGE", "image_url": "https://p3-pigeon-sign.ecombdimg.com/xx.jpg"}) + "\n"
    assert detect_platform_from_leyan_jsonl(raw) == "douyin"


def test_detect_platform_by_key_hint():
    assert detect_platform_from_leyan_jsonl("", key="leyan/douyin/2026-02-05/x.jsonl.gz") == "douyin"


def test_detect_platform_none_when_no_signal():
    raw = _mk_line({"format": "TEXT", "content": "hello"}) + "\n"
    assert detect_platform_from_leyan_jsonl(raw) is None
