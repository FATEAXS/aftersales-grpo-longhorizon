"""数据生成确定性 + HTTP 服务契约 + 盲评守卫 + 数据集渲染。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient

from aftersales_grpo.simulator.models import ENVIRONMENT_VERSION
from aftersales_grpo.simulator.server import app, load_world, world_fingerprint

DATA_DIR = ROOT / "environments/aftersalesim/data"


@pytest.fixture(scope="module")
def client():
    load_world(DATA_DIR)
    with TestClient(app) as test_client:
        yield test_client


def test_world_data_present():
    manifest = json_load(DATA_DIR / "world_manifest.json")
    assert manifest["counts"]["tasks"] == 600
    assert manifest["splits"]["train"]["tasks"] == 400
    assert manifest["splits"]["evaluation"]["tasks"] == 100


def json_load(path):
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def test_task_split_zero_overlap():
    task_ids = {}
    for split, count in [("train", 400), ("validation", 100), ("evaluation", 100)]:
        ids = [
            json_loads(line)["task_id"]
            for line in open(DATA_DIR / "tasks" / f"{split}.jsonl", encoding="utf-8")
            if line.strip()
        ]
        assert len(ids) == count
        task_ids[split] = set(ids)
    assert not (task_ids["train"] & task_ids["validation"])
    assert not (task_ids["train"] & task_ids["evaluation"])
    assert not (task_ids["validation"] & task_ids["evaluation"])


def json_loads(text):
    import json

    return json.loads(text)


def test_health_contract(client):
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert data["environment_version"] == ENVIRONMENT_VERSION
    assert data["world_fingerprint"] == world_fingerprint(DATA_DIR)


def test_session_action_and_grading(client):
    fact = first_fact("train")
    created = client.post("/v1/sessions", json={"task_id": fact["task_id"]}).json()
    assert created["environment_version"] == ENVIRONMENT_VERSION
    assert "user_message" in created["observation"]
    sid = created["session_id"]

    result = client.post(
        f"/v1/sessions/{sid}/action",
        json={"tool": "query_order", "arguments": {"order_id": fact["order_id"]}},
    )
    assert result.status_code == 200
    assert result.json()["ok"]

    outcome = client.get(f"/v1/sessions/{sid}/result").json()
    assert "reward" in outcome and "summary" in outcome
    client.post(f"/v1/sessions/{sid}/release")


def test_unknown_task_404(client):
    assert client.post("/v1/sessions", json={"task_id": "nope"}).status_code == 404


def first_fact(split):
    with open(DATA_DIR / "tasks" / f"{split}.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                return json_loads(line)
    raise AssertionError("no facts")


def test_blind_guard_detects_leak():
    from aftersales_grpo.evaluation.artifacts import guard_blind_trajectory

    record = {"task_id": "t", "messages": []}
    assert guard_blind_trajectory(record) == []
    record["expected_resolution"] = "return_refund"
    assert "expected_resolution" in guard_blind_trajectory(record)


def test_dataset_rendering_with_fake_tokenizer():
    from aftersales_grpo.environment.tools import TOOLS
    from aftersales_grpo.training.sft.dataset import build_supervised_example

    class FakeTokenizer:
        """以空白字符编号的确定性 tokenizer：token = ' ' + word。"""

        def __call__(self, text, add_special_tokens=False):
            words = text.split(" ")
            return {"input_ids": [hash_of(w) for w in words]}

        def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False):
            parts = []
            for m in messages:
                parts.append(f"<{m['role']}>")
                if m.get("content"):
                    parts.append(str(m["content"]))
                for call in m.get("tool_calls") or []:
                    parts.append(f"<tool:{call['function']['name']}>")
            if add_generation_prompt:
                parts.append("<assistant>")
            return " ".join(parts)

    def hash_of(word):
        # 稳定伪 token id（不同词几乎不碰撞）
        return sum(bytearray(word, "utf-8")) * 31 + len(word)

    messages = [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "user message"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "query_order", "arguments": "{\"order_id\": \"O1\"}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "{}"},
        {"role": "assistant", "content": "done"},
    ]
    example = build_supervised_example(messages, TOOLS, FakeTokenizer(), max_length=4096)
    assert example is not None
    trained = [i for i, label in zip(example["input_ids"], example["labels"]) if label != -100]
    assert trained, "assistant 回合必须参与 loss"
    # assistant token 数量有限：不应是全部 token 都参与训练
    assert len(trained) < len(example["input_ids"])
