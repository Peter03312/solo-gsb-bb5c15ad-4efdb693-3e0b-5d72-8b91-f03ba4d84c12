"""批量幂等投递判定（POST /deliveries/batch）：

一次提交 1–100 笔投递编号 + 收件地址，返回与输入顺序一致的逐项判定
（成功含终点路径，失败含原错误详情及状态）。整批先校验：无效地址、批内重复编号、
既有编号绑定其他规范化地址、尚无默认策略且存在新编号——任一命中即整批拒绝且不写入。
已有同编号同地址的判定原样重放；新判定在同一事务中共用受理时的策略修订号与比例，
分别稳定分桶、解析并持久化（路由失败仍保存）；全部写入后仅评估一次候选侧熔断，
触发只影响后续请求，重放不重复计数。
"""
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

# 必须在导入 app 之前设置，config 在导入时读取
os.environ.setdefault("ROUTER_DB", os.path.join(tempfile.mkdtemp(prefix="router-test-"), "test.db"))

import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import app


@pytest.fixture()
def client():
    db.init_db()
    conn = db.connect()
    conn.executescript(
        "DELETE FROM draft_rules;"
        "DELETE FROM delivery_policy;"
        "DELETE FROM delivery_decisions;"
        "DELETE FROM circuit_breaker_settings;"
        "DELETE FROM circuit_breaks;"
        "DELETE FROM version_rules;"
        "DELETE FROM versions;"
        "DELETE FROM sqlite_sequence WHERE name = 'versions';"
        "DELETE FROM draft_revisions;"
        "INSERT INTO draft_revisions (revision, rules, rule_count, created_at)"
        " VALUES (0, '[]', 0, strftime('%Y-%m-%dT%H:%M:%S+00:00', 'now'));"
        "UPDATE draft_meta SET revision = 0 WHERE id = 1;"
    )
    conn.close()
    with TestClient(app) as c:
        yield c


BASE_RULES = [
    {
        "match_type": "exact",
        "match_key": "shared@corp.example.com",
        "action": "route",
        "targets": [{"kind": "mailbox", "value": "base@mail.example.com"}],
    },
    {
        "match_type": "domain",
        "match_key": "corp.example.com",
        "action": "route",
        "targets": [{"kind": "mailbox", "value": "base-catchall@mail.example.com"}],
    },
]

CANDIDATE_RULES = [
    {
        "match_type": "exact",
        "match_key": "shared@corp.example.com",
        "action": "route",
        "targets": [{"kind": "mailbox", "value": "cand@mail.example.com"}],
    },
    # 候选里 blocked@ 命中拒收；候选无域兜底 → 其他地址 no_route
    {
        "match_type": "exact",
        "match_key": "blocked@corp.example.com",
        "action": "reject",
        "targets": [],
    },
]


def put_rules(client, revision, rules):
    return client.put("/draft", json={"revision": revision, "rules": rules})


def publish(client, revision):
    return client.post("/publish", json={"revision": revision})


def publish_two_versions(client):
    assert put_rules(client, 0, BASE_RULES).status_code == 200
    assert publish(client, 1).status_code == 200
    assert put_rules(client, 1, CANDIDATE_RULES).status_code == 200
    resp = publish(client, 2)
    assert resp.status_code == 200, resp.text
    return 1, 2


def set_policy(client, revision, baseline, candidate, percent):
    return client.put("/policy", json={
        "revision": revision,
        "baseline_version": baseline,
        "candidate_version": candidate,
        "candidate_percent": percent,
    })


def set_breaker(client, revision, threshold=50, min_samples=3, window_minutes=10):
    return client.put("/policy/circuit-breaker", json={
        "revision": revision,
        "failure_threshold_percent": threshold,
        "min_candidate_samples": min_samples,
        "window_minutes": window_minutes,
    })


def submit(client, delivery_id, address):
    return client.post("/deliveries", json={"delivery_id": delivery_id, "address": address})


def submit_batch(client, items):
    return client.post("/deliveries/batch", json={"items": items})


def drop_policy_row():
    """白盒：直接删除策略行，构造“已有判定但尚无默认策略”的场景。"""
    conn = db.connect()
    try:
        conn.execute("DELETE FROM delivery_policy")
    finally:
        conn.close()


# ---------- 首次批量受理：逐项判定、顺序一致、共用受理时策略 ----------

def test_batch_submit_returns_per_item_decisions_in_input_order(client):
    v1, v2 = publish_two_versions(client)
    set_policy(client, 0, v1, v2, 100)  # 全部走候选
    items = [
        {"delivery_id": "b-1", "address": " Shared@Corp.Example.COM "},   # 候选成功
        {"delivery_id": "b-2", "address": "blocked@corp.example.com"},    # 候选拒收
        {"delivery_id": "b-3", "address": "stranger@corp.example.com"},   # 候选无路由
    ]
    resp = submit_batch(client, items)
    assert resp.status_code == 200, resp.text
    results = resp.json()["items"]
    assert [r["delivery_id"] for r in results] == ["b-1", "b-2", "b-3"]  # 与输入顺序一致

    ok = results[0]
    assert ok["status"] == "ok"
    assert ok["address"] == "shared@corp.example.com"  # 规范化地址
    assert [e["mailbox"] for e in ok["endpoints"]] == ["cand@mail.example.com"]
    assert ok["error"] is None
    assert ok["selection"] == "candidate" and ok["version"] == v2

    rejected = results[1]
    assert rejected["status"] == "failed"
    assert rejected["endpoints"] is None
    assert rejected["error"]["error"] == "rejected"  # 原错误详情
    assert rejected["error"]["delivery_id"] == "b-2"
    assert rejected["error"]["version"] == v2 and rejected["error"]["selection"] == "candidate"

    no_route = results[2]
    assert no_route["status"] == "failed"
    assert no_route["error"]["error"] == "no_route"

    # 同一批共用受理时的策略修订号与比例
    assert {r["policy_revision"] for r in results} == {1}
    assert {r["candidate_percent"] for r in results} == {100}
    # 成功与失败判定都已持久化，可按编号查询
    for item in results:
        stored = client.get(f"/deliveries/{item['delivery_id']}").json()
        assert stored == item


def test_batch_items_share_acceptance_policy_and_bucket_independently(client):
    v1, v2 = publish_two_versions(client)
    set_policy(client, 0, v1, v2, 50)
    items = [{"delivery_id": f"b-x{i}", "address": f"user{i}@corp.example.com"} for i in range(10)]
    resp = submit_batch(client, items)
    assert resp.status_code == 200, resp.text
    results = resp.json()["items"]
    assert {r["policy_revision"] for r in results} == {1}  # 共用受理时的策略修订号
    assert {r["candidate_percent"] for r in results} == {50}
    assert {r["selection"] for r in results} == {"baseline", "candidate"}  # 各自稳定分桶
    for item in results:
        # 各项按自身地址稳定分桶，与 /deliver 的即时判定一致（含候选侧失败）
        deliver = client.post("/deliver", json={"address": item["address"]})
        if deliver.status_code == 200:
            assert item["status"] == "ok"
            assert item["selection"] == deliver.json()["selection"]
            assert item["version"] == deliver.json()["version"]
        else:  # 候选无域兜底：no_route 失败同样作为判定持久化
            detail = deliver.json()["detail"]
            assert item["status"] == "failed"
            assert item["selection"] == detail["selection"]
            assert item["version"] == detail["version"]
            assert item["error"]["error"] == detail["error"]


def test_batch_replay_returns_identical_response(client):
    v1, v2 = publish_two_versions(client)
    set_policy(client, 0, v1, v2, 100)
    items = [
        {"delivery_id": "b-r1", "address": "shared@corp.example.com"},
        {"delivery_id": "b-r2", "address": "blocked@corp.example.com"},
    ]
    first = submit_batch(client, items)
    assert first.status_code == 200
    # 策略随后改变：重放仍返回首次判定
    assert set_policy(client, 1, v1, v2, 0).status_code == 200
    replay = submit_batch(client, items)
    assert replay.status_code == 200
    assert replay.json() == first.json()
    assert replay.json()["items"][0]["policy_revision"] == 1
    assert replay.json()["items"][0]["version"] == v2


def test_batch_mixed_replay_and_new_items(client):
    v1, v2 = publish_two_versions(client)
    set_policy(client, 0, v1, v2, 100)
    # 既有判定（经单笔接口受理，含失败判定）
    ok_single = submit(client, "b-m1", "shared@corp.example.com")
    assert ok_single.status_code == 200
    failed_single = submit(client, "b-m2", "blocked@corp.example.com")
    assert failed_single.status_code == 422
    # 批量中混排：既有同编号同地址原样重放，新编号当场判定
    resp = submit_batch(client, [
        {"delivery_id": "b-m3", "address": "shared@corp.example.com"},   # 新
        {"delivery_id": "b-m1", "address": " SHARED@CORP.EXAMPLE.COM "},  # 重放（规范化后相同）
        {"delivery_id": "b-m2", "address": "blocked@corp.example.com"},  # 重放失败判定
    ])
    assert resp.status_code == 200, resp.text
    items = resp.json()["items"]
    assert [i["delivery_id"] for i in items] == ["b-m3", "b-m1", "b-m2"]
    assert items[0]["status"] == "ok"
    assert items[1] == client.get("/deliveries/b-m1").json()  # 与首次判定逐字段相同
    assert items[2]["status"] == "failed"
    assert items[2]["error"] == failed_single.json()["detail"]  # 原错误详情
    # 全部项目共用同一策略修订号（重放保留首次受理时的修订号）
    assert {i["policy_revision"] for i in items} == {1}


# ---------- 整批校验：任一不满足即整批拒绝且不写入 ----------

def test_batch_invalid_address_rejects_whole_batch(client):
    v1, v2 = publish_two_versions(client)
    set_policy(client, 0, v1, v2, 100)
    resp = submit_batch(client, [
        {"delivery_id": "b-i1", "address": "shared@corp.example.com"},
        {"delivery_id": "b-i2", "address": "not-an-address"},
    ])
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "invalid_address"
    assert detail["delivery_id"] == "b-i2"
    # 整批不写入：其余编号未被占用
    assert client.get("/deliveries/b-i1").status_code == 404
    assert client.get("/deliveries/b-i2").status_code == 404
    ok = submit(client, "b-i1", "shared@corp.example.com")
    assert ok.status_code == 200


def test_batch_duplicate_id_rejects_whole_batch(client):
    v1, v2 = publish_two_versions(client)
    set_policy(client, 0, v1, v2, 100)
    resp = submit_batch(client, [
        {"delivery_id": "b-d1", "address": "shared@corp.example.com"},
        {"delivery_id": "b-d2", "address": "shared@corp.example.com"},
        {"delivery_id": "b-d1", "address": "shared@corp.example.com"},  # 批内重复（即使同地址）
    ])
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "duplicate_delivery_id"
    assert detail["delivery_id"] == "b-d1"
    assert client.get("/deliveries/b-d1").status_code == 404  # 整批不写入
    assert client.get("/deliveries/b-d2").status_code == 404


def test_batch_existing_id_with_different_address_rejects_whole_batch(client):
    v1, v2 = publish_two_versions(client)
    set_policy(client, 0, v1, v2, 100)
    assert submit(client, "b-c1", "shared@corp.example.com").status_code == 200
    resp = submit_batch(client, [
        {"delivery_id": "b-c2", "address": "shared@corp.example.com"},      # 新编号
        {"delivery_id": "b-c1", "address": "other@corp.example.com"},       # 既有编号绑定其他地址
    ])
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["error"] == "delivery_id_conflict"
    assert detail["conflict_ids"] == ["b-c1"]
    # 整批不写入：新编号未被占用，原判定不改写
    assert client.get("/deliveries/b-c2").status_code == 404
    assert client.get("/deliveries/b-c1").json()["address"] == "shared@corp.example.com"


def test_batch_missing_policy_rejects_when_new_items_exist(client):
    v1, v2 = publish_two_versions(client)
    resp = submit_batch(client, [
        {"delivery_id": "b-p1", "address": "shared@corp.example.com"},
        {"delivery_id": "b-p2", "address": "shared@corp.example.com"},
    ])
    assert resp.status_code == 409
    assert resp.json()["detail"]["error"] == "policy_not_configured"
    assert client.get("/deliveries/b-p1").status_code == 404  # 编号未被占用
    set_policy(client, 0, v1, v2, 100)
    ok = submit_batch(client, [{"delivery_id": "b-p1", "address": "shared@corp.example.com"}])
    assert ok.status_code == 200


def test_batch_replay_only_needs_no_policy(client):
    v1, v2 = publish_two_versions(client)
    set_policy(client, 0, v1, v2, 100)
    assert submit(client, "b-n1", "shared@corp.example.com").status_code == 200
    drop_policy_row()  # 构造“已有判定但当前无默认策略”
    # 全部是既有编号 + 相同地址：原样重放，不需要策略
    replay = submit_batch(client, [{"delivery_id": "b-n1", "address": "SHARED@CORP.EXAMPLE.COM"}])
    assert replay.status_code == 200
    assert replay.json()["items"][0]["delivery_id"] == "b-n1"
    assert replay.json()["items"][0]["status"] == "ok"
    # 只要存在新编号就整批拒绝，且重放项也不返回
    resp = submit_batch(client, [
        {"delivery_id": "b-n1", "address": "shared@corp.example.com"},
        {"delivery_id": "b-n2", "address": "shared@corp.example.com"},
    ])
    assert resp.status_code == 409
    assert resp.json()["detail"]["error"] == "policy_not_configured"
    assert client.get("/deliveries/b-n2").status_code == 404


def test_batch_shape_validation(client):
    v1, v2 = publish_two_versions(client)
    set_policy(client, 0, v1, v2, 100)
    assert submit_batch(client, []).status_code == 422                      # 空批
    assert submit_batch(client, [{"delivery_id": f"b-s{i}", "address": "a@b.example.com"}
                                 for i in range(101)]).status_code == 422  # 超过 100 笔
    assert submit_batch(client, [{"delivery_id": "", "address": "a@b.example.com"}]).status_code == 422
    assert submit_batch(client, [{"delivery_id": "b-s1"}]).status_code == 422
    assert client.post("/deliveries/batch", json={}).status_code == 422


# ---------- 熔断：新判定全部写入后仅评估一次，重放不重复计数 ----------

def test_batch_circuit_breaker_evaluated_once_after_all_writes(client):
    v1, v2 = publish_two_versions(client)
    set_policy(client, 0, v1, v2, 100)
    assert set_breaker(client, 1, threshold=50, min_samples=3).status_code == 200
    items = [{"delivery_id": f"b-b{i}", "address": "blocked@corp.example.com"} for i in range(3)]
    resp = submit_batch(client, items)
    assert resp.status_code == 200, resp.text
    results = resp.json()["items"]
    # 本批仍返回已持久化的原判定（候选侧失败，受理时的策略修订号）
    assert all(r["status"] == "failed" for r in results)
    assert all(r["error"]["error"] == "rejected" for r in results)
    assert {r["policy_revision"] for r in results} == {1}
    assert {r["selection"] for r in results} == {"candidate"}
    # 全部写入后只评估一次：候选比例置 0、策略修订号 +1、恰好一条熔断记录
    policy = client.get("/policy").json()
    assert policy["revision"] == 2
    assert policy["candidate_percent"] == 0
    breaker = client.get("/policy/circuit-breaker").json()
    assert len(breaker["breaks"]) == 1
    brk = breaker["breaks"][0]
    assert brk["policy_revision"] == 1
    assert brk["sample_count"] == 3
    assert brk["failure_count"] == 3
    # 重放整批：不重复计数、不再触发
    replay = submit_batch(client, items)
    assert replay.status_code == 200
    assert replay.json() == resp.json()
    assert len(client.get("/policy/circuit-breaker").json()["breaks"]) == 1
    assert client.get("/policy").json()["revision"] == 2


def test_batch_below_breaker_threshold_does_not_trigger(client):
    v1, v2 = publish_two_versions(client)
    set_policy(client, 0, v1, v2, 100)
    assert set_breaker(client, 1, threshold=50, min_samples=5).status_code == 200
    items = [{"delivery_id": f"b-t{i}", "address": "blocked@corp.example.com"} for i in range(2)]
    resp = submit_batch(client, items)
    assert resp.status_code == 200
    policy = client.get("/policy").json()
    assert policy["revision"] == 1          # 样本不足，未触发
    assert policy["candidate_percent"] == 100
    assert client.get("/policy/circuit-breaker").json()["breaks"] == []


# ---------- 并发与既有接口兼容 ----------

def test_concurrent_identical_batches_single_decision_set(client):
    v1, v2 = publish_two_versions(client)
    set_policy(client, 0, v1, v2, 100)
    items = [
        {"delivery_id": "b-cc1", "address": "shared@corp.example.com"},
        {"delivery_id": "b-cc2", "address": "blocked@corp.example.com"},
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(lambda _: submit_batch(client, items), range(8)))
    assert all(r.status_code == 200 for r in responses)
    assert {r.text for r in responses} == {responses[0].text}  # 完全相同的判定
    assert client.get("/deliveries/b-cc1").status_code == 200
    assert client.get("/deliveries/b-cc2").status_code == 200


def test_single_and_batch_endpoints_interoperate(client):
    # 单笔接口创建的判定在批量接口原样重放，反之亦然；单笔接口行为不变
    v1, v2 = publish_two_versions(client)
    set_policy(client, 0, v1, v2, 100)
    batch = submit_batch(client, [{"delivery_id": "b-w1", "address": "shared@corp.example.com"}])
    assert batch.status_code == 200
    single_replay = submit(client, "b-w1", "shared@corp.example.com")
    assert single_replay.status_code == 200
    assert single_replay.json()["delivery_id"] == "b-w1"
    assert single_replay.json()["endpoints"] == batch.json()["items"][0]["endpoints"]

    single = submit(client, "b-w2", "blocked@corp.example.com")
    assert single.status_code == 422
    batch_replay = submit_batch(client, [{"delivery_id": "b-w2", "address": "blocked@corp.example.com"}])
    assert batch_replay.status_code == 200
    item = batch_replay.json()["items"][0]
    assert item["status"] == "failed"
    assert item["error"] == single.json()["detail"]
