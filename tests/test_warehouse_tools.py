import db
import mcp_server
import seed


def seeded(tmp_path):
    path = str(tmp_path / "logi.db")
    db.init_db(path)
    seed.seed(path)
    return path


def test_search_warehouse_filters_space(tmp_path):
    path = seeded(tmp_path)
    res = mcp_server.search_warehouse_impl(min_pallets=50, db_path=path)
    assert res["status"] == "OK"
    ids = [w["id"] for w in res["warehouses"]]  # free capacity: W1 80, W2 60, W4 140; W3 has 5
    assert "W3" not in ids
    assert "W1" in ids


def test_allocate_space_success(tmp_path):
    path = seeded(tmp_path)
    res = mcp_server.allocate_space_impl(
        warehouse_id="W1", pallets=10, order_ref="ORD-1", db_path=path
    )
    assert res["status"] == "OK"
    with db.get_session(path) as s:
        assert s.get(db.Warehouse, "W1").used_pallets == 50


def test_allocate_space_capacity_error(tmp_path):
    path = seeded(tmp_path)
    res = mcp_server.allocate_space_impl(
        warehouse_id="W1", pallets=100, order_ref="ORD-1", db_path=path  # free = 80
    )
    assert res["status"] == "ERROR"
    assert res["reason"] == "capacity_exceeded"


def test_allocate_unknown_warehouse(tmp_path):
    path = seeded(tmp_path)
    res = mcp_server.allocate_space_impl(
        warehouse_id="NOPE", pallets=1, order_ref="ORD-1", db_path=path
    )
    assert res == {"status": "ERROR", "reason": "not_found", "detail": "warehouse NOPE"}


def test_log_self_heal(tmp_path):
    path = seeded(tmp_path)
    res = mcp_server.log_self_heal_impl(
        agent="freight_agent", error_reason="capacity_exceeded",
        remediation="re-booked on T2", outcome="SUCCESS", db_path=path,
    )
    assert res["status"] == "OK"
    with db.get_session(path) as s:
        row = s.query(db.SelfHealLog).one()
        assert row.agent == "freight_agent"
        assert row.error_reason == "capacity_exceeded"
