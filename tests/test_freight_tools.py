import db
import mcp_server
import seed


def seeded(tmp_path):
    path = str(tmp_path / "logi.db")
    db.init_db(path)
    seed.seed(path)
    return path


def test_search_freight_filters_capacity(tmp_path):
    path = seeded(tmp_path)
    res = mcp_server.search_freight_impl(min_capacity_kg=3400, db_path=path)
    assert res["status"] == "OK"
    ids = [t["id"] for t in res["trucks"]]
    # ORDER BY rate_per_km: T1=120, T3=130, T2=210, T4=260
    assert ids == ["T1", "T3", "T2", "T4"]


def test_search_freight_excludes_unavailable(tmp_path):
    path = seeded(tmp_path)
    with db.get_session(path) as s:
        s.get(db.Truck, "T1").available = False
        s.commit()
    res = mcp_server.search_freight_impl(min_capacity_kg=3400, db_path=path)
    assert "T1" not in [t["id"] for t in res["trucks"]]


def test_book_shipment_success(tmp_path):
    path = seeded(tmp_path)
    res = mcp_server.book_shipment_impl(
        truck_id="T1", origin="Chennai Port", destination="Guindy",
        pallets=10, db_path=path,
    )
    assert res["status"] == "OK"
    assert res["cost"] == 120 * 12  # rate_per_km * CORRIDOR_KM["Chennai Port"]["Guindy"]
    with db.get_session(path) as s:
        assert s.get(db.Truck, "T1").available is False


def test_book_shipment_over_weight_is_capacity_error(tmp_path):
    path = seeded(tmp_path)
    res = mcp_server.book_shipment_impl(
        truck_id="T1", origin="Chennai Port", destination="Guindy",
        pallets=1000, db_path=path,  # 1000*250 = 250000kg > 3500
    )
    assert res["status"] == "ERROR"
    assert res["reason"] == "capacity_exceeded"


def test_book_shipment_unknown_truck(tmp_path):
    path = seeded(tmp_path)
    res = mcp_server.book_shipment_impl(
        truck_id="NOPE", origin="Chennai Port", destination="Guindy",
        pallets=2, db_path=path,
    )
    assert res == {"status": "ERROR", "reason": "not_found", "detail": "truck NOPE"}


def test_book_shipment_unavailable_truck(tmp_path):
    path = seeded(tmp_path)
    with db.get_session(path) as s:
        s.get(db.Truck, "T2").available = False
        s.commit()
    res = mcp_server.book_shipment_impl(
        truck_id="T2", origin="Chennai Port", destination="Sriperumbudur",
        pallets=4, db_path=path,
    )
    assert res["status"] == "ERROR"
    assert res["reason"] == "truck_unavailable"


def test_cancel_shipment_restores_availability(tmp_path):
    path = seeded(tmp_path)
    booking = mcp_server.book_shipment_impl(
        truck_id="T1", origin="Chennai Port", destination="Guindy",
        pallets=10, db_path=path,
    )
    res = mcp_server.cancel_shipment_impl(shipment_id=booking["shipment_id"], db_path=path)
    assert res["status"] == "OK"
    with db.get_session(path) as s:
        assert s.get(db.Truck, "T1").available is True
        assert s.get(db.Shipment, booking["shipment_id"]).status == "CANCELLED"


def test_corridor_fallback_distance(tmp_path):
    path = seeded(tmp_path)
    res = mcp_server.book_shipment_impl(
        truck_id="T1", origin="Guindy", destination="Sriperumbudur",
        pallets=2, db_path=path,
    )
    assert res["cost"] == 120 * 33  # CORRIDOR_KM["Guindy"]["Sriperumbudur"]
