import db
import mcp_server
import seed


def seeded(tmp_path):
    path = str(tmp_path / "logi.db")
    db.init_db(path)
    seed.seed(path)
    return path


def test_verify_clean_booking(tmp_path):
    path = seeded(tmp_path)
    booking = mcp_server.book_shipment_impl(
        truck_id="T1", origin="Chennai Port", destination="Guindy",
        pallets=10, db_path=path,
    )
    res = mcp_server.verify_booking_impl(booking["shipment_id"], db_path=path)
    assert res == {"status": "OK", "violations": []}


def test_verify_detects_double_book(tmp_path):
    path = seeded(tmp_path)
    with db.get_session(path) as s:
        s.add(db.Shipment(truck_id="T1", origin="Chennai Port",
                          destination="Guindy", status="BOOKED", cost=100))
        s.add(db.Shipment(truck_id="T1", origin="Chennai Port",
                          destination="Guindy", status="BOOKED", cost=150))
        s.commit()
    res = mcp_server.verify_booking_impl(1, db_path=path)
    assert res["status"] == "ERROR"
    assert res["reason"] == "verifier_violations"
    assert any("double_book" in v for v in res["violations"])


def test_verify_unknown_shipment(tmp_path):
    path = seeded(tmp_path)
    res = mcp_server.verify_booking_impl(999, db_path=path)
    assert res["status"] == "ERROR"
    assert res["reason"] == "not_found"


def test_verify_flags_negative_cost(tmp_path):
    path = seeded(tmp_path)
    with db.get_session(path) as s:
        s.add(db.Shipment(truck_id="T1", origin="A", destination="B",
                          status="BOOKED", cost=-5))
        s.commit()
    res = mcp_server.verify_booking_impl(1, db_path=path)
    assert any("negative_cost" in v for v in res["violations"])


def test_verify_flags_available_truck_with_no_bookings(tmp_path):
    path = seeded(tmp_path)
    booking = mcp_server.book_shipment_impl(
        truck_id="T1", origin="Chennai Port", destination="Guindy",
        pallets=10, db_path=path,
    )
    with db.get_session(path) as s:
        s.get(db.Truck, "T1").available = True  # inconsistent flag
        s.commit()
    res = mcp_server.verify_booking_impl(booking["shipment_id"], db_path=path)
    assert any("truck_flag" in v for v in res["violations"])
