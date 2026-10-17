import sqlalchemy

import db
import seed


def make_db(tmp_path):
    path = str(tmp_path / "logistics.db")
    db.init_db(path)
    seed.seed(path)
    return path


def test_seed_trucks_and_warehouses(tmp_path):
    path = make_db(tmp_path)
    with db.get_session(path) as s:
        assert s.query(db.Truck).count() == 4
        assert s.query(db.Warehouse).count() == 4
        assert s.get(db.Truck, "T1").type == "LCV"
        assert all(t.available for t in s.query(db.Truck).all())


def test_seed_idempotent(tmp_path):
    path = make_db(tmp_path)
    seed.seed(path)  # second call must not duplicate rows
    with db.get_session(path) as s:
        assert s.query(db.Truck).count() == 4
        assert s.query(db.Warehouse).count() == 4


def test_all_tables_created(tmp_path):
    path = make_db(tmp_path)
    names = set(sqlalchemy.inspect(db.get_engine(path)).get_table_names())
    assert {"trucks", "shipments", "warehouses", "allocations", "receipts",
            "self_heal_log"} <= names
