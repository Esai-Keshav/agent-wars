"""Load fake Chennai logistics data into the DB. Idempotent (spec 5)."""
import argparse

import config
import db


def seed(db_path: str = "") -> dict:
    db.init_db(db_path)
    with db.get_session(db_path) as s:
        trucks_added = 0
        if s.query(db.Truck).count() == 0:
            for t in config.SEED_TRUCKS:
                s.add(db.Truck(**t))
            trucks_added = len(config.SEED_TRUCKS)
        whs_added = 0
        if s.query(db.Warehouse).count() == 0:
            for w in config.SEED_WAREHOUSES:
                s.add(db.Warehouse(**w))
            whs_added = len(config.SEED_WAREHOUSES)
        s.commit()
    return {"status": "OK", "trucks": trucks_added, "warehouses": whs_added}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Load fake Chennai logistics data")
    ap.add_argument("--db", default=config.DB_PATH, help="path to logistics.db")
    args = ap.parse_args()
    print(seed(args.db))
