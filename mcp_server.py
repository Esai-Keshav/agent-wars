"""Custom MCP server exposing logistics ORM operations to ADK agents (spec 4).

Every tool never raises raw exceptions (spec 6): it returns a structured dict.
Launch over stdio: python mcp_server.py
"""
import time

from mcp.server.mcpserver import MCPServer
from sqlalchemy.exc import OperationalError

import config
import db
import seed

mcp = MCPServer("logistics")

_DB_LOCK_RETRIES = 1  # internal db_locked retry, before agent-level ladder


def _err(reason: str, detail: str) -> dict:
    return {"status": "ERROR", "reason": reason, "detail": detail}


def search_freight_impl(min_capacity_kg: float, db_path: str = "") -> dict:
    try:
        with db.get_session(db_path) as s:
            trucks = (
                s.query(db.Truck)
                .filter(db.Truck.available.is_(True),
                        db.Truck.capacity_kg >= min_capacity_kg)
                .order_by(db.Truck.rate_per_km)
                .all()
            )
            payload = [{"id": t.id, "type": t.type, "capacity_kg": t.capacity_kg,
                        "rate_per_km": t.rate_per_km} for t in trucks]
        return {"status": "OK", "trucks": payload}
    except OperationalError as exc:
        if "locked" in str(exc) and _DB_LOCK_RETRIES:
            time.sleep(config.RETRY_DELAY_S)
            return search_freight_impl(min_capacity_kg, db_path)
        return _err("db_error", str(exc))


def corridor_km(origin: str, destination: str) -> int:
    return config.CORRIDOR_KM.get(origin, {}).get(
        destination, config.DEFAULT_CORRIDOR_KM
    )


def book_shipment_impl(
    truck_id: str, origin: str, destination: str, pallets: float, db_path: str = ""
) -> dict:
    weight = pallets * config.PALLET_WEIGHT_KG
    km = corridor_km(origin, destination)
    try:
        with db.get_session(db_path) as s:
            truck = s.get(db.Truck, truck_id)
            if truck is None:
                return _err("not_found", f"truck {truck_id}")
            if not truck.available:
                return _err("truck_unavailable", f"truck {truck_id} not available")
            if weight > truck.capacity_kg:
                return _err(
                    "capacity_exceeded",
                    f"need {weight}kg, truck {truck_id} holds {truck.capacity_kg}kg",
                )
            shipment = db.Shipment(truck_id=truck_id, origin=origin,
                                   destination=destination, status="BOOKED",
                                   cost=km * truck.rate_per_km)
            s.add(shipment)
            truck.available = False
            s.commit()
            shipment_id = shipment.id
            cost = shipment.cost
        return {
            "status": "OK", "shipment_id": shipment_id, "truck_id": truck_id,
            "origin": origin, "destination": destination, "weight_kg": weight,
            "distance_km": km, "cost": cost,
        }
    except OperationalError as exc:
        if "locked" in str(exc) and _DB_LOCK_RETRIES:
            time.sleep(config.RETRY_DELAY_S)
            return book_shipment_impl(truck_id, origin, destination, pallets, db_path)
        return _err("db_error", str(exc))


def cancel_shipment_impl(shipment_id: int, db_path: str = "") -> dict:
    try:
        with db.get_session(db_path) as s:
            ship = s.get(db.Shipment, shipment_id)
            if ship is None:
                return _err("not_found", f"shipment {shipment_id}")
            truck_id_bak = ship.truck_id
            ship.status = "CANCELLED"
            other_booked = (
                s.query(db.Shipment)
                .filter(db.Shipment.truck_id == ship.truck_id,
                        db.Shipment.status == "BOOKED",
                        db.Shipment.id != shipment_id)
                .count()
            )
            if other_booked == 0:
                truck = s.get(db.Truck, truck_id_bak)
                if truck is not None:
                    truck.available = True
            s.commit()
        return {"status": "OK", "shipment_id": shipment_id, "truck_id": truck_id_bak}
    except OperationalError as exc:
        if "locked" in str(exc) and _DB_LOCK_RETRIES:
            time.sleep(config.RETRY_DELAY_S)
            return cancel_shipment_impl(shipment_id, db_path)
        return _err("db_error", str(exc))


@mcp.tool()
def search_freight(min_capacity_kg: float) -> dict:
    """Find available trucks (fake Chennai fleet) that can carry the given weight."""
    return search_freight_impl(min_capacity_kg)


@mcp.tool()
def book_shipment(truck_id: str, origin: str, destination: str, pallets: float) -> dict:
    """Book a truck for a shipment; marks the truck unavailable and returns cost."""
    return book_shipment_impl(truck_id, origin, destination, pallets)


@mcp.tool()
def cancel_shipment(shipment_id: int) -> dict:
    """Cancel a booked shipment and free the truck."""
    return cancel_shipment_impl(shipment_id)


if __name__ == "__main__":
    seed.seed(config.DB_PATH)
    mcp.run()
