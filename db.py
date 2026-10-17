"""SQLAlchemy ORM models + session helpers for logistics.db (spec 5)."""
from sqlalchemy import (Boolean, DateTime, Float, Integer, String, create_engine,
                        func)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

import config


class Base(DeclarativeBase):
    pass


class Truck(Base):
    __tablename__ = "trucks"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    type: Mapped[str] = mapped_column(String)
    capacity_kg: Mapped[float] = mapped_column(Float)
    rate_per_km: Mapped[float] = mapped_column(Float)
    available: Mapped[bool] = mapped_column(Boolean, default=True)


class Shipment(Base):
    __tablename__ = "shipments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    truck_id: Mapped[str] = mapped_column(String)
    origin: Mapped[str] = mapped_column(String)
    destination: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String)
    cost: Mapped[float] = mapped_column(Float)


class Warehouse(Base):
    __tablename__ = "warehouses"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    area_chennai: Mapped[str] = mapped_column(String)
    capacity_pallets: Mapped[int] = mapped_column(Integer)
    used_pallets: Mapped[int] = mapped_column(Integer, default=0)


class Allocation(Base):
    __tablename__ = "allocations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    warehouse_id: Mapped[str] = mapped_column(String)
    pallets: Mapped[int] = mapped_column(Integer)
    order_ref: Mapped[str] = mapped_column(String)


class Receipt(Base):
    __tablename__ = "receipts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_path: Mapped[str] = mapped_column(String)
    shipment_id: Mapped[int | None] = mapped_column(Integer, nullable=True)


class SelfHealLog(Base):
    __tablename__ = "self_heal_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent: Mapped[str] = mapped_column(String)
    error_reason: Mapped[str] = mapped_column(String)
    remediation: Mapped[str] = mapped_column(String)
    outcome: Mapped[str] = mapped_column(String)
    ts: Mapped[object] = mapped_column(DateTime, server_default=func.now())


_ENGINES = {}


def get_engine(db_path: str = ""):
    path = db_path or config.DB_PATH
    if path not in _ENGINES:
        _ENGINES[path] = create_engine(f"sqlite:///{path}", future=True)
    return _ENGINES[path]


def init_db(db_path: str = "") -> None:
    Base.metadata.create_all(get_engine(db_path))


def get_session(db_path: str = "") -> Session:
    return Session(bind=get_engine(db_path), future=True)
