"""Single source of truth for every hardcoded/configurable value (spec 3.1)."""

import os

from dotenv import load_dotenv

load_dotenv()

# Model
MODEL_NAME = "gemini-3.5-flash-lite"
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# Paths
DB_PATH = os.environ.get("LOGISTICS_DB_PATH", "logistics.db")
RECEIPTS_DIR = os.environ.get("RECEIPTS_DIR", "receipts")
SAMPLE_RECEIPT = os.path.join(RECEIPTS_DIR, "sample_challan.jpg")

# Self-healing
MAX_RETRIES = 2
RETRY_DELAY_S = 1.0

# Latency
MAX_E2E_SECONDS = 30

# Fake Chennai geography: origin -> destination road-distance km
CORRIDOR_KM = {
    "Chennai Port": {
        "Sriperumbudur": 45,
        "Avadi": 22,
        "Ambattur": 20,
        "Guindy": 12,
    },
    "Guindy": {
        "Sriperumbudur": 33,
        "Avadi": 15,
        "Ambattur": 14,
    },
}
DEFAULT_CORRIDOR_KM = 30  # fallback when pair not in CORRIDOR_KM

# Seed data (fake Chennai operations)
SEED_TRUCKS = [
    {
        "id": "T1",
        "type": "LCV",
        "capacity_kg": 3500,
        "rate_per_km": 120,
        "available": True,
    },
    {
        "id": "T2",
        "type": "Container",
        "capacity_kg": 9000,
        "rate_per_km": 210,
        "available": True,
    },
    {
        "id": "T3",
        "type": "LCV",
        "capacity_kg": 4000,
        "rate_per_km": 130,
        "available": True,
    },
    {
        "id": "T4",
        "type": "Tipper",
        "capacity_kg": 12000,
        "rate_per_km": 260,
        "available": True,
    },
]
SEED_WAREHOUSES = [
    {
        "id": "W1",
        "name": "Sriperumbudur Hub",
        "area_chennai": "Sriperumbudur",
        "capacity_pallets": 120,
        "used_pallets": 40,
    },
    {
        "id": "W2",
        "name": "Avadi Depot",
        "area_chennai": "Avadi",
        "capacity_pallets": 80,
        "used_pallets": 20,
    },
    {
        "id": "W3",
        "name": "Ambattur Warehouse",
        "area_chennai": "Ambattur",
        "capacity_pallets": 60,
        "used_pallets": 55,
    },
    {
        "id": "W4",
        "name": "Guindy Cold-Chain",
        "area_chennai": "Guindy",
        "capacity_pallets": 150,
        "used_pallets": 10,
    },
]

# Booking assumptions
PALLET_WEIGHT_KG = 250
