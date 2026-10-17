import subprocess
import sys

import db
import mcp_server
import seed


def seeded(tmp_path):
    path = str(tmp_path / "logi.db")
    db.init_db(path)
    seed.seed(path)
    return path


def test_remediation_ladder_t2_fallback(tmp_path):
    """Agent picks T2 when T1 unavailable: tool layer must allow it."""
    path = seeded(tmp_path)
    with db.get_session(path) as s:
        s.get(db.Truck, "T1").available = False
        s.commit()
    first = mcp_server.book_shipment_impl(
        truck_id="T1", origin="Chennai Port", destination="Guindy",
        pallets=10, db_path=path,
    )
    assert first["status"] == "ERROR"
    second = mcp_server.book_shipment_impl(
        truck_id="T2", origin="Chennai Port", destination="Guindy",
        pallets=10, db_path=path,
    )
    assert second["status"] == "OK"


def test_rollback_then_rebook_on_double_book(tmp_path):
    path = seeded(tmp_path)
    b1 = mcp_server.book_shipment_impl(
        truck_id="T1", origin="Chennai Port", destination="Guindy",
        pallets=10, db_path=path,
    )
    bad = mcp_server.book_shipment_impl(
        truck_id="T1", origin="Chennai Port", destination="Guindy",
        pallets=10, db_path=path,
    )
    assert bad["status"] == "ERROR"  # ladder enters here in real agent flow
    check = mcp_server.verify_booking_impl(b1["shipment_id"], db_path=path)
    assert check["status"] == "OK"
    assert mcp_server.cancel_shipment_impl(b1["shipment_id"], db_path=path)["status"] == "OK"
    rebook = mcp_server.book_shipment_impl(
        truck_id="T1", origin="Chennai Port", destination="Guindy",
        pallets=10, db_path=path,
    )
    assert rebook["status"] == "OK"


def test_retry_sentinel_never_exceeds_two(tmp_path):
    """Config MAX_RETRIES must stay 2 (latency guard, spec 6)."""
    import config
    assert config.MAX_RETRIES == 2


def test_mcp_server_boots_and_lists_tools():
    """MCP server answers initialize over stdio (integration; needs installed deps)."""
    script = """
import asyncio, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(command=sys.executable, args=["mcp_server.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            expected = {"search_freight", "book_shipment", "cancel_shipment",
                        "search_warehouse", "allocate_space", "verify_booking",
                        "log_self_heal"}
            assert expected <= names, names
            print("MCP_TOOLS_OK")

asyncio.run(main())
"""
    res = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=60,
        cwd="/home/esai/Desktop/devfest/war",
    )
    assert res.returncode == 0, res.stderr
    assert "MCP_TOOLS_OK" in res.stdout
