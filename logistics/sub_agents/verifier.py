"""Booking verification agent (spec 4, self-healing checkpoint)."""
import os
import sys

import config
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

_SERVER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "mcp_server.py"
)

verifier_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=sys.executable, args=[os.path.abspath(_SERVER)]
        ),
        timeout=5,
    ),
    tool_filter=["verify_booking", "log_self_heal"],
)

verifier_agent = LlmAgent(
    name="verifier_agent",
    model=config.MODEL_NAME,
    instruction=(
        "You verify logistics bookings are internally consistent."
        " After every booking reported by other agents, call verify_booking on"
        " the shipment_id."
        " If the result is status OK with no violations, reply VERIFIED and"
        " include the shipment summary."
        " If status ERROR with reason verifier_violations, list every violation"
        " exactly as given so the coordinator can trigger rollback and re-book."
        " Violations you will see include double_book, capacity_exceeded,"
        " negative_cost, truck_flag, bad_status, missing_truck"
        " (rollback = cancel_shipment on the bad booking, then a fresh booking)."
        " Call log_self_heal with agent='verifier_agent' whenever a violation"
        " you reported leads to a fix."
        " If reason is not_found, re-check the id once; you may retry at most"
        f" {config.MAX_RETRIES} times, then escalate with status ERROR."
        " Never raise or fabricate results; only report tool outputs."
    ),
    tools=[verifier_toolset],
)
