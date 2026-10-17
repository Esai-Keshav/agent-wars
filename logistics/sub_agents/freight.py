"""Freight booking specialist agent (spec 4)."""
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

freight_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=sys.executable, args=[os.path.abspath(_SERVER)]
        ),
        timeout=5,
    ),
    tool_filter=["search_freight", "book_shipment", "cancel_shipment", "log_self_heal"],
)

freight_agent = LlmAgent(
    name="freight_agent",
    model=config.MODEL_NAME,
    instruction=(
        "You book truck freight for shipments in the Chennai area."
        " Tools search_freight, book_shipment, cancel_shipment, log_self_heal"
        " come from an MCP server backed by SQLite."
        " Workflow: 1) search_freight with the payload weight (pallets * 250 kg)."
        " 2) Book the cheapest truck that fits, with the exact origin/destination."
        " 3) If a tool returns status ERROR, remediate (max 2 retries):"
        " capacity_exceeded/truck_unavailable -> search again and book the next best truck;"
        " not_found -> re-check arguments once; db_error -> retry once."
        " 4) After any ERROR you remediated, call log_self_heal with agent='freight_agent',"
        " the error reason, what you did, and whether it succeeded."
        " 5) If retries are exhausted, reply with status ERROR and the original reason."
        " Never raise or fabricate results; only report tool outputs."
        f" You MUST retry at most {config.MAX_RETRIES} times, then escalate."
        " For photo challans, the user gives an image path; call read_receipt(path),"
        " then proceed with the extracted pallets/weight/destination as a normal booking."
    ),
    tools=[freight_toolset],
)
