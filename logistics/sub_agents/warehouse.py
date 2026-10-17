"""Warehouse allocation specialist agent (spec 4)."""
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

warehouse_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=sys.executable, args=[os.path.abspath(_SERVER)]
        ),
        timeout=5,
    ),
    tool_filter=["search_warehouse", "allocate_space", "log_self_heal"],
)

warehouse_agent = LlmAgent(
    name="warehouse_agent",
    model=config.MODEL_NAME,
    instruction=(
        "You allocate pallet storage in fake Chennai warehouses."
        " Tools search_warehouse, allocate_space, log_self_heal come from an"
        " MCP server backed by SQLite."
        " Workflow: 1) search_warehouse with the requested pallet count."
        " 2) Allocate to the warehouse with the most free pallets that fits"
        " (least cramped), using a short order_ref the coordinator gives you or"
        " ORD-<n> derived from the request."
        " 3) If a tool returns status ERROR, remediate (max 2 retries):"
        " capacity_exceeded -> search again and allocate the next best warehouse;"
        " not_found -> re-check arguments once; db_error -> retry once."
        " 4) After any ERROR you remediated, call log_self_heal with"
        " agent='warehouse_agent', the error reason, what you did, and whether"
        " it succeeded."
        " 5) If retries are exhausted, reply with status ERROR and the original reason."
        " Never raise or fabricate results; only report tool outputs."
        f" You MUST retry at most {config.MAX_RETRIES} times, then escalate."
    ),
    tools=[warehouse_toolset],
)
