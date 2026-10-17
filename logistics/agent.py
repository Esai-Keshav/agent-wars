"""Root coordinator: routes freight/warehouse work, verifies, summarizes (spec 4)."""
import config
from google.adk.agents import LlmAgent
from google.adk.tools.google_search_tool import GoogleSearchTool
from logistics.sub_agents import freight, verifier, warehouse

# Gemini <3.0 cannot combine built-in search with sub-agent delegation unless
# the multi-tool limit is bypassed (ADK requirement, see GoogleSearchTool docs).
google_search_tool = GoogleSearchTool(bypass_multi_tools_limit=True)

root_agent = LlmAgent(
    name="logistics_coordinator",
    model=config.MODEL_NAME,
    instruction=(
        "You coordinate logistics for the Chennai region.\n"
        "Routing: freight/booking problems -> freight_agent; storage/inventory"
        " placement -> warehouse_agent; whenever a specialist reports a booking,"
        " ask verifier_agent to verify_booking the shipment_id.\n"
        "For a combined request (storage + freight), delegate in this order:"
        " 1) warehouse_agent to allocate space, 2) freight_agent to book transport,"
        " 3) verifier_agent on the resulting shipment_id.\n"
        "You MUST produce the final answer yourself. After a specialist reports,"
        " continue the pipeline; never let a specialist's reply be the final answer.\n"
        f"Self-healing (mandatory): if a specialist reports status ERROR, tell it"
        f" to remediate (its instructions allow max {config.MAX_RETRIES} retries)."
        f" If retries are exhausted, tell the user exactly what failed and why,"
        f" honestly.\n"
        "Final answer: one short summary with truck + warehouse ids, order ref,"
        " total cost, verification result, and how many failures were"
        " auto-recovered (from log_self_heal entries the specialists report).\n"
        "Use google_search only to look up current Chennai diesel/toll/rate"
        " context if a user asks about market rates; never block a booking on it."
    ),
    tools=[google_search_tool],
    sub_agents=[freight.freight_agent, warehouse.warehouse_agent, verifier.verifier_agent],
)
