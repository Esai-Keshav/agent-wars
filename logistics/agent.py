"""Root coordinator: routes freight/warehouse work, verifies, summarizes (spec 4)."""
import config
from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from logistics.sub_agents import freight, verifier, warehouse

root_agent = LlmAgent(
    name="logistics_coordinator",
    model=config.MODEL_NAME,
    instruction=(
        "You coordinate logistics for the Chennai region.\n"
        "Routing: freight/booking problems -> freight_agent; storage/inventory"
        " placement -> warehouse_agent; whenever a specialist reports a booking,"
        " ask verifier_agent to verify_booking the shipment_id.\n"
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
    tools=[google_search],
    sub_agents=[freight.freight_agent, warehouse.warehouse_agent, verifier.verifier_agent],
)
