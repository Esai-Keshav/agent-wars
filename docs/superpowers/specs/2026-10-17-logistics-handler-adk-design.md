# Logistics Handler — ADK Multi-Agent System (Design Spec)

**Date:** 2026-10-17
**Event:** Hackathon / AI Agent Challenge (per `req.md`)
**Status:** Approved design, pending implementation plan

## 1. Problem

Build an agentic system that automates a logistics workflow end-to-end. Scope chosen: **procurement and booking** — freight booking plus warehouse allocation for shipments originating in Chennai, using fake seed data. Judges interact via a chat interface; agents research, book, verify, and recover from failures autonomously.

## 2. Requirements (from req.md)

- Multi-agent architecture is mandatory (single agents are discouraged by the brief).
- A feedback/self-healing loop is mandatory: failures must be diagnosed (why) and auto-remediated.
- MCP integration and API calls are scored incentives.
- Multimodal capability (images, documents) earns architectural-complexity points.
- Latency is a scored metric; an end-to-end runnable pipeline is required (missing it disqualifies).
- Submission: GitHub repo + README with an architecture diagram + `.env` instructions for local run.

## 3. Stack

- Python 3.10+, `google-adk`
- Gemini 2.0 Flash via Google AI Studio (`GOOGLE_API_KEY`), built-in `google_search` tool
- SQLAlchemy 2.0 (`logistics.db`) as the fake-data store, models in `db.py`, seeded by `seed.py`
- SQLite MCP server exposed to agents via ADK `McpToolset`
- `adk web` dev UI as the demo interface (no custom frontend)
- `config.py` — single module holding every hardcoded/configuration value (see §3.1)

### 3.1 Configuration (`config.py`)

All hardcoded and tunable values live in one `config.py`; no magic strings or numbers elsewhere. Values come from `os.environ` with defaults defined in `config.py`, so `.env` overrides anything without code changes.

| Group | Contents |
|---|---|
| Model | `MODEL_NAME` (default `"gemini-2.0-flash"`), `GOOGLE_API_KEY` (env only) |
| Paths | `DB_PATH` (`logistics.db`), `RECEIPTS_DIR` (`receipts/`), seed data location |
| Self-heal | `MAX_RETRIES` (2), `RETRY_DELAY_S`, verifier trigger toggle |
| Latency | `MAX_E2E_SECONDS` (30), per-turn duration log toggle |
| Data limits | pallet/truck capacity bounds used by tools and seed script |
| Prompt constants | agent names, instruction fragments kept out of agent files where reused |

## 4. Architecture

```
root coordinator
├── freight_agent
├── warehouse_agent
├── verifier_agent
└── tools: google_search (live Chennai diesel/toll/rate context)
```

All DB access flows through the SQLite MCP server. All persistence tools are MCP tools.

| Agent | Type | Role | Tools |
|---|---|---|---|
| `root` | LlmAgent | Understands the request, routes to specialists, assembles the final answer, folds in live web-search context | `google_search`; transfers to sub-agents |
| `freight_agent` | LlmAgent | Finds trucks/routes (Chennai Port, Avadi, Sriperumbudur corridors), quotes rates, books shipments, cancels on re-book | MCP: `search_freight`, `book_shipment`, `cancel_shipment` |
| `warehouse_agent` | LlmAgent | Allocates storage space at Chennai warehouses, checks pallet availability | MCP: `search_warehouse`, `allocate_space` |
| `verifier_agent` | LlmAgent | Re-reads the DB after every booking; detects capacity conflicts, double bookings, negative stock; flags violations to root | MCP: `verify_booking` |
| (tool, not agent) | — | Reads uploaded receipt/loading-challan images via Gemini Vision, extracts sender/weight/pallet count as structured JSON | `receipt_reader` |

**Flow:** user prompt → root → specialist → SQLite (MCP) → result → root → verifier → root → user confirmation.

## 5. Data Model (SQLite, fake Chennai seed data)

- `trucks` — id, type, capacity_kg, rate_per_km, available
- `shipments` — id, truck_id, origin, destination, status, cost
- `warehouses` — id, name, area_chennai, capacity_pallets, used_pallets
- `allocations` — id, warehouse_id, pallets, order_ref
- `receipts` — id, image_path, shipment_id
- `self_heal_log` — id, agent, error_reason, remediation, outcome, timestamp

Seed script populates trucks and warehouses with plausible Chennai-area values (port, Avadi, Sriperumbudur, Ambattur, Guindy).

## 6. Self-Healing Design (mandatory per req.md)

Every persistence tool is wrapped so it never raises raw exceptions; it returns structured errors:

```json
{"status": "ERROR", "reason": "capacity_exceeded", "detail": "..."}
```

Remediation ladder, max 2 retries per failure:

| Error | Remediation |
|---|---|
| `capacity_exceeded` | Re-search alternative truck/warehouse, retry |
| `double_book` (caught by verifier) | Verifier flags root; root delegates to freight_agent: rollback via `cancel_shipment`, then re-book |
| `db_locked` / timeout | Retry once after short delay |
| unknown | Escalate to root; root reports the failure honestly to the user |

Verifier runs after every successful booking. A verifier catch is treated as an error and re-enters the ladder. Every remediation is recorded in `self_heal_log`; the final answer reports the recovery count to the user. Retry budget is capped so worst-case end-to-end latency stays bounded (~10s overhead).

## 7. Multimodal Flow

User uploads a receipt/loading-challan photo → saved to `receipts/` → `receipt_reader` (Gemini Vision) extracts sender, weight, pallet count → structured JSON → freight/warehouse agents book against the extracted values.

## 8. Demo Script (judge path)

1. "Book shipment of 4 pallets from Chennai Port to a warehouse in Sriperumbudur" → warehouse allocates, freight books, verifier confirms, root summarizes with cost.
2. Upload a challan image → auto-extracted order → booked.
3. "Truck T3 just broke down" → agent cancels, re-books, shows self-heal log entries.

## 9. Verification

- `pytest`: seed DB exists; each tool returns success and structured errors correctly; verifier catches a double-book; retry ladder never exceeds depth 2.
- Smoke test: full demo script runs end-to-end in `adk web`.
- Latency: agent turns log duration; target end-to-end under 30 seconds.

## 10. Deliverables

- Source tree with agents, tools, MCP wiring, seed script
- `README.md` with mermaid architecture diagram (agents, API/tool calls, DB, MCP paths), setup instructions, demo script
- `.env.example`, `requirements.txt`
- Tests (`pytest`)

## 11. Build Order (thin slices)

1. Project scaffold + `config.py` + seed DB → verify: pytest seed test green
2. Freight agent + tools → verify: books a shipment via pytest
3. Warehouse agent + tools → verify: allocates pallets via pytest
4. Root + hierarchy wiring → verify: demo prompt 1 works in `adk web`
5. Verifier + self-healing ladder → verify: fault-injection tests green
6. Receipt tool (multimodal) → verify: demo prompt 2 works
7. README + diagram + .env.example → verify: fresh-clone run from README alone
8. Polish + latency check → verify: e2e under 30s
