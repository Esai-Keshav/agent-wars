# Hackathon / AI Agent Challenge: Technical Requirements & Event Brief

---

## 1. Problem Statement
* **Domain:** Logistics & Supply Chain Automation ("Logistics Handler")
* **Core Objective:** Build an AI Agent (or multi-agent system) capable of fully automating an end-to-end logistics workflow.
* **Scope & Examples:**
  * Warehouse management and inventory dispatch.
  * Goods tracking, sorting, and delivery routing (e.g., similar to operations at Amazon, Swiggy, Zomato, etc.).
  * Automated order processing, stock movements, and end-of-line fulfillment.
* *Note:* The problem statement in the qualifier stage is intentionally open-ended within logistics to encourage creative architectural solutions.

---

## 2. Technical & Agent Architecture Requirements

### 2.1 Multi-Agent & Orchestration
* **System Design:** Single-agent implementations ("one-man army") are strongly discouraged due to poor scaling and lack of modularity. Multi-agent architecture is expected.
* **Agent Types:**
  * **Multimodal Agents:** Agents capable of ingesting and handling diverse input/output data (text, images, sensor/inventory data, etc.).
  * **Conversational Agents:** Natural language interaction capabilities where applicable.
* **Complexity vs. Efficiency:** More complex architectures yield higher scoring potential, provided execution efficiency is maintained.

### 2.2 Integrations & Tooling
* **API Calls:** Seamless integration with any relevant REST/third-party APIs required by your problem statement.
* **MCP (Model Context Protocol):** Support for MCP integration is explicitly welcomed and incentivized.

### 2.3 Feedback Loop & Self-Healing
* **Mandatory Loop:** The system must incorporate a feedback/evaluation mechanism.
* **Self-Healing Capabilities:**
  * If an agent fails, encounters an exception, or gets an invalid response, it must diagnose the failure cause (`why`) and attempt automated remediation/retry.
  * Self-healing is critical for advancing beyond qualifiers into the semifinals and finals.

---

## 3. Evaluation & Scoring Metrics

| Metric | Details |
| :--- | :--- |
| **Architectural Complexity** | Weight given to multi-agent design, specialized roles, multimodal support, and tool integrations (MCP/APIs). |
| **Execution Time / Latency** | AI execution time is a core quantitative parameter. Faster end-to-end execution at equivalent complexity is scored higher. |
| **Self-Healing & Robustness** | Ability to gracefully handle edge cases, tool call errors, and agent failure states. |
| **Flow Completeness** | Systems missing an end-to-end runnable pipeline are disqualified immediately. |
| **Submission Speed** | First-come, first-served bonus/tie-breaker; early submissions are strongly favored. |

---

## 4. Repository & Submission Guidelines

### 4.1 GitHub Repository
* **Platform:** GitHub (Public or Private repository).
  * *If private:* Explicit read access must be granted to the evaluation team.
* **Execution Options:**
  1. **Hosted:** Live deployed endpoint or hosted environment.
  2. **Local Evaluation:** Provide full repository code along with environment variables (`.env` file / instructions / required API keys) so evaluators can run the solution locally via automation scripts.

### 4.2 Documentation (`README.md`)
* The `README.md` acts as the primary evaluation overview for your submission.
* **Mandatory Diagram:** Must include an architecture diagram / flowchart / illustration depicting:
  * Complete starting point to ending point.
  * Agent communication pathways.
  * API calls, database interactions, and MCP server integrations.

### 4.3 Form Submission
* Submit via the designated Google Form / QR code link.
* Ensure accurate repository link and contact phone number are provided for selection calls.

---

## 5. Team Composition & Event Schedule

### 5.1 Team Rules
* **Team Size:** Minimum 2 members, maximum 4 members.
  * *Solo entries:* Permitted but discouraged due to strict time limits.
  * *Cross-functional teams:* Student-professional collaborations are actively encouraged.

### 5.2 Rounds & Elimination Structure
* **Qualifiers:**
  * Hard cutoff: **Top 50%** of teams advance; the bottom half is eliminated.
  * Qualifier submissions close at approximately **2:00 PM** (hard deadline before 2:30 PM).
* **Semifinals:**
  * Time window: **2:30 PM – 4:30 PM**.
  * Evaluates the **Top 5** qualifying teams.
* **Finals:**
  * Final showdown between the **Top 2** (or Top 3, schedule permitting) teams with an advanced problem statement.
