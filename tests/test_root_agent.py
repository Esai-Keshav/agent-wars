from google.adk.agents import LlmAgent

import logistics


def test_root_agent_exists():
    assert isinstance(logistics.root_agent, LlmAgent)
    assert logistics.root_agent.name == "logistics_coordinator"


def test_sub_agents_wired():
    names = [a.name for a in logistics.root_agent.sub_agents]
    assert names == ["freight_agent", "warehouse_agent", "verifier_agent"]


def test_root_includes_search_tool():
    from google.adk.tools import google_search
    assert google_search in logistics.root_agent.tools


def test_root_instruction_routes():
    assert "freight_agent" in logistics.root_agent.instruction
    assert "verifier_agent" in logistics.root_agent.instruction
