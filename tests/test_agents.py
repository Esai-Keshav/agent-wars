import config
from google.adk.agents import LlmAgent

from logistics.sub_agents import freight, verifier, warehouse


def test_specialists_are_llm_agents():
    for a in [freight.freight_agent, warehouse.warehouse_agent, verifier.verifier_agent]:
        assert isinstance(a, LlmAgent)


def test_model_comes_from_config():
    for a in [freight.freight_agent, warehouse.warehouse_agent, verifier.verifier_agent]:
        assert a.model == config.MODEL_NAME


def test_freight_has_mcp_toolset():
    from google.adk.tools.mcp_tool import McpToolset
    assert isinstance(freight.freight_toolset, McpToolset)


def test_verifier_instruction_has_ladder():
    assert "capacity_exceeded" in verifier.verifier_agent.instruction


def test_warehouse_instruction_has_remediation():
    assert "log_self_heal" in warehouse.warehouse_agent.instruction
