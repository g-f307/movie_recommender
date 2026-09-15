"""Agentes e utilidades para experimentos sintéticos."""
from cinebot_ml.simulation.agents import (
    AgentConfigError, LatentPreference, SyntheticAgent, SyntheticJudgment,
    build_agent, build_agent_cohort, load_agent_config,
)

__all__ = ["AgentConfigError", "LatentPreference", "SyntheticAgent", "SyntheticJudgment", "build_agent", "build_agent_cohort", "load_agent_config"]
