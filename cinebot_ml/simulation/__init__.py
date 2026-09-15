"""Agentes e utilidades para experimentos sintéticos."""
from cinebot_ml.simulation.agents import (
    AgentConfigError, LatentPreference, SyntheticAgent, SyntheticJudgment,
    build_agent, build_agent_cohort, load_agent_config,
)
from cinebot_ml.simulation.temporal import (
    SimulationConfig,
    SimulationResult,
    SimulationScenario,
    SimulationStep,
    TemporalSimulationError,
    TemporalSimulator,
    load_simulation_config,
    write_simulation_result,
)

__all__ = [
    "AgentConfigError", "LatentPreference", "SimulationConfig", "SimulationResult",
    "SimulationScenario", "SimulationStep", "SyntheticAgent", "SyntheticJudgment",
    "TemporalSimulationError", "TemporalSimulator", "build_agent", "build_agent_cohort",
    "load_agent_config", "load_simulation_config", "write_simulation_result",
]
