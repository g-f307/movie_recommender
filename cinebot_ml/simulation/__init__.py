"""Agentes e utilidades para experimentos sintéticos."""
from cinebot_ml.simulation.agents import (
    AgentConfigError, DeclaredPreference, LatentPreference, ProfileObservationConfig,
    SyntheticAgent, SyntheticJudgment, build_agent, build_agent_cohort,
    build_declared_preference, load_agent_config, load_profile_observation_config,
    profile_observation_manifest,
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
    "AgentConfigError", "DeclaredPreference", "LatentPreference",
    "ProfileObservationConfig", "SimulationConfig", "SimulationResult",
    "SimulationScenario", "SimulationStep", "SyntheticAgent", "SyntheticJudgment",
    "TemporalSimulationError", "TemporalSimulator", "build_agent", "build_agent_cohort",
    "build_declared_preference", "load_agent_config", "load_profile_observation_config",
    "load_simulation_config", "profile_observation_manifest", "write_simulation_result",
]
