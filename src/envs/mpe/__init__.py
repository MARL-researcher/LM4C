from . import core, scenario

from gym.envs.registration import register
from . import scenarios

import importlib
# Multiagent envs
# ----------------------------------------

_particles = {
    "simple_tag_0vis_colli": "SimpleTag_0vis_colli-v0",
    "simple_spread_0vis": "SimpleSpread_0vis-v0",
}

for scenario_name, gymkey in _particles.items():
    scenario_module = importlib.import_module("envs.mpe.scenarios."+scenario_name)
    scenario_aux = scenario_module.Scenario()
    world = scenario_aux.make_world()

    # Registers multi-agent particle environments:
    register(
        gymkey,
        entry_point="envs.mpe.environment:MultiAgentEnv",
        kwargs={
            "world": world,
            "reset_callback": scenario_aux.reset_world,
            "reward_callback": scenario_aux.reward,
            "observation_callback": scenario_aux.observation,
            "world_info_callback": getattr(scenario_aux, "world_benchmark_data", None)
        },
    )