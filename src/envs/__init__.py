from functools import partial
import sys
import os

from .multiagentenv import MultiAgentEnv

from .matrix_game import OneStepMatrixGame
from .stag_hunt import StagHunt
from .mpe.mpe_wrapper import MPEWrapper
from .traffic_junction import TrafficJunctionEnv
from .gymma import LBFWrapper, RWareWrapper
from .hallway import HallwayEnv

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}

REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)
REGISTRY["one_step_matrix_game"] = partial(env_fn, env=OneStepMatrixGame)
REGISTRY["mpe"] = partial(env_fn, env=MPEWrapper)
REGISTRY["traffic_junction"] = partial(env_fn, env=TrafficJunctionEnv)
REGISTRY["lbf"] = partial(env_fn, env=LBFWrapper)
REGISTRY["rware"] = partial(env_fn, env=RWareWrapper)
REGISTRY["hallway"] = partial(env_fn, env=HallwayEnv)

# The corresponding SMAC environment for dynamic registration
def register_smacv1():
    from envs.smacv1 import smacv1
    REGISTRY["sc2_v1"] = partial(env_fn, env=smacv1)
    if sys.platform == 'linux':
        os.environ.setdefault("SC2PATH", "~/StarCraftII")

def register_smacv2():
    from envs.smacv2 import smacv2
    REGISTRY["sc2_v2"] = partial(env_fn, env=smacv2)
    if sys.platform == 'linux':
        os.environ.setdefault("SC2PATH", "~/StarCraftII")
