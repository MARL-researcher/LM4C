from functools import partial
import sys
import os

from .multiagentenv import MultiAgentEnv

from .mpe.mpe_wrapper import MPEWrapper
from .hallway import HallwayEnv

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}

REGISTRY["mpe"] = partial(env_fn, env=MPEWrapper)
REGISTRY["hallway"] = partial(env_fn, env=HallwayEnv)

def register_smacv2():
    from envs.smacv2 import smacv2
    REGISTRY["sc2_v2"] = partial(env_fn, env=smacv2)
    if sys.platform == 'linux':
        os.environ.setdefault("SC2PATH", "~/StarCraftII")
