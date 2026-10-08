from .cola_run import run as cola_run
from .sr_run import run as sr_run
from .t2mac_run import run as t2mac_run
from .tgcnet_run import run as tgcnet_run
from .masia_run import msra_run as masia_run
from .laies_run import run as laies_run
from .lm4c_run import run as lm4c_run

REGISTRY = {}

REGISTRY["cola_run"] = cola_run
REGISTRY["sr_run"] = sr_run
REGISTRY["t2mac_run"] = t2mac_run
REGISTRY["tgcnet_run"] = tgcnet_run
REGISTRY["masia_run"] = masia_run
REGISTRY["laies_run"] = laies_run
REGISTRY["lm4c_run"] = lm4c_run
