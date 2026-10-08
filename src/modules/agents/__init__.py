REGISTRY = {}

from .cola_agent import COLAAgent
from .sr_agent import SRAgent
from .t2mac_agent import T2MACAgent
from .tgcnet_agent import TGCNetAgent
from .masia_agent import MASIAAgent
from .laies_agent import LAIESAgent
from .lm4c_agent import LM4CAgent

REGISTRY['cola'] = COLAAgent
REGISTRY["sr_agent"] = SRAgent
REGISTRY["t2mac"] = T2MACAgent
REGISTRY["tgcnet"] = TGCNetAgent
REGISTRY["masia"] = MASIAAgent
REGISTRY["laies"] = LAIESAgent
REGISTRY["lm4c"] = LM4CAgent
