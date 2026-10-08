REGISTRY = {}

from .cola_controller import COLAMAC
from .sr_controller import SRMAC
from .t2mac_controller import T2MACMAC
from .tgcnet_controller import TGCNetMAC
from .masia_controller import MASIAMAC
from .laies_controller import LAIESMAC
from .lm4c_controller import LM4CMAC

REGISTRY['cola_mac'] = COLAMAC
REGISTRY['sr_mac'] = SRMAC
REGISTRY['t2mac_mac'] = T2MACMAC
REGISTRY['tgcnet_mac'] = TGCNetMAC
REGISTRY["masia_mac"] = MASIAMAC
REGISTRY["laies_mac"] = LAIESMAC
REGISTRY["lm4c_mac"] = LM4CMAC

