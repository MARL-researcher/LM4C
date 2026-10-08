from .centralV import CentralVCritic
from .research_centralV import ResearchCentralVCritic
from .maa2c_research_critic import MAA2CResearchCritic
from .centralV_rnn_critic import CentralVRNNCritic

from .centralV_rcritic import RCentralVCritic

REGISTRY = {}

REGISTRY["cv_critic"] = CentralVCritic
REGISTRY["research_cv_critic"] = ResearchCentralVCritic
REGISTRY["maa2c_research_critic"] = MAA2CResearchCritic
REGISTRY["centralV_rnn_critic"] = CentralVRNNCritic
REGISTRY["rcritic"] = RCentralVCritic