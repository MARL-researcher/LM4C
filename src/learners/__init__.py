from .cola_learner import COLALearner
from .sr_learner import SRLearner
from .t2mac_learner import T2MACLearner
from .tgc_learner import TGCLearner
from .masia_learner import MASIALearner
from .laies_learner import LAIESLearner
from .lm4c_learner import LM4CLearner

REGISTRY = {}

REGISTRY["cola_learner"] = COLALearner
REGISTRY["sr_learner"] = SRLearner
REGISTRY["t2mac_learner"] = T2MACLearner
REGISTRY["tgc_learner"] = TGCLearner
REGISTRY["masia_learner"] = MASIALearner
REGISTRY["laies_learner"] = LAIESLearner
REGISTRY["lm4c_learner"] = LM4CLearner