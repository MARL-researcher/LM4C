REGISTRY = {}

from .cola_parallel_runner import ParallelRunner as ColaParallelRunner
REGISTRY['cola_parallel'] = ColaParallelRunner

from .sr_parallel_runner import ParallelRunner as SRParallelRunner
REGISTRY['sr_parallel'] = SRParallelRunner

from .t2mac_parallel_runner import ParallelRunner as T2MACParallelRunner
REGISTRY['t2mac_parallel'] = T2MACParallelRunner

from .tgcnet_parallel_runner import ParallelRunner as TGCNetParallelRunner
REGISTRY['tgcnet_parallel'] = TGCNetParallelRunner

from .masia_parallel_runner import ParallelRunner as MASIAParallelRunner
REGISTRY['masia_parallel'] = MASIAParallelRunner

from .lm4c_parallel_runner import ParallelRunner as LM4CParallelRunner
REGISTRY['lm4c_parallel'] = LM4CParallelRunner

from .laies_parallel_runner import ParallelRunner as LaiESParallelRunner
REGISTRY['laies_parallel'] = LaiESParallelRunner
