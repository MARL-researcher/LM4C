REGISTRY = {}

from .lm4c_parallel_runner import ParallelRunner as LM4CParallelRunner
REGISTRY['lm4c_parallel'] = LM4CParallelRunner
