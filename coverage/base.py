from typing import List, Tuple
from core import Env

class CoverageOptimizer:
    def __init__(self, env: Env):
        self.env = env

    def process(self) -> List[Tuple[int, int]]:
        raise NotImplementedError
    
    