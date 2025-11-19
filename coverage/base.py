from typing import List, Tuple
from core import Env

class CoverageOptimizer:
    def __init__(self, env: Env):
        self.env = env
        self.w, self.h = env.w, env.h
        self.robots_number = env.robots_number

    def process(self) -> List[Tuple[int, int]]:
        raise NotImplementedError
    
    def evaluate(self, positions: List[Tuple[float, float]]) -> float:
        raise NotImplementedError
    
    def in_obstacle(self, x: float, y: float) -> bool:
        return self.env.is_obstacle(int(x), int(y))

    def clamp(self, x: float, y: float) -> Tuple[float, float]:
        return (
            max(0, min(self.w - 1, x)),
            max(0, min(self.h - 1, y)),
        )