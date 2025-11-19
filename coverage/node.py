import random
from typing import List, Tuple
from core import Env

class Particle:
    def __init__(self, env: Env):
        self.w, self.h = env.shape
        self.robots_number = env.robots_number

        self.position: List[Tuple[float, float]] = [
            self.random_position(self.w, self.h)
            for _ in range(self.robots_number)
        ]

        self.velocity: List[Tuple[float, float]] = [
            (0.0, 0.0)
            for _ in range(self.robots_number)
        ]

        self.best_position = list(self.position)
        self.best_cost = float("inf")

    @staticmethod
    def random_position(w, h) -> Tuple[float, float]:
        x = random.uniform(0, w - 1)
        y = random.uniform(0, h - 1)
        return (x, y)
