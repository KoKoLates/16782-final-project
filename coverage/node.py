import random
from typing import List, Tuple
from core import Env

class Particle:
    def __init__(self, robots_number: int, env: Env):
        self.env = env
        self.robots_number = robots_number

        self.position: List[Tuple[float, float]] = [
            self.random_position()
            for _ in range(robots_number)
        ]

        self.velocity: List[Tuple[float, float]] = [
            (0.0, 0.0)
            for _ in range(robots_number)
        ]

        self.best_position = list(self.position)
        self.best_cost = float("inf")

    def random_position(self) -> Tuple[float, float]:
        x = random.uniform(0, self.env.w - 1)
        y = random.uniform(0, self.env.h - 1)
        return (x, y)
