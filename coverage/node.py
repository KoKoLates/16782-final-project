import random
from typing import List, Tuple
from core import Env

class Particle:
    def __init__(self, env: Env, robot_signal_range: int):
        self.w, self.h = env.shape
        self.robots_number = env.robots_number
        self.robot_signal_range = robot_signal_range

        self.position: List[Tuple[float, float]] = [
            self.random_position(self.w, self.h, self.robot_signal_range)
            for _ in range(self.robots_number)
        ]

        self.velocity: List[Tuple[float, float]] = [
            (0.0, 0.0)
            for _ in range(self.robots_number)
        ]

        self.best_position = list(self.position)
        self.best_cost = float("inf")

    @staticmethod
    def random_position(w, h, robot_signal_range) -> Tuple[float, float]:
        x = random.uniform(robot_signal_range, w - robot_signal_range)
        y = random.uniform(robot_signal_range, h - robot_signal_range)
        return (x, y)
