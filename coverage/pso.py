import random
from typing import List, Tuple
from core import Env
from .base import CoverageOptimizer
from .node import Particle

class ParticleSwarmOptimizer(CoverageOptimizer):
    def __init__(
        self, 
        env: Env, 
        particle_count: int = 20,
        max_iter: int = 100,
        omega: float = 0.7,
        c1: float = 1.4,
        c2: float = 1.4
    ):
        super().__init__(env)
        self.particle_count = particle_count
        self.max_iter = max_iter
        self.omega = omega
        self.c1 = c1
        self.c2 = c2

        self.particles = [
            Particle(self.env.robots_number, env)
            for _ in range(particle_count)
        ]

        self.global_best_position = [None] * self.env.robots_number
        self.global_best_cost = float('inf')

    def evaluate(self, positions: List[Tuple[float,float]]) -> float:
        return random.random()
        # cost = 0
        # ...
        # return cost

    def update_particle(self, particle: Particle):
        new_positions = []
        new_velocities = []

        for i in range(particle.robots_number):

            px, py = particle.position[i]
            vx, vy = particle.velocity[i]

            bpx, bpy = particle.best_position[i]
            gbx, gby = self.global_best_position[i]

            r1, r2 = random.random(), random.random()

            new_vx = (
                self.omega * vx +
                self.c1 * r1 * (bpx - px) +
                self.c2 * r2 * (gbx - px)
            )

            new_vy = (
                self.omega * vy +
                self.c1 * r1 * (bpy - py) +
                self.c2 * r2 * (gby - py)
            )

            new_x = px + new_vx
            new_y = py + new_vy

            new_x = max(0, min(self.env.w - 1, new_x))
            new_y = max(0, min(self.env.h - 1, new_y))

            new_positions.append((new_x, new_y))
            new_velocities.append((new_vx, new_vy))

        particle.position = new_positions
        particle.velocity = new_velocities
    
    def process(self) -> List[Tuple[int, int]]:
        self.global_best_position = list(self.particles[0].position)
        self.global_best_cost = self.evaluate(self.global_best_position)
        for iteration in range(self.max_iter):
            for particle in self.particles:
                cost = self.evaluate(particle.position)

                if cost < particle.best_cost:
                    particle.best_cost = cost
                    particle.best_position = list(particle.position)

                if cost < self.global_best_cost:
                    self.global_best_cost = cost
                    self.global_best_position = list(particle.position)
            
            for particle in self.particles:
                self.update_particle(particle)

        result = [(int(x), int(y)) for (x, y) in self.global_best_position]
        return result
        