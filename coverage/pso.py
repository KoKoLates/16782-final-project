import random
import math
from typing import List, Tuple
from core import Env
from .base import CoverageOptimizer
from .node import Particle

class ParticleSwarmOptimizer(CoverageOptimizer):
    def __init__(
        self, 
        env: Env, 
        particle_count: int = 50,
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
    
    def cost_obstacle(self, positions):
        cost = 0
        for (x, y) in positions:
            if self.env.is_obstacle(int(x), int(y)):
                cost += 1e9
        return cost
    
    def cost_connectivity(self, positions):
        tower = (self.env.w / 2, self.env.h / 2)
        tower_range = 3.0
        robot_range = 3.0

        n = len(positions)
        connected = [False] * n

        roots = []
        for i, p in enumerate(positions):
            if math.dist(p, tower) < tower_range:
                roots.append(i)
                connected[i] = True

        if not roots:
            return 1e6

        queue = roots[:]
        while queue:
            i = queue.pop(0)
            for j in range(n):
                if not connected[j] and math.dist(positions[i], positions[j]) < robot_range:
                    connected[j] = True
                    queue.append(j)

        penalty = 0
        for ok in connected:
            if not ok:
                penalty += 1e5

        return penalty

    def evaluate(self, positions: List[Tuple[float,float]]) -> float:
        cost  = 0
        cost += self.cost_obstacle(positions)
        cost += self.cost_connectivity(positions)
        # cost += self.cost_coverage(positions)
        return cost

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
        history_global_best_cost = self.global_best_cost
        stall = 0
        epsilon = 1e-6
        patience = 10

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

            delta_cost = abs(history_global_best_cost - self.global_best_cost)

            if delta_cost < epsilon:
                stall += 1
            else:
                stall = 0

            if stall >= patience:
                break

            history_global_best_cost = self.global_best_cost
        
        print("Iteration Count:", iteration)
        print("Final Best Cost:", self.global_best_cost)
        result = [(int(x), int(y)) for (x, y) in self.global_best_position]
        return result
        